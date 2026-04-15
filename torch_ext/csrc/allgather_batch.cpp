// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PrivateUse1 (NPU) and Meta implementations for
// torch.ops.custom_comm.allgather_batch.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/StringUtil.h>
#include <mutex>
#include <unordered_map>

#include "hccl_custom_allgather_batch.h"
#include "common.h"

// ============================================================
// Forward declarations (symbols from libhcomm.so)
// Avoids including hcom.h which has heavy transitive deps.
// ============================================================

extern "C" {
HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle);
}

// ============================================================
// torch_npu stream API (provided by torch_npu headers via NpuExtension)
// ============================================================

#ifndef CUSTOM_COMM_NO_NPU
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <hccl/hccl.h>  // for HcclAllGather
#endif

// ============================================================
// Torch error-checking macro for HCCL calls
// ============================================================

#define HCCL_TORCH_CHECK(call)                                      \
    do {                                                            \
        HcclResult _ret = (call);                                   \
        TORCH_CHECK(_ret == HCCL_SUCCESS,                           \
            "HCCL error: ", static_cast<int>(_ret),                 \
            " at ", __FILE__, ":", __LINE__);                        \
    } while (0)

// ============================================================
// Comm handle cache + dtype mapping
// ============================================================

namespace {

HcclComm GetCachedComm(c10::string_view group) {
    // Fast path: thread-local single-entry cache, no lock.
    // Covers the common case of repeated calls with the same group name.
    thread_local std::string tl_key;
    thread_local HcclComm    tl_comm = nullptr;
    if (C10_LIKELY(tl_comm != nullptr &&
                   group.size() == tl_key.size() &&
                   std::memcmp(group.data(), tl_key.data(), tl_key.size()) == 0)) {
        return tl_comm;
    }

    // Slow path: first call or group name changed.
    static std::mutex mu;
    static std::unordered_map<std::string, HcclComm> cache;

    std::string key(group.data(), group.size());
    HcclComm comm = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(key);
        if (it != cache.end()) {
            comm = it->second;
        }
    }
    if (!comm) {
        HCCL_TORCH_CHECK(HcomGetCommHandleByGroup(key.c_str(), &comm));
        TORCH_CHECK(comm != nullptr,
                    "Failed to get HcclComm for group: ", key);
        std::lock_guard<std::mutex> lk(mu);
        cache[key] = comm;
    }

    tl_key = std::move(key);
    tl_comm = comm;
    return comm;
}

HcclDataType ScalarTypeToHcclDtype(c10::ScalarType st) {
    switch (st) {
        case c10::ScalarType::Char:   return HCCL_DATA_TYPE_INT8;   // int8
        case c10::ScalarType::Byte:   return HCCL_DATA_TYPE_UINT8;  // uint8
        case c10::ScalarType::Short:  return HCCL_DATA_TYPE_INT16;
        case c10::ScalarType::Half:   return HCCL_DATA_TYPE_FP16;
        case c10::ScalarType::Float:  return HCCL_DATA_TYPE_FP32;
        case c10::ScalarType::Int:    return HCCL_DATA_TYPE_INT32;
        case c10::ScalarType::Long:   return HCCL_DATA_TYPE_INT64;
        case c10::ScalarType::Double: return HCCL_DATA_TYPE_FP64;
        case c10::ScalarType::BFloat16: return HCCL_DATA_TYPE_BFP16;
        case c10::ScalarType::Float8_e4m3fn:  return HCCL_DATA_TYPE_FP8E4M3;
        case c10::ScalarType::Float8_e5m2:    return HCCL_DATA_TYPE_FP8E5M2;
        default:
            TORCH_CHECK(false, "Unsupported dtype for allgather_batch: ",
                        c10::toString(st));
    }
}

}  // namespace

// ============================================================
// PrivateUse1 (NPU device) implementation
// ============================================================

std::vector<at::Tensor> allgather_batch_npu(
    const std::vector<at::Tensor> &inputs,
    c10::string_view hcom,
    int64_t world_size) {
#ifdef CUSTOM_COMM_NO_NPU
    TORCH_CHECK(false, "allgather_batch_npu requires NPU device");
#else
    RECORD_FUNCTION("custom_comm::allgather_batch", {});

    TORCH_CHECK(!inputs.empty(), "inputs must be non-empty");
    TORCH_CHECK(inputs.size() <= MAX_DESC_COUNT,
                "inputs.size() (", inputs.size(), ") exceeds MAX_DESC_COUNT (", MAX_DESC_COUNT, ")");
    TORCH_CHECK(world_size > 0, "world_size must be positive");

    const uint32_t descCount = static_cast<uint32_t>(inputs.size());

    // 1. Resolve hcom group name -> HcclComm handle (cached after first call)
    HcclComm comm = GetCachedComm(hcom);

    // 2. Get current NPU stream
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);

    // 3. Pack tensors as uint8 rows, cat, single AllGather, split back.
    //    Uses PyTorch caching allocator (no aclrtMalloc/Free per call).
    //    Unpack: narrow(dim=1) is non-contiguous, so .contiguous() copies
    //    each output tensor on device (not zero-copy).
    const int64_t N = inputs[0].size(0);
    std::vector<at::Tensor> byte_views;
    std::vector<int64_t> byte_widths;
    std::vector<std::vector<int64_t>> orig_shapes;

    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto &inp = inputs[i];
        TORCH_CHECK(inp.dim() >= 1 && inp.is_contiguous());
        TORCH_CHECK(inp.size(0) == N, "All inputs must have same dim-0");
        int64_t bw = static_cast<int64_t>(inp.nbytes()) / N;
        byte_views.push_back(
            at::from_blob(inp.data_ptr(), {N, bw},
                          inp.options().dtype(at::kByte)));
        byte_widths.push_back(bw);
        orig_shapes.push_back(inp.sizes().vec());
    }

    auto packed = at::cat(byte_views, /*dim=*/1);            // (N, sum_bw)
    auto gathered = at::empty({N * world_size, packed.size(1)},
                              packed.options());

    // Single AllGather call (same as what torch.distributed uses)
    HCCL_TORCH_CHECK(HcclAllGather(
        packed.data_ptr(), gathered.data_ptr(),
        static_cast<uint64_t>(packed.numel()),
        HCCL_DATA_TYPE_UINT8, comm, stream));

    // 4. Split gathered along columns and view back to original dtypes
    std::vector<at::Tensor> outputs;
    int64_t col = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        int64_t bw = byte_widths[i];
        auto slice = gathered.narrow(/*dim=*/1, col, bw).contiguous();
        col += bw;
        auto out_shape = inputs[i].sizes().vec();
        out_shape[0] *= world_size;
        outputs.push_back(
            slice.view(inputs[i].scalar_type()).reshape(out_shape));
    }
    return outputs;
#endif
}

// ============================================================
// Meta implementation (shape inference, no device access)
// ============================================================

static std::vector<at::Tensor> allgather_batch_meta(
    const std::vector<at::Tensor> &inputs,
    c10::string_view /*hcom*/,
    int64_t world_size) {

    TORCH_CHECK(!inputs.empty(), "inputs must be non-empty");
    TORCH_CHECK(world_size > 0, "world_size must be positive");

    std::vector<at::Tensor> outputs;
    outputs.reserve(inputs.size());

    for (const auto &input : inputs) {
        TORCH_CHECK(input.dim() >= 1,
                    "allgather_batch inputs must be at least 1-dimensional");
        auto out_sizes = input.sizes().vec();
        out_sizes[0] *= world_size;
        outputs.push_back(at::empty(out_sizes, input.options()));
    }
    return outputs;
}

// ============================================================
// Registration
// ============================================================

TORCH_LIBRARY_IMPL(custom_comm, PrivateUse1, m) {
    m.impl("allgather_batch", &allgather_batch_npu);
}

TORCH_LIBRARY_IMPL(custom_comm, Meta, m) {
    m.impl("allgather_batch", &allgather_batch_meta);
}

// ============================================================
// pybind11 direct entry (bypasses Dispatcher for eager mode)
// ============================================================

#include <torch/python.h>

// In-place variant: outputs pre-allocated by caller, no return value overhead.
static void allgather_batch_inplace(
    const std::vector<at::Tensor> &inputs,
    std::vector<at::Tensor> &outputs,
    c10::string_view hcom,
    int64_t world_size) {
#ifndef CUSTOM_COMM_NO_NPU
    TORCH_CHECK(!inputs.empty() && inputs.size() == outputs.size());
    TORCH_CHECK(world_size > 0);
    HcclComm comm = GetCachedComm(hcom);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);

    const int64_t N = inputs[0].size(0);
    std::vector<at::Tensor> byte_views;
    std::vector<int64_t> bws;
    for (const auto &inp : inputs) {
        int64_t bw = static_cast<int64_t>(inp.nbytes()) / N;
        byte_views.push_back(
            at::from_blob(inp.data_ptr(), {N, bw},
                          inp.options().dtype(at::kByte)));
        bws.push_back(bw);
    }
    auto packed = at::cat(byte_views, 1);
    auto gathered = at::empty({N * world_size, packed.size(1)},
                              packed.options());
    HCCL_TORCH_CHECK(HcclAllGather(
        packed.data_ptr(), gathered.data_ptr(),
        static_cast<uint64_t>(packed.numel()),
        HCCL_DATA_TYPE_UINT8, comm, stream));

    int64_t col = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto src = gathered.narrow(1, col, bws[i]);
        at::from_blob(outputs[i].data_ptr(), {N * world_size, bws[i]},
                      gathered.options())
            .copy_(src);
        col += bws[i];
    }
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("allgather_batch_eager", &allgather_batch_npu);
    m.def("allgather_batch_inplace", &allgather_batch_inplace);
}
