// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PrivateUse1 (NPU) and Meta implementations for
// custom_comm::allgather_batch.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/StringUtil.h>
#include <mutex>
#include <unordered_map>

#include "hccl_custom_allgather_batch.h"
#include "common.h"
#include "log_util.h"

// ============================================================
// Forward declarations (symbols from libhcomm.so)
// ============================================================

extern "C" {
HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle);
}

// ============================================================
// torch_npu headers (NPU device only)
// ============================================================

#ifndef CUSTOM_COMM_NO_NPU
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <acl/acl_rt.h>
#endif

// ============================================================
// Error-checking macros
// ============================================================

#define HCCL_TORCH_CHECK(call)                                      \
    do {                                                            \
        HcclResult _ret = (call);                                   \
        if (_ret != HCCL_SUCCESS) {                                 \
            CC_LOG_ERROR("HCCL_TORCH_CHECK failed: %s -> %d",       \
                         #call, static_cast<int>(_ret));            \
        }                                                           \
        TORCH_CHECK(_ret == HCCL_SUCCESS,                           \
            "HCCL error ", static_cast<int>(_ret),                  \
            " from ", #call,                                        \
            " at ", __FILE__, ":", __LINE__);                       \
    } while (0)

#define ACL_TORCH_CHECK(call)                                       \
    do {                                                            \
        auto _r = (call);                                           \
        TORCH_CHECK(_r == 0, #call " failed: ", static_cast<int>(_r)); \
    } while (0)

// ============================================================
// Comm handle cache + dtype mapping
// ============================================================

namespace {

// If `group` is a decimal representation of an integer, treat it as the raw
// HcclComm pointer value (i.e. what ProcessGroupHCCL::getHcclComm(rank)
// returns stringified). This bypasses HcomGetCommHandleByGroup entirely,
// which on some torch_npu versions can't resolve pg labels like "group_name_0".
static HcclComm TryResolveCommFromInt(c10::string_view group) {
    if (group.empty()) return nullptr;
    for (char c : group) {
        if (c < '0' || c > '9') return nullptr;
    }
    uint64_t value = 0;
    for (char c : group) {
        value = value * 10 + static_cast<uint64_t>(c - '0');
    }
    if (value == 0) return nullptr;
    return reinterpret_cast<HcclComm>(static_cast<uintptr_t>(value));
}

HcclComm GetCachedComm(c10::string_view group) {
    thread_local std::string tl_key;
    thread_local HcclComm    tl_comm = nullptr;
    if (C10_LIKELY(tl_comm != nullptr &&
                   group.size() == tl_key.size() &&
                   std::memcmp(group.data(), tl_key.data(), tl_key.size()) == 0)) {
        return tl_comm;
    }
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
        // Fast path: parse numeric key as a raw HcclComm pointer value
        // (conftest now passes str(backend.get_hccl_comm(rank)) — an int
        // handle produced by ProcessGroupHCCL::getHCCLComm). Falls back to
        // HcomGetCommHandleByGroup for legacy string-group usage.
        comm = TryResolveCommFromInt(group);
        if (!comm) {
            HCCL_TORCH_CHECK(HcomGetCommHandleByGroup(key.c_str(), &comm));
            CC_LOG_INFO("GetCachedComm: resolved '%s' via HcomGetCommHandleByGroup -> %p",
                        key.c_str(), comm);
        } else {
            CC_LOG_INFO("GetCachedComm: parsed '%s' as HcclComm handle -> %p",
                        key.c_str(), comm);
        }
        TORCH_CHECK(comm != nullptr, "Failed to resolve HcclComm for: ", key);
        std::lock_guard<std::mutex> lk(mu);
        cache[key] = comm;
    }
    tl_comm = comm;
    tl_key = std::move(key);
    return comm;
}

HcclDataType ScalarTypeToHcclDtype(c10::ScalarType t) {
    switch (t) {
        case c10::ScalarType::Char:          return HCCL_DATA_TYPE_INT8;
        case c10::ScalarType::Byte:          return HCCL_DATA_TYPE_UINT8;
        case c10::ScalarType::Short:         return HCCL_DATA_TYPE_INT16;
        case c10::ScalarType::Half:          return HCCL_DATA_TYPE_FP16;
        case c10::ScalarType::Float:         return HCCL_DATA_TYPE_FP32;
        case c10::ScalarType::Int:           return HCCL_DATA_TYPE_INT32;
        case c10::ScalarType::Long:          return HCCL_DATA_TYPE_INT64;
        case c10::ScalarType::Double:        return HCCL_DATA_TYPE_FP64;
        case c10::ScalarType::BFloat16:      return HCCL_DATA_TYPE_BFP16;
        case c10::ScalarType::Float8_e4m3fn: return HCCL_DATA_TYPE_FP8E4M3;
        case c10::ScalarType::Float8_e5m2:   return HCCL_DATA_TYPE_FP8E5M2;
        default:
            TORCH_CHECK(false, "Unsupported dtype for allgather_batch: ",
                        c10::toString(t));
    }
}

// ============================================================
// aclGraph workaround: dispatch HCCL on a cached comm stream.
//
// Direct HCCL calls on the compute stream are not captured by
// aclGraph. torch_npu's ProcessGroupHCCL works around this by
// using a dedicated HCCL stream + event-based sync. We do the
// same here, caching the stream/events per thread to avoid
// per-call create/destroy overhead.
//
// TODO: remove when torch_npu fixes aclGraph capture for
// direct HCCL calls on the compute stream.
// ============================================================

#ifndef CUSTOM_COMM_NO_NPU
struct CommStreamCtx {
    aclrtStream stream = nullptr;
    aclrtEvent  pre    = nullptr;
    aclrtEvent  post   = nullptr;

    void init() {
        if (stream) return;
        ACL_TORCH_CHECK(aclrtCreateStream(&stream));
        ACL_TORCH_CHECK(aclrtCreateEvent(&pre));
        ACL_TORCH_CHECK(aclrtCreateEvent(&post));
    }
};

static void DispatchHcclOnCommStream(
    const HcclAllGatherDesc *descs,
    uint32_t descCount,
    HcclComm comm,
    aclrtStream computeStream) {

    // Cached per-thread: avoid aclrtCreate on every call.
    static thread_local CommStreamCtx ctx;
    ctx.init();

    // compute → comm: wait for pending work on compute stream
    ACL_TORCH_CHECK(aclrtRecordEvent(ctx.pre, computeStream));
    ACL_TORCH_CHECK(aclrtStreamWaitEvent(ctx.stream, ctx.pre));

    // HCCL on dedicated comm stream
    HCCL_TORCH_CHECK(HcclAllGatherBatch(descs, descCount, comm, ctx.stream));

    // comm → compute: wait for HCCL to finish before compute resumes
    ACL_TORCH_CHECK(aclrtRecordEvent(ctx.post, ctx.stream));
    ACL_TORCH_CHECK(aclrtStreamWaitEvent(computeStream, ctx.post));
}
#endif  // CUSTOM_COMM_NO_NPU

}  // namespace

// ============================================================
// PrivateUse1 (NPU) implementation
// ============================================================

std::vector<at::Tensor> allgather_batch_npu(
    const std::vector<at::Tensor> &inputs,
    c10::string_view hcom,
    int64_t world_size) {
#ifdef CUSTOM_COMM_NO_NPU
    TORCH_CHECK(false, "allgather_batch_npu requires NPU device");
#else
    RECORD_FUNCTION("custom_comm::allgather_batch", {});

    fprintf(stderr, "[custom_comm] allgather_batch  n=%zu  world_size=?  hcom=%.*s\n",
            inputs.size(), (int)hcom.size(), hcom.data());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto &t = inputs[i];
        fprintf(stderr, "  input[%zu]: dtype=%s  numel=%ld  shape=[",
                i, c10::toString(t.scalar_type()), (long)t.numel());
        for (int d = 0; d < t.dim(); ++d)
            fprintf(stderr, "%s%ld", d ? "," : "", (long)t.size(d));
        fprintf(stderr, "]\n");
    }

    TORCH_CHECK(!inputs.empty(), "inputs must be non-empty");
    TORCH_CHECK(inputs.size() <= MAX_DESC_COUNT,
                "inputs.size() (", inputs.size(), ") exceeds MAX_DESC_COUNT");
    TORCH_CHECK(world_size > 0, "world_size must be positive");
    TORCH_CHECK(!hcom.empty(), "hcom (HCCL group name) must be non-empty");
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto &t = inputs[i];
        TORCH_CHECK(t.defined(), "inputs[", i, "] is undefined");
        TORCH_CHECK(t.dim() >= 1, "inputs[", i, "] must be at least 1-D");
        TORCH_CHECK(t.numel() > 0, "inputs[", i, "] is empty");
    }

    const uint32_t descCount = static_cast<uint32_t>(inputs.size());
    HcclComm comm = GetCachedComm(hcom);
    TORCH_CHECK(comm != nullptr, "failed to resolve HcclComm from hcom=",
                std::string(hcom.data(), hcom.size()));
    aclrtStream computeStream = c10_npu::getCurrentNPUStream().stream(false);
    TORCH_CHECK(computeStream != nullptr, "NPU compute stream is null");

    // Build descriptors and pre-allocate outputs
    HcclAllGatherDesc descs[MAX_DESC_COUNT];
    std::vector<at::Tensor> outputs;
    for (uint32_t i = 0; i < descCount; ++i) {
        const auto &inp = inputs[i];
        TORCH_CHECK(inp.is_contiguous());
        auto out_shape = inp.sizes().vec();
        out_shape[0] *= world_size;
        auto out = at::empty(out_shape, inp.options());

        descs[i].sendBuf   = inp.data_ptr();
        descs[i].sendCount = static_cast<uint64_t>(inp.numel());
        descs[i].dataType  = ScalarTypeToHcclDtype(inp.scalar_type());
        descs[i].recvBuf   = out.data_ptr();
        outputs.push_back(std::move(out));
    }

    // WORKAROUND: aclGraph capture requires HCCL on a separate stream
    // with event-based sync. See DispatchHcclOnCommStream comment.
    DispatchHcclOnCommStream(descs, descCount, comm, computeStream);

    return outputs;
#endif
}

// ============================================================
// Meta implementation (shape inference, no NPU)
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
        TORCH_CHECK(input.dim() >= 1);
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

// In-place variant: caller provides pre-allocated output tensors.
static void allgather_batch_inplace(
    const std::vector<at::Tensor> &inputs,
    std::vector<at::Tensor> &outputs,
    c10::string_view hcom,
    int64_t world_size) {
#ifndef CUSTOM_COMM_NO_NPU
    TORCH_CHECK(!inputs.empty() && inputs.size() == outputs.size());
    TORCH_CHECK(world_size > 0);
    HcclComm comm = GetCachedComm(hcom);
    TORCH_CHECK(comm != nullptr, "Failed to resolve HCCL comm from hcom string");
    aclrtStream computeStream = c10_npu::getCurrentNPUStream().stream(false);

    const uint32_t n = static_cast<uint32_t>(inputs.size());
    HcclAllGatherDesc descs[MAX_DESC_COUNT];
    for (uint32_t i = 0; i < n; ++i) {
        descs[i].sendBuf   = inputs[i].data_ptr();
        descs[i].sendCount = static_cast<uint64_t>(inputs[i].numel());
        descs[i].dataType  = ScalarTypeToHcclDtype(inputs[i].scalar_type());
        descs[i].recvBuf   = outputs[i].data_ptr();
    }
    DispatchHcclOnCommStream(descs, n, comm, computeStream);
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("allgather_batch_eager", &allgather_batch_npu,
          "allgather_batch via pybind11 (returns new tensors)");
    m.def("allgather_batch_inplace", &allgather_batch_inplace,
          "allgather_batch into pre-allocated outputs");
}
