// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch extension: PrivateUse1 (NPU) and Meta implementations for
// custom_comm::allgather_batch.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <mutex>
#include <unordered_map>
#include <cstring>

#include "hccl_custom_allgather_batch.h"
#include "common.h"

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
    thread_local std::string tl_key;
    thread_local HcclComm    tl_comm = nullptr;
    if (C10_LIKELY(tl_comm != nullptr &&
                   group.size() == tl_key.size() &&
                   std::memcmp(group.data(), tl_key.data(), group.size()) == 0)) {
        return tl_comm;
    }

    // Slow path: mutex-guarded process-wide map.
    static std::mutex mu;
    static std::unordered_map<std::string, HcclComm> cache;

    std::string key(group.data(), group.size());
    HcclComm comm = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(key);
        if (it != cache.end()) {
            tl_key = key;
            tl_comm = it->second;
            return tl_comm;
        }
    }
    HCCL_TORCH_CHECK(HcomGetCommHandleByGroup(key.c_str(), &comm));
    TORCH_CHECK(comm != nullptr, "Failed to get HcclComm for group: ", key);
    {
        std::lock_guard<std::mutex> lk(mu);
        cache[key] = comm;
    }
    tl_comm = comm;
    tl_key = std::move(key);
    return comm;
}

HcclDataType ScalarTypeToHcclDtype(c10::ScalarType t) {
    switch (t) {
        case c10::ScalarType::Char:   return HCCL_DATA_TYPE_INT8;
        case c10::ScalarType::Byte:   return HCCL_DATA_TYPE_UINT8;
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
                        c10::toString(t));
    }
}

#ifndef CUSTOM_COMM_NO_NPU
// ============================================================
// aclGraph-compatible HCCL call: run HcclAllGatherBatch on a
// dedicated comm stream with event sync, so that aclGraph capture
// sees the event record/wait pair it requires.
//
// WORKAROUND: Directly calling HCCL ops on the compute stream
// is not captured by aclGraph. ProcessGroupHCCL works around
// this by dispatching HCCL ops to a dedicated comm stream with
// event-based synchronisation. We replicate that pattern here.
// TODO: reuse a cached comm stream + events to avoid per-call
//       aclrtCreate/Destroy overhead.
// ============================================================
static void DispatchHcclOnCommStream(
    const HcclAllGatherDesc *descs,
    uint32_t descCount,
    HcclComm comm,
    aclrtStream computeStream) {

    // Create a temporary comm stream + events for cross-stream sync.
    // TODO(perf): cache these per-device to eliminate create/destroy cost.
    aclrtStream commStream = nullptr;
    aclrtCreateStream(&commStream);

    aclrtEvent preEvent = nullptr, postEvent = nullptr;
    aclrtCreateEvent(&preEvent);
    aclrtCreateEvent(&postEvent);

    // Sync: compute → comm (ensure inputs are ready)
    aclrtRecordEvent(preEvent, computeStream);
    aclrtStreamWaitEvent(commStream, preEvent);

    // Run the batched AllGather on the comm stream
    auto ret = HcclAllGatherBatch(descs, descCount, comm, commStream);
    TORCH_CHECK(ret == HCCL_SUCCESS,
                "HcclAllGatherBatch failed: ", static_cast<int>(ret));

    // Sync: comm → compute (ensure outputs are visible)
    aclrtRecordEvent(postEvent, commStream);
    aclrtStreamWaitEvent(computeStream, postEvent);

    // Cleanup
    // TODO(perf): pool these instead of destroying each call
    aclrtDestroyEvent(preEvent);
    aclrtDestroyEvent(postEvent);
    aclrtDestroyStream(commStream);
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

    TORCH_CHECK(!inputs.empty(), "inputs must be non-empty");
    TORCH_CHECK(inputs.size() <= MAX_DESC_COUNT,
                "inputs.size() (", inputs.size(), ") exceeds MAX_DESC_COUNT (", MAX_DESC_COUNT, ")");
    TORCH_CHECK(world_size > 0, "world_size must be positive");

    const uint32_t descCount = static_cast<uint32_t>(inputs.size());
    HcclComm comm = GetCachedComm(hcom);
    aclrtStream computeStream = c10_npu::getCurrentNPUStream().stream(false);

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

    // Dispatch via comm stream + event sync (aclGraph-compatible)
    DispatchHcclOnCommStream(descs, descCount, comm, computeStream);

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

// In-place variant: caller provides pre-allocated outputs.
static void allgather_batch_inplace(
    const std::vector<at::Tensor> &inputs,
    std::vector<at::Tensor> &outputs,
    c10::string_view hcom,
    int64_t world_size) {
#ifndef CUSTOM_COMM_NO_NPU
    TORCH_CHECK(!inputs.empty() && inputs.size() == outputs.size());
    TORCH_CHECK(world_size > 0);
    const uint32_t n = static_cast<uint32_t>(inputs.size());
    HcclComm comm = GetCachedComm(hcom);
    aclrtStream computeStream = c10_npu::getCurrentNPUStream().stream(false);

    HcclAllGatherDesc descs[MAX_DESC_COUNT];
    for (uint32_t i = 0; i < inputs.size(); ++i) {
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
          "allgather_batch via direct pybind11 call (no Dispatcher)");
    m.def("allgather_batch_inplace", &allgather_batch_inplace,
          "allgather_batch writing into pre-allocated outputs");
}
