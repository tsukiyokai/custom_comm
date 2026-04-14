// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PrivateUse1 (NPU) and Meta implementations for
// torch.ops.custom_comm.allgather_batch.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/StringUtil.h>

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

#ifndef CUSTOM_COMM_NO_NPU  // macOS syntax-check: guard NPU-specific headers
#include "torch_npu/csrc/core/npu/NPUStream.h"
#endif

// ============================================================
// PyTorch ScalarType -> HcclDataType mapping
// ============================================================

namespace {

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
// PrivateUse1 (NPU device) implementation
// ============================================================

static std::vector<at::Tensor> allgather_batch_npu(
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

    // 1. Resolve hcom group name -> HcclComm handle
    HcclComm comm = nullptr;
    std::string hcom_str(hcom.data(), hcom.size());
    HCCL_TORCH_CHECK(HcomGetCommHandleByGroup(hcom_str.c_str(), &comm));
    TORCH_CHECK(comm != nullptr, "Failed to get HcclComm for group: ", hcom_str);

    // 2. Get current NPU stream
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);

    // 3. Allocate output tensors and build descriptor array
    std::vector<at::Tensor> outputs;
    outputs.reserve(descCount);

    HcclAllGatherDesc descs[MAX_DESC_COUNT];

    for (uint32_t i = 0; i < descCount; ++i) {
        const at::Tensor &input = inputs[i];
        TORCH_CHECK(input.dim() >= 1,
                    "input[", i, "] must be at least 1-dimensional");
        TORCH_CHECK(input.is_contiguous(), "input[", i, "] must be contiguous");
        TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
                    "input[", i, "] must be on NPU device");

        // Output shape: dim 0 scaled by world_size
        auto out_sizes = input.sizes().vec();
        out_sizes[0] *= world_size;
        at::Tensor output = at::empty(out_sizes, input.options());

        descs[i].sendBuf   = input.data_ptr();
        descs[i].sendCount = static_cast<uint64_t>(input.numel());
        descs[i].dataType  = ScalarTypeToHcclDtype(input.scalar_type());
        descs[i].recvBuf   = output.data_ptr();

        outputs.push_back(std::move(output));
    }

    // 4. Call C API
    HCCL_TORCH_CHECK(HcclAllGatherBatch(descs, descCount, comm, stream));

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
