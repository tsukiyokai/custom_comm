// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUSTOM_COMM_COMMON_H_
#define CUSTOM_COMM_COMMON_H_

#include <cstdint>
#include <hccl/hccl_types.h>

// ============================================================
// Constants
// ============================================================

constexpr uint32_t MAX_DESC_COUNT = 8;

// ============================================================
// Error checking macros (CHK_RET style, ref-survey.md:68-79)
// ============================================================

#define HCCL_CHECK(call)                                              \
    do {                                                              \
        HcclResult _ret = (call);                                     \
        if (_ret != HCCL_SUCCESS) {                                   \
            return _ret;                                              \
        }                                                             \
    } while (0)

// ============================================================
// Data type utilities
// ============================================================

// Returns byte size per element for a given HcclDataType.
// Returns 0 on unknown type (caller must check).
// Covers all 17 values in hccl_types.h:90-108 (0-12 + 14-17, skip 13).
inline uint64_t DtypeSize(HcclDataType dtype) {
    switch (dtype) {
        case HCCL_DATA_TYPE_INT8:    return 1;
        case HCCL_DATA_TYPE_UINT8:   return 1;
        case HCCL_DATA_TYPE_INT16:   return 2;
        case HCCL_DATA_TYPE_UINT16:  return 2;
        case HCCL_DATA_TYPE_FP16:    return 2;
        case HCCL_DATA_TYPE_BFP16:   return 2;
        case HCCL_DATA_TYPE_INT32:   return 4;
        case HCCL_DATA_TYPE_UINT32:  return 4;
        case HCCL_DATA_TYPE_FP32:    return 4;
        case HCCL_DATA_TYPE_INT64:   return 8;
        case HCCL_DATA_TYPE_UINT64:  return 8;
        case HCCL_DATA_TYPE_FP64:    return 8;
        case HCCL_DATA_TYPE_INT128:  return 16;
        case HCCL_DATA_TYPE_HIF8:    return 1;  // 14
        case HCCL_DATA_TYPE_FP8E4M3: return 1;  // 15
        case HCCL_DATA_TYPE_FP8E5M2: return 1;  // 16
        case HCCL_DATA_TYPE_FP8E8M0: return 1;  // 17
        default: return 0;
    }
}

// ============================================================
// CCU task argument (Phase 2 data structure, frozen contract)
// ============================================================

#include "hccl_custom_allgather_batch.h"

namespace custom_comm {

struct AllGatherBatchTaskArg {
    uint32_t          descCount;
    uint32_t          rankId;
    uint32_t          rankSize;
    HcclAllGatherDesc descs[MAX_DESC_COUNT];
};

}  // namespace custom_comm

#endif  // CUSTOM_COMM_COMMON_H_
