// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// FROZEN CONTRACT -- do not change after Phase 1 ships.

#ifndef CUSTOM_COMM_ALLGATHER_BATCH_H_
#define CUSTOM_COMM_ALLGATHER_BATCH_H_

#include <cstdint>
#include <hccl/hccl_types.h>
#include <acl/acl_base_rt.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// C API data structures
// ============================================================

typedef struct {
    void        *sendBuf;   // device memory, non-null
    uint64_t     sendCount; // element count (not bytes)
    HcclDataType dataType;  // hccl_types.h:90-108
    void        *recvBuf;   // device memory, size >= sendCount * sizeof(dataType) * worldSize
} HcclAllGatherDesc;

// ============================================================
// C API entry point
// ============================================================

// Semantically equivalent to descCount independent HcclAllGather calls,
// but executed as a single operation (Phase 2: single CCU kernel).
//
// Phase dispatch: if env CUSTOM_COMM_USE_CCU == "1", takes Phase 2 (CCU) path;
// otherwise takes Phase 1 (decomposed byte-packing) path.
//
// Constraints:
//   - 1 <= descCount <= MAX_DESC_COUNT (8)
//   - Each desc's sendBuf/recvBuf must be valid device memory
//   - comm must be an initialized HcclComm
//   - stream must be a valid aclrtStream
HcclResult HcclAllGatherBatch(
    const HcclAllGatherDesc *descs,
    uint32_t descCount,
    HcclComm comm,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_COMM_ALLGATHER_BATCH_H_
