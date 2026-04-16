// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Phase 1 decomposed strategy:
//   1. Compute total bytes across all descs
//   2. Allocate packed send buffer, memcpy each desc's sendBuf into it
//   3. Allocate gathered recv buffer (totalBytes * worldSize)
//   4. Single HcclAllGatherInner(packed, gathered, totalBytes, UINT8, comm, stream)
//   5. Copy each rank-i slice back to the corresponding desc's recvBuf
//
// This is the correctness oracle -- Phase 2 CCU output is verified bit-exact
// against this path.

#include "hccl_custom_allgather_batch.h"
#include "common.h"

#include <hccl/hccl_types.h>
#include <acl/acl_rt.h>

// Forward-declare internal HCCL APIs instead of #include <hccl/hccl_comm.h>
// and <hccl/hccl_inner.h>. Those SDK headers pull in hccl_types.h from the
// CANN toolkit, which may conflict with the version bundled in torch_npu
// (different struct layouts / enum values behind the same include guard).
extern "C" {
HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize);
HcclResult HcclAllGatherInner(const void *sendBuf, void *recvBuf,
                              uint64_t count, HcclDataType dataType,
                              HcclComm comm, aclrtStream stream);
}

#include <cstdint>
#include <vector>

namespace custom_comm {

// ============================================================
// Byte-size computation with overflow check
// ============================================================

static HcclResult ComputeSendBytes(
    const HcclAllGatherDesc *descs, uint32_t descCount,
    std::vector<uint64_t> &perDescBytes, uint64_t &totalBytes) {

    totalBytes = 0;
    perDescBytes.resize(descCount);

    for (uint32_t i = 0; i < descCount; ++i) {
        uint64_t elemSize = DtypeSize(descs[i].dataType);
        if (elemSize == 0) return HCCL_E_PARA;

        // Overflow check: sendCount * elemSize
        if (descs[i].sendCount > UINT64_MAX / elemSize) {
            return HCCL_E_PARA;
        }
        perDescBytes[i] = descs[i].sendCount * elemSize;

        // Overflow check: running total
        if (totalBytes > UINT64_MAX - perDescBytes[i]) {
            return HCCL_E_PARA;
        }
        totalBytes += perDescBytes[i];
    }
    return HCCL_SUCCESS;
}

// ============================================================
// Decomposed AllGather: pack -> single AG(UINT8) -> unpack
// ============================================================

HcclResult DecomposedAllGatherBatch(
    const HcclAllGatherDesc *descs, uint32_t descCount,
    HcclComm comm, aclrtStream stream) {

    // 1. Compute byte sizes
    std::vector<uint64_t> perDescBytes;
    uint64_t totalBytes = 0;
    HCCL_CHECK(ComputeSendBytes(descs, descCount, perDescBytes, totalBytes));

    if (totalBytes == 0) return HCCL_SUCCESS;

    // 2. Get world size from communicator
    uint32_t worldSize = 0;
    HCCL_CHECK(HcclGetRankSize(comm, &worldSize));
    if (worldSize == 0) return HCCL_E_PARA;

    // Overflow check: totalBytes * worldSize
    if (totalBytes > UINT64_MAX / worldSize) {
        return HCCL_E_PARA;
    }
    uint64_t gatheredBytes = totalBytes * worldSize;

    // 3. Allocate device buffers for pack/gather
    void *packedSend = nullptr;
    void *packedRecv = nullptr;
    HcclResult result = HCCL_SUCCESS;
    aclError aclRet;

    aclRet = aclrtMalloc(&packedSend, totalBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != 0) return HCCL_E_MEMORY;

    aclRet = aclrtMalloc(&packedRecv, gatheredBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != 0) {
        (void)aclrtFree(packedSend);
        return HCCL_E_MEMORY;
    }

    // 4. Pack: memcpy each desc's sendBuf into packed buffer (device-to-device)
    uint64_t offset = 0;
    for (uint32_t i = 0; i < descCount; ++i) {
        aclRet = aclrtMemcpyAsync(
            static_cast<char *>(packedSend) + offset, perDescBytes[i],
            descs[i].sendBuf, perDescBytes[i],
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        if (aclRet != 0) { result = HCCL_E_RUNTIME; goto cleanup; }
        offset += perDescBytes[i];
    }

    // 5. Single AllGather on packed uint8 buffer
    result = HcclAllGatherInner(
        packedSend, packedRecv, totalBytes,
        HCCL_DATA_TYPE_UINT8, comm, stream);
    if (result != HCCL_SUCCESS) goto cleanup;

    // 6. Unpack: copy gathered data back to each desc's recvBuf
    //
    // packedRecv layout (worldSize copies of packed data):
    //   [rank0: desc0_bytes | desc1_bytes | ...]
    //   [rank1: desc0_bytes | desc1_bytes | ...]
    //   ...
    //
    // Each desc's recvBuf layout:
    //   [rank0_data | rank1_data | ... | rankN_data]
    //
    // Transpose: for each desc, gather its slice from each rank.

    for (uint32_t i = 0; i < descCount; ++i) {
        uint64_t descOffset = 0;
        for (uint32_t j = 0; j < i; ++j) {
            descOffset += perDescBytes[j];
        }

        for (uint32_t r = 0; r < worldSize; ++r) {
            const char *src = static_cast<const char *>(packedRecv)
                              + r * totalBytes + descOffset;
            char *dst = static_cast<char *>(descs[i].recvBuf)
                        + r * perDescBytes[i];

            aclRet = aclrtMemcpyAsync(
                dst, perDescBytes[i], src, perDescBytes[i],
                ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            if (aclRet != 0) { result = HCCL_E_RUNTIME; goto cleanup; }
        }
    }

cleanup:
    // Sync stream before freeing -- async ops may still be in flight
    (void)aclrtSynchronizeStream(stream);
    (void)aclrtFree(packedSend);
    (void)aclrtFree(packedRecv);
    return result;
}

}  // namespace custom_comm
