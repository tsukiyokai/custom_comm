// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "hccl_custom_allgather_batch.h"
#include "common.h"
#include "engine_ctx.h"

#include <hccl/hccl_comm.h>

#include <cstdlib>
#include <cstring>

#ifndef __APPLE__
#include <acl/acl_prof.h>
#include <acl/acl_rt.h>
#include <hccl/hccl_res.h>
// RT internal: add slave stream to capture graph model
extern "C" int rtStreamAddToModel(void *stream, void *model);
#endif

// Phase 1 decomposed strategy (decomposed_strategy.cc)
namespace custom_comm {
HcclResult DecomposedAllGatherBatch(
    const HcclAllGatherDesc *descs, uint32_t descCount,
    HcclComm comm, aclrtStream stream);
}  // namespace custom_comm

// ============================================================
// Parameter validation
// ============================================================

static HcclResult ValidateParams(
    const HcclAllGatherDesc *descs, uint32_t descCount) {
    if (descs == nullptr) {
        return HCCL_E_PTR;
    }
    if (descCount == 0 || descCount > MAX_DESC_COUNT) {
        return HCCL_E_PARA;
    }
    for (uint32_t i = 0; i < descCount; ++i) {
        if (descs[i].sendBuf == nullptr || descs[i].recvBuf == nullptr) {
            return HCCL_E_PTR;
        }
        if (DtypeSize(descs[i].dataType) == 0) {
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

// ============================================================
// Environment variable check for Phase 2 CCU path
// ============================================================

static bool UseCcuPath() {
    const char *val = std::getenv("CUSTOM_COMM_USE_CCU");
    if (val == nullptr) return false;
    // Only "1" or "true" (case-sensitive) enables CCU path.
    // Avoids atoi to prevent undefined behavior on non-numeric strings.
    return (std::strcmp(val, "1") == 0 || std::strcmp(val, "true") == 0);
}

// ============================================================
// Profiling helpers
// ============================================================

#ifndef __APPLE__
static inline void ProfMark(const char *msg, aclrtStream stream) {
    // Best-effort: ignore errors (profiling is non-critical).
    (void)aclprofMarkEx(msg, __builtin_strlen(msg), stream);
}
#else
static inline void ProfMark(const char * /*msg*/, void * /*stream*/) {}
#endif

// ============================================================
// Core dispatch (called between balanced ProfMark begin/end)
// ============================================================

static HcclResult HcclAllGatherBatchImpl(
    const HcclAllGatherDesc *descs,
    uint32_t descCount,
    HcclComm comm,
    aclrtStream stream) {

    if (UseCcuPath()) {
        // Phase 2: CCU batched zero-copy AllGather
        uint32_t rankSize = 0;
        HCCL_CHECK(HcclGetRankSize(comm, &rankSize));
        if (rankSize <= 1) {
            // Single rank: no peers to exchange with, use Phase 1 self-copy
            return custom_comm::DecomposedAllGatherBatch(
                descs, descCount, comm, stream);
        }

        HCCL_CHECK(custom_comm::InitCcuContext(comm));

#ifndef __APPLE__
        // aclGraph capture: register CCU slave stream into the capture model
        // so that HcclCcuKernelLaunch operations are recorded into the graph.
        {
            aclmdlRICaptureStatus captureStatus = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
            aclmdlRI rtModel = nullptr;
            aclError aclRet = aclmdlRICaptureGetInfo(stream, &captureStatus, &rtModel);
            if (aclRet == ACL_SUCCESS &&
                captureStatus == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE && rtModel != nullptr) {
                uint64_t threadHandle = 0;
                HCCL_CHECK(custom_comm::GetCcuThreadHandle(comm, &threadHandle));
                aclrtStream slaveStream = nullptr;
                uint32_t len = sizeof(slaveStream);
                HCCL_CHECK(HcclThreadResGetInfo(
                    comm, threadHandle, THREAD_RES_TYPE_STREAM,
                    len, reinterpret_cast<void **>(&slaveStream)));
                if (slaveStream != nullptr) {
                    rtStreamAddToModel(slaveStream, rtModel);
                }
            }
        }
#endif

        uint32_t rankId = 0;
        HCCL_CHECK(HcclGetRankId(comm, &rankId));

        custom_comm::AllGatherBatchTaskArg taskArg{};
        taskArg.descCount = descCount;
        taskArg.rankId    = rankId;
        taskArg.rankSize  = rankSize;
        for (uint32_t i = 0; i < descCount; ++i) {
            taskArg.descs[i] = descs[i];
        }

        ProfMark("custom_comm::ccu_launch::begin", stream);
        HcclResult result = custom_comm::LaunchCcuKernel(comm, &taskArg);
        ProfMark("custom_comm::ccu_launch::end", stream);
        return result;
    }

    // Phase 1: decomposed byte-packing strategy
    return custom_comm::DecomposedAllGatherBatch(descs, descCount, comm, stream);
}

// ============================================================
// Entry point
// ============================================================

HcclResult HcclAllGatherBatch(
    const HcclAllGatherDesc *descs,
    uint32_t descCount,
    HcclComm comm,
    aclrtStream stream) {

    HCCL_CHECK(ValidateParams(descs, descCount));

    ProfMark("custom_comm::allgather_batch::begin", stream);
    HcclResult result = HcclAllGatherBatchImpl(descs, descCount, comm, stream);
    ProfMark("custom_comm::allgather_batch::end", stream);
    return result;
}
