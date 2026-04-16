// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Phase 2: CCU context management -- one-time registration + cached launch.
//
// InitCcuContext: HcclEngineCtxCreate -> HcclChannelAcquire -> CcuKernelRegister
//                 -> HcclThreadAcquire -> KernelRegisterFinish  (first call)
//                 HcclEngineCtxGet (subsequent calls, cached)
//
// LaunchCcuKernel: HcclEngineCtxGet -> LaunchBatchedAGKernel

#include "engine_ctx.h"
#include "common.h"

#include <hccl/hccl_comm.h>
#include <hccl/hccl_res.h>
#include <hcomm/ccu/ccu_kernel.h>
#include <hcomm/ccu/hccl_ccu_res.h>

#include <cstdint>
#include <vector>

namespace custom_comm {

// ============================================================
// Forward declarations for helpers in ccu_kernel_ag_batch_mesh1d.cc
// ============================================================

HcclResult RegisterBatchedAGKernel(
    HcclComm comm, CcuKernelHandle *handle,
    uint32_t rankId, uint32_t rankSize,
    const std::vector<ChannelHandle> &channels);

HcclResult LaunchBatchedAGKernel(
    HcclComm comm, ThreadHandle thread, CcuKernelHandle kernel,
    const AllGatherBatchTaskArg &taskArg);

// ============================================================
// CcuContext -- stored in HcclEngineCtx, cached per (comm, tag)
// ============================================================

static constexpr const char *CTX_TAG = "custom_comm_ag_batch";

// XN IDs used: TOKEN(0), RECV_ADDR(1..MAX_DESC_COUNT), POST_SYNC(MAX_DESC_COUNT+1)
static constexpr uint32_t NOTIFY_COUNT = 1 + MAX_DESC_COUNT + 1;  // 10

struct CcuContext {
    CcuKernelHandle kernelHandle{};
    ThreadHandle    threadHandle{};
    bool initialized = false;
};

// ============================================================
// InitCcuContext
// ============================================================

HcclResult InitCcuContext(HcclComm comm) {
    // Fast path: return cached context
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, CTX_TAG, COMM_ENGINE_CCU,
                         &ctx, &ctxSize) == HCCL_SUCCESS && ctx != nullptr) {
        auto *ccuCtx = static_cast<CcuContext *>(ctx);
        if (ccuCtx->initialized) return HCCL_SUCCESS;
        // Partial init from a prior failed attempt: SDK resource cleanup
        // semantics are opaque, so retrying is unsafe.  Surface the error.
        return HCCL_E_INTERNAL;
    }

    // Slow path: first call -- allocate + register

    // 1. Create engine context slot
    HCCL_CHECK(HcclEngineCtxCreate(comm, CTX_TAG, COMM_ENGINE_CCU,
                                   sizeof(CcuContext), &ctx));
    auto *ccuCtx = static_cast<CcuContext *>(ctx);

    // 2. Get topology info
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    HCCL_CHECK(HcclGetRankId(comm, &rankId));
    HCCL_CHECK(HcclGetRankSize(comm, &rankSize));

    const uint32_t numPeers = rankSize - 1;

    // 3. Acquire one channel per peer, ordered by ascending remote rank
    std::vector<HcclChannelDesc> channelDescs(numPeers);
    HCCL_CHECK(HcclChannelDescInit(channelDescs.data(), numPeers));

    uint32_t peerIdx = 0;
    for (uint32_t r = 0; r < rankSize; ++r) {
        if (r == rankId) continue;
        channelDescs[peerIdx].remoteRank = r;
        channelDescs[peerIdx].notifyNum  = NOTIFY_COUNT;
        ++peerIdx;
    }

    std::vector<ChannelHandle> channels(numPeers);
    HCCL_CHECK(HcclChannelAcquire(comm, COMM_ENGINE_CCU,
                                   channelDescs.data(), numPeers,
                                   channels.data()));

    // 4. Register CCU kernel (compiles Algorithm -> microcode IR)
    HCCL_CHECK(RegisterBatchedAGKernel(comm, &ccuCtx->kernelHandle,
                                       rankId, rankSize, channels));

    // 5. Acquire CCU thread with enough notification slots
    HCCL_CHECK(HcclThreadAcquire(comm, COMM_ENGINE_CCU,
                                  1, NOTIFY_COUNT,
                                  &ccuCtx->threadHandle));

    // 6. Finalize: CcuKernelMgr translates IR to hardware microcode
    HCCL_CHECK(HcclCcuKernelRegisterFinish(comm));

    ccuCtx->initialized = true;
    return HCCL_SUCCESS;
}

// ============================================================
// LaunchCcuKernel
// ============================================================

HcclResult LaunchCcuKernel(HcclComm comm, const void *taskArg) {
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    HCCL_CHECK(HcclEngineCtxGet(comm, CTX_TAG, COMM_ENGINE_CCU,
                                &ctx, &ctxSize));

    auto *ccuCtx = static_cast<CcuContext *>(ctx);
    auto *arg    = static_cast<const AllGatherBatchTaskArg *>(taskArg);

    return LaunchBatchedAGKernel(comm, ccuCtx->threadHandle,
                                ccuCtx->kernelHandle, *arg);
}

// ============================================================
// GetCcuThreadHandle -- expose thread handle for aclGraph capture
// ============================================================

HcclResult GetCcuThreadHandle(HcclComm comm, uint64_t *threadHandle) {
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    HCCL_CHECK(HcclEngineCtxGet(comm, CTX_TAG, COMM_ENGINE_CCU,
                                &ctx, &ctxSize));

    if (ctx == nullptr) return HCCL_E_INTERNAL;
    auto *ccuCtx = static_cast<CcuContext *>(ctx);
    if (!ccuCtx || ccuCtx->threadHandle == 0) return HCCL_E_INTERNAL;
    *threadHandle = ccuCtx->threadHandle;
    return HCCL_SUCCESS;
}

}  // namespace custom_comm
