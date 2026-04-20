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
#include "log_util.h"

#include <hccl/hccl_comm.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res.h>
#include <hcomm/ccu/ccu_kernel.h>
#include <hcomm/ccu/hccl_ccu_res.h>

#include <acl/acl.h>

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
    if (comm == nullptr) {
        CC_LOG_ERROR("InitCcuContext: comm is null");
        return HCCL_E_PARA;
    }

    // Fast path: return cached context
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, CTX_TAG, COMM_ENGINE_CCU,
                         &ctx, &ctxSize) == HCCL_SUCCESS && ctx != nullptr) {
        auto *ccuCtx = static_cast<CcuContext *>(ctx);
        if (ccuCtx->initialized) {
            CC_LOG_DEBUG("InitCcuContext: cached ctx hit");
            return HCCL_SUCCESS;
        }
        // Partial init from a prior failed attempt: SDK resource cleanup
        // semantics are opaque, so retrying is unsafe.  Surface the error.
        CC_LOG_ERROR("InitCcuContext: partial context detected; refusing retry");
        return HCCL_E_INTERNAL;
    }

    // Slow path: first call -- allocate + register
    CC_LOG_INFO("InitCcuContext: first-time registration");

    // 1. Create engine context slot
    HCCL_CHECK(HcclEngineCtxCreate(comm, CTX_TAG, COMM_ENGINE_CCU,
                                   sizeof(CcuContext), &ctx));
    auto *ccuCtx = static_cast<CcuContext *>(ctx);

    // 2. Get topology info
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    HCCL_CHECK(HcclGetRankId(comm, &rankId));
    HCCL_CHECK(HcclGetRankSize(comm, &rankSize));

    if (rankSize < 2) {
        CC_LOG_ERROR("InitCcuContext: rankSize=%u too small for CCU (need >=2)", rankSize);
        return HCCL_E_PARA;
    }
    if (rankId >= rankSize) {
        CC_LOG_ERROR("InitCcuContext: rankId=%u out of range (rankSize=%u)", rankId, rankSize);
        return HCCL_E_INTERNAL;
    }

    const uint32_t numPeers = rankSize - 1;

    // Build channelDesc for every link to every remote rank. CCU requires
    // memHandles=nullptr (no user buffer exchange — HCCL owns the ccl buf)
    // and notifyNum=16 (hardcoded by ccu_urma_channel.cc).
    std::vector<HcclChannelDesc> channelDescs;
    for (uint32_t r = 0; r < rankSize; ++r) {
        if (r == rankId) continue;
        uint32_t linkNum = 0;
        CommLink *links = nullptr;
        HCCL_CHECK(HcclRankGraphGetLinks(comm, /*netLayer=*/0, rankId, r, &links, &linkNum));
        if (links == nullptr || linkNum == 0) {
            CC_LOG_ERROR("InitCcuContext: no links for rank=%u peer=%u (linkNum=%u)",
                         rankId, r, linkNum);
            return HCCL_E_INTERNAL;
        }
        for (uint32_t i = 0; i < linkNum; ++i) {
            HcclChannelDesc desc{};
            HcclChannelDescInit(&desc, 1);
            desc.remoteRank           = r;
            // CCU engine requires notifyNum=16 exactly and rejects any
            // user-supplied memHandles (it uses the HCCL shared buffer).
            desc.notifyNum            = 16;
            desc.memHandles           = nullptr;
            desc.memHandleNum         = 0;
            desc.localEndpoint        = links[i].srcEndpointDesc;
            desc.remoteEndpoint       = links[i].dstEndpointDesc;
            desc.channelProtocol      = links[i].linkAttr.linkProtocol;
            channelDescs.push_back(desc);
        }
    }

    if (channelDescs.empty()) {
        CC_LOG_ERROR("InitCcuContext: no channel descriptors built (rankSize=%u)",
                     rankSize);
        return HCCL_E_INTERNAL;
    }

    std::vector<ChannelHandle> channels(channelDescs.size());
    HCCL_CHECK(HcclChannelAcquire(comm, COMM_ENGINE_CCU,
                                  channelDescs.data(), channelDescs.size(),
                                  channels.data()));
    for (size_t i = 0; i < channels.size(); ++i) {
        if (channels[i] == 0) {
            CC_LOG_ERROR("InitCcuContext: channels[%zu] is null after Acquire", i);
            return HCCL_E_INTERNAL;
        }
    }

    // 4. Register CCU kernel (RegisterBatchedAGKernel)
    HCCL_CHECK(RegisterBatchedAGKernel(comm, &ccuCtx->kernelHandle,
                                       rankId, rankSize, channels));

    // 5. Acquire CCU thread with enough notification slots
    HCCL_CHECK(HcclThreadAcquire(comm, COMM_ENGINE_CCU,
                                  1, NOTIFY_COUNT,
                                  &ccuCtx->threadHandle));

    // 6. Finalize: CcuKernelMgr translates IR to hardware microcode
    HCCL_CHECK(HcclCcuKernelRegisterFinish(comm));

    ccuCtx->initialized = true;
    CC_LOG_INFO("InitCcuContext: ready (rank=%u/%u, peers=%u)",
                rankId, rankSize, numPeers);
    return HCCL_SUCCESS;
}

// ============================================================
// LaunchCcuKernel
// ============================================================

HcclResult LaunchCcuKernel(HcclComm comm, const void *taskArg) {
    if (comm == nullptr || taskArg == nullptr) {
        CC_LOG_ERROR("LaunchCcuKernel: null comm or taskArg");
        return HCCL_E_PARA;
    }
    auto *arg = static_cast<const AllGatherBatchTaskArg *>(taskArg);
    CC_LOG_INFO("LaunchCcuKernel: descCount=%u rank=%u/%u",
                arg->descCount, arg->rankId, arg->rankSize);
    if (arg->descCount == 0 || arg->descCount > MAX_DESC_COUNT) {
        CC_LOG_ERROR("LaunchCcuKernel: descCount=%u out of range (max=%u)",
                     arg->descCount, MAX_DESC_COUNT);
        return HCCL_E_PARA;
    }

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    HCCL_CHECK(HcclEngineCtxGet(comm, CTX_TAG, COMM_ENGINE_CCU,
                                &ctx, &ctxSize));
    if (ctx == nullptr || ctxSize < sizeof(CcuContext)) {
        CC_LOG_ERROR("LaunchCcuKernel: invalid ctx (ptr=%p, size=%llu)",
                     ctx, static_cast<unsigned long long>(ctxSize));
        return HCCL_E_INTERNAL;
    }
    auto *ccuCtx = static_cast<CcuContext *>(ctx);
    if (!ccuCtx->initialized) {
        CC_LOG_ERROR("LaunchCcuKernel: ctx not initialized (did InitCcuContext succeed?)");
        return HCCL_E_INTERNAL;
    }

    return LaunchBatchedAGKernel(comm, ccuCtx->threadHandle,
                                ccuCtx->kernelHandle, *arg);
}

// ============================================================
// GetCcuThreadHandle -- expose thread handle for aclGraph capture
// ============================================================

HcclResult GetCcuThreadHandle(HcclComm comm, uint64_t *threadHandle) {
    if (comm == nullptr || threadHandle == nullptr) {
        CC_LOG_ERROR("GetCcuThreadHandle: null comm or out-param");
        return HCCL_E_PARA;
    }
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    HCCL_CHECK(HcclEngineCtxGet(comm, CTX_TAG, COMM_ENGINE_CCU,
                                &ctx, &ctxSize));
    if (ctx == nullptr || ctxSize < sizeof(CcuContext)) {
        CC_LOG_ERROR("GetCcuThreadHandle: ctx missing or too small");
        return HCCL_E_INTERNAL;
    }
    auto *ccuCtx = static_cast<CcuContext *>(ctx);
    if (!ccuCtx->initialized || ccuCtx->threadHandle == 0) {
        CC_LOG_ERROR("GetCcuThreadHandle: ctx not initialized");
        return HCCL_E_INTERNAL;
    }
    *threadHandle = ccuCtx->threadHandle;
    return HCCL_SUCCESS;
}

}  // namespace custom_comm
