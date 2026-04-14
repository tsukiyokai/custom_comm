// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// FROZEN CONTRACT -- do not change after Phase 1 ships.

#ifndef CUSTOM_COMM_CCU_KERNEL_AG_BATCH_MESH1D_H_
#define CUSTOM_COMM_CCU_KERNEL_AG_BATCH_MESH1D_H_

// Phase 2 CCU kernel declaration.
// Phase 1: header-only stub, no implementation linked.
//
// CcuKernelAllGatherBatchMesh1D inherits hcomm::CcuKernel (ccu_kernel.h:42-67)
// and implements Algorithm() + GeneArgs() for batched AllGather on Mesh1D topology.
//
// Not included by Phase 1 code; exists to freeze the class interface.

// Phase 2 will add:
//   #include <hcomm/ccu/ccu_kernel.h>
//   class CcuKernelAllGatherBatchMesh1D : public hcomm::CcuKernel { ... };

#endif  // CUSTOM_COMM_CCU_KERNEL_AG_BATCH_MESH1D_H_
