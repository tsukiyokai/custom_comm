# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""4-way benchmark: compare AllGather strategies.

  A) 3x dist.all_gather (list API, allocates per call)
  B) 3x dist.all_gather_into_tensor (NPUCommunicator pattern)
  C) Python packed: view-as-uint8 + cat + single AG (spike approach)
  D) custom_comm.allgather_batch (C API)

Usage:
    torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
"""

import time
import torch
import torch.distributed as dist

HAS_NPU = False
try:
    import torch_npu  # noqa: F401
    import custom_comm  # noqa: F401
    HAS_NPU = True
except ImportError:
    pass

WARMUP = 50
ITERS = 200


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6


def main():
    if not HAS_NPU:
        print("NPU not available")
        return

    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)

    pg = dist.distributed_c10d._get_default_group()
    hcom = pg._get_backend(device).get_hccl_comm_name(rank)

    # OPT-AG-04: INT8(N,H) + FP32(N,) + INT32(N,K)
    N, H, K = 32, 7168, 8
    x = torch.randint(0, 127, (N, H), dtype=torch.int8, device="npu")
    s = torch.randn(N, dtype=torch.float32, device="npu")
    ids = torch.randint(0, 8, (N, K), dtype=torch.int32, device="npu")
    ws_ = dist.get_world_size()

    # ---- A: 3x dist.all_gather (list API) ----
    def method_a():
        for t in [x, s, ids]:
            out = [torch.empty_like(t) for _ in range(ws_)]
            dist.all_gather(out, t)

    # ---- B: 3x all_gather_into_tensor (matching NPUCommunicator) ----
    def method_b():
        for t in [x, s, ids]:
            out = torch.empty(t.shape[0] * ws_, *t.shape[1:],
                              dtype=t.dtype, device=t.device)
            dist.all_gather_into_tensor(out, t)

    # ---- C: Python packed (spike approach) ----
    def method_c():
        tensors = [x, s, ids]
        n = tensors[0].shape[0]
        bv = [t.reshape(n, -1).contiguous().view(torch.uint8) for t in tensors]
        packed = torch.cat(bv, dim=1)
        out = torch.empty(n * ws_, packed.shape[1],
                          dtype=torch.uint8, device=packed.device)
        dist.all_gather_into_tensor(out, packed)

    # ---- D: custom_comm.allgather_batch (C API) ----
    xf = x.reshape(-1)
    sf = s.reshape(-1)
    idf = ids.reshape(-1)

    def method_d():
        torch.ops.custom_comm.allgather_batch([xf, sf, idf], hcom, ws_)

    ta = timed(method_a)
    tb = timed(method_b)
    tc = timed(method_c)
    td = timed(method_d)

    if rank == 0:
        print(f"\nOPT-AG-04 Benchmark (W={ws}, N={N}, H={H}, K={K})")
        print(f"  A) 3x dist.all_gather (list)        : {ta:8.1f} us")
        print(f"  B) 3x all_gather_into_tensor         : {tb:8.1f} us")
        print(f"  C) 1x Python packed AG               : {tc:8.1f} us")
        print(f"  D) 1x custom_comm.allgather_batch    : {td:8.1f} us")
        print()
        print(f"  D vs A speedup: {ta/td:.2f}x")
        print(f"  D vs B speedup: {tb/td:.2f}x")
        print(f"  C vs A speedup: {ta/tc:.2f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
