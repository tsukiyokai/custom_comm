#!/usr/bin/env python3
# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""4-way benchmark: compare AllGather strategies for OPT-AG-04.

  A) 3x dist.all_gather (list API)
  B) 3x dist.all_gather_into_tensor (matches NPUCommunicator.all_gather)
  C) Python byte-packing: view-as-uint8 + cat + single AG (spike approach)
  D) custom_comm.allgather_batch (C API, Phase 1 decomposed)

Usage:
    torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
"""
import os, time, torch, torch.distributed as dist

HAS_NPU = False
try:
    import torch_npu, custom_comm  # noqa: F401
    HAS_NPU = True
except ImportError:
    pass

WARMUP, ITERS = 50, 200


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
        return
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    dev = torch.device(f"npu:{rank}")
    torch.npu.set_device(rank)
    hcom = dist.distributed_c10d._get_default_group() \
              ._get_backend(dev).get_hccl_comm_name(rank)

    N, H, K = 32, 7168, 8
    x   = torch.randint(0, 127, (N, H), dtype=torch.int8,    device=dev)
    s   = torch.randn(N,                 dtype=torch.float32, device=dev)
    ids = torch.randint(0, 8,   (N, K),  dtype=torch.int32,   device=dev)

    # A) 3x dist.all_gather (list API)
    def method_a():
        for t in [x, s, ids]:
            out = [torch.empty_like(t) for _ in range(ws)]
            dist.all_gather(out, t)

    # B) 3x dist.all_gather_into_tensor (NPUCommunicator pattern)
    def method_b():
        for t in [x, s, ids]:
            out = torch.empty(t.shape[0]*ws, *t.shape[1:], dtype=t.dtype, device=dev)
            dist.all_gather_into_tensor(out, t)

    # C) Python packed: view-uint8 + cat + single AG + split
    def method_c():
        n = N
        u8 = [t.reshape(n,-1).contiguous().view(torch.uint8) for t in [x, s, ids]]
        packed = torch.cat(u8, dim=1)
        out = torch.empty(n*ws, packed.shape[1], dtype=torch.uint8, device=dev)
        dist.all_gather_into_tensor(out, packed)

    # D) custom_comm C API
    xf, sf, kf = x.reshape(-1), s.reshape(-1), ids.reshape(-1)
    def method_d():
        torch.ops.custom_comm.allgather_batch([xf, sf, kf], hcom, ws)

    ta = timed(method_a)
    tb = timed(method_b)
    tc = timed(method_c)
    td = timed(method_d)

    if rank == 0:
        results = [("A) 3x all_gather (list)", ta),
                   ("B) 3x all_gather_into_tensor", tb),
                   ("C) 1x Python packed AG", tc),
                   ("D) 1x allgather_batch (C)", td)]
        mx = max(t for _, t in results)
        print(f"\nOPT-AG-04 Benchmark  W={ws}  N={N} H={H} K={K}")
        print("-" * 60)
        for label, us in results:
            bar = "█" * int(us / mx * 30) if mx > 0 else ""
            print(f"  {label:<34s} {us:8.1f} us  {bar}")
        print()
        print(f"  D vs A: {ta/td:.2f}x   D vs B: {tb/td:.2f}x   D vs C: {tc/td:.2f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
