# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: allgather_batch vs N separate AllGather calls.

Usage:
    torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
    torchrun --nproc_per_node=8 tests/bench_allgather_batch.py --ag04
"""

import os
import time
import argparse

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
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6


def bench_ag04(hcom, world_size, device):
    """OPT-AG-04: INT8(N,H) + FP32(N) + INT32(N,K), 3 AG -> 1."""
    N, H, K = 32, 7168, 8

    x = torch.randint(0, 127, (N, H), dtype=torch.int8, device=device)
    s = torch.randn(N, dtype=torch.float32, device=device)
    ids = torch.randint(0, 100, (N, K), dtype=torch.int32, device=device)

    # Pre-allocate outputs for baseline
    ox = torch.empty(N * 8, H, dtype=torch.int8, device=device)
    os_ = torch.empty(N * 8, dtype=torch.float32, device=device)
    ok = torch.empty(N * 8, K, dtype=torch.int32, device=device)

    def baseline():
        dist.all_gather_into_tensor(ox, x)
        dist.all_gather_into_tensor(os_, s)
        dist.all_gather_into_tensor(ok, ids)

    xf = x.contiguous().view(-1)
    sf = s.contiguous().view(-1)
    kf = ids.contiguous().view(-1)

    def batched():
        torch.ops.custom_comm.allgather_batch([xf, sf, kf], hcom, world_size)

    world_size = dist.get_world_size()
    t_base = timed(baseline)
    t_batch = timed(batched)
    return t_base, t_batch


def bench_homogeneous(hcom, world_size, device):
    """Homogeneous benchmark: varying desc count and size."""
    results = []
    for desc_count in [1, 2, 4, 8]:
        for label, nbytes in [("4KB", 4096), ("256KB", 256*1024), ("2.5MB", 2560000)]:
            inputs = [torch.zeros(nbytes, dtype=torch.int8, device=device)
                      for _ in range(desc_count)]
            us = timed(lambda: torch.ops.custom_comm.allgather_batch(
                inputs, hcom, world_size))
            results.append((desc_count, label, us))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ag04", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    args, _ = parser.parse_known_args()

    if not HAS_NPU:
        print("NPU not available")
        return

    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)

    pg = dist.group.WORLD
    hcom = pg._get_backend(device).get_hccl_comm_name(rank)
    phase = "CCU" if os.environ.get("CUSTOM_COMM_USE_CCU") == "1" else "Decomposed"

    if args.ag04:
        t_base, t_batch = bench_ag04(hcom, world_size, device)
        if rank == 0:
            speedup = t_base / t_batch if t_batch > 0 else 0
            print(f"\nOPT-AG-04 Benchmark (Phase: {phase}, W={world_size})")
            print(f"  3x separate AG : {t_base:10.1f} us")
            print(f"  1x allgather_batch : {t_batch:10.1f} us")
            print(f"  Speedup        : {speedup:10.2f}x")
            print(f"  Saved          : {t_base - t_batch:10.1f} us ({(1 - t_batch/t_base)*100:.1f}%)")
    else:
        results = []
        for n_desc in [1, 2, 4, 8]:
            for label, size in [("4KB", 4096), ("256KB", 262144), ("2.5MB", 2500000)]:
                inputs = [torch.zeros(size, dtype=torch.int8, device=device)
                          for _ in range(n_desc)]
                us = timed(lambda: torch.ops.custom_comm.allgather_batch(
                    inputs, hcom, world_size))
                results.append((n_desc, label, us))
        if rank == 0:
            print(f"\nAllGather Batch Benchmark (W={world_size})")
            print(f"  {'descs':>5} {'size':>10} {'us':>10}")
            for dc, lb, us in results:
                print(f"  {dc:>5} {lb:>10} {us:>10.1f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
