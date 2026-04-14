# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: allgather_batch vs N separate AllGather calls.

Usage:
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
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ag04", action="store_true")
    args = parser.parse_args()

    if not HAS_NPU:
        print("NPU not available")
        return

    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)

    pg = torch.distributed.group.WORLD
    hcom = pg._get_backend(device).get_hccl_comm_name(rank)
    phase = "CCU" if os.environ.get("CUSTOM_COMM_USE_CCU") == "1" else "Decomposed"

    N, H, K = 32, 7168, 8
    x = torch.randint(0, 127, (N, H), dtype=torch.int8, device=device)
    s = torch.randn(N, dtype=torch.float32, device=device)
    ids = torch.randint(0, 8, (N, K), dtype=torch.int32, device=device)

    ws = world_size

    # Baseline: 3 separate dist.all_gather_into_tensor
    ox = torch.empty(N * ws, H, dtype=torch.int8, device=device)
    os_ = torch.empty(N * ws, dtype=torch.float32, device=device)
    ok = torch.empty(N * ws, K, dtype=torch.int32, device=device)

    def baseline():
        dist.all_gather_into_tensor(ox, x)
        dist.all_gather_into_tensor(os_, s)
        dist.all_gather_into_tensor(ok, ids)

    # allgather_batch (C API)
    xf = x.contiguous().view(-1)
    sf = s.contiguous().view(-1)
    kf = ids.contiguous().view(-1)

    def batched():
        torch.ops.custom_comm.allgather_batch([xf, sf, kf], hcom, ws)

    t_base = timed(baseline)
    t_batch = timed(batched)

    if rank == 0:
        speedup = t_base / t_batch if t_batch > 0 else 0
        saved = t_base - t_batch
        pct = saved / t_base * 100 if t_base > 0 else 0
        print(f"\nOPT-AG-04 Benchmark (Phase1 Decomposed, W={ws})")
        print(f"  N=32, H=7168, K=8 (same as spike)")
        print(f"  3x dist.all_gather_into_tensor : {t_base:8.1f} us")
        print(f"  1x custom_comm.allgather_batch : {t_batch:8.1f} us")
        print(f"  Speedup                        : {t_base/t_batch:8.2f}x")
        print(f"  Saved                          : {t_base - t_batch:8.1f} us ({pct:.1f}%)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
