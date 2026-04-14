# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for allgather_batch: homogeneous and OPT-AG-09 scenarios.

Usage:
    torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
    torchrun --nproc_per_node=8 tests/bench_allgather_batch.py --ag09
    CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
"""

import os
import time
import argparse

import torch

HAS_NPU = False
try:
    import torch_npu  # noqa: F401
    import custom_comm  # noqa: F401
    HAS_NPU = True
except ImportError:
    pass

WARMUP = 10
REPEAT = 100

# ---- Homogeneous benchmark (INT8, varying size and desc count) ----

MSG_SIZES = [
    ("4KB",   4 * 1024),
    ("256KB", 256 * 1024),
    ("2.5MB", 2_500_000),
    ("10MB",  10 * 1024 * 1024),
]
DESC_COUNTS = [1, 2, 4, 8]


def bench_homogeneous(hcom, world_size, device):
    results = []
    for desc_count in DESC_COUNTS:
        for label, msg_bytes in MSG_SIZES:
            inputs = [
                torch.zeros(msg_bytes, dtype=torch.int8, device=device)
                for _ in range(desc_count)
            ]
            for _ in range(WARMUP):
                torch.ops.custom_comm.allgather_batch(inputs, hcom, world_size)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                torch.ops.custom_comm.allgather_batch(inputs, hcom, world_size)
            torch.npu.synchronize()
            us = (time.perf_counter() - t0) / 100 * 1e6
            results.append((desc_count, label, us))
    return results


# ---- OPT-AG-09: INT8 + FP32 scale + INT32 topk_ids ----

def bench_ag09(hcom, world_size, device):
    """Three-tensor batched AllGather mimicking quantized MoE AGRS."""
    configs = [
        ("2.5MB+scale+ids", 2560000, 56, 8),  # ~2.5MB int8, 56 fp32, 8 int32
        ("256KB+scale+ids",  262144, 56, 8),
        ("64KB+scale+ids",    65536, 56, 8),
    ]
    results = []
    for label, n_int8, n_fp32, n_int32 in configs:
        x = torch.zeros(n_int8, dtype=torch.int8, device=device)
        s = torch.zeros(n_fp32, dtype=torch.float32, device=device)
        ids = torch.zeros(n_int32, dtype=torch.int32, device=device)
        for _ in range(WARMUP):
            torch.ops.custom_comm.allgather_batch([x, s, ids], hcom, world_size)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            torch.ops.custom_comm.allgather_batch([x, s, ids], hcom, world_size)
        torch.npu.synchronize()
        us = (time.perf_counter() - t0) / 100 * 1e6
        results.append((label, us))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ag09", action="store_true", help="Run OPT-AG-09 scenario")
    args, _ = parser.parse_known_args()

    if not HAS_NPU:
        print("NPU not available, skipping benchmark")
        return

    torch.distributed.init_process_group(backend="hccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)

    pg = torch.distributed.group.WORLD
    hcom = pg._get_backend(device).get_hccl_comm_name(rank)
    phase = "CCU" if os.environ.get("CUSTOM_COMM_USE_CCU") == "1" else "Decomposed"

    if args.ag09:
        results = bench_ag09(hcom, world_size, device)
        if rank == 0:
            print(f"\nOPT-AG-09 Benchmark (Phase: {phase}, W={world_size})")
            print(f"  {'config':<30s}  {'us':>10s}")
            print(f"  {'-'*30}  {'-'*10}")
            for label, us in results:
                print(f"  {label:<30s}  {us:>10.1f}")
    else:
        results = bench_homogeneous(hcom, world_size, device)
        if rank == 0:
            print(f"\nAllGatherBatch Benchmark (Phase: {phase}, W={world_size})")
            print(f"  {'descs':>5} {'size':>10} {'us':>10}")
            for dc, label, us in results:
                print(f"  {dc:>5} {label:>10} {us:>10.1f}")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
