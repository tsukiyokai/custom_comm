# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmark: Phase 1 (decomposed) vs Phase 2 (CCU batched).

Usage:
  torchrun --nproc_per_node=8 tests/bench_allgather_batch.py

Environment:
  CUSTOM_COMM_USE_CCU=0  -> Phase 1 (default)
  CUSTOM_COMM_USE_CCU=1  -> Phase 2
"""

import os
import time
import argparse

import torch

try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

# ============================================================
# Benchmark parameters
# ============================================================

MSG_SIZES = [
    ("4KB",   4 * 1024),
    ("256KB", 256 * 1024),
    ("2.5MB", 2_500_000),
    ("10MB",  10 * 1024 * 1024),
]

DESC_COUNTS = [1, 2, 4, 8]
WARMUP = 10
REPEAT = 100


def bench_allgather_batch(rank, world_size, hcom, device):
    """Run benchmark matrix."""
    results = []

    for desc_count in DESC_COUNTS:
        for label, msg_bytes in MSG_SIZES:
            # Each desc gets msg_bytes of INT8 data
            inputs = [
                torch.zeros(msg_bytes, dtype=torch.int8, device=device)
                for _ in range(desc_count)
            ]

            # Warmup
            for _ in range(WARMUP):
                torch.ops.custom_comm.allgather_batch(inputs, hcom, world_size)
            torch.npu.synchronize()

            # Timed
            t0 = time.perf_counter()
            for _ in range(REPEAT):
                torch.ops.custom_comm.allgather_batch(inputs, hcom, world_size)
            torch.npu.synchronize()
            elapsed_us = (time.perf_counter() - t0) / REPEAT * 1e6

            results.append((desc_count, label, elapsed_us))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
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
    backend = pg._get_backend(device)
    hcom = backend.get_hccl_comm_name(rank)

    phase = "CCU" if os.environ.get("CUSTOM_COMM_USE_CCU") == "1" else "Decomposed"

    results = bench_allgather_batch(rank, world_size, hcom, device)

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"AllGatherBatch Benchmark  (Phase: {phase}, W={world_size})")
        print(f"{'=' * 60}")
        print(f"{'descs':>6} {'msg_size':>10} {'latency_us':>12}")
        print(f"{'-' * 6:>6} {'-' * 10:>10} {'-' * 12:>12}")
        for desc_count, label, lat in results:
            print(f"{desc_count:>6} {label:>10} {lat:>12.1f}")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
