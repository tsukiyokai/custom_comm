#!/usr/bin/env python3
# Copyright (c) 2026 custom_comm authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""5-way AG benchmark for OPT-AG-04.

  A) 3x dist.all_gather            (list API)
  B) 3x all_gather_into_tensor     (baseline)
  C) Python packed AG + unpack      (reference)
  D) allgather_batch via Dispatcher (torch.ops)
  E) allgather_batch via pybind11   (direct call)

Usage:
    torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
"""
import time, torch, torch.distributed as dist

HAS_NPU = False
try:
    import torch_npu  # noqa: F401
    import custom_comm  # noqa: F401
    import custom_comm._C as _C
    HAS_NPU = True
except ImportError:
    pass

WARMUP, ITERS = 50, 200


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    dist.barrier()
    t0 = torch.npu.Event(enable_timing=True)
    t1 = torch.npu.Event(enable_timing=True)
    t0.record()
    for _ in range(ITERS):
        fn()
    t1.record()
    torch.npu.synchronize()
    return t0.elapsed_time(t1) * 1000 / ITERS  # us


def main():
    if not HAS_NPU:
        return
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.npu.set_device(rank)
    dev = torch.device(f"npu:{rank}")
    hcom = (dist.distributed_c10d._get_default_group()
            ._get_backend(dev).get_hccl_comm_name(rank))

    N, H, K = 32, 7168, 8
    x   = torch.randint(0, 127, (N, H), dtype=torch.int8,    device=dev)
    s   = torch.randn(N,                 dtype=torch.float32, device=dev)
    ids = torch.randint(0, 1000, (N, K), dtype=torch.int32,   device=dev)
    tensors = [x, s, ids]
    bws = [t.nbytes // N for t in tensors]

    # ── A) 3x dist.all_gather (list API) ───────────────────────
    def method_a():
        for t in tensors:
            out = [torch.empty_like(t) for _ in range(ws)]
            dist.all_gather(out, t)

    # ── B) 3x all_gather_into_tensor ───────────────────────────
    def method_b():
        for t in tensors:
            out = torch.empty(N * ws, *t.shape[1:], dtype=t.dtype, device=dev)
            dist.all_gather_into_tensor(out, t)

    # ── C) Python packed AG + unpack ───────────────────────────
    def method_c():
        u8 = [t.reshape(N, -1).contiguous().view(torch.uint8) for t in tensors]
        packed = torch.cat(u8, dim=1)
        out = torch.empty(N * ws, packed.shape[1], dtype=torch.uint8, device=dev)
        dist.all_gather_into_tensor(out, packed)
        col = 0
        for i, t in enumerate(tensors):
            bw = bws[i]
            sl = out[:, col:col + bw].contiguous()
            sl.view(t.dtype).reshape([N * ws] + list(t.shape[1:]))
            col += bw

    # ── D) custom_comm via Dispatcher ──────────────────────────
    def method_d():
        torch.ops.custom_comm.allgather_batch(tensors, hcom, ws)

    # ── E) custom_comm via pybind11 (no Dispatcher) ────────────
    def method_e():
        custom_comm._C.allgather_batch_eager(tensors, hcom, ws)

    ta = timed(method_a)
    tb = timed(method_b)
    tc = timed(method_c)
    td = timed(method_d)
    te = timed(method_e)

    if rank == 0:
        W = 40
        results = [
            ("A) 3x all_gather (list)",      ta),
            ("B) 3x all_gather_into_tensor",  tb),
            ("C) Python packed AG + unpack",   tc),
            ("D) allgather_batch (Dispatcher)", td),
            ("E) allgather_batch (pybind11)",   te),
        ]
        mx = max(v for _, v in results)
        print(f"\nOPT-AG-04 Benchmark  W={ws}  N={N}")
        print("-" * 64)
        for label, us in results:
            bar = "\u2588" * int(us / mx * 30)
            print(f"  {label:<{W}} {us:8.1f} us  {bar}")
        print()
        print(f"  D vs E (Dispatcher overhead):  {td - te:+.1f} us")
        print(f"  E vs C (remaining C++ tax):    {te - tc:+.1f} us")
        print(f"  D vs B (total speedup):        {tb/td:.2f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
