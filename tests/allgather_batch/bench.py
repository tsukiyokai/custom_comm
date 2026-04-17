#!/usr/bin/env python3
# Copyright (c) 2026 custom_comm authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: compare AllGather strategies for allgather_batch.

    torchrun --nproc_per_node=8 tests/allgather_batch/bench.py
"""
import torch, torch.distributed as dist

HAS_NPU = False
try:
    import torch_npu, custom_comm, custom_comm._C as _C  # noqa: F401,E401
    HAS_NPU = True
except ImportError:
    pass

WARMUP, ITERS = 50, 200


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    t0 = torch.npu.Event(enable_timing=True)
    t1 = torch.npu.Event(enable_timing=True)
    t0.record()
    for _ in range(ITERS):
        fn()
    t1.record()
    torch.npu.synchronize()
    return t0.elapsed_time(t1) * 1000 / ITERS


def main():
    if not HAS_NPU:
        return
    dist.init_process_group(backend="hccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.npu.set_device(rank)
    dev = torch.device(f"npu:{rank}")
    hcom = (dist.distributed_c10d._get_default_group()
            ._get_backend(dev).get_hccl_comm_name(rank))

    N, H, K = 32, 7168, 8
    x   = torch.randint(0, 127, (N, H), dtype=torch.int8,    device=dev)
    s   = torch.randn(N,                 dtype=torch.float32, device=dev)
    ids = torch.randint(0, 8,   (N, K),  dtype=torch.int32,   device=dev)
    tensors = [x, s, ids]
    bws = [t.nbytes // N for t in tensors]

    outs_f = [torch.empty(N * ws, *t.shape[1:], dtype=t.dtype, device=dev)
              for t in tensors]

    def method_a():
        for t in tensors:
            dist.all_gather([torch.empty_like(t) for _ in range(ws)], t)

    def method_b():
        for t in tensors:
            dist.all_gather_into_tensor(
                torch.empty(N * ws, *t.shape[1:], dtype=t.dtype, device=dev), t)

    def method_c():
        u8 = [t.reshape(N, -1).contiguous().view(torch.uint8) for t in tensors]
        packed = torch.cat(u8, dim=1)
        out = torch.empty(N * ws, packed.shape[1], dtype=torch.uint8, device=dev)
        dist.all_gather_into_tensor(out, packed)
        col = 0
        for t in tensors:
            bw = t.nbytes // N
            out[:, col:col + bw].contiguous().view(t.dtype).reshape(
                [N * ws] + list(t.shape[1:]))
            col += bw

    def method_d():
        torch.ops.custom_comm.allgather_batch(tensors, hcom, ws)

    def method_e():
        _C.allgather_batch_eager(tensors, hcom, ws)

    def method_f():
        _C.allgather_batch_inplace(tensors, outs_f, hcom, ws)

    ta, tb, tc = timed(method_a), timed(method_b), timed(method_c)
    td, te, tf = timed(method_d), timed(method_e), timed(method_f)

    if rank == 0:
        W = 36
        rows = [("A) 3 all_gather(list),ds3",       ta),
                ("B) 3 all_gather_into_tensor,pg2",  tb),
                ("C) 1 ag_packed,pure py",           tc),
                ("D) 1 agb,torch.ops(Dispatcher)",   td),
                ("E) 1 agb,pybind11(eager)",         te),
                ("F) 1 agb,pybind11(in-place)",      tf)]
        mx = max(v for _, v in rows)
        print(f"\nOPT-AG-04 Benchmark  W={ws}  N={N}")
        print("-" * 60)
        for label, us in rows:
            bar = "\u2588" * int(us / mx * 30)
            print(f"  {label:<{W}} {us:8.1f} us  {bar}")
        print()
        print(f"  me vs ds:  {ta/tf:.2f}x  (saved {ta - tf:.0f} us)")
        print(f"  me vs pg:  {tb/tf:.2f}x  (saved {tb - tf:.0f} us)")
        print(f"  Dispatcher:   {td - te:+.1f} us")
        print(f"  ReturnValue:  {te - tf:+.1f} us")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
