# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eager-mode tests for the allgather_batch operator.

Layered as:
  TestMetaKernel     — Meta-dispatch shape inference (no NPU)
  TestNpuFunctional  — Phase 1 decomposed HCCL path (needs NPU + torchrun)
  TestCcuPath        — Phase 2 CCU kernel path (needs NPU + torchrun, CUSTOM_COMM_USE_CCU=1)

Graph-mode counterpart: ./test_graph.py
Performance numbers:    ./bench.py

Usage:
  pytest tests/                                         # runs meta-only (others auto-skip)
  torchrun --nproc_per_node=N -m pytest tests/          # full eager suite
"""

import os
import pytest
import torch

DTYPES = [torch.int8, torch.float16, torch.float32, torch.bfloat16]


def make_input(shape, dtype, device="meta"):
    if device == "meta":
        return torch.empty(shape, dtype=dtype, device="meta")
    return torch.randn(shape, device=device).to(dtype)


# ============================================================
# Meta kernel tests (no NPU required)
# ============================================================

@pytest.mark.ext
class TestMetaKernel:
    """Shape inference via Meta dispatch. Runs anywhere."""

    def _call(self, inputs, world_size):
        return torch.ops.custom_comm.allgather_batch(inputs, "dummy", world_size)

    @pytest.mark.parametrize("world_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_single_desc(self, world_size, dtype):
        inp = make_input((128, 64), dtype)
        [out] = self._call([inp], world_size)
        assert out.shape == (128 * world_size, 64)
        assert out.dtype == dtype

    @pytest.mark.parametrize("world_size", [2, 4, 8])
    def test_heterogeneous_dtypes(self, world_size):
        """Multiple tensors with different dtypes."""
        inp_i8 = make_input((256,), torch.int8)
        inp_f32 = make_input((4,), torch.float32)
        outs = self._call([inp_i8, inp_f32], world_size)
        assert outs[0].shape == (256 * world_size,)
        assert outs[0].dtype == torch.int8
        assert outs[1].shape == (4 * world_size,)
        assert outs[1].dtype == torch.float32

    def test_ag09_meta(self):
        """OPT-AG-04: INT8 data + FP32 scale + INT32 topk_ids (3 descs)."""
        data = torch.empty(2048, dtype=torch.int8, device="meta")
        scale = torch.empty(56, dtype=torch.float32, device="meta")
        ids = torch.empty(8, dtype=torch.int32, device="meta")
        outs = torch.ops.custom_comm.allgather_batch(
            [data, scale, ids], "dummy", 8
        )
        assert outs[0].shape == (2048 * 8,)
        assert outs[1].shape == (56 * 8,)
        assert outs[2].shape == (8 * 8,)

    def test_max_desc_count(self):
        inputs = [torch.empty(32, 16, device="meta") for _ in range(8)]
        outs = self._call(inputs, 4)
        assert len(outs) == 8
        assert all(o.shape == (128, 16) for o in outs)

    def _call(self, inputs, ws):
        return torch.ops.custom_comm.allgather_batch(inputs, "hcom", ws)

    def test_empty_dim0(self):
        [out] = self._call([torch.empty(0, 64, device="meta")], 4)
        assert out.shape == (0, 64)

    def test_preserves_dtype(self):
        for dt in [torch.int8, torch.float16, torch.float32, torch.bfloat16]:
            [out] = self._call([torch.empty(16, dtype=dt, device="meta")], 2)
            assert out.dtype == dt


# ============================================================
# NPU functional tests (require device + multi-rank)
# ============================================================

@pytest.mark.npu
@pytest.mark.ext
@pytest.mark.dist
class TestNpuFunctional:
    """Phase 1 (decomposed) eager-mode correctness on NPU.

    Run: torchrun --nproc_per_node=N pytest tests/ -k TestNpuFunctional
    """

    @pytest.fixture(autouse=True)
    def _bind(self, dist_ctx):
        self.rank = dist_ctx.rank
        self.world_size = dist_ctx.world_size
        self.device = dist_ctx.device
        self.hcom = dist_ctx.hcom

    # ---- basic correctness ----

    @pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.float32, torch.bfloat16])
    def test_single_desc(self, dtype):
        n = 128
        # Build on CPU then transfer to avoid NPU arange dtype limitations.
        inp = (torch.arange(n) + self.rank * n).to(dtype).to(self.device).contiguous()
        outs = torch.ops.custom_comm.allgather_batch([inp], self.hcom, self.world_size)
        assert outs[0].shape == (n * self.world_size,)
        for r in range(self.world_size):
            expected = (torch.arange(n) + r * n).to(dtype).to(self.device).contiguous()
            assert torch.equal(outs[0][r * n:(r + 1) * n], expected)

    def test_heterogeneous_int8_fp32(self):
        """OPT-AG-09 core: INT8 + FP32 scale."""
        data = torch.full((2048,), self.rank + 1, dtype=torch.int8, device=self.device)
        scale = torch.full((4,), (self.rank + 1) * 0.5, dtype=torch.float32, device=self.device)
        outs = torch.ops.custom_comm.allgather_batch([data, scale], self.hcom, self.world_size)
        assert len(outs) == 2
        assert outs[0].shape == (2048 * self.world_size,)
        assert outs[1].shape == (4 * self.world_size,)

    def test_three_tensor_pack(self):
        """OPT-AG-04/09: INT8 data + FP32 scale + INT32 topk_ids."""
        x = torch.randint(0, 127, (2048,), dtype=torch.int8, device=self.device)
        s = torch.randn(56, dtype=torch.float32, device=self.device)
        ids = torch.randint(0, 1000, (8,), dtype=torch.int32, device=self.device)
        outs = torch.ops.custom_comm.allgather_batch(
            [x, s, ids], self.hcom, self.world_size
        )
        assert outs[0].shape == (2048 * self.world_size,)
        assert outs[1].shape == (56 * self.world_size,)
        assert outs[2].shape == (8 * self.world_size,)

    def test_repeated_calls(self):
        data = torch.ones(64, device=self.device, dtype=torch.float16)
        for _ in range(100):
            torch.ops.custom_comm.allgather_batch([data], self.hcom, self.world_size)


# ============================================================
# CCU path tests (Phase 2, CUSTOM_COMM_USE_CCU=1)
# ============================================================

@pytest.mark.npu
@pytest.mark.ext
@pytest.mark.dist
class TestCcuPath:
    """Phase 2 CCU kernel tests. Run with CUSTOM_COMM_USE_CCU=1."""

    @pytest.fixture(autouse=True)
    def _bind(self, dist_ctx):
        self.rank = dist_ctx.rank
        self.world_size = dist_ctx.world_size
        self.device = dist_ctx.device
        self.hcom = dist_ctx.hcom

    def test_ccu_matches_decomposed(self):
        """CCU output must match decomposed output bit-exactly."""
        import os
        data = (torch.arange(256) + self.rank).to(torch.int8).to(self.device)
        scale = torch.randn(4, device=self.device, dtype=torch.float32)

        # Phase 1
        os.environ.pop("CUSTOM_COMM_USE_CCU", None)
        out_p1 = torch.ops.custom_comm.allgather_batch([data, scale], self.hcom, self.world_size)

        # Phase 2
        os.environ["CUSTOM_COMM_USE_CCU"] = "1"
        out_p2 = torch.ops.custom_comm.allgather_batch([data, scale], self.hcom, self.world_size)
        os.environ.pop("CUSTOM_COMM_USE_CCU", None)

        assert torch.equal(out_p2[0], out_p1[0])
        assert torch.equal(out_p2[1], out_p1[1])
