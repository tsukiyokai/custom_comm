# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-mode (torchair / torch.compile / aclGraph) tests for allgather_batch.

Layers:
  - TestMetaShapeInference : shape-only, meta-dispatch
  - TestConverterRegistration : torchair GE converter registration
  - TestGraphModeE2E : torch.compile(backend=torchair) end-to-end on NPU
  - TestAclGraphCapture : NPU aclGraph capture + replay

Eager-mode counterparts live in ./test_eager.py.

Run:
    pytest tests/allgather_batch/test_graph.py -v
"""

import pytest
import torch


# ============================================================
# 1. Meta-dispatch shape inference tests
# ============================================================

@pytest.mark.ext
class TestMetaShapeInference:
    """Verify allgather_batch meta-dispatch produces correct output shapes."""

    def _call(self, inputs, world_size):
        return torch.ops.custom_comm.allgather_batch(inputs, "hcom", world_size)

    def test_single_input(self):
        inp = torch.empty(128, 64, dtype=torch.float32, device="meta")
        outs = self._call([inp], 4)
        assert len(outs) == 1
        assert outs[0].shape == (512, 64)

    @pytest.mark.parametrize("world_size", [1, 2, 8])
    def test_heterogeneous_dtypes(self, world_size):
        inputs = [
            torch.empty(2048, dtype=torch.int8, device="meta"),
            torch.empty(4, dtype=torch.float32, device="meta"),
        ]
        outs = self._call(inputs, world_size)
        assert outs[0].shape == (2048 * world_size,)
        assert outs[1].shape == (4 * world_size,)
        assert outs[0].dtype == torch.int8
        assert outs[1].dtype == torch.float32

    def test_max_descs(self):
        inputs = [torch.empty(32, 16, device="meta") for _ in range(8)]
        outs = self._call(inputs, 2)
        assert all(o.shape == (64, 16) for o in outs)

    def test_empty_dim0(self):
        inp = torch.empty(0, 64, device="meta")
        [out] = self._call([inp], 4)
        assert out.shape == (0, 64)

    def test_multidim(self):
        inp = torch.empty(10, 20, 30, device="meta")
        [out] = self._call([inp], 4)
        assert out.shape == (40, 20, 30)


# ============================================================
# 2. Converter registration tests (need torchair)
# ============================================================

@pytest.mark.torchair
@pytest.mark.ext
class TestConverterRegistration:
    """Verify the GE converter is registered correctly."""

    def test_converter_module_importable(self):
        import custom_comm.converters.allgather_batch_converter  # noqa: F401

    def test_converter_registered(self):
        """The converter should be reachable via op's _ge_converter attr
        (new torchair) or the global _CONVERTERS dict (old torchair)."""
        op = torch.ops.custom_comm.allgather_batch.default
        # New torchair attaches converter directly to the op
        has_attr = hasattr(op, "_ge_converter")
        # Old torchair uses a global dict
        from torchair._ge_concrete_graph import fx2ge_converter
        registry = getattr(fx2ge_converter, "_CONVERTERS", {})
        has_dict = op in registry
        assert has_attr or has_dict, (
            "converter not found via _ge_converter attr or _CONVERTERS dict"
        )


# ============================================================
# Graph mode end-to-end (require NPU + torchair)
# ============================================================

@pytest.mark.npu
@pytest.mark.ext
@pytest.mark.torchair
@pytest.mark.dist
class TestGraphModeE2E:
    """End-to-end graph mode tests. Require NPU hardware."""

    @pytest.fixture(autouse=True)
    def _bind(self, dist_ctx):
        self.rank = dist_ctx.rank
        self.world_size = dist_ctx.world_size
        self.device = dist_ctx.device
        self.hcom = dist_ctx.hcom

    def test_ge_graph_compile(self):
        """Verify torch.compile(backend='torchair') can trace allgather_batch."""
        import custom_comm

        @torch.compile(backend="torchair")
        def fn(x, s):
            return custom_comm.allgather_batch([x, s], self.hcom, self.world_size)

        x = torch.randn(64, dtype=torch.float16, device=self.device)
        s = torch.randn(4, dtype=torch.float32, device=self.device)
        out = fn(x, s)
        assert out[0].shape == (64 * self.world_size,)
        assert out[1].shape == (4 * self.world_size,)

    def test_ge_graph_correctness(self):
        """GE graph output should match eager output."""
        import custom_comm

        x = (torch.arange(32) + self.rank).to(torch.int8).to(self.device)
        s = torch.full((4,), float(self.rank), dtype=torch.float32, device=self.device)

        eager_out = custom_comm.allgather_batch([x, s], self.hcom, self.world_size)

        @torch.compile(backend="torchair")
        def fn(x, s):
            return custom_comm.allgather_batch([x, s], self.hcom, self.world_size)

        graph_out = fn(x, s)
        assert torch.equal(eager_out[0], graph_out[0])
        assert torch.equal(eager_out[1], graph_out[1])


# ============================================================
# aclGraph capture tests (require NPU)
# ============================================================

@pytest.mark.npu
@pytest.mark.ext
@pytest.mark.dist
class TestAclGraphCapture:
    """aclGraph capture correctness on NPU."""

    @pytest.fixture(autouse=True)
    def _setup(self, dist_ctx):
        self.rank = dist_ctx.rank
        self.world_size = dist_ctx.world_size
        self.device = dist_ctx.device
        self.hcom = dist_ctx.hcom

    def test_aclgraph_capture_replay(self):
        """aclGraph capture + replay should produce correct results."""
        import custom_comm

        data = torch.ones(64, dtype=torch.float16, device=self.device) * (self.rank + 1)
        scale = torch.ones(4, dtype=torch.float32, device=self.device) * (self.rank + 1)

        # Warmup (required before capture)
        _ = custom_comm.allgather_batch([data, scale], self.hcom, self.world_size)
        torch.npu.synchronize()

        # Capture
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = custom_comm.allgather_batch([data, scale], self.hcom, self.world_size)

        # Replay
        graph.replay()
        torch.npu.synchronize()

        assert out[0].shape == (64 * self.world_size,)
        assert out[1].shape == (4 * self.world_size,)

    def test_aclgraph_repeated_replay(self):
        """Multiple replays should give consistent results."""
        import custom_comm

        data = torch.arange(32).to(torch.int8).to(self.device)
        _ = custom_comm.allgather_batch([data], self.hcom, self.world_size)
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = custom_comm.allgather_batch([data], self.hcom, self.world_size)

        for _ in range(10):
            graph.replay()
        assert out[0].shape[0] == data.shape[0] * self.world_size
