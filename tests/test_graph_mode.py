# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for graph mode (GE converter) and shape inference.

Test tiers:
  1. Meta kernel -- shape inference via Meta dispatch (no NPU/torchair needed)
  2. Converter registration -- verify decorator wired up (needs torchair)
  3. Converter structure -- signature and decomposition logic (needs torchair)

All tests are collectable on macOS; NPU/torchair-dependent tests are skipped
with clear messages when dependencies are absent.

Run:
  pytest --collect-only tests/test_graph_mode.py   # verify collection
  pytest tests/test_graph_mode.py -v               # run available tests
"""

import inspect

import pytest
import torch

# ---- Capability probes ----

HAS_EXT = False
try:
    import custom_comm._C  # noqa: F401 -- triggers TORCH_LIBRARY registration
    HAS_EXT = True
except ImportError:
    try:
        import custom_comm  # noqa: F401
        HAS_EXT = hasattr(torch.ops, "custom_comm")
    except ImportError:
        pass

HAS_TORCHAIR = False
try:
    import torchair  # noqa: F401
    HAS_TORCHAIR = True
except ImportError:
    pass

requires_ext = pytest.mark.skipif(not HAS_EXT, reason="custom_comm._C not built")
requires_torchair = pytest.mark.skipif(not HAS_TORCHAIR, reason="torchair not installed")


# ============================================================
# 1. Meta-dispatch shape inference (no torchair needed)
# ============================================================

@requires_ext
class TestMetaShapeInference:
    """Verify Meta kernel produces correct output shapes and dtypes."""

    @pytest.mark.parametrize("world_size", [1, 2, 4, 8])
    def test_single_input(self, world_size):
        inp = torch.empty(128, 64, dtype=torch.float32, device="meta")
        outs = torch.ops.custom_comm.allgather_batch([inp], "hcom", world_size)
        assert len(outs) == 1
        assert outs[0].shape == (128 * world_size, 64)
        assert outs[0].dtype == torch.float32

    @pytest.mark.parametrize("world_size", [1, 2, 8])
    def test_heterogeneous_dtypes(self, world_size):
        """OPT-AG-09 scenario: INT8 data + FP32 scale."""
        data = torch.empty(2048, dtype=torch.int8, device="meta")
        scale = torch.empty(4, dtype=torch.float32, device="meta")
        outs = torch.ops.custom_comm.allgather_batch(
            [data, scale], "hcom", world_size
        )
        assert len(outs) == 2
        assert outs[0].shape == (2048 * world_size,)
        assert outs[0].dtype == torch.int8
        assert outs[1].shape == (4 * world_size,)
        assert outs[1].dtype == torch.float32

    def test_max_desc_count(self):
        """8 descriptors (MAX_DESC_COUNT)."""
        inputs = [torch.empty(16, device="meta") for _ in range(8)]
        outputs = torch.ops.custom_comm.allgather_batch(inputs, "hcom", 2)
        assert len(outputs) == 8
        for out in outputs:
            assert out.shape == (32,)

    def test_empty_dim0(self):
        """Zero-size dim0 is passed through (0 * world_size = 0)."""
        x = torch.empty(0, 64, device="meta")
        [out] = torch.ops.custom_comm.allgather_batch([x], "hcom", 4)
        assert out.shape == (0, 64)

    def test_multidim_tensor(self):
        """Multi-dimensional tensors: only dim 0 is scaled."""
        x = torch.empty(10, 20, 30, device="meta")
        [out] = torch.ops.custom_comm.allgather_batch([x], "hcom", 3)
        assert out.shape == (30, 20, 30)


# ---- Converter registration tests (require torchair) ----

@pytest.mark.skipif(not HAS_TORCHAIR, reason="torchair not installed")
@pytest.mark.skipif(not HAS_EXT, reason="custom_comm extension not installed")
class TestConverterRegistration:
    """Verify the GE converter is importable and correctly registered."""

    def test_converter_module_importable(self):
        import custom_comm.converters.allgather_batch_converter as conv
        assert callable(conv.convert_allgather_batch)

    def test_converter_signature(self):
        """Converter must accept (inputs, hcom, world_size, *, meta_outputs)."""
        import inspect
        from custom_comm.converters.allgather_batch_converter import convert_allgather_batch

        sig = inspect.signature(convert_allgather_batch)
        params = list(sig.parameters.keys())
        assert "inputs" in params
        assert "hcom" in params
        assert "world_size" in params

    def test_converter_in_registry(self):
        """The converter should be registered in torchair's converter registry."""
        from torchair._ge_concrete_graph import fx2ge_converter
        op = torch.ops.custom_comm.allgather_batch.default
        # The registry key varies by torchair version; check both known locations
        registry = getattr(fx2ge_converter, "_CONVERTERS", None) or \
                   getattr(fx2ge_converter, "converters", {})
        assert op in registry, (
            f"allgather_batch converter not found in torchair registry. "
            f"Sample keys: {list(registry.keys())[:5]}"
        )
