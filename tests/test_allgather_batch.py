# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for custom_comm.allgather_batch.

Test layers:
  - Meta kernel tests: run anywhere (macOS / CPU), verify shape inference.
  - NPU functional tests: require NPU device, verify bit-exact correctness.

Usage:
  pytest --collect-only tests/  # macOS: verify collection
  pytest tests/                 # NPU: run all
"""

import pytest
import torch

# ============================================================
# Fixtures & Helpers
# ============================================================

HAS_NPU = False
try:
    import torch_npu  # noqa: F401
    HAS_NPU = torch.npu.is_available()
except ImportError:
    pass

# Load extension to register torch.ops.custom_comm schema
HAS_EXT = False
try:
    import custom_comm  # noqa: F401
    HAS_EXT = True
except ImportError:
    pass

requires_npu = pytest.mark.skipif(not HAS_NPU, reason="NPU device not available")
requires_ext = pytest.mark.skipif(not HAS_EXT, reason="custom_comm extension not built")

DTYPES_META = [torch.int8, torch.float16, torch.float32, torch.bfloat16]
WORLD_SIZES = [2, 4, 8]


def make_input(shape, dtype, device="meta"):
    """Create a test tensor on the given device."""
    if device == "meta":
        return torch.empty(shape, dtype=dtype, device="meta")
    return torch.randn(shape, device=device).to(dtype)


# ============================================================
# Meta kernel tests (no device required, but extension must be loaded)
# ============================================================

@requires_ext
class TestMetaKernel:
    """Verify allgather_batch_meta produces correct output shapes."""

    @pytest.mark.parametrize("dtype", DTYPES_META)
    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_single_desc_shape(self, dtype, world_size):
        """Single input tensor: output dim0 = input dim0 * world_size."""
        inp = make_input((128, 64), dtype, device="meta")
        outputs = torch.ops.custom_comm.allgather_batch([inp], "dummy", world_size)
        assert len(outputs) == 1
        assert outputs[0].shape == (128 * world_size, 64)
        assert outputs[0].dtype == dtype

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_heterogeneous_dtypes(self, world_size):
        """Multiple inputs with different dtypes."""
        inp_int8 = make_input((256,), torch.int8, device="meta")
        inp_fp32 = make_input((4,), torch.float32, device="meta")

        outputs = torch.ops.custom_comm.allgather_batch(
            [inp_int8, inp_fp32], "dummy", world_size
        )
        assert len(outputs) == 2
        assert outputs[0].shape == (256 * world_size,)
        assert outputs[0].dtype == torch.int8
        assert outputs[1].shape == (4 * world_size,)
        assert outputs[1].dtype == torch.float32

    def test_max_desc_count(self):
        """MAX_DESC_COUNT=8 inputs should work."""
        inputs = [make_input((32, 16), torch.float16, device="meta") for _ in range(8)]
        outputs = torch.ops.custom_comm.allgather_batch(inputs, "dummy", 2)
        assert len(outputs) == 8
        for out in outputs:
            assert out.shape == (64, 16)

    def test_empty_tensor(self):
        """Zero-element tensor: output should also be zero-element."""
        inp = make_input((0, 64), torch.float32, device="meta")
        outputs = torch.ops.custom_comm.allgather_batch([inp], "dummy", 4)
        assert len(outputs) == 1
        assert outputs[0].shape == (0, 64)
        assert outputs[0].dtype == torch.float32

    def test_empty_tensor_mixed(self):
        """Mix of empty and non-empty tensors."""
        empty = make_input((0, 32), torch.float16, device="meta")
        nonempty = make_input((16, 32), torch.float16, device="meta")
        outputs = torch.ops.custom_comm.allgather_batch(
            [empty, nonempty], "dummy", 2
        )
        assert outputs[0].shape == (0, 32)
        assert outputs[1].shape == (32, 32)

    def test_multidimensional(self):
        """3D tensor: only dim0 is scaled."""
        inp = make_input((10, 20, 30), torch.float32, device="meta")
        outputs = torch.ops.custom_comm.allgather_batch([inp], "dummy", 4)
        assert outputs[0].shape == (40, 20, 30)

    def test_preserves_dtype(self):
        """Each output preserves its input's dtype."""
        inputs = [
            make_input((16,), torch.int8, device="meta"),
            make_input((16,), torch.float16, device="meta"),
            make_input((16,), torch.bfloat16, device="meta"),
            make_input((16,), torch.float32, device="meta"),
        ]
        outputs = torch.ops.custom_comm.allgather_batch(inputs, "dummy", 2)
        for inp, out in zip(inputs, outputs):
            assert out.dtype == inp.dtype


# ============================================================
# NPU functional tests (require device + multi-rank)
# ============================================================

@requires_npu
class TestNpuFunctional:
    """Functional tests on NPU device.

    These tests require a multi-rank environment.
    Run with: torchrun --nproc_per_node=N pytest tests/test_allgather_batch.py::TestNpuFunctional
    """

    @pytest.fixture(autouse=True)
    def setup_dist(self):
        """Initialize distributed environment if not already done."""
        if not torch.distributed.is_initialized():
            pytest.skip("Distributed environment not initialized")
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.device = torch.device(f"npu:{self.rank}")
        torch.npu.set_device(self.device)

        # Get hcom group name
        pg = torch.distributed.group.WORLD
        backend = pg._get_backend(self.device)
        self.hcom = backend.get_hccl_comm_name(self.rank)

    @pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.float32, torch.bfloat16])
    def test_single_desc_correctness(self, dtype):
        """Single desc: verify gathered output is correct."""
        n = 128
        inp = torch.arange(n, device=self.device, dtype=dtype) + self.rank * n

        outputs = torch.ops.custom_comm.allgather_batch(
            [inp], self.hcom, self.world_size
        )

        assert outputs[0].shape == (n * self.world_size,)
        # Verify each rank's slice
        for r in range(self.world_size):
            expected = torch.arange(n, device=self.device, dtype=dtype) + r * n
            actual = outputs[0][r * n : (r + 1) * n]
            assert torch.equal(actual, expected), f"Mismatch at rank {r}"

    def test_heterogeneous_dtype_correctness(self):
        """INT8 data + FP32 scale: core use case (OPT-AG-09)."""
        data = torch.full((2048,), self.rank + 1, device=self.device, dtype=torch.int8)
        scale = torch.full((4,), (self.rank + 1) * 0.5, device=self.device, dtype=torch.float32)

        outputs = torch.ops.custom_comm.allgather_batch(
            [data, scale], self.hcom, self.world_size
        )

        assert len(outputs) == 2
        assert outputs[0].shape == (2048 * self.world_size,)
        assert outputs[1].shape == (4 * self.world_size,)

        for r in range(self.world_size):
            expected_data = torch.full((2048,), r + 1, device=self.device, dtype=torch.int8)
            assert torch.equal(outputs[0][r * 2048 : (r + 1) * 2048], expected_data)

            expected_scale = torch.full((4,), (r + 1) * 0.5, device=self.device, dtype=torch.float32)
            assert torch.equal(outputs[1][r * 4 : (r + 1) * 4], expected_scale)

    def test_repeated_calls_stability(self):
        """100 repeated calls should not leak or crash."""
        inp = torch.ones(64, device=self.device, dtype=torch.float16)
        for _ in range(100):
            outputs = torch.ops.custom_comm.allgather_batch(
                [inp], self.hcom, self.world_size
            )
            assert outputs[0].shape[0] == 64 * self.world_size
