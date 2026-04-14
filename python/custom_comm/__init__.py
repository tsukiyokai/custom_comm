# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""custom_comm: Custom communication operators for Ascend NPU."""

import torch

# Load the C++ extension .so which registers torch.ops.custom_comm.*
try:
    import custom_comm._C  # noqa: F401
except ImportError:
    pass  # Allow import for type-checking / docs without built extension

from custom_comm.ops import allgather_batch

# Register GE converters when torchair is available (graph mode support).
try:
    import custom_comm.converters  # noqa: F401
except ImportError:
    pass

__all__ = ["allgather_batch"]
