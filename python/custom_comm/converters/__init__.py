# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GE graph-mode converters for custom_comm operators.

Registers torchair FX-to-GE converters so that torch.compile(backend="torchair")
can lower custom_comm ops into GE IR for graph-mode execution on Ascend NPU.

Import this module after both custom_comm._C (schema) and torchair are available.
"""

try:
    import custom_comm.converters.allgather_batch_converter  # noqa: F401
except ImportError:
    # torchair not installed -- graph mode unavailable, eager mode still works.
    pass
