# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type-hinted wrappers for custom_comm operators."""

from typing import List

import torch


def allgather_batch(
    inputs: List[torch.Tensor],
    hcom: str,
    world_size: int,
) -> List[torch.Tensor]:
    """Batched AllGather for heterogeneous-dtype tensors.

    Args:
        inputs:     List of contiguous NPU tensors (may differ in dtype/shape).
        hcom:       HCCL communicator group name.
        world_size: Number of ranks in the communication group.

    Returns:
        List of gathered tensors. Each output[i] has shape
        [world_size * inputs[i].shape[0], *inputs[i].shape[1:]].
    """
    return torch.ops.custom_comm.allgather_batch(inputs, hcom, world_size)
