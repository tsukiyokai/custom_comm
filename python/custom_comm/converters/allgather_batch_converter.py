# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GE converter for torch.ops.custom_comm.allgather_batch.

Graph-mode limitation (REPLAN):
    There is no native GE op for batched heterogeneous AllGather. The eager-mode
    implementation fuses N descriptors into a single CCU kernel launch, but the
    GE graph IR only exposes HcomAllGather for single-tensor AllGather.

    This converter decomposes the batched op into N individual HcomAllGather
    calls.  This is functionally correct but loses the single-CCU-task batching
    benefit in graph mode.  Other torch.compile optimizations (operator fusion,
    memory planning) still apply.

    If a future CANN release provides a native GE "AllGatherBatch" op that can
    schedule CCU kernels, this converter should be upgraded to the single-op path.
"""

import logging
from typing import Any, List

import torch

from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec

logger = logging.getLogger(__name__)

# NOTE: This module must be imported after custom_comm._C has been loaded
# (which registers the torch.ops.custom_comm schema).  The normal import path
# via custom_comm.__init__ -> custom_comm.converters guarantees this ordering.
# Do NOT import custom_comm here -- it would create a circular import.


def _record_hcom_group(hcom: str, world_size: int) -> str:
    """Best-effort recording of the HCCL group in the GE graph.

    torchair's standard path uses get_group_name_and_record(tag, rank_list,
    group_size), but our op receives a pre-resolved hcom string without the
    original tag / rank_list.  We attempt to record using the world group
    assumption (rank_list = [0..world_size-1]).

    For non-world groups the caller should ensure the group is already recorded
    in the GE graph (e.g. by having another collective op on the same group
    earlier in the graph, which is the common case in practice).
    """
    try:
        from torchair._ge_concrete_graph.hcom_utils import get_default_ge_graph
        rank_list = list(range(world_size))
        get_default_ge_graph().record_process_group(hcom, rank_list, "")
    except Exception:
        logger.debug(
            "Could not record hcom group '%s' in GE graph; "
            "proceeding without recording (group may already be known).",
            hcom,
        )
    return hcom


@register_fx_node_ge_converter(torch.ops.custom_comm.allgather_batch.default)
def convert_allgather_batch(
    inputs: List[Tensor],
    hcom: str,
    world_size: int,
    *,
    out: List[Tensor] = None,
    meta_outputs: List[TensorSpec] = None,
):
    """Decompose allgather_batch into N individual HcomAllGather ops.

    Each input[i] is independently gathered via ge.HcomAllGather, producing
    an output whose dim-0 is scaled by world_size.  The result list preserves
    input ordering and per-element dtype.
    """
    group_name = _record_hcom_group(hcom, world_size)

    outputs = []
    for inp in inputs:
        gathered = ge.HcomAllGather(inp, group=group_name, rank_size=world_size)
        outputs.append(gathered)
    return outputs
