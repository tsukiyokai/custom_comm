# custom_comm

Custom high-performance communication operators for Ascend NPUs. Built on top of
HCCL/HComm, custom_comm provides batched heterogeneous-dtype collective operations
as PyTorch custom ops via torch_npu, with full support for eager mode, graph mode
(torchair GE and aclGraph), and multi-level profiling.

## Operators

### allgather_batch

Gathers up to 8 tensors of different dtypes and shapes in a single kernel launch,
avoiding the overhead of issuing multiple independent AllGather calls. This is the
core primitive for MoE quantized inference where INT8 activations, FP32 scales,
and INT32 routing indices must be gathered together.

## Prerequisites

- Ascend NPU with CANN 9.0+ toolkit (Atlas A5 / Ascend 950)
- PyTorch 2.9 (see requirements.txt)
- torch_npu 2.9
- Python 3.10+

## Installation

### From source

```bash
source ~/Ascend/set_env.sh
pip install -r requirements.txt
pip install -e .
```

### C++ library only

```bash
cmake -B build && cmake --build build
```

## Quick Start

```python
import torch, torch_npu, custom_comm

torch.distributed.init_process_group(backend="hccl")
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.npu.set_device(rank)

pg = torch.distributed.group.WORLD
hcom = pg._get_backend(torch.device(f"npu:{rank}")).get_hccl_comm_name(rank)

# Gather INT8 data + FP32 scale + INT32 indices in one call
data = torch.randn(2048, 4096, device=f"npu:{rank}").to(torch.int8)
scale = torch.randn(2048, device=f"npu:{rank}", dtype=torch.float32)
ids = torch.randint(0, 8, (2048,), device=f"npu:{rank}", dtype=torch.int32)

results = custom_comm.allgather_batch([data, scale, ids], hcom, world_size)
```

```bash
torchrun --nproc_per_node=8 example.py
```

## Eager Mode

Two execution strategies, selected via environment variable:

### Phase 1: Decomposed (default)

Packs all tensors into a contiguous byte buffer, performs a single `HcclAllGather`,
then unpacks into per-tensor output buffers. Works on all CANN versions.

### Phase 2: CCU Batched

Registers a custom CCU kernel that performs multi-descriptor zero-copy RDMA gathers.
Each descriptor's data is gathered directly into the output buffer without packing or
copying. Requires CANN 9.0+ and HComm CCU API.

```bash
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 your_script.py
```

## Graph Mode

### GE Graph Mode (torchair)

custom_comm registers a GE converter via `@register_fx_node_ge_converter` so that
`allgather_batch` can be lowered into the GE IR when using `torch.compile`:

```python
@torch.compile(backend="torchair")
def fused_gather(x, s, ids, hcom, ws):
    return custom_comm.allgather_batch([x, s, ids], hcom, ws)
```

The converter decomposes `allgather_batch` into multiple `HcomAllGather` GE ops,
preserving per-tensor dtype and shape. This allows the operator to participate in
GE graph optimization passes (operator fusion, memory planning, etc.).

### aclGraph Capture

For the Phase 2 CCU path, custom_comm supports aclGraph capture. When the main
stream enters capture mode (`aclmdlRICaptureBegin`), the operator detects the
capture state, retrieves the CCU slave stream via `HcclThreadResGetInfo`, and
registers it into the graph via `rtStreamAddToModel`. This enables CCU kernel
operations to be recorded and replayed as part of the captured graph.

## Profiling and DFX

custom_comm provides profiling hooks at multiple levels:

- `RECORD_FUNCTION`: `torch.profiler` timeline integration. The op appears as
  `custom_comm::allgather_batch` in PyTorch profiler traces and Chrome traces.
- `aclprofMarkEx`: CANN profiler markers. Begin/end markers bracket the entire
  operation and the CCU kernel launch separately, visible in Ascend Insight.
- **Slave stream events**: When using Phase 2 (CCU), `aclrtEvent` pairs are
  recorded on the CCU slave stream, providing precise device-side kernel
  duration measurement independent of host-side timing overhead.

## Testing

```bash
# Smoke test: verify basic HCCL connectivity
torchrun --nproc_per_node=8 tests/smoke_test.py

# Meta-device shape inference (no NPU required)
pytest tests/ -k "meta or Meta"

# NPU functional tests (Phase 1 and Phase 2)
torchrun --nproc_per_node=8 pytest tests/test_allgather_batch.py

# Graph mode tests
torchrun --nproc_per_node=8 pytest tests/test_graph_mode.py

# Performance benchmarks
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py --ag09
CUSTOM_COMM_USE_CCU=1 torchrun --nproc_per_node=8 tests/bench_allgather_batch.py
```

## License

Apache-2.0
