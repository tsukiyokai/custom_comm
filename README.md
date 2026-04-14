# custom_comm

面向昇腾 NPU 的高性能自定义通信算子库。在 HCCL/HComm 之上实现批量异构集合通信，
通过 PyTorch custom op 接口集成，支持 eager mode 和 graph mode (torchair)。

## 前置依赖

- CANN 9.0+ (Atlas A5 / Ascend 910_95)
- Python 3.10+
- PyTorch 2.6+ / torch_npu 2.6+

## 安装

预编译 wheel（推荐）：

```bash
pip install custom_comm
```

从源码安装（需要 CANN toolkit）：

```bash
source /path/to/Ascend/set_env.sh
pip install -e .
```

## 快速开始

```python
import torch, torch_npu, custom_comm

torch.distributed.init_process_group(backend="hccl")
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.npu.set_device(rank)

pg = torch.distributed.group.WORLD
hcom = pg._get_backend(torch.device(f"npu:{rank}")).get_hccl_comm_name(rank)

# 三合一 AllGather: INT8 数据 + FP32 scale + INT32 topk_ids
data = torch.randn(2048, 4096, dtype=torch.int8, device=f"npu:{rank}")
scale = torch.randn(2048, dtype=torch.float32, device=f"npu:{rank}")
ids = torch.randint(0, 8, (2048,), dtype=torch.int32, device=f"npu:{rank}")

gathered = custom_comm.allgather_batch([data, scale, ids], hcom, world_size)
```

### Graph Mode

```python
@torch.compile(backend="npu")
def fn(x, s, hcom, ws):
    return custom_comm.allgather_batch([x, s], hcom, ws)
```

## 测试

```bash
pytest tests/ -k "meta"                                  # shape 推导（无需 NPU）
torchrun --nproc_per_node=8 -m pytest tests/             # NPU 功能测试
torchrun --nproc_per_node=8 tests/bench_allgather_batch.py --ag09  # 性能基准
```

## License

Apache-2.0
