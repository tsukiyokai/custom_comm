# 测试矩阵

## Eager Mode

| 测试 | 文件 | 状态 |
|------|------|------|
| shape 推导 (meta dispatch, 4 dtype x 3 world_size) | TestMetaKernel::test_single_desc | 通过 |
| 异构 dtype shape (INT8+FP32) | TestMetaKernel::test_heterogeneous_dtypes | 通过 |
| OPT-AG-09 三 desc shape (int8+fp32+int32) | TestMetaKernel::test_ag09_meta | 通过 |
| 8 desc 极限 | TestMetaKernel::test_max_desc_count | 通过 |
| 空 tensor / dtype 保持 | TestMetaKernel::test_empty_tensor, test_preserves_dtype | 通过 |
| Phase 1 单 desc 正确性 (4 dtype) | TestNpuFunctional::test_single_desc | 待 NPU |
| Phase 1 INT8+FP32 正确性 | TestNpuFunctional::test_heterogeneous_int8_fp32 | 待 NPU |
| Phase 1 三合一 (AG-04/09) | TestNpuFunctional::test_three_tensor_pack | 待 NPU |
| Phase 1 重复调用稳定性 | TestNpuFunctional::test_repeated_calls | 待 NPU |
| Phase 2 CCU vs Decomposed | TestCcuPath::test_ccu_matches_decomposed | 待 NPU |

## Graph Mode

| 测试 | 文件 | 状态 |
|------|------|------|
| GE converter 注册 | TestConverterRegistration | PASS |
| GE graph 编译 | TestGraphModeE2E::test_ge_graph_compile | 待 NPU |
| GE graph 正确性 | TestGraphModeE2E::test_ge_graph_correctness | 待 NPU |

## aclGraph Capture

| 测试 | 文件 | 状态 |
|------|------|------|
| capture + replay 正确性 | TestAclGraphCapture::test_capture_and_replay | 待 NPU |
| 重复 replay 一致性 | TestAclGraphCapture::test_repeated_replay | 待 NPU |

## Benchmark

| 场景 | 命令 | 状态 |
|------|------|------|
| 同构 INT8 (1/2/4/8 desc, 4K-10M) | `bench_allgather_batch.py` | 待 NPU |
| AG-09 三合一 | `bench_allgather_batch.py --ag09` | 待 NPU |
| Phase 2 CCU | `CUSTOM_COMM_USE_CCU=1 bench_allgather_batch.py` | 待 NPU |
