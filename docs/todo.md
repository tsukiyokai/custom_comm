# 测试矩阵

## Eager Mode

| 测试                                             | 文件                                                    | 状态  |
| ------------------------------------------------ | ------------------------------------------------------- | ----- |
| shape推导(meta dispatch, 4 dtype x 3 world_size) | TestMetaKernel::test_single_desc                        | 通过  |
| 异构dtype shape (INT8+FP32)                      | TestMetaKernel::test_heterogeneous_dtypes               | 通过  |
| OPT-AG-09 三desc shape (int8+fp32+int32)         | TestMetaKernel::test_ag09_meta                          | 通过  |
| 8 desc极限                                       | TestMetaKernel::test_max_desc_count                     | 通过  |
| 空tensor / dtype保持                             | TestMetaKernel::test_empty_tensor, test_preserves_dtype | 通过  |
| Phase 1单desc正确性(4 dtype)                     | TestNpuFunctional::test_single_desc                     | 待NPU |
| Phase 1 INT8+FP32 正确性                         | TestNpuFunctional::test_heterogeneous_int8_fp32         | 待NPU |
| Phase 1三合一(AG-04/09)                          | TestNpuFunctional::test_three_tensor_pack               | 待NPU |
| Phase 1重复调用稳定性                            | TestNpuFunctional::test_repeated_calls                  | 待NPU |
| Phase 2 CCU vs Decomposed                        | TestCcuPath::test_ccu_matches_decomposed                | 待NPU |

## Graph Mode

| 测试             | 文件                                        | 状态  |
| ---------------- | ------------------------------------------- | ----- |
| GE converter注册 | TestConverterRegistration                   | PASS  |
| GE graph编译     | TestGraphModeE2E::test_ge_graph_compile     | 待NPU |
| GE graph正确性   | TestGraphModeE2E::test_ge_graph_correctness | 待NPU |

## aclGraph Capture

| 测试                   | 文件                                         | 状态  |
| ---------------------- | -------------------------------------------- | ----- |
| capture + replay正确性 | TestAclGraphCapture::test_capture_and_replay | 待NPU |
| 重复replay一致性       | TestAclGraphCapture::test_repeated_replay    | 待NPU |

## Benchmark

| 场景                            | 命令                                             | 状态  |
| ------------------------------- | ------------------------------------------------ | ----- |
| 同构 N (1/2/4/8 desc, 4K-10M) | `tests/allgather_batch/bench.py`                       | 待NPU |
| AG-09 三合一                  | `tests/allgather_batch/bench.py --ag09`                | 待NPU |
| Phase 2 CCU                   | `CUSTOM_COMM_USE_CCU=1 tests/allgather_batch/bench.py` | 待NPU |
