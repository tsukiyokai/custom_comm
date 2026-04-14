# Code Review: custom_comm v0.1.0

审查范围: 全部 17 个源文件（Phase 1 + Phase 2 + Phase 3），交叉校验 design-doc.md。
审查日期: 2026-04-14

## 变更概述

custom_comm 是一个独立的华为昇腾自定义通信算子仓库，以 PyTorch 自定义算子形式交付
HcclAllGatherBatch（多类型批量 AllGather）。项目分四层：

- C API 层 (ops/allgather_batch/): 通信核心逻辑 + CCU kernel，4 头文件 + 4 源文件
- Torch Binding (torch_ext/csrc/): TORCH_LIBRARY schema + PrivateUse1/Meta dispatch，2 文件
- Python (python/custom_comm/): .so 加载 + 类型提示 + GE converter，4 文件
- Build System: CMakeLists.txt + FindCANN.cmake + setup.py，3 文件
- Tests: 功能测试 + 图模式测试 + 性能基准，3 文件

总计约 1400 行代码（不含设计文档）。

## 审查发现

共发现 11 个问题（严重 4 / 一般 4 / 建议 3）。4 个严重问题已全部修复 (RESOLVED)。

---

### #1 [严重] [RESOLVED] setup.py 缺少 C API 源文件，torch extension 无法解析核心符号

- 位置: `setup.py:31-33`
- 规则: 模式 3 (构建系统注册)
- 置信度: 确定

问题代码:

```python
sources=[
    "torch_ext/csrc/ops_registration.cpp",
    "torch_ext/csrc/allgather_batch.cpp",
],
```

`allgather_batch.cpp:128` 调用 `HcclAllGatherBatch()`，该符号定义在
`ops/allgather_batch/src/all_gather_batch.cc`（CMake 构建目标 `custom_comm_ops`）。
但 setup.py 的 `sources` 列表仅包含 torch_ext/ 下的两个文件，`libraries` 仅链接
`hcomm`。`HcclAllGatherBatch` 既不在 sources 中，也不在 libhcomm.so 中。

已验证: 通过 `nm` 检查 build/libcustom_comm_ops.dylib 确认该符号仅存在于
custom_comm_ops 库中。Python 层无任何 dlopen 机制加载此库。

运行时表现: Linux 下 `dlopen(RTLD_LAZY)` 使 import 成功但首次调用崩溃；
macOS 下链接阶段即失败。

修复建议:

```python
sources=[
    "torch_ext/csrc/ops_registration.cpp",
    "torch_ext/csrc/allgather_batch.cpp",
    "ops/allgather_batch/src/all_gather_batch.cc",
    "ops/allgather_batch/src/decomposed_strategy.cc",
    "ops/allgather_batch/src/engine_ctx.cc",
    "ops/allgather_batch/src/ccu_kernel_ag_batch_mesh1d.cc",
],
```

---

### #2 [严重] [RESOLVED] CCU 路径缺少 sendCount * elemSize 溢出检查

- 位置: `ops/allgather_batch/src/ccu_kernel_ag_batch_mesh1d.cc:321`
- 规则: 红线 1.3 (整数溢出)
- 置信度: 确定

问题代码:

```cpp
uint64_t bytes = taskArg.descs[d].sendCount * elemSize;
```

Decomposed 路径在 `decomposed_strategy.cc:42-44` 有完整的溢出检查:

```cpp
if (descs[i].sendCount > UINT64_MAX / elemSize) {
    return HCCL_E_PARA;
}
```

但 CCU 路径的 `LaunchBatchedAGKernel()` 完全跳过了此检查。同一函数的
`selfOffset` 计算 (line 325) 也存在类似问题: `rankId * bytes` 在 bytes
已溢出的前提下会产生二次错误。

入口 `ValidateParams()` (`all_gather_batch.cc:36-41`) 仅验证 `DtypeSize != 0`，
不检查乘积溢出，因此无法依赖上游保护。

修复建议:

```cpp
uint64_t elemSize = DtypeSize(taskArg.descs[d].dataType);
if (elemSize == 0 || taskArg.descs[d].sendCount > UINT64_MAX / elemSize) {
    return HCCL_E_PARA;
}
uint64_t bytes = taskArg.descs[d].sendCount * elemSize;
```

---

### #3 [严重] [RESOLVED] CcuContext 缺少 initialized 标志，部分初始化后误判为成功

- 位置: `ops/allgather_batch/src/engine_ctx.cc:47-50, 56-63`
- 规则: 红线 1.4 (变量未初始化)
- 置信度: 较确定
- 反证: 已检查 HcclEngineCtxCreate 是否零初始化分配内存。SDK API 行为不透明，
  无法确认。即使零初始化，零值 CcuKernelHandle/ThreadHandle 传入
  HcclCcuKernelLaunch 的行为未定义。设计文档 line 848 明确提到此风险并建议
  添加 initialized flag，但代码未实现。反证条件不成立。

问题代码:

```cpp
struct CcuContext {
    CcuKernelHandle kernelHandle;  // 无默认值
    ThreadHandle    threadHandle;  // 无默认值
    // 缺少: bool initialized = false;
};

HcclResult InitCcuContext(HcclComm comm) {
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, CTX_TAG, COMM_ENGINE_CCU,
                         &ctx, &ctxSize) == HCCL_SUCCESS && ctx != nullptr) {
        return HCCL_SUCCESS;  // 误判: ctx 存在但可能部分初始化
    }
    // ...
    HCCL_CHECK(HcclEngineCtxCreate(...));  // 成功
    HCCL_CHECK(HcclChannelAcquire(...));   // 成功
    HCCL_CHECK(RegisterBatchedAGKernel(...)); // 若此处失败 →
    // ctx 已存在但 kernelHandle 无效
    // 下次调用 InitCcuContext: HcclEngineCtxGet 命中，返回 SUCCESS
    // LaunchCcuKernel 使用无效 handle → UB
```

修复建议:

```cpp
struct CcuContext {
    CcuKernelHandle kernelHandle{};
    ThreadHandle    threadHandle{};
    bool initialized = false;
};

// InitCcuContext fast path:
if (...HCCL_SUCCESS && ctx != nullptr) {
    auto *ccuCtx = static_cast<CcuContext *>(ctx);
    if (ccuCtx->initialized) return HCCL_SUCCESS;
    // else: partial init, fall through to retry
}
// ... slow path ...
ccuCtx->initialized = true;  // 全部步骤成功后才置位
```

---

### #4 [严重] [RESOLVED] 设计文档 GeneArgs 编码布局与实现严重不一致

- 位置: `design-doc.md:343-366` vs `ops/allgather_batch/src/ccu_kernel_ag_batch_mesh1d.cc:44-57`
- 规则: 模式 6 (Host-Kernel 一致性)
- 置信度: 确定

设计文档描述的 GeneArgs 编码:

    头部: [descCount, rankId, rankSize] (3 slots)
    每路: [sendBuf, recvBuf, sendCount, dataType, sendBytes, reserved] (6 slots)
    总计: 3 + 6*8 = 51 slots, 4 SQEs

实际实现的 GeneArgs 编码:

    头部: [token] (1 slot)
    每路: [sendAddr, recvAddr, sendBytes, selfOffset] (4 slots)
    总计: 1 + 4*8 = 33 slots, 3 SQEs

关键差异:
1. 设计文档无 RDMA token，实现以 token 为首 slot
2. 设计文档传 descCount/rankId/rankSize，实现通过 rankSize 编译期固化 +
   sendBytes==0 跳过替代 descCount
3. 设计文档传 sendCount+dataType，实现改为 Host 预计算 sendBytes+selfOffset
4. SQE 数量从 4 降至 3（减少硬件调度开销）

代码本身的 GeneArgs 编码与 Algorithm() Load() 顺序经验证一致，不存在编解码
错配。实际实现比设计文档方案更优。但设计文档未同步更新，会误导后续维护者。

修复建议: 更新 design-doc.md 的 GeneArgs 序列化章节，使其与实现一致。

---

### #5 [一般] ccu_kernel_ag_batch_mesh1d.h 仍是 Phase 1 空 stub

- 位置: `ops/allgather_batch/inc/ccu_kernel_ag_batch_mesh1d.h:17-19`
- 规则: 1.3.3 (TODO/注释与代码不符)
- 置信度: 确定

问题代码:

```cpp
// Phase 2 will add:
//   #include <hcomm/ccu/ccu_kernel.h>
//   class CcuKernelAllGatherBatchMesh1D : public hcomm::CcuKernel { ... };
```

Phase 2 已完成，CcuKernelAllGatherBatchMesh1D 类已在 `.cc` 文件中实现。
但头文件仍是空 stub，注释声称 "Phase 2 will add"。标记为 FROZEN CONTRACT
但实际内容过时。

修复建议: 将类声明移入头文件，或将注释更新为 "Implementation in .cc file"
并移除 FROZEN CONTRACT 标记（空头文件没有冻结价值）。

---

### #6 [一般] HCCL_CHECK 宏无错误日志

- 位置: `ops/allgather_batch/inc/common.h:20-26`
- 规则: DFX (错误可观测性)
- 置信度: 确定

问题代码:

```cpp
#define HCCL_CHECK(call)                                              \
    do {                                                              \
        HcclResult _ret = (call);                                     \
        if (_ret != HCCL_SUCCESS) {                                   \
            return _ret;                                              \
        }                                                             \
    } while (0)
```

设计文档 line 828 注释 `/* 日志: 函数名 + 错误码 + 文件位置 */`，但实现中无任何
日志输出。所有 C API 层的错误都是"静默返回错误码"，在多层调用链中极难定位失败点。
对比 HCCL_TORCH_CHECK (`allgather_batch.cpp:63-69`) 至少包含 file:line 信息。

修复建议: 添加 HCCL 日志宏或 fprintf(stderr):

```cpp
#define HCCL_CHECK(call)                                              \
    do {                                                              \
        HcclResult _ret = (call);                                     \
        if (_ret != HCCL_SUCCESS) {                                   \
            HCCL_ERROR("[custom_comm] %s failed: %d at %s:%d",        \
                       #call, _ret, __FILE__, __LINE__);              \
            return _ret;                                              \
        }                                                             \
    } while (0)
```

---

### #7 [一般] [RESOLVED] setup.py 链接库列表不完整

- 位置: `setup.py:44`
- 规则: 模式 3 (构建系统注册)
- 置信度: 较确定
- 反证: 已检查 `all_gather_batch.cc:66` 的 `aclprofMarkEx` 调用。此符号在
  `libascendcl.so` 中。如果 #1 修复后将 C API 源文件纳入 torch extension，
  则 `aclprofMarkEx` 需要链接 ascendcl。当前 setup.py 仅链接 hcomm。
  设计文档 line 513 的 setup.py 示例包含 `libraries=["hcomm", "ascendcl"]`。
  反证条件不成立。

问题代码:

```python
libraries=["hcomm"],
```

修复建议:

```python
libraries=["hcomm", "ascendcl"],
```

注: 此问题在 #1 修复前被掩盖（C API 源文件未编入 torch extension），
修复 #1 后必须同步修复。

---

### #8 [一般] ProfMark macOS stub 参数类型不匹配

- 位置: `ops/allgather_batch/src/all_gather_batch.cc:69`
- 规则: 2.7.1 (类型安全)
- 置信度: 确定

问题代码:

```cpp
#else
static inline void ProfMark(const char * /*msg*/, void * /*stream*/) {}
#endif
```

非 macOS 版本的签名为 `ProfMark(const char *msg, aclrtStream stream)`，
macOS stub 将第二个参数改为 `void*`。虽然 `aclrtStream` 底层可能是指针类型，
但显式使用 `void*` 破坏了类型一致性。

修复建议: macOS stub 也应使用 `aclrtStream`，或在 macOS 下 typedef
`using aclrtStream = void*;`。

---

### #9 [建议] 测试缺少 error path 覆盖

- 位置: `tests/test_allgather_batch.py`
- 规则: 测试完备性
- 置信度: 确定

当前测试仅覆盖 happy path。缺少以下 error case 测试:

- `inputs.size() > MAX_DESC_COUNT (8)`: 应抛出 TORCH_CHECK 异常
- 空 inputs 列表: 应抛出 "inputs must be non-empty"
- 非连续 tensor: 应抛出 "must be contiguous"
- world_size <= 0: 应抛出 "must be positive"
- 0 维 tensor: 应抛出 "must be at least 1-dimensional"

这些 error path 在 `allgather_batch.cpp:84-109` 中都有 TORCH_CHECK，
但缺少测试验证。Meta dispatch 测试可在 macOS 上覆盖前 4 个场景。

修复建议: 添加 `TestMetaKernel` 下的 `test_error_*` 用例，使用
`pytest.raises(RuntimeError)`。

---

### #10 [建议] Benchmark 未覆盖核心异构 dtype 场景

- 位置: `tests/bench_allgather_batch.py:49-52`
- 规则: 测试完备性
- 置信度: 确定

问题代码:

```python
inputs = [
    torch.zeros(msg_bytes, dtype=torch.int8, device=device)
    for _ in range(desc_count)
]
```

所有 benchmark 用例使用同构 INT8。OPT-AG-09 的核心场景是 INT8 data + FP32 scale
的异构 dtype 组合。同构 INT8 无法反映异构场景下 GeneArgs 序列化和多路 WriteNb
的真实开销。

修复建议: 增加 OPT-AG-09 场景 benchmark:

```python
# OPT-AG-09: 2.5MB INT8 + 4KB FP32
inputs = [
    torch.zeros(2_500_000, dtype=torch.int8, device=device),
    torch.zeros(1024, dtype=torch.float32, device=device),
]
```

---

### #11 [建议] GE converter 的 `out` 参数声明但未使用

- 位置: `python/custom_comm/converters/allgather_batch_converter.py:68`
- 规则: 2.1.3 (冗余代码)
- 置信度: 待确认
- 反证: torchair 的 `register_fx_node_ge_converter` 框架可能通过 keyword
  argument 传递 `out` 参数。未能确认 torchair 是否在某些执行路径中使用此参数。
  如果框架要求 converter 签名必须包含 `out`，则此参数是必要的，移除会导致
  TypeError。需人工确认 torchair converter 协议。

问题代码:

```python
def convert_allgather_batch(
    inputs: List[Tensor],
    hcom: str,
    world_size: int,
    *,
    out: List[Tensor] = None,    # 声明但未使用
    meta_outputs: List[TensorSpec] = None,
):
```

修复建议: 确认 torchair converter 协议后，若 `out` 非必需则移除。

---

## 交叉校验: 设计文档一致性

| 检查项 | 设计文档 | 实际实现 | 判定 |
|--------|---------|---------|------|
| C API 签名 | design-doc.md:526-531 | hccl_custom_allgather_batch.h:43-47 | 一致 |
| HcclAllGatherDesc 布局 | design-doc.md:311-322 | hccl_custom_allgather_batch.h:21-27 | 一致 |
| AllGatherBatchTaskArg | design-doc.md:330-337 | common.h:66-71 | 不一致 (设计文档继承 CcuTaskArg, 实现未继承; 合理简化) |
| GeneArgs 编码 | design-doc.md:343-366 | ccu_kernel_ag_batch_mesh1d.cc:44-57 | 不一致 (见 #4) |
| EngineCtx: CcuContext | design-doc.md:374-381 | engine_ctx.cc:47-50 | 不一致 (缺 channels, initialized) |
| TORCH_LIBRARY schema | design-doc.md:551-553 | ops_registration.cpp:9-12 | 一致 |
| Meta impl 签名 | design-doc.md:579-591 | allgather_batch.cpp:138-157 | 一致 |
| Phase dispatch 机制 | design-doc.md:240-241 | all_gather_batch.cc:51-57, 82 | 一致 |
| Error checking 宏 | design-doc.md:825-834 | common.h:20-26 | 不一致 (缺日志, 见 #6) |
| DFX: initialized flag | design-doc.md:848 | engine_ctx.cc:47-50 | 未实现 (见 #3) |
| setup.py 链接库 | design-doc.md:513 | setup.py:44 | 不一致 (缺 ascendcl, 见 #7) |
| GE converter | design-doc.md:284-289 | allgather_batch_converter.py:62-83 | 一致 (均为 decomposed) |

## 交叉校验: 约束合规性

| 约束 (back.txt / design-doc.md) | 验证结果 |
|------|---------|
| 不侵入 PTA/HCCL/HCOMM | 通过: 仅使用 SDK 头文件和 libhcomm.so 公开/内部 API |
| descCount 1-8 | 通过: ValidateParams 检查 (all_gather_batch.cc:33) |
| sendBuf/recvBuf 非空 | 通过: ValidateParams 检查 (all_gather_batch.cc:37) |
| CUSTOM_COMM_USE_CCU 分发 | 通过: UseCcuPath() 正确实现 (all_gather_batch.cc:51-57) |
| macOS 语法检查模式 | 通过: `__APPLE__` guard + `-undefined dynamic_lookup` |
| Phase 1 作为 correctness oracle 保留 | 通过: Decomposed 路径永久保留 |
| 支持 eager mode | 通过: PrivateUse1 dispatch 正确注册 |
| 支持入图 (GE) | 通过: GE converter 已注册，decomposed 降级策略合理 |
| RECORD_FUNCTION profiling | 通过: allgather_batch.cpp:82 |

## 交叉校验: 安全红线

| 红线项 | 检查结果 |
|--------|---------|
| RED-01 除零 | 通过: DtypeSize 返回 0 时 ValidateParams 拦截 |
| RED-02 数组越界 | 通过: descCount <= MAX_DESC_COUNT 检查后用于数组索引 |
| RED-03 整数溢出 | 通过 (已修复): CCU 路径增加溢出检查 (#2 RESOLVED) |
| RED-04 变量初始化 | 通过 (已修复): CcuContext 成员值初始化 + initialized 标志 (#3 RESOLVED) |
| RED-05 空指针 | 通过: 入口检查 descs/sendBuf/recvBuf 非空 |
| RED-06 资源匹配 | 通过: Decomposed 路径有 cleanup label; EngineCtx 由 HCCL 管理生命周期 |
| RED-07 数据竞争 | 通过: HCCL comm 串行调用约束在调用方保证，UseCcuPath() 的 getenv 读取可接受 |

## 总结

GATE_PASS (amended)

4 个严重问题已全部修复:
- #1: setup.py 补齐 C API 源文件 + ascendcl 链接库 (同时修复 #7)
- #2: CCU 路径增加 sendCount * elemSize 溢出检查，与 decomposed 路径一致
- #3: CcuContext 增加 initialized 标志，部分初始化时返回 HCCL_E_INTERNAL
- #4: design-doc.md GeneArgs 章节与 CcuContext 结构已同步至实际实现

代码的整体架构设计良好：层间职责清晰，Phase dispatch 机制正确，CCU kernel
的 Algorithm()/GeneArgs() 编解码一致，Python 层的 import guard 和 fallback
设计合理。测试框架的 Meta/NPU 分层策略使 macOS 开发可行。

剩余 3 个一般问题 (#5, #6, #8) 和 3 个建议级问题 (#9, #10, #11) 可在后续迭代中处理。
