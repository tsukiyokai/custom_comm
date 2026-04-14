# HcclAllGatherBatch 需求分析

日期: 2026-04-13
输入: back.txt (项目范围), design-notes.md (技术设计, 612行)
方法: 基于性能分析视角的需求分解与风险评估

---

# ==== 1. 功能分解

## 1.1 C API: HcclAllGatherBatch

```c
typedef struct {
    void        *sendBuf;
    uint64_t     sendCount;
    HcclDataType dataType;
    void        *recvBuf;
} HcclAllGatherDesc;

HcclResult HcclAllGatherBatch(
    const HcclAllGatherDesc *descs, uint32_t descCount,
    HcclComm comm, aclrtStream stream);
```

语义: 等价于descCount次独立HcclAllGather，但在单个CCU task内完成。

关键约束:
- descCount上限: MAX_DESC_COUNT = 8，与Mesh1D rankSize对齐
  (design-notes.md:423)
- 各路dtype可异构(int8 + fp32等)
- sendBuf/recvBuf必须是device memory
- recvBuf大小 = sendCount * sizeof(dataType) * worldSize

背景(OPT-AG-09): MoE量化路径对INT8数据和FP32 scale分别执行独立
AllGather。scale仅占总数据量约3%，但触发完整AG调度(约8-10us)。
HcclAllGatherBatch消除这种冗余launch开销。(design-notes.md:12-15)


## 1.2 Torch Extension

注册三层impl:

```cpp
TORCH_LIBRARY(custom_comm, m) {
    m.def("allgather_batch(Tensor[] inputs, str hcom, "
          "int world_size) -> Tensor[]");
}
// PrivateUse1: NPU设备实现，调用C API
// Meta:        shape推导 (output_i = [world_size * input_i[0], ...])
// Autograd:    不需要(通信算子无梯度)
```

hcom参数获取路径(公开API，无需修改PTA):
- Python: `pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)`
- C++: 接收 `c10::string_view hcom` 参数
- 实现位置: ProcessGroupHCCL.cpp:3475-3524, pybind11 Init.cpp:409-417
  (design-notes.md:127-128)


## 1.3 Python Wrapper

```python
gathered = torch.ops.custom_comm.allgather_batch(
    [x_int8, x_scale],   # Tensor[] - 不同dtype的input列表
    hcom,                  # str - group name
    world_size,            # int
)  # -> Tensor[] - gathered后的tensor列表
```

TORCH_LIBRARY的schema自动生成Python binding，无需额外wrapper层。
op-plugin已有成熟模式可参照(AllGatherBaseMatmulKernelNpuOpApi.cpp)。


## 1.4 Graph Mode

两个层次:

GE converter(Phase 3):
```python
@register_fx_node_ge_converter(
    torch.ops.custom_comm.allgather_batch.default)
def convert_allgather_batch(inputs, hcom, world_size, meta_outputs=None):
    group_name = get_group_name_and_record(tag, rank_list, group_size)
    return [ge.custom_op("AllGatherBatch", ...)]
```
- hcom处理: torchair hcom_utils.py的get_group_name_and_record()
- cache场景: codegen_refresh_cache_pgname()自动处理
  (design-notes.md:132-133)

aclGraph capture(Phase 3): CCU kernel入图需要CaptureSlaveStreams机制:
1. 主流发起aclmdlRICaptureBegin
2. 算子入口检测capture状态(aclmdlRICaptureGetInfo)
3. HcclThreadResGetInfo获取CCU thread绑定的slave stream
4. rtStreamAddToModel将slave stream添加到主流model
5. 后续HcclCcuKernelLaunch自动被录制

(design-notes.md:546-557)

待确认项:
- Tensor[]在GE converter中的映射方式(可变长度input list如何映射
  多个GE输入?)
- ge.custom_op能否调度CCU kernel?
- HcclThreadResGetInfo API在CANN 9.0.0中的可用性
  (文档标注CANN V100R001C17 Eco版本新增)


## 1.5 Profiling / DFX

分层策略，每层独立可用:

| 层 | 机制 | 输出 | Phase |
|----|------|------|-------|
| PyTorch | RECORD_FUNCTION | profiler timeline op标记 | 1+ |
| Host粗计时 | gettimeofday | 端到端us | 1+ |
| msprof | aclprofMarkEx(stream) | MindStudio命名标记 | 2+ |
| CCU内部 | AddCcuProfiling | 逐路opName/dataSize | 2+ |
| Device精确 | slave stream + aclrtEvent | CCU真实device耗时 | 3 |

关键注意: user stream上的aclrtRecordEvent只反映host提交时间，
不反映CCU执行时间。精确device计时需要在CCU的slave stream上打event，
依赖HcclThreadResGetInfo(与aclGraph capture同一API依赖)。
Phase 1-2用host gettimeofday + aclrtSynchronizeStream做粗糙计时。
(design-notes.md:486-493)


---

# ==== 2. 非功能约束

## 2.1 平台与SDK

- SDK: CANN 9.0.0，`--devel`或`--full`安装
- 硬件: A5 (910_95) only
- 通信机制: CCU(硬件微码调度)，非AIV-MTE
- 不区分训练/推理

SDK安装路径通过ASCEND_CANN_PACKAGE_PATH定位
(默认/usr/local/Ascend/ascend-toolkit/latest)。
CCU kernel是纯C++代码(继承CcuKernel)，不需要bisheng/ccec编译器。
bisheng仅在未来扩展AIV计算kernel时需要。(design-notes.md:372-377)


## 2.2 非侵入约束

三个不侵入:
- PTA (torch_npu): 不修改任何源码
- HCCL: 不修改HCCL库
- HCOMM: 不修改HCOMM库

允许使用的扩展点:

| 扩展点 | 路径 | 稳定性 | 用途 |
|--------|------|--------|------|
| HCCL公开API | include/hccl/ | 公开稳定 | 通信资源获取 |
| HCOMM公开API | include/hcomm/ | 公开稳定 | 数据面原语 |
| CCU编程框架 | pkg_inc/hcomm/ccu/ | 内部API | CcuKernel继承 |
| 运行时链接 | lib64/libhcomm.so | 稳定 | 符号链接 |
| 算子注册 | TORCH_LIBRARY | PyTorch公开 | eager mode |
| 图模式注册 | GE converter | torchair公开 | graph mode |


## 2.3 构建约束

- 独立repo，独立构建和发布，不依赖HCCL构建系统
- CMake构建，通过ASCEND_CANN_PACKAGE_PATH发现SDK
- 参照omni-ops的目录结构和构建模式(design-notes.md:56-57)
- Torch extension通过NpuExtension(setup.py)构建


---

# ==== 3. 风险评估

## 3.1 CCU Kernel复杂度

风险等级: 高

WriteNb对齐约束:
- MTE路径要求1024B块对齐(design-notes.md:205)
- CCU WriteNb是否有相同约束: 待确认
- 若有对齐约束，小buffer(如4KB fp32 scale)的sendCount须为
  对齐粒度的整数倍，或需padding
- 验证方式: 参照HCCL源码中AllGather的CCU实现
  (hccl/src/ops/all_gather/template/ccu/)

GeneArgs序列化限制:
- 每个SQE slot容量有限(推测13个, design-notes.md:567)
- 编码: 固定头部3 slots + 每路描述符6 slots
- 随descCount线性增长:

| descCount | 总slots | SQE数 | 额外调度开销 |
|-----------|---------|-------|-------------|
| 1 | 9 | 1 | 无 |
| 2 | 15 | 2 | 约1-2us(未确认) |
| 4 | 27 | 3 | 约2-4us(未确认) |
| 8 | 51 | 4 | 约3-6us(未确认) |

- SQE拆分的额外调度开销需实测确认

CcuRep DSL:
- 无官方文档，唯一参考是HCCL源码(hccl/src/ops/*/template/ccu/)
- 微码生成通过CcuKernelMgr::Translate，行为不透明
- 调试手段极有限: CCU微程序在硬件上执行，无printf/断点能力
- 正确性风险 > 可用性风险(design-notes.md:166-167)

缓解策略: Phase 1先验证正确性(不碰CCU)，Phase 2用Phase 1
输出作为bit-exact oracle。CCU kernel逐步引入，从最简单的
WriteNb循环开始。


## 3.2 跨平台开发

风险等级: 中

开发环境(macOS)与目标平台(aarch64 + A5 NPU)的能力边界:

| 能力 | macOS | aarch64 + NPU |
|------|-------|---------------|
| C++语法检查 | 可 | 可 |
| TORCH_LIBRARY注册逻辑 | 可 | 可 |
| Meta impl(shape推导) | 可 | 可 |
| Python wrapper/测试框架 | 可 | 可 |
| 链接libhcomm.so | 不可 | 可 |
| CCU kernel执行 | 不可 | 可 |
| 多卡通信测试 | 不可 | 可 |
| Profiling | 不可 | 可 |

缓解:
- CMake条件编译: macOS下stub掉NPU调用
- 单元测试分层: shape推导在macOS可测，通信逻辑仅在NPU环境测
- 代码组织: NPU相关代码隔离到单独编译单元


## 3.3 SDK API稳定性

风险等级: 中(短期低，长期中-高)

API稳定性分层:

| 层 | 路径 | 稳定性 | Phase依赖 |
|----|------|--------|-----------|
| 公开API | include/hccl/ | 稳定 | 全Phase |
| 公开API | include/hcomm/ | 稳定 | 全Phase |
| 内部API | pkg_inc/hccl/ | 实验性 | Phase 2+ |
| 内部API | pkg_inc/hcomm/ccu/ | 无跨版本承诺 | Phase 2+ |

项目锁定CANN 9.0.0，短期风险可控。
接口已版本化(_v1后缀, design-notes.md:165)，
迁移到CANN 10.x时可能新增_v2而保留_v1。

关键依赖点(任何变更都需适配):
- CcuKernel基类(继承接口)
- WriteNb/ReadNb/NotifyRecord/Wait原语(通信操作核心)
- HcclCcuKernelRegister/Launch C API(注册和启动入口)

CommLib.xml确认全部32个CCU头文件以install_type="devel"发布
(design-notes.md:148-151)。libhcomm.so导出了CcuKernel的全部
成员方法(design-notes.md:153-158)，可达性已验证。


## 3.4 Phase 1 Decomposed vs Phase 2 CCU 性能权衡

风险等级: 低(性能层面Phase 2必然更优，风险在开发层面)

核心判断: Phase 1和Phase 2不是运行时的二选一，而是开发迭代路径。
Phase 2在运行时永远优于Phase 1，因为零额外HBM流量。
"风险"在于Phase 2的CCU开发能否按期交付(见3.1)。

量化对比(OPT-AG-09场景: buf1=2.5MB int8, buf2=4KB fp32, W=8):

| 指标 | Phase 1 | Phase 2 |
|------|---------|---------|
| kernel launch次数 | 3 | 1 |
| 额外HBM流量 | 约45MB | 约5MB(仅self-copy) |
| 额外延迟 | pack约3us + unpack约20us | 约0us |
| 端到端开销 | AG时间 + 约53us | AG时间 + 约10us |

(数据来源: design-notes.md:253, 265, 288, 297)

Phase 1作为correctness oracle的价值不可替代: 即使Phase 2
性能更优，Phase 1的bit-exact输出是Phase 2正确性验证的唯一基准。


---

# ==== 4. 性能特征分析

## 4.1 Phase 1: Decomposed策略

两种分解方式及其性能特征:

方式A -- N次独立HcclAllGather(最简单但非选定方案):
- 延迟: N * (T_launch + T_ag(size_i))
- T_launch约8-10us/次(design-notes.md:13)
- descCount=2时总launch开销约16-20us
- 各次调用在同一stream上保序，无需显式sync
- 缺点: launch开销随descCount线性增长，小buffer放大效应显著
  (4KB scale的AG通信本身约1us，但launch约8-10us)

方式B -- Byte-packing(选定方案, design-notes.md策略1):
- 步骤: view-as-uint8 + cat -> HcclAllGather(UINT8) -> split + contiguous
- launch次数: 固定3次(pack kernel + AG + unpack kernel)，
  不随descCount增长
- 额外HBM流量: pack约sum(size_i), unpack约sum(size_i) * W
  - OPT-AG-09: pack约2.5MB, unpack约20MB (design-notes.md:253)
- 优势: launch开销封顶，大descCount场景优于方式A
- 劣势: unpack需.contiguous()触发全量拷贝(non-contiguous view不能
  直接作为后续kernel输入)
- stream sync: 仅在最终需要结果时sync(aclrtSynchronizeStream)

方式B优于方式A的临界点:
- descCount=1: 方式A 1次launch，方式B 3次launch -> 方式A更优
- descCount=2: 方式A 2次launch(约16-20us)，方式B 3次launch(约24-30us)
  但方式B有pack/unpack开销 -> 方式A仍更优
- descCount >= 4: 方式A 4次launch(约32-40us)，方式B仍3次 -> 方式B
  开始追平(但要加上pack/unpack的HBM开销)
- 实际选择方式B的理由: OPT-AG-09场景descCount=2，方式B虽然launch
  更多，但与已验证的Python byte-packing方案bit-exact对齐，
  降低验证成本


## 4.2 Phase 2: CCU Batched策略(零拷贝)

单CCU kernel内完成所有desc的AllGather:

```
CCU Algorithm():
    PreSync()                       // 硬件同步屏障
    for i in 0..descCount-1:
        WriteNb * (W-1)             // 发送到所有远端peer
        LocalCopyNb * 1             // 本地self-copy
    PostSync()                      // 硬件同步屏障
```

性能模型(alpha-beta):
- T_phase2 = T_ccu_launch + sum(alpha_i + size_i / BW_link) + T_sync
- T_ccu_launch: 约8-10us(单次, design-notes.md:13)
- alpha_i: 每路WriteNb启动延迟(CCU硬件调度，远小于host launch)
- BW_link: HCCS链路带宽(A5 910_95精确值未确认)
- T_sync: PreSync + PostSync硬件同步，约2us(design-notes.md:297)

零拷贝收益:
- 无workspace allocation
- 数据直达最终目的地址: sendBuf -> 远端recvBuf对应位置
- 仅self-copy产生本地HBM流量: 约5MB for W=8(design-notes.md:288)
- 对比Phase 1的约45MB，减少约89%本地HBM流量

各路串行的代价:
- CCU微程序内for循环，各desc顺序执行
- 小buffer(4KB scale): 传输约1us，串行代价可忽略
- 大buffer(2.5MB int8): 传输约15us(估算，依赖HCCS带宽，未确认)
- CCU硬件调度粒度远小于host调度，串行开销被压缩

首次调用初始化:
- HcclEngineCtxCreate + HcclChannelAcquire + HcclCcuKernelRegister
- 估算约200-500us一次性开销(含CcuRep->微码翻译，未确认)
- 后续调用通过HcclEngineCtxGet幂等缓存，无额外初始化


## 4.3 Crossover分析

运行时crossover: 不存在。Phase 2在运行时始终优于Phase 1。
- Phase 1额外HBM流量 >= 0(pack + unpack)
- Phase 2额外HBM流量 约等于 0(零拷贝)
- 两者的通信部分(AllGather本身)耗时相同(底层同样走CCU路径)

按消息大小分析绝对收益:

| 消息大小(per desc) | Phase 1 unpack开销 | Phase 2额外开销 | 节省 |
|--------------------|-------------------|----------------|------|
| < 1KB | < 1us | 约0us | 可忽略 |
| 1KB - 10MB | 1 - 50us | 约0us | 明确 |
| > 100MB | > 500us(未确认) | 约0us | 显著 |

按descCount分析:

| descCount | Phase 1(方式B)launch | Phase 2 GeneArgs SQE | Phase 2优势 |
|-----------|---------------------|---------------------|-------------|
| 1 | 3次 | 1个 | launch节省约16-20us |
| 2 | 3次 | 2个 | launch + unpack节省 |
| 8 | 3次 | 4个 | 同上，SQE开销略增 |

OPT-AG-09典型场景(descCount=2, 2.5MB+4KB, W=8):
- Phase 1总额外开销: 约53us(3次launch + pack + unpack)
- Phase 2总额外开销: 约12us(1次launch + sync)
- 节省约41us/call，约77%的额外开销
- 在每forward pass调用一次的场景，累计节省可观

真正的crossover是开发决策:
- Phase 1: 约2周开发，运行时有可量化的performance tax
- Phase 2: 约3-4周开发，运行时最优，但CCU编程风险(见3.1)
- Phase 1作为correctness oracle无法省略
- 渐进路线: Phase 1验证正确性，Phase 2替换为最优实现


---

# ==== 5. 各Phase验收标准

## Phase 0: 环境验证

- [ ] 独立repo CMake构建成功(不依赖HCCL构建系统)
- [ ] TORCH_LIBRARY注册可编译运行(import后torch.ops.custom_comm可见)
- [ ] HcclAllGather host API可调用(链接libhcomm.so成功)
- [ ] CCU API可达: HcclCcuKernelRegister链接成功
- [ ] pkg_inc/hcomm/ccu/头文件在devel安装下可达


## Phase 1: Decomposed Eager Mode

功能:
- [ ] C API HcclAllGatherBatch入口可调用
- [ ] byte-packing结果与OPT-AG-09 Python方案A bit-exact一致
- [ ] Meta impl shape推导正确
      (output_i.shape = [world_size * input_i.shape[0], ...])
- [ ] 2卡/4卡/8卡正确性通过

性能:
- [ ] RECORD_FUNCTION在torch.profiler timeline上可见
- [ ] Host gettimeofday计时在预期范围内
      (AG通信时间 + pack/unpack overhead < 策略1预期值)

稳定性:
- [ ] 重复调用100次无崩溃/泄漏
- [ ] 不同dtype组合均通过:
      int8+fp32, fp16+fp32, int8+fp16+fp32


## Phase 2: CCU Batched AllGather

功能:
- [ ] CcuKernel注册(HcclCcuKernelRegister)和启动(Launch)成功
- [ ] 零拷贝验证: 无中间workspace分配，recvBuf地址即最终目的
- [ ] 输出与Phase 1 bit-exact一致(Phase 1为oracle)

性能:
- [ ] kernel launch次数从3降至1(profiling确认)
- [ ] 端到端延迟优于Phase 1(排除首次调用初始化开销)
- [ ] CCU AddCcuProfiling报告逐路metrics
- [ ] aclprofMarkEx在MindStudio timeline上可见

稳定性:
- [ ] 重复调用1000次无崩溃/泄漏
- [ ] HcclCommDestroy后无悬空handle
- [ ] 多stream并发场景无竞争


## Phase 3: Graph Mode + Optimization

功能:
- [ ] GE converter注册，torch.compile路径编译通过
- [ ] aclGraph capture成功(CaptureSlaveStreams机制)
- [ ] slave stream aclrtEvent精确device计时可用

性能:
- [ ] 图模式执行性能 >= eager模式(无退化)
- [ ] 可选: LoopGroup升级后大消息(> 100MB)性能提升


---

# ==== 6. 依赖关系

## 6.1 构建依赖链

层次从底向上:

1. SDK头文件 (CANN 9.0.0 devel安装提供)
2. C API (ops/allgather_batch/) -- 依赖层1
3. Torch Extension (torch_extension/csrc/) -- 依赖层2 + PyTorch + torch_npu
4. Python Binding (自动生成) -- 依赖层3
5. Graph Mode (torch_extension/converters/) -- 依赖层3 + torchair


## 6.2 各层依赖明细

| 层 | 依赖项 | 提供 |
|----|--------|------|
| SDK头文件 | CANN 9.0.0 --devel | include/hccl/, pkg_inc/hcomm/ccu/, libhcomm.so |
| C API | SDK头文件 + libhcomm.so + ACL | HcclAllGatherBatch(), HcclAllGatherDesc |
| Torch Ext | C API + PyTorch + torch_npu | torch.ops.custom_comm.allgather_batch |
| Python | Torch Ext (.so加载) | Python可调用接口 |
| Graph Mode | Torch Ext + torchair | GE converter, aclGraph capture |


## 6.3 关键外部API依赖(按Phase递增)

Phase 1(仅公开稳定API):
- HcclAllGather -- include/hccl/hccl_comm.h
- HcclDataType -- include/hccl/hccl_types.h
- TORCH_LIBRARY -- PyTorch C++ API
- aclrtSynchronizeStream -- ACL runtime

Phase 2(新增内部API):
- HcclEngineCtxCreate/Get -- include/hccl/hccl_res.h
- HcclChannelAcquire -- include/hccl/hccl_res.h
- HcclCcuKernelRegister/Launch -- pkg_inc/hcomm/ccu/hccl_ccu_res.h
- CcuKernel基类 -- pkg_inc/hcomm/ccu/ccu_kernel.h
- WriteNb/LocalCopyNb等原语 -- CcuKernel成员方法

Phase 3(新增):
- HcclThreadResGetInfo -- 内部API(版本依赖待确认)
- aclmdlRICaptureBegin/GetInfo -- ACL runtime
- register_fx_node_ge_converter -- torchair扩展点


## 6.4 运行时环境

- CANN 9.0.0 Toolkit (devel安装)
- PyTorch + torch_npu (PTA)
- torchair (图模式, Phase 3)
- A5 (910_95) NPU设备
- 已初始化的ProcessGroupHCCL (提供HcclComm handle)


---

# ==== 附录: 术语对照

| 缩写 | 全称 | 说明 |
|------|------|------|
| CCU | Collective Communication Unit | 集合通信单元，A5硬件微码调度引擎 |
| AIV | AI Vector core | A5向量计算核 |
| MTE | Memory Transfer Engine | 内存搬移引擎 |
| SQE | Submission Queue Entry | CCU任务提交队列条目 |
| HCCS | Huawei Cache Coherence System | 昇腾片间互连 |
| PTA | PyTorch Adapter (torch_npu) | PyTorch昇腾适配层 |
| GE | Graph Engine | 华为图编译引擎 |
| DFX | Design for X | 可测试/可调试/可维护性 |
