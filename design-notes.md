# 自定义通信算子仓设计笔记

日期: 2026-04-11 (初版) / 2026-04-12 (SDK解包验证刷新) / 2026-04-13 (目标算子切换)
状态: 设计探索阶段

---

## 1. 目标与约束 (来自back.txt)

目标: 以自定义算子形式端到端交付昇腾亲和高性能自定义通信算子。
首个交付算子: HcclAllGatherBatch(多类型批量AllGather)。

背景(OPT-AG-09): MoE量化路径对INT8数据和FP32 scale分别执行独立AllGather
(2次AG)。scale仅占总数据量3%，但触发完整AG调度(~8-10us)。
HcclAllGatherBatch在单个CCU task内完成多路异构数据的AllGather��
消除冗余kernel launch开销。

约束:
- 基于CANN 9.0.0
- 当前仅考虑A5(910_95)平台
- 不区分训练/推理
- 不侵入式修改PTA或HCCL/HCOMM
- 支持eager mode
- 支持入图(aclgraph和GE)
- 支持profiling和DFX

---

## 2. 根本矛盾

CCU编程复杂度与零拷贝性能收益之间的张力:
- HcclAllGatherBatch是纯通信算子(无计算融合)，CCU是A5上的自然选择
- 要达到零拷贝(每路buffer直接WriteNb到远端recvBuf)，需要自定义CcuKernel
- CcuKernel编程依赖pkg_inc/hcomm/ccu/(内部API，无官方文档，参照HCCL源码)
- 高阶原语GroupBroadcast在CcuKernelAlgBase中(hccl源码内部，SDK未导出)

结论: HCOMM的CcuKernel API(pkg_inc/hcomm/ccu/)和HCCL资源API(include/hccl/)
是官方扩展点。CcuKernel的WriteNb/LocalCopyNb原语足以实现AllGather，
不需要依赖CcuKernelAlgBase。LoopGroup升级路径通过pkg_inc中的
LoopGroupCall等积木可自行搭建(~100行)。

---

## 3. 四条技术路线评估

### 路线A: HCCL Custom Ops框架

通过hccl_res.h公开API获取通信资源(HcclEngineCtxCreate,
HcclChannelAcquire, HcclCommMemReg等)。
在TemplateType枚举中预留了1000-2000的自定义range。

- 优势: 能触及最底层通信原语，最优融合
- 代价: 与HCCL构建系统和部署模型耦合

### 路线B: 独立PyTorch自定义算子 (omni-ops模式)

TORCH_LIBRARY注册 + Ascend C kernel + GE converter。
参照omni-ops的目录结构和构建模式。

- 优势: 完全独立构建和发布，迭代快
- 代价: 通信资源获取依赖公开API的完备性

### 路线C: torchcomms Backend

torchcomms定义了TorchCommBackend抽象基类，支持动态加载。

- 优势: 前瞻性，社区对齐
- 代价: experimental状态，无HCCL backend，bootstrapping工作量大
- 结论: 短期不适合做主干，长期观察

### 路线D: Op-Plugin层 (aclnn API)

参照op-plugin中AllGatherMatmul/MatmulAllReduce的模式。

- 优势: 最简单，已有成熟模式
- 代价: 受限于aclnn层暴露的能力

### 选择

路线B为主干架构(独立构建+发布解耦)。路线A的CCU API为核心实现层:
HcclAllGatherBatch是纯通信算子，直接继承hcomm::CcuKernel实现，
不需要AIV计算kernel，不需要Ascend C编译器。

---

## 4. 关键风险验证结果

### 4.1 Ascend C HCCL Device API在A5上的可用性

ascendc-hccl.md文档所有接口"支持型号"只标"Atlas A2训练系列产品"。
但HCCL example 05 README明确支持Ascend 950PR/950DT(即A5)。
MC2在A5上使用GetHcclContext获取通信上下文，证实API可用。

关键区别 -- A5有两套通信机制:

    A2路径: Hccl<>::AllGather() Prepare/Commit/Wait
    A5路径: GetHcclContext -> MTE窗口 / CCU channel

- A2: AICPU-based server调度
- A5 MTE: AIV核直接操作MTE窗口，软同步(status window轮询)
- A5 CCU: 硬件微码调度，channel-based WriteNb/ReadNb

FP8数据类型支持:
- hccl_types.h:104-107 定义了 HCCL_DATA_TYPE_FP8E4M3(15), FP8E5M2(16), FP8E8M0(17)
- AllGather文档说"支持HcclDataType枚举中的所有类型" -> 包含FP8
- AllReduce/ReduceScatter只列FP32/FP16/INT8/INT16/INT32/BFP16，不含FP8
- MTE路径按字节搬移，类型解释由kernel自己负责，不受此限制
- 注意: hcomm_primitives.h中FP8枚举值有`#ifndef OPEN_BUILD_PROJECT` guard,
  但hccl_types.h中无此guard，FP8类型始终可用

### 4.2 独立repo构建可行性

已验证可行(从CANN 9.0.0 .run包解包确认):
- omni-ops通过ASCEND_CANN_PACKAGE_PATH定位SDK
- HCCL example 05自包含，只依赖SDK安装路径
- Ascend C编译器: `${ASCEND_HOME_PATH}/aarch64-linux/ccec_compiler/bin/bisheng`
  (bisheng和ccec是同一binary; CMake ASC模块通过FindProgram("bisheng")定位)
- AIV kernel编译标志: --cce-aicore-arch=dav-c310-vec
- Ascend C CMake模块: `${ASCEND_HOME_PATH}/aarch64-linux/ascendc_kernel_cmake/`
  提供完整的ASC语言支持(CMakeDetermineASCCompiler, ascendc.cmake等)

需要: CANN Toolkit以`--devel`或`--full`模式安装

### 4.3 hcom group name获取路径

公开API，无需修改PTA:

    Python侧: pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    C++侧: 接收 c10::string_view hcom 参数

- 实现位置: ProcessGroupHCCL.cpp:3475-3524, pybind11绑定在Init.cpp:409-417
- op-plugin已有成熟模式(AllGatherBaseMatmulKernelNpuOpApi.cpp)
- 图模式: torchair hcom_utils.py的get_group_name_and_record()处理
- cache场景: codegen_refresh_cache_pgname()自动处理，自定义op只要以string接收hcom即可

### 4.4 已有多类型AllGather实现 -- GAP确认

已有:
- HcclAllGather: 单dtype单buffer，标准集合通信
- OPT-AG-09方案A: Python层byte-packing(view-as-uint8 + cat)，已验证bit-exact正确性
- HcclGroupStart/HcclGroupEnd: experimental API，可能批量提交但不保证融合为单CCU task

GAP: 无原生多类型AllGather原语(接收多个异构dtype buffer，单次CCU task完成gather)
方案A的局限: 纯Python操作，每次调用都有view+cat+split开销;
不能进入图编译优化; 不能利用CCU硬件调度; unpack产生全量数据重排

### 4.5 CCU API可达性 (本session新增)

CommLib.xml (hcomm打包清单) 确认:
- 全部32个CCU头文件显式列出，install_type="devel"
- 安装路径: `${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu/`
- 客户以`--devel`或`--full`安装CANN 9.0.0即可获得

libhcomm.so符号导出验证:
- C API: HcclCcuKernelRegister, HcclCcuKernelRegisterFinish, HcclCcuKernelLaunch
- CcuKernel C++类: 构造/析构(C1/C2/D0/D1/D2), vtable(_ZTV), typeinfo(_ZTI)
- 全部成员方法: WriteNb, ReadNb, NotifyRecord/Wait, LocalCopyNb, LocalReduceNb,
  CreateVariable, CreateLocalAddr/RemoteAddr, CreateCcuBuf, GeneTaskParam, Func, Loop等
- CcuKernelMgr: Register, AllocRes, Translate(Rep->微码)

结论: 直接`-I${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu -lhcomm`可用。
不需要从HCOMM源码抄头文件。

注意事项:
- pkg_inc/非public API，无跨版本兼容承诺 (项目锁定CANN 9.0.0，风险可控)
- _v1后缀说明接口已版本化
- 正确性风险 > 可用性风险: CcuRep DSL生成硬件微码，无官方文档，
  唯一参考是HCCL源码中`src/ops/*/template/ccu/`的实现

---

## 5. A5通信机制: CCU vs AIV-MTE

### CCU (Collective Communication Unit)

硬件微码调度引擎，HCCL内建集合算法的生产路径。

核心能力:
- 数据搬移: channel-based WriteNb/ReadNb
- 本地操作: LocalCopyNb, LocalReduceNb
- 硬件同步: NotifyRecord/NotifyWait
- 循环优化: LoopGroup自动展开和并行化
- Die感知: 支持多Die拓扑 (CCU_MAX_IODIE_NUM=2)

不能做: 任意计算(如量化/反量化)

关键API (pkg_inc/hcomm/ccu/hccl_ccu_res.h, extern "C"):
- HcclCcuKernelRegister(comm, kernelHandle, kernelCreator, kernelArg)
- HcclCcuKernelRegisterFinish(comm)
- HcclCcuKernelLaunch(comm, threadHandle, kernelHandle, taskArgs)

编程模型: 继承CcuKernel基类，实现Algorithm()方法，使用CcuRep DSL描述通信pattern。
CcuKernel在pkg_inc/hcomm/ccu/ccu_kernel.h中定义，namespace hcomm。

使用者:
- HCCL内部: 全部内建集合算法的CCU实现(`hccl/src/ops/*/template/ccu/`)
- 外部: 官方example无CCU用例，但头文件+符号均可达(见4.5)

### AIV-MTE (官方Custom Ops推荐路径)

软件管理的MTE窗口机制，HCCL example 05展示的模式。

核心能力:
- 远端数据窗口: buffersIn[](读远端), buffersOut[](状态同步)
- 软同步: 写status window通知，轮询等待
- 分块搬移: 1024B块对齐，多AIV核负载均衡
- 计算融合: AIV核可在通信间隙执行向量运算(量化/反量化)

Example 05的模式:
1. HcclEngineCtxCreate(comm, tag, COMM_ENGINE_AIV, ...) -- 创建AIV上下文
2. HcclCommMemReg(comm, tag, &mem, &handle) -- 注册本地内存
3. HcclChannelAcquire(comm, COMM_ENGINE_AIV, descs, n, channels) -- 获取通道
4. HcclChannelGetRemoteMems(comm, channel, ...) -- 获取远端内存地址
5. 将buffersIn/buffersOut打包传入AIV kernel
6. AIV kernel内部通过地址直接操作MTE窗口

全程只用`include/hccl/`下的公开API，不涉及CCU。

HCCL上下文结构 (moe_distribute_comm_ctx.h):

    struct HcclCombinOpParam {
        uint64_t workSpace, workSpaceSize;
        uint32_t rankId, rankDim;
        uint64_t winSize;
        uint64_t windowsIn[HCCL_MTE_MAX_RANK_NUM];
        uint64_t windowsOut[HCCL_MTE_MAX_RANK_NUM];
    };

### 两者关系

不是互相替代，是并行路径:
- CCU: HCCL内建算法的默认路径(isMc2=false时); 硬件调度，最优纯通信性能
- AIV-MTE: MC2融合算子/custom ops的路径(isMc2=true时); 可融合计算+通信
- 优先级链: CCU_MS -> CCU_SCHED -> AIV -> AICPU

对custom_comm的意义:
- HcclAllGatherBatch是纯通信算子(无quant/dequant计算)，CCU是自然选择
- AIV-MTE路径不需要: 没有计算融合需求
- 全Phase走CCU: Phase 1间接使用(HcclAllGather内部走CCU),
  Phase 2直接编程(自定义CcuKernel)

---

## 6. HcclAllGatherBatch的三种A5实现策略

### 策略1: Decomposed (host pack + HcclAllGather + host unpack)

    [at::cat(view-as-uint8)] -> [HcclAllGather(packed_uint8)] -> [split + view back]

- 3次操作(pack device kernel + AG + unpack device kernel)
- pack: view-as-uint8 + cat(dim=1)，复用OPT-AG-09验证过的byte-packing逻辑
- HcclAllGather(UINT8): HCCL_DATA_TYPE_UINT8 = 7 (hccl_types.h:98)，内部自动走CCU
- unpack: split返回non-contiguous view，需.contiguous()触发全量拷贝
- 代价: pack+unpack产生额外本地HBM搬移(W=8, N=2048时unpack约20MB ~20us)
- 适用: 快速验证正确性和API形态

### 策略2: Packed CCU AllGather (LocalCopyNb pack + GroupBroadcast + unpack)

    CCU Algorithm(): LocalCopyNb(pack) -> GroupBroadcast(packed) -> LocalCopyNb(unpack)

- 单kernel launch，硬件调度
- flat-concat打包(先input1全部，再input2全部): 2次LocalCopyNb
- unpack: 2*W次LocalCopyNb(每rank每buffer一次，W=8时16次)
- 需要workspace: pack buffer + gathered buffer
- 代价: 与策略3相比，多pack(~3us) + unpack(~20us)的本地HBM搬移
- 注意: per-row打包(dim=1 interleave)在CCU中需N次LocalCopyNb，不可行;
  flat-concat是CCU上唯一可行的打包方式

### 策略3: Batched CCU AllGather (zero-copy) -- 选定

    CCU Algorithm():
      PreSync()
      for i in 0..descCount-1:
        DoAllGather(buf_i)   // WriteNb直接写到远端recvBuf
      PostSync()

- 单kernel launch，每路buffer独立执行AllGather(WriteNb * (W-1) + LocalCopyNb * 1)
- 零额外内存拷贝: 无pack/unpack，无workspace，数据直达最终目的地址
- 各路串行执行(CCU微程序内for循环)，但小buffer(如4KB scale)传输时间极短
- 继承hcomm::CcuKernel，用WriteNb循环实现; 不依赖CcuKernelAlgBase
- LoopGroup升级路径: 大消息场景可用pkg_inc中的LoopGroupCall积木自行搭建

### 策略对比

| 维度 | 策略1 Decomposed | 策略2 CCU+Pack | 策略3 CCU Batched |
|------|-----------------|---------------|------------------|
| kernel launch | 3 | 1 | 1 |
| workspace | packed buf | packed+gathered | 无 |
| 本地HBM额外流量(W=8,N=2048) | ~45MB | ~45MB | ~5MB(仅self-copy) |
| CCU编程 | 不需要 | 需要 | 需要 |
| 小buffer表现 | 好(合并传输) | 好(合并传输) | 可接受(独立WriteNb) |
| 图模式 | 需处理3个op | 单op | 单op |

### 选择: 策略3

理由: 零拷贝带来约20us的性能优势(来自消除unpack全量重排)。
OPT-AG-09场景(buf1=2.5MB int8, buf2=4KB fp32)下策略3的额外
GroupBroadcast开销(~2us)远小于策略2的unpack代价(~20us)。

渐进路线: 策略1用于Phase 1验证正确性(与Python byte-packing bit-exact对比),
策略3作为Phase 2的交付实现。

---

## 7. SDK依赖分析 (本session从.run解包验证)

### 7.1 完整依赖清单

客户以`--devel`模式安装CANN 9.0.0 Toolkit后可用:

    ${ASCEND_CANN_PACKAGE_PATH}/  (默认 /usr/local/Ascend/ascend-toolkit/latest)
    ├── include/                      # 公开API头文件
    │   ├── hccl/                     # HCCL: hccl_types.h, hccl_comm.h, hccl_res.h
    │   ├── hcomm/                    # HCOMM: hcomm_primitives.h(symlink), hcomm_res.h
    │   └── securec.h                 # c_sec库
    ├── pkg_inc/                      # 内部API (devel安装)
    │   ├── hccl/                     # hccl_ex.h, hccl_inner.h, hccl_res_expt.h
    │   └── hcomm/ccu/               # CCU编程框架 (32个头文件)
    ├── lib64/                        # 运行时库
    │   ├── libhcomm.so              # HCCL+HCOMM+CCU统一入口
    │   └── libhccl_alg.so, ...      # 算法库
    ├── aarch64-linux/
    │   ├── ccec_compiler/bin/        # bisheng, ccec, ld.lld
    │   ├── asc/include/              # kernel_operator.h (Ascend C入口)
    │   ├── ascendc/include/          # Ascend C impl头文件
    │   ├── ascendc_kernel_cmake/     # CMake ASC语言支持模块
    │   └── tikcpp/tikcfw/            # TIK C++框架
    ├── runtime/include/external/
    │   └── acl/                      # ACL运行时: acl.h, acl_rt.h, acl_op.h
    └── ops_base/                     # aclnn框架 + op_host tiling

### 7.2 各子包用途映射

| 子包 | 安装内容 | 用于 |
|---|---|---|
| hcomm | include/hccl/, pkg_inc/hcomm/ccu/, lib64/libhcomm.so | 通信API全部 |
| npu-runtime | runtime/include/external/acl/, lib64/libascendcl.so | ACL运行时 |
| bisheng-compiler | aarch64-linux/ccec_compiler/ | Ascend C编译器 |
| asc-devkit | aarch64-linux/asc/, ascendc/, ascendc_kernel_cmake/ | Ascend C开发 |
| opbase | ops_base/aclnn/, ops_base/pkg_inc/op_common/ | aclnn框架+tiling |
| metadef | metadef/include/register/, exe_graph/runtime/ | 算子注册+GE IR |
| ge-compiler | ge-compiler/include/ge/ | GE编译器(图模式) |
| tbe-tik | TBE运行时(Python侧) | 非直接依赖 |
| acl-extend | media相关头文件 | 不需要 |

### 7.3 API稳定性分层

    include/hccl/         公开稳定    所有Phase使用
    include/hcomm/        公开稳定    数据面原语(Write/Read/Notify)
    pkg_inc/hccl/         内部API     hccl_res_expt.h(实验性资源API)
    pkg_inc/hcomm/ccu/    内部API     CcuKernel + LoopGroupCall, devel安装可达, 无跨版本兼容承诺

### 7.4 CcuKernelAlgBase不可达分析 (本session验证)

GroupBroadcast高阶原语有两套实现:

| 实现 | 位置 | .so | 符号导出 | 头文件 |
|------|------|-----|---------|--------|
| ops_hccl::CcuKernelAlgBase | hccl/src/ops/op_common/template/ccu/ | libhccl.so | 未导出 | 仅源码树 |
| Hccl::CcuContext | hcomm/src/legacy/unified_platform/ | libhccl_v2.so | 已导出 | 仅源码树 |

两者均无法从SDK直接使用: CcuKernelAlgBase符号未导出，
Hccl::CcuContext头文件不在SDK中且参数类型不兼容(CcuTransport* vs ChannelHandle)。

解决方案: 不依赖CcuKernelAlgBase。
继承hcomm::CcuKernel(pkg_inc可达)，用WriteNb循环实现AllGather。
GroupBroadcast的核心是"WriteNb * (W-1) + LocalCopyNb + sync"，
在OPT-AG-09消息大小(MB级)下不需要LoopGroup分块流水线。
未来大消息场景: pkg_inc中的LoopGroupCall/CcuBuf/Executor等积木
均可达(与CcuKernel同层)，可自行搭建~100行LoopGroup编排逻辑。

### 7.5 Ascend C编译器依赖

HcclAllGatherBatch是纯通信算子，无AIV计算kernel。
CCU kernel是纯C++代码(继承CcuKernel, 编译为.so)，不需要bisheng/ccec编译器。
bisheng仅在未来扩展AIV计算kernel时需要(如量化通信融合场景)。

---

## 8. 具体架构

### 目录结构

    custom_comm/
    ├── ops/
    │   └── allgather_batch/
    │       ├── inc/
    │       │   ├── hccl_custom_allgather_batch.h     # C API: HcclAllGatherBatch()
    │       │   ├── common.h                           # OpParam, CcuContext
    │       │   └── ccu_kernel_ag_batch_mesh1d.h       # CCU kernel类声明
    │       ├── op_host/
    │       │   ├── all_gather_batch.cc                # 入口 + InitCcuContext + LaunchCcuKernel
    │       │   └── ccu_kernel_ag_batch_mesh1d.cc      # CCU kernel实现(Algorithm, GeneArgs)
    │       └── tests/
    ├── torch_extension/
    │   ├── csrc/
    │   │   ├── ops_registration.cpp       # TORCH_LIBRARY(custom_comm, m)
    │   │   └── allgather_batch.cpp        # PyTorch API impl(调用C API)
    │   ├── converters/
    │   │   └── allgather_batch_converter.py  # GE converter
    │   ├── __init__.py
    │   └── setup.py                       # NpuExtension构建
    ├── CMakeLists.txt
    ├── cmake/
    │   └── config.cmake                   # ASCEND_CANN_PACKAGE_PATH发现
    ├── .sdk/9.0.0/                        # 解包的SDK (开发参考, gitignored)
    └── .gitignore

### C API (参考HcclAllGatherBatch_CustomOp_design.md)

    typedef struct {
        void        *sendBuf;
        uint64_t     sendCount;
        HcclDataType dataType;
        void        *recvBuf;
    } HcclAllGatherDesc;

    HcclResult HcclAllGatherBatch(
        const HcclAllGatherDesc *descs, uint32_t descCount,
        HcclComm comm, aclrtStream stream);

语义等价于descCount次独立HcclAllGather，但在单个CCU task内完成。
MAX_DESC_COUNT = 8 (与Mesh1D rankSize上限对齐)。

### PyTorch API

    gathered = torch.ops.custom_comm.allgather_batch(
        [x_int8, x_scale],   # Tensor[] - 不同dtype的input列表
        hcom,                  # str - group name
        world_size,           # int
    )
    # -> Tensor[] - gathered后的tensor列表

### CCU Kernel结构

    class CcuKernelAllGatherBatchMesh1D : public hcomm::CcuKernel {
    protected:
        HcclResult Algorithm() override;  // PreSync -> 逐路Broadcast -> PostSync
        std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;
    private:
        void Broadcast(CcuRep::LocalAddr& src,
                       std::vector<CcuRep::RemoteAddr>& dst,
                       CcuRep::Variable& size);  // WriteNb*(W-1) + LocalCopyNb
    };

继承hcomm::CcuKernel(pkg_inc可达)，不依赖CcuKernelAlgBase。
Broadcast()用WriteNb循环实现，对OPT-AG-09消息大小足够。

### Host侧调用模式

    首次调用: HcclEngineCtxCreate -> HcclChannelAcquire -> HcclCcuKernelRegister
    后续调用: HcclEngineCtxGet(命中) -> 构造CcuTaskArg -> HcclCcuKernelLaunch

资源通过HcclEngineCtxGet幂等缓存。CcuContext包含kernelHandle + threadHandle。

### 注册链

eager模式:

    TORCH_LIBRARY(custom_comm, m) {
        m.def("allgather_batch(Tensor[] inputs, str hcom, "
              "int world_size) -> Tensor[]");
    }
    TORCH_LIBRARY_IMPL(custom_comm, PrivateUse1, m) {
        m.impl("allgather_batch", &allgather_batch_npu);
    }
    TORCH_LIBRARY_IMPL(custom_comm, Meta, m) {
        m.impl("allgather_batch", &allgather_batch_meta);
    }

图模式 (GE converter):

    @register_fx_node_ge_converter(
        torch.ops.custom_comm.allgather_batch.default)
    def convert_allgather_batch(inputs, hcom, world_size,
                                 meta_outputs=None):
        group_name = get_group_name_and_record(tag, rank_list, group_size)
        return [ge.custom_op("AllGatherBatch", ...)]

### Profiling分层

| 层 | 机制 | 给出什么 | Phase |
|---|------|---------|-------|
| PyTorch | RECORD_FUNCTION | torch.profiler timeline上的op标记 | 1 |
| Host | gettimeofday包裹(同example 05) | 粗粒度端到端us | 1 |
| msprof | aclprofMarkEx(stream) | MindStudio timeline上的命名标记 | 2 |
| CCU内部 | CcuKernel::AddCcuProfiling | 逐路opName/dataSize/channelId | 2 |
| Device精确 | HcclThreadResGetInfo取slave stream + aclrtEvent | CCU真实device耗时 | 3 |

注意: aclrtRecordEvent在user stream上只反映host提交时间，不反映CCU执行时间。
精确device计时需要在CCU的slave stream上打event(依赖HcclThreadResGetInfo，
与aclGraph capture同一API依赖)。Phase 1-2用host gettimeofday + aclrtSynchronizeStream
做粗糙计时，Phase 3获得slave stream后做精确计时。

### 上游集成点

- omni-npu AGRS路径: 替换2次独立AllGather为1次allgather_batch
- vllm-ascend: 量化推理TP gather场景
- torch.compile: GE converter自动识别
- MindStudio: RECORD_FUNCTION + aclprofMarkEx + CCU profiling

---

## 9. 开发路线图

    Phase 0: 环境验证 (~1周)
      - 独立repo CMake构建
      - TORCH_LIBRARY注册能编译运行
      - HcclAllGather host API可调用(通过链接libhcomm.so)
      - CCU API可达性验证: HcclCcuKernelRegister链接测试
      - 确认pkg_inc/hcomm/ccu/头文件devel安装可达

    Phase 1: allgather_batch eager mode - 策略1 (~2周)
      - C API: HcclAllGatherBatch入口
      - Host侧byte-packing: view(uint8) + cat(dim=1) -> HcclAllGather(UINT8)
      - TORCH_LIBRARY注册 + Meta impl(shape推导)
      - 2卡/4卡/8卡正确性测试(与OPT-AG-09 Python方案A bit-exact对比)
      - Profiling: RECORD_FUNCTION + host gettimeofday计时

    Phase 2: CCU Batched AllGather - 策略3 (~3-4周)
      - 继承hcomm::CcuKernel(pkg_inc/hcomm/ccu/ccu_kernel.h)
      - InitCcuContext: HcclEngineCtxCreate + HcclChannelAcquire + HcclCcuKernelRegister
      - CcuKernelAllGatherBatchMesh1D: Algorithm()内逐路WriteNb循环
      - GeneArgs: 固定头部(3) + 每路描述符(6 * descCount) 序列化
      - 性能对比Phase 1，验证kernel launch开销节省
      - Profiling: aclprofMarkEx标记 + CCU AddCcuProfiling(逐路粒度)

    Phase 3: 图模式 + 优化 (~2-3周)
      - GE图模式: torchair GE converter注册 + 编译测试
      - aclGraph capture: 实现CaptureSlaveStreams(见10.1), 从流入图
      - Profiling: HcclThreadResGetInfo取slave stream + aclrtEvent精确device计时
      - 可选: LoopGroup升级(大消息分块流水线)

    持续: 上游集成
      - omni-npu AGRS路径集成验证
      - vllm-ascend推理场景验证

---

## 10. 待确认事项

### 10.1 aclGraph capture: CCU从流入图机制

HcclCcuKernelLaunch本身无stream参数，但CCU thread内部绑定了底层stream。
HCCL已有设计方案(ccu_aclgraph设计文档)通过CaptureSlaveStreams机制支持入图:

    1. PTA主流发起aclmdlRICaptureBegin
    2. 自定义算子入口检测主流capture状态(aclmdlRICaptureGetInfo)
    3. HcclThreadResGetInfo获取CCU thread绑定的slave stream
    4. rtStreamAddToModel将slave stream添加到主流model
    5. 后续HcclCcuKernelLaunch在slave stream上的操作自动被录制

自定义CCU kernel入图所需:
- 在HcclAllGatherBatch入口实现CaptureSlaveStreams逻辑
- 依赖HcclThreadResGetInfo API (CANN V100R001C17 Eco版本新增)
- 仅支持A5平台

参考: ~/repo/hccl_8/hccl-docs/wiki/关键特性/AclGraph/ccu_aclgraph设计文档.md

### 10.2 其他待确认

- Tensor[]在GE converter中的映射方式(可变长度input list -> 多个GE输入?)
- GE compiler对自定义CCU kernel的编排能力(ge.custom_op能否调度CCU?)
- CCU WriteNb单次最大传输大小(是否有硬件上限? 决定是否需要分块)
- CCU LocalCopyNb对齐约束(1024B块对齐?)
- CcuKernel外部注册时的comm生命周期管理(HcclCommDestroy后handle悬空)
- InitCcuContext部分失败时的上下文污染恢复(参考设计文档E4)
- GeneArgs超过13个slot时的SQE拆分开销(descCount=2时15 slots -> 2个SQE)
- HcclGroupStart/End是否能融合多次HcclAllGather为单CCU task(策略1的潜在优化)

---

## 11. 参考代码位置

HcclAllGatherBatch设计:
- 设计文档(非最终版): ~/Downloads/HcclAllGatherBatch_CustomOp_design.md
- OPT-AG-09 spike: ~/note/proj/pangu-comm/spike/allgather/OPT-AG-09/spike.md
- OPT-AG-09验证脚本: ~/note/proj/pangu-comm/spike/allgather/OPT-AG-09/scripts/
- CCU aclGraph设计: ~/repo/hccl_8/hccl-docs/wiki/关键特性/AclGraph/ccu_aclgraph设计文档.md
- aclGraph参考API: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/appdevg/acldevg/aclcppdevg_000519.html

华为侧:
- HCCL custom ops example: ~/repo/cann/hccl/examples/05_custom_ops_allgather/
- HCCL custom ops P2P: ~/repo/cann/hccl/examples/04_custom_ops_p2p/
- MC2 QuantAllReduce: ~/repo/cann/ops-transformer/mc2/quant_all_reduce/
- MC2 MTE通信层: ~/repo/cann/ops-transformer/mc2/quant_reduce_scatter/op_kernel/mte_comm.h
- MC2 HCCL context: ~/repo/cann/ops-transformer/mc2/common/op_kernel/moe_distribute_comm_ctx.h
- HCCL CCU AllGather: ~/repo/cann/hccl/src/ops/all_gather/template/ccu/
- HCOMM CCU API: ~/repo/cann/hcomm/pkg_inc/hcomm/ccu/hccl_ccu_res.h
- HCOMM CcuKernel: ~/repo/cann/hcomm/pkg_inc/hcomm/ccu/ccu_kernel.h
- HCOMM LoopGroupCall: ~/repo/cann/hcomm/pkg_inc/hcomm/ccu/ccu_loopgroupcall_v1.h
- HCCL CcuKernelAlgBase: ~/repo/cann/hccl/src/ops/op_common/template/ccu/ccu_kernel_alg_base.{h,cc}
- HCOMM CcuContext(v2): ~/repo/cann/hcomm/src/legacy/unified_platform/pub_inc/ccu/ccu_ctx.h
- HCOMM打包清单: ~/repo/cann/hcomm/scripts/package/module/ascend/CommLib.xml
- HCCL dlsym基础设施: ~/repo/cann/hccl/src/common/hcomm_dlsym/
- omni-ops(目录结构参考): ~/repo/vllm-project/omniai/omni-ops/
- op-plugin AllGatherMatmul: ~/repo/ascend/op-plugin/.../AllGatherBaseMatmulKernelNpuOpApi.cpp
- torchair hcom_utils: ~/repo/torch/torch_npu/third_party/torchair/.../hcom_utils.py
- torchair GE converters: ~/repo/torch/torch_npu/third_party/torchair/.../ge_converter/experimental/

竞品侧:
- NVIDIA TransformerEngine: ~/repo/nvidia/transformerengine/ (FP8 comm参考)
- torchcomms: ~/repo/torch/torchcomms/ (backend抽象参考)

SDK解包参考 (本地, gitignored):
- .sdk/9.0.0/hcomm/ -- HCCL/HCOMM头文件+库
- .sdk/9.0.0/npu-runtime/ -- ACL运行时
- .sdk/9.0.0/bisheng-compiler/ -- Ascend C编译器
- .sdk/9.0.0/asc-devkit/ -- Ascend C开发头文件
- .sdk/9.0.0/opbase/ -- aclnn框架
- .sdk/9.0.0/metadef/ -- GE IR/算子注册
- .sdk/9.0.0/ge-compiler/ -- GE编译器
