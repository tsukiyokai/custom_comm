# Reference Repository Survey for custom_comm

Date: 2026-04-13

---

## 1. HCCL (~/repo/cann/hccl)

### 1.1 Directory Structure

```
hccl/
├── CMakeLists.txt              # ENABLE_CUSTOM 支持外部 custom ops
├── include/
│   └── hccl/
│       ├── hccl.h              # Public collective API
│       └── hccl_mc2.h          # MC2 hooks
├── examples/
│   ├── 04_custom_ops_p2p/      # AICPU P2P custom op 示例
│   └── 05_custom_ops_allgather/# AIV AllGather custom op 示例
│       ├── CMakeLists.txt
│       ├── inc/
│       │   ├── common.h        # OpParam struct, error macros
│       │   ├── log.h           # CHK_RET, HCCL_ERROR 等
│       │   ├── extra_args.h    # ExtraArgs struct
│       │   └── aiv_*.h         # AIV kernel headers
│       ├── op_host/            # Host-side entry, kernel launch
│       └── op_kernel/          # AIV device kernel
├── src/
│   └── ops/
│       ├── all_gather/         # AllGather variants
│       │   └── template/ccu/   # CCU kernel 实现
│       │       └── ccu_kernel_all_gather_mesh_1d.cc
│       └── op_common/
│           └── template/ccu/
│               └── ccu_kernel_alg_base.h  # GroupBroadcast 等高阶原语
└── test/
```

### 1.2 Custom Ops Framework (Example 05)

OpParam 结构 (`inc/common.h:30-65`):

```cpp
namespace hccl_custom {
struct OpParam {
    uint64_t sendBuf;
    uint64_t recvBuf;
    uint32_t rank, rankSize;
    uint64_t count;
    HcclDataType dataType;
    ExtraArgs extraArgs;  // opaque extension
};
}
```

Resource lifecycle (op_host/all_gather.cc):

```
1. HcclEngineCtxCreate(comm, tag, COMM_ENGINE_AIV, ...)
2. HcclCommMemReg(comm, tag, &mem, &handle)
3. HcclChannelAcquire(comm, engine, descs, num, channels)
4. HcclChannelGetRemoteAddr(comm, channel, rank, &addr)
5. // ... kernel launch ...
6. // channels released implicitly on context destroy
```

Error checking pattern (`inc/log.h`):

```cpp
#define CHK_RET(expr)                                        \
    do {                                                     \
        HcclResult _ret = (expr);                            \
        if (_ret != HCCL_SUCCESS) {                          \
            HCCL_ERROR("[%s] ret=%d", __func__, _ret);       \
            return _ret;                                     \
        }                                                    \
    } while (0)
```

### 1.3 CCU Kernel Pattern (src/ops/all_gather/)

AllGather CCU template (ccu_kernel_all_gather_mesh_1d.cc):

```
class AllGatherCcuKernel : public hcomm::CcuKernel {
    Algorithm() override {
        // DSL: define sync, data movement
        PreSync();
        for each remote rank:
            WriteNb(channel, remoteAddr, localAddr, size, event);
        LocalCopyNb(localDst, localSrc, size, event);
        PostSync();
    }
    GeneArgs(const CcuTaskArg &arg) override {
        // Pack runtime args into vector<uint64_t>
        return {inputAddr, outputAddr, count, rankId, rankSize, ...};
    }
};
```

CCU resource lifecycle (from HCCL examples/internal):

1. `HcclCcuKernelRegister(comm, creator, arg, &handle)` - register kernel
2. `HcclCcuKernelRegisterFinish(comm)` - compile to microcode
3. `HcclCcuKernelLaunch(comm, handle, taskArg)` - execute

### 1.4 Build System

```cmake
# examples/05_custom_ops_allgather/CMakeLists.txt
enable_language(ASC)  # Ascend C compiler
find_package(HCCL REQUIRED)
target_link_libraries(custom_op PRIVATE hcomm ascendcl runtime)
target_include_directories(custom_op PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl)
```

---

## 2. HCOMM (~/repo/cann/hcomm)

### 2.1 Directory Structure

```
hcomm/
  include/hccl/              # Public stable API
    hccl_types.h             # Data types, error codes
    hccl_comm.h              # Communicator management
    hccl_res.h               # Resource management (channels, threads, engine ctx)
  pkg_inc/hcomm/ccu/         # Extension API for CCU kernel development
    ccu_kernel.h             # CcuKernel base class (core)
    ccu_kernel_arg.h         # CcuKernelArg (constructor args)
    ccu_kernel_signature.h   # Kernel type identity
    ccu_task_arg_v1.h        # CcuTaskArg (per-launch args)
    ccu_task_param_v1.h      # CcuTaskParam (SQE hardware struct)
    ccu_common.h             # CCU_MAX_IODIE_NUM=2
    ccu_kernel_resource.h    # CcuRepResource definitions
    ccu_rep_context_v1.h     # DSL context
    hccl_ccu_res.h           # Register/Launch C API
    ... (33 headers total)
  include/
    hccl/
      hccl_types.h           # HcclResult, HcclDataType
      hccl_res.h             # HcclEngineCtxCreate, etc.
```

### 2.2 CcuKernel Base Class

File: `pkg_inc/hcomm/ccu/ccu_kernel.h` (lines 103-106)

Required virtual methods:

```cpp
virtual HcclResult Algorithm() = 0;          // Define CCU program
virtual vector<uint64_t> GeneArgs(const CcuTaskArg&) = 0;  // Map task->SQE args
```

Key inherited primitives:

| Method | Purpose | Category |
|---|---|---|
| `WriteNb(channel, dst, src, len, event)` | RDMA write | Data move |
| `ReadNb(channel, dst, src, len, event)` | RDMA read | Data move |
| `LocalCopyNb(dst, src, len, event)` | Local memcpy | Data move |
| `NotifyRecord(channel, idx)` | Signal peer | Sync |
| `NotifyWait(channel, idx)` | Wait for peer | Sync |
| `CreateVariable()` | Allocate CCU variable | Resource |
| `CreateLocalAddr()` / `CreateRemoteAddr()` | Address handles | Resource |
| `Load(var)` | Load arg from SQE | Arg passing |
| `Func(...)` / `Loop(...)` | Control flow | DSL |

### 2.3 CCU Registration/Launch API

File: `pkg_inc/hcomm/ccu/hccl_ccu_res.h`

```cpp
extern "C" {
// Step 1: Register kernel (returns handle)
HcclResult HcclCcuKernelRegister(
    HcclComm comm,
    KernelCreator creator,  // lambda: (CcuKernelArg&) -> unique_ptr<CcuKernel>
    const CcuKernelArg &arg,
    CcuKernelHandle *handle);

// Step 2: Finalize registration (compiles to microcode)
HcclResult HcclCcuKernelRegisterFinish(HcclComm comm);

// Step 3: Launch (per-invocation)
HcclResult HcclCcuKernelLaunch(
    HcclComm comm,
    CcuKernelHandle handle,
    const CcuTaskArg &taskArg,
    ThreadHandle thread);
}
```

### 2.4 SQE Argument Constraint

```cpp
// pkg_inc/hcomm/ccu/ccu_task_param_v1.h
constexpr uint32_t CCU_SQE_ARGS_LEN = 13;

struct CcuTaskParam {
    uint32_t instrStartIdx;
    uint32_t instrCount;
    uint64_t args[CCU_SQE_ARGS_LEN];  // max 13 u64 per SQE
};
```

GeneArgs() returns vector<uint64_t>. If len > 13, the runtime splits into
multiple SQEs automatically. For AllGatherBatch:
- Fixed overhead: 3 args (descCount, rankSize, rankId)
- Per-desc: ~4 args (srcAddr, dstAddr, count, dtype)
- Limit: floor((13 - 3) / 4) = 2 descs per SQE if packed naively
- Strategy: pass descriptor array base addr + count, let kernel indirect-load

---

## 3. ops-transformer (~/repo/cann/ops-transformer)

### 3.1 Directory Structure

```
ops-transformer/
├── mc2/                        # MC2 fused ops (comm+compute)
│   ├── all_gather_matmul/
│   │   ├── op_host/            # OpDef + ACLNN adapter
│   │   ├── op_api/             # C API wrapper
│   │   └── op_kernel/          # Ascend C kernel
│   └── ...
├── attention/, ffn/, moe/      # Domain-specific ops
├── common/include/             # Shared error macros, types
├── torch_extension/            # Torch<->ACLNN bridge
│   └── npu_ops_transformer/
│       ├── op_builder/builder.py
│       └── ops/                # Per-op Python + C++
├── cmake/
│   ├── func.cmake              # op_add_subdirectory()
│   └── config.cmake            # SDK path discovery
└── experimental/
    └── npu_ops_transformer_ext/  # TORCH_LIBRARY examples
```

### 3.2 OpDef Registration Pattern (CANN native)

```cpp
// op_host/all_gather_matmul_def.cpp
class AllGatherMatmul : public OpDef {
    AllGatherMatmul() {
        this->Input("x1").DataType({DT_FLOAT16, DT_BF16});
        this->Input("x2").DataType({DT_FLOAT16, DT_BF16});
        this->Output("y").DataType({DT_FLOAT16, DT_BF16});
        this->Attr("group").AttrType(REQUIRED).String();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(AllGatherMatmul);
```

### 3.3 ACLNN Two-Step API

```cpp
aclnnStatus aclnnAllGatherMatmulGetWorkspaceSize(
    const aclTensor *x1, ..., uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnAllGatherMatmul(void *workspace, uint64_t wsSize,
    aclOpExecutor *executor, aclrtStream stream);
```

### 3.4 Torch Extension (OpBuilder pattern)

```python
# torch_extension/npu_ops_transformer/ops/{op_name}.py
class MyOpBuilder(OpBuilder):
    def sources(self): return ['csrc/my_op.cpp']
    def schema(self): return "my_op(Tensor x, ...) -> Tensor"
    # OpBuilder handles: CppExtension, include_dirs, library_dirs
```

### 3.5 TORCH_LIBRARY Pattern (experimental/)

```cpp
// experimental/ascend_ops/npu_ops_def.cpp
TORCH_LIBRARY(npu_ops_transformer_ext, m) {
    m.def("op_name(Tensor x, ...) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m) {
    m.impl("op_name", &op_name_npu);
}

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m) {
    m.impl("op_name", &op_name_meta);
}
```

### 3.6 Build System

CMake SDK discovery:
```cmake
set(ASCEND_CANN_PACKAGE_PATH
    ${CUSTOM_PATH}
    $ENV{ASCEND_HOME_PATH}
    "/usr/local/Ascend/latest")
```

---

## 4. omni-ops (~/repo/vllm-project/omniai/omni-ops)

### 4.1 Project Layout

```
omni-ops/training/ascendc/
├── CMakeLists.txt
├── cmake/                      # FindCANN-style modules
├── src/ops-transformer/        # Kernel source
├── torch_ops_extension/
│   ├── setup.py                # pip install entry point
│   └── omni_training_custom_ops/
│       ├── __init__.py         # Loads .so, exports to torch_npu
│       ├── csrc_base/
│       │   ├── ops_common.h    # ACL/torch includes, ACLNN_CMD macro
│       │   ├── ops_def_registration.cpp  # TORCH_LIBRARY_FRAGMENT
│       │   └── function.h
│       └── ops_transformer/
│           └── {category}/{op_name}/
│               ├── csrc/npu_*.cpp   # TORCH_LIBRARY_IMPL per op
│               ├── __init__.py
│               └── test/
```

### 4.2 TORCH_LIBRARY Registration

Central schema (csrc_base/ops_def_registration.cpp):
```cpp
TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.def("npu_flash_attention(Tensor q, Tensor k, ...) -> Tensor");
    m.def("npu_scatter_update(Tensor self, ...) -> Tensor");
    // ... all ops registered here
}
```

Per-op dispatch (csrc/npu_op.cpp):
```cpp
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_flash_attention", &npu_flash_attention_impl);
}
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_flash_attention", &npu_flash_attention_meta);
}
```

### 4.3 setup.py

```python
from setuptools import setup, find_packages
from torch_npu.utils.cpp_extension import NpuExtension

ext = NpuExtension(
    name="omni_training_custom_ops.custom_ops_lib",
    sources=glob('csrc_base/*.cpp') + glob('ops_transformer/**/csrc/*.cpp'),
    include_dirs=[f"{cann_path}/include", ...],
)

setup(
    name="omni_training_custom_ops",
    packages=find_packages(),
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
)
```

### 4.3 Test Pattern

```python
import torch, torch_npu
from omni_training_custom_ops import custom_ops_lib  # triggers C++ lib load

def test_op():
    x = torch.randn(4, 8, dtype=torch.bfloat16).npu()
    out = torch.ops.custom.npu_op_name(x)
    assert out.shape == expected_shape
```

### 4.4 GE Converter Pattern (for torch.compile / graph mode)

```python
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter

@register_fx_node_ge_converter(torch.ops.custom.npu_op_name.default)
def convert_npu_op_name(x, *, out, meta_outputs=None):
    return ge.CustomOp("NpuOpName", x)
```

---

## 5. Cross-Repo Pattern Summary

### 5.1 Naming Conventions

| Convention | Example | Where |
|---|---|---|
| TORCH_LIBRARY namespace | `custom_comm` | setup.py + ops_registration.cpp |
| Op name prefix | `custom_comm::allgather_batch` | schema def |
| C API prefix | `HcclCustomComm*` or `CustomComm*` | C headers |
| Python module | `custom_comm.ops` | __init__.py |

### 5.2 Build System Patterns

SDK path discovery (all repos use same pattern):
```cmake
set(ASCEND_CANN_PACKAGE_PATH
    "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN SDK root")
# Sub-paths:
#   ${SDK}/include/          -- public headers
#   ${SDK}/lib64/            -- shared libraries
#   ${SDK}/tools/            -- op build tools
```

Torch extension build:
```python
NpuExtension(name, sources, include_dirs=[
    f'{SDK}/include', f'{SDK}/include/aclnn'])
```

### 5.3 Dispatch Key Convention

| Key | Purpose |
|---|---|
| `PrivateUse1` | NPU device implementation |
| `Meta` | Shape inference (torch.compile) |
| `AutogradPrivateUse1` | Custom autograd |

---

## Cross-Cutting: Applicable to custom_comm

### Op Registration Pattern (recommended for custom_comm)

```cpp
// ops_def_registration.cpp
TORCH_LIBRARY(custom_comm, m) {
    m.def("allgather_batch(Tensor[] inputs, str hcom, int rank_size) -> Tensor[]");
}

// allgather_batch_npu.cpp
TORCH_LIBRARY_IMPL(custom_comm, PrivateUse1, m) {
    m.impl("allgather_batch", allgather_batch_npu);
}

// allgather_batch_meta.cpp
TORCH_LIBRARY_IMPL(custom_comm, Meta, m) {
    m.impl("allgather_batch", allgather_batch_meta);
}
```

### Error Handling Pattern

```cpp
// Wrap HCCL calls
#define HCCL_CHECK(call)                                          \
    do {                                                          \
        HcclResult ret = (call);                                  \
        TORCH_CHECK(ret == HCCL_SUCCESS,                          \
            "HCCL error: ", ret, " at ", __FILE__, ":", __LINE__);\
    } while (0)
```

### CCU Kernel Skeleton (Phase 2)

```cpp
class AllGatherBatchMesh1d : public hcomm::CcuKernel {
    HcclResult Algorithm() override {
        // 1. CreateVariable() for each desc
        // 2. Load() args from GeneArgs vector
        // 3. Loop: WriteNb per remote rank + LocalCopyNb for self
        // 4. NotifyRecord/NotifyWait for sync
    }
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override {
        // Pack: baseAddr, perDescInfo[], rankSize, rankId
        // Must fit CCU_SQE_ARGS_LEN=13 or use indirect buffer
    }
};
```

### Build System Skeleton for custom_comm

```cmake
cmake_minimum_required(VERSION 3.18)
project(custom_comm CXX)
set(CMAKE_CXX_STANDARD 17)

# SDK discovery
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CANN_PACKAGE_PATH}
    CACHE PATH "CANN toolkit root")
include_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/hcomm/hcomm/include
    ${ASCEND_CANN_PACKAGE_PATH}/hcomm/hcomm/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/npu-runtime/runtime/include
)
link_directories(${ASCEND_CANN_PACKAGE_PATH}/lib64)

# C++ library
add_library(custom_comm_ops SHARED
    ops/allgather_batch.cc
    ops/ccu_kernel_ag_batch.cc
    ops/engine_ctx.cc
)
target_link_libraries(custom_comm_ops hccl hcomm ascendcl)
```

### setup.py Pattern

```python
from torch_npu.utils.cpp_extension import NpuExtension
from torch.utils.cpp_extension import BuildExtension
from setuptools import setup

setup(
    name="custom_comm",
    packages=["custom_comm"],
    ext_modules=[NpuExtension(
        "custom_comm._C",
        sources=["csrc/ops_registration.cpp", "csrc/allgather_batch.cpp"],
        include_dirs=[...],
        library_dirs=[...],
        libraries=["hccl", "hcomm"],
    )],
    cmdclass={"build_ext": BuildExtension},
)
```
