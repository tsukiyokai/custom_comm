# FAQ: CANN 环境编译与部署

## 1. CANN 9.0 的 libhccl.so 在哪？

CANN 9.0 将 HCCL 运行时拆分为多个库（`libhccl_v2.so`, `libhccl_alg.so` 等），
同时保留了聚合入口 `libhccl.so`。但这些文件位于 `x86_64-linux/lib64/` 而非
`set_env.sh` 默认配置的 `lib64/`。

    # set_env.sh 仅设置了 cann-9.0.0/lib64/，需要追加：
    export LD_LIBRARY_PATH=$ASCEND_HOME/x86_64-linux/lib64:$LD_LIBRARY_PATH

## 2. torch_npu import 失败：libhccl.so not found

原因：`set_env.sh` 的 `LD_LIBRARY_PATH` 缺少 `x86_64-linux/lib64/`。补上即可：

    source ~/Ascend/set_env.sh
    export LD_LIBRARY_PATH=~/Ascend/cann-9.0.0/x86_64-linux/lib64:$LD_LIBRARY_PATH

## 3. pip install torch_npu 报版本不匹配

编译服务器上装的是 `torch==2.9.0+cpu`（CPU-only wheel），而 PyPI 上的
`torch_npu` 要求精确匹配 `torch==2.9.0`。用 `--no-deps` 跳过依赖检查：

    pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
    pip install torch_npu==2.9.0 --no-deps

手动补装依赖（torch_npu 运行时需要 `pyyaml` 和 `numpy`）：

    pip install pyyaml numpy

## 4. inspkg.sh -c 不含运行时库

`inspkg.sh -c`（COMPILE_ONLY 模式）只安装编译所需的 headers 和工具链，
不包含 `libhccl.so` 等运行时库。如需完整安装：

    bash inspkg.sh -V 9.0.0 -s 950    # 不加 -c

## 5. FindCANN.cmake 如何适配不同 SDK 布局

项目的 `cmake/FindCANN.cmake` 支持两种布局：

- Layout A（安装版）：headers 在 `${SDK}/include/hccl/`，由 inspkg.sh 或 Docker 安装
- Layout B（开发版）：headers 在 `${SDK}/hcomm/hcomm/include/hccl/`，macOS 本地 .sdk 目录

CMake 自动检测。如果两种都找不到，设置 `-DASCEND_CANN_PACKAGE_PATH=/path/to/sdk`。

## 6. Docker 容器遗留 root 权限文件

Docker 中运行 `cmake --build` 或 `pip install` 后，`build/` 和 `*.egg-info/`
的 owner 是 root，宿主机 user 无法删除。解决方法：

    # 在容器内
    chown -R $(id -u):$(id -g) /workspace/custom_comm

    # 或重新 clone 到干净目录
    git clone ... ~/code/custom_comm_clean

## 7. SSH 反向隧道代理加速下载

蓝区服务器下载 PyPI 包较慢时，可通过 Mac 代理加速：

    # ~/.ssh/config
    Host bluezone
        HostName 1.95.171.223
        User fan33
        IdentityFile ~/.ssh/id_ed25519
        RemoteForward 10077 127.0.0.1:10077

    # 蓝区使用:
    export http_proxy=http://127.0.0.1:10077
    export https_proxy=http://127.0.0.1:10077

## 8. FindCANN.cmake 双布局支持

CANN SDK 有两种路径布局：

- 安装版（inspkg/Docker）：`${SDK}/include/hccl/hccl_types.h`
- 开发版（macOS .sdk）：`${SDK}/hcomm/hcomm/include/hccl/hccl_types.h`

CMake 自动检测。如果都找不到，设置 `-DASCEND_CANN_PACKAGE_PATH=` 指向 SDK 根目录。

## 9. setup.py 与 NpuExtension 的 ACL header 冲突

NpuExtension 自动注入 `torch_npu/include/third_party/acl/inc/`。
如果 setup.py 同时添加 `${SDK}/include/`，两套 `acl_base.h` 会导致
符号重定义。解法：只添加 `${SDK}/include/hccl/`、`${SDK}/include/hcomm/`
和 `${SDK}/pkg_inc/`，避免引入 SDK 的 acl 目录。

项目头文件中的 `#include <acl/acl_base_rt.h>` 改为 forward-declare
`typedef void *aclrtStream;`。

## 10. torchair converter 注册 API 变更（torch_npu >= 2.9）

`@register_fx_node_ge_converter` 行为变化：
- 装饰后返回 `Converter` 对象而非原始函数
- converter 挂在 `op._ge_converter` 属性上
- 不再使用全局 `_CONVERTERS` dict

测试应检查 `hasattr(op, "_ge_converter")` 而非查 dict。

## 11. CANN toolkit 4 月版本缺少 HcclThreadResGetInfo

从 mirror 下载的 CANN 9.0 toolkit（含 4 月 8 日构建）不包含 `HcclThreadResGetInfo`
API（头文件和 libhcomm.so 均无）。需要从 hcomm 源码仓库编译安装才能获得该符号。

    cd hcomm && bash build.sh --full --pkg  # 在 Docker 容器中编译
    bash build_out/cann-hcomm_*.run --full --install-path=$HOME/Ascend

## 12. hcomm 源码编译需要 Docker 环境

宿主机直接编译 hcomm 会遇到 `__sk__` 类型错误（SuperKernel 编译器问题）。
需要在 `cann-build-9.0.0` Docker 容器中编译，容器内的 ccec 编译器版本与源码匹配。

    docker exec cc-build bash -c 'source /usr/local/Ascend/set_env.sh && \
        cd /workspace/hcomm && bash build.sh --full --pkg'

## 13. hccl 源码编译依赖新版 hcomm

hccl 编译必须在安装了最新 hcomm 之后进行（hccl 依赖 hcomm 的头文件和库）。
安装顺序：hcomm -> hccl。

## 14. rsync 同步代码比 git push/pull 更快

通过 rsync 直接同步本地代码到蓝区，比 push 到 GitHub 再 pull 更快：

    bash sync.sh    # 使用项目根目录的 sync.sh

## 15. pip install 后 C 扩展丢失

rsync 同步会覆盖 `*.egg-info`，导致 pip editable install 的符号链接失效。
每次 rsync 后需要重新 `pip install -e .`。

## 16. CCU 路径编译报 HcclCommConfig 字段缺失

现象：编译 CCU 代码路径时（默认行为，由 `CUSTOM_COMM_ENABLE_CCU` 宏开启），CANN 9.0 SDK 的 `hccl_comm.h`
内联初始化函数报一串 `'HcclCommConfig' has no member named 'hcclAlgo' /
'hcclBufferName' / 'aclGraphZeroCopyEnable' / ...` 错误。

根因：torch_npu 2.9.0 在 `torch_npu/include/third_party/hccl/inc/hccl/` 下
bundle 了较老版本的 `hccl_types.h`，其中 `HcclCommConfig` 只有约 10 个字段。
CANN 9.0 SDK 把该 struct 扩展到 14+ 个字段，且在 `hccl_comm.h` 里把写入这些
新字段的逻辑 inline 在 `HcclCommConfigInit()` 里。NpuExtension 默认的 `-I`
顺序让 torch_npu 的 bundle header 先于 SDK header 被解析，导致新字段在
struct 定义里找不到。

解法：在 `setup.py` 中把 `${ASCEND_HOME}/include/` 从 `-isystem` 提到 `-I`
最前面，让 CANN SDK 的 `hccl_types.h` 先被找到。

    _inc = [
        os.path.join(SDK, "include"),   # 最高优先级，覆盖 torch_npu bundle
        os.path.join(SDK, "pkg_inc"),
    ]

编译 CCU 路径时还需要额外加 `pkg_inc/hcomm`、`pkg_inc/hcomm/ccu`、`include/hccl`，
因为 SDK 里的 header 相互使用相对路径 include（setup.py 默认就会加上）:

    for sub in ["hcomm", "hcomm/ccu"]:
        _inc.append(os.path.join(SDK, "pkg_inc", sub))
    _inc.append(os.path.join(SDK, "include", "hccl"))

关于依赖版本：torch_npu 2.9.0 是 PyPI 上最新稳定版（2.10.0rc1 / 2.11.0rc1
都还是 RC），即使升级也不会解决问题。bundle 的 `hccl_types.h` 本来就是
torch_npu 自己内部编译的 snapshot，不保证与当前 CANN SDK 一致。正确做法
是让 SDK 的 header 覆盖 bundle，让编译期 struct 布局与运行时 `libhccl.so`
的 ABI 一致。

对应 commit：`aac0d7d`。
