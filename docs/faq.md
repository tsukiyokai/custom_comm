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

## 8. FindCANN.cmake 安装版需要额外 include 路径

CANN SDK 的 `pkg_inc/hcomm/ccu/` 内部头文件使用裸 `#include "hcomm_primitives.h"`，
该文件实际位于 `include/hccl/`。CMake 的 include 路径需要同时包含
`${SDK}/include` 和 `${SDK}/include/hccl`，否则编译失败。
