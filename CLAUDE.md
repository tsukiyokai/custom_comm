# CLAUDE.md

This file provides guidance to Claude Code for working with this repository.

## Project Overview

custom_comm is a library of high-performance custom communication operators for
Ascend NPU (CANN ecosystem). It provides batched collective operations as PyTorch
custom ops via torch_npu, with support for eager mode, graph mode (torchair), and
heterogeneous-dtype tensors.

Target platform: Atlas A5 (Ascend 910_95), CANN 9.0+.

## Architecture

The first operator is `allgather_batch` -- gather up to 8 heterogeneous-dtype
tensors in a single collective call. Two execution paths:

- Phase 1 (decomposed): packs tensors into a flat buffer, calls HcclAllGather once, unpacks.
- Phase 2 (CCU): launches a single CCU kernel for zero-copy RDMA gather per descriptor.

Selected at runtime via `CUSTOM_COMM_USE_CCU=1`.

## Repository Layout

    CMakeLists.txt              C++ build (standalone libcustom_comm_ops.so)
    setup.py                    Python package (torch NpuExtension)
    cmake/FindCANN.cmake        SDK detection (installed toolkit or dev-tree)
    ops/allgather_batch/
      inc/                      C/C++ headers (C API, common defs, CCU kernel)
      src/                      Implementation (dispatch, decomposed, CCU, engine ctx)
    torch_ext/csrc/             PyTorch C++ extension (op registration + NPU/Meta impl)
    python/custom_comm/         Python package
      ops.py                    torch.autograd wrapper
      converters/               torchair GE graph-mode converter
    tests/                      pytest (meta + NPU functional + benchmark)
    docs/                       Design docs, architecture diagrams

## Build

C++ only (syntax check):

    cmake -B build && cmake --build build

Python extension (requires CANN SDK + torch_npu):

    source ~/Ascend/set_env.sh
    export LD_LIBRARY_PATH=~/Ascend/cann-9.0.0/x86_64-linux/lib64:$LD_LIBRARY_PATH
    pip install -e .

## Testing

    pytest tests/ -k "meta"           # Meta-device shape tests (no NPU needed)
    pytest tests/ -v                   # All available tests
    torchrun --nproc_per_node=N pytest tests/  # NPU functional (needs hardware)

## Blue Zone (Remote Build)

Blue zone is for compilation and meta tests (no NPU hardware).
NPU functional tests run on yellow zone separately.

    workflow:
        local edit
            -> rsync
                -> build on blue zone
                    -> pytest meta
                        -> commit & push

    # 1. Sync local changes to blue zone
    rsync -avz --exclude='build/' --exclude='*.so' --exclude='__pycache__' \
        ./ bluezone:~/code/custom_comm/

    # 2. SSH in and build
    ssh bluezone
    source ~/Ascend/cann-9.0.0/set_env.sh
    source ~/venv/bin/activate
    cd ~/code/custom_comm && pip install -e . --no-build-isolation

    # 3. Run non-NPU tests
    pytest tests/ -k meta

Connection and environment:
- Host alias: `bluezone` (in ~/.ssh/config)
- CANN SDK: ~/Ascend/cann-9.0.0 (do NOT use /usr/local/Ascend, that is an older version)
- Python venv: ~/venv (has torch 2.x + torch_npu)
- Code path: ~/code/custom_comm/

## Key Constraints

- CANN 9.0 SDK required (headers in `include/hccl/`, `pkg_inc/`, runtime in `x86_64-linux/lib64/`)
- torch_npu NpuExtension injects its own ACL headers; do NOT add `${SDK}/include` broadly to
  setup.py include_dirs -- only `include/hccl/`, `include/hcomm/`, `pkg_inc/`
- `hccl_custom_allgather_batch.h` uses forward-declared `aclrtStream` (not `#include <acl/acl_base_rt.h>`)
  to avoid conflicts with torch_npu's bundled ACL headers
