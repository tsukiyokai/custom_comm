# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build script for custom_comm: custom communication operators for Ascend NPU."""

import os
from setuptools import setup, find_packages

# ==== CANN SDK discovery ====
SDK = os.environ.get(
    "ASCEND_HOME_PATH",
    os.environ.get("ASCEND_CANN_PACKAGE_PATH",
                   "/usr/local/Ascend/ascend-toolkit/latest"),
)

# Two possible SDK layouts:
#   A) Installed toolkit: ${SDK}/include/hccl/hccl_types.h
#   B) Dev-SDK tree:      ${SDK}/hcomm/hcomm/include/hccl/hccl_types.h
_hcomm = os.path.join(SDK, "hcomm", "hcomm")
if os.path.isfile(os.path.join(SDK, "include", "hccl", "hccl_types.h")):
    # NpuExtension auto-adds torch_npu/include/third_party/acl/inc (ACL headers).
    # Only add HCCL and pkg_inc here; avoid SDK/include/ to prevent ACL redefinition.
    _inc = [
        os.path.join(SDK, "include", "hccl"),
        os.path.join(SDK, "include", "hcomm"),
        os.path.join(SDK, "pkg_inc"),
    ]
    _lib = [
        os.path.join(SDK, "lib64"),
        os.path.join(SDK, "x86_64-linux", "lib64"),
    ]
elif os.path.isfile(os.path.join(_hcomm, "include", "hccl", "hccl_types.h")):
    _inc = [
        os.path.join(_hcomm, "include"),
        os.path.join(_hcomm, "include", "hccl"),
        os.path.join(_hcomm, "pkg_inc"),
        os.path.join(SDK, "npu-runtime", "include", "external"),
    ]
    _lib = [os.path.join(_hcomm, "lib64"), os.path.join(SDK, "lib64")]
else:
    _inc = []
    _lib = []

# ==== Extension (requires torch + torch_npu) ====

ext_modules = []
cmdclass = {}

try:
    from torch_npu.utils.cpp_extension import NpuExtension
    from torch.utils.cpp_extension import BuildExtension

    ext_modules = [NpuExtension(
        name="custom_comm._C",
        sources=[
            "torch_ext/csrc/ops_registration.cpp",
            "torch_ext/csrc/allgather_batch.cpp",
            "ops/allgather_batch/src/all_gather_batch.cc",
            "ops/allgather_batch/src/decomposed_strategy.cc",
            "ops/allgather_batch/src/engine_ctx.cc",
            "ops/allgather_batch/src/ccu_kernel_ag_batch_mesh1d.cc",
        ],
        include_dirs=_inc + [
            os.path.join(os.path.dirname(__file__), "ops", "allgather_batch", "inc"),
        ],
        library_dirs=_lib,
        libraries=["hcomm", "ascendcl"] if _lib else [],
        extra_compile_args=["-std=c++17"],
    )]
    cmdclass = {"build_ext": BuildExtension}
except ImportError:
    pass  # No torch_npu: metadata-only install

# ==== Package setup ====

setup(
    name="custom_comm",
    version="0.1.0",
    package_dir={"": "python"},
    packages=["custom_comm", "custom_comm.converters"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.8",
)
