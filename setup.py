# Copyright (c) 2026 custom_comm Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build script for custom_comm torch extension."""

import os
from setuptools import setup, find_packages

# SDK root (CANN 9.0.0 --devel install)
SDK = os.environ.get(
    "ASCEND_CANN_PACKAGE_PATH",
    os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"),
)
HCOMM = os.path.join(SDK, "hcomm", "hcomm")

# ---- Extension setup ----
# NpuExtension auto-adds torch_npu include paths and NPU compile flags.
# Import guarded: on macOS (no torch_npu), fall back to metadata-only install.

ext_modules = []
cmdclass = {}

try:
    from torch_npu.utils.cpp_extension import NpuExtension
    from torch.utils.cpp_extension import BuildExtension

    ext_modules = [
        NpuExtension(
            "custom_comm._C",
            sources=[
                "torch_ext/csrc/ops_registration.cpp",
                "torch_ext/csrc/allgather_batch.cpp",
                "ops/allgather_batch/src/all_gather_batch.cc",
                "ops/allgather_batch/src/decomposed_strategy.cc",
                "ops/allgather_batch/src/engine_ctx.cc",
                "ops/allgather_batch/src/ccu_kernel_ag_batch_mesh1d.cc",
            ],
            include_dirs=[
                os.path.join(HCOMM, "include"),       # hccl/hccl_types.h
                os.path.join(HCOMM, "pkg_inc"),        # hccl/hccl_inner.h, hccl/hcom.h
                os.path.join(SDK, "npu-runtime", "runtime", "include", "external"),  # acl/
                os.path.join(os.path.dirname(__file__), "ops", "allgather_batch", "inc"),
            ],
            library_dirs=[
                os.path.join(HCOMM, "lib64"),
                os.path.join(SDK, "lib64"),
            ],
            libraries=["hcomm", "ascendcl"],
            extra_compile_args=["-std=c++17"],
        ),
    ]
    cmdclass = {"build_ext": BuildExtension}
except ImportError:
    pass  # macOS: no torch_npu, skip extension build

setup(
    name="custom_comm",
    version="0.1.0",
    description="Custom communication operators for Ascend NPU",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.8",
)
