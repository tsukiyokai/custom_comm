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
    # SDK/include is needed for hccl/hccl_comm.h (not bundled in torch_npu).
    # NpuExtension also injects torch_npu's own ACL/HCCL headers, but ours
    # come first in the search order.  Our header (hccl_custom_allgather_batch.h)
    # forward-declares aclrtStream to avoid pulling in acl_base_rt.h.
    _inc = [
        os.path.join(SDK, "include"),
        os.path.join(SDK, "include", "hccl"),
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

    # Strategy sources: decomposed is always included; CCU is optional.
    _agb_sources = [
        "ops/allgather_batch/src/all_gather_batch.cc",
        "ops/allgather_batch/src/decomposed/decomposed_strategy.cc",
    ]
    _extra_macros = []
    if os.environ.get("CUSTOM_COMM_ENABLE_CCU", "0") == "1":
        _agb_sources += [
            "ops/allgather_batch/src/ccu/engine_ctx.cc",
            "ops/allgather_batch/src/ccu/ccu_kernel_ag_batch_mesh1d.cc",
        ]
        _extra_macros.append(("CUSTOM_COMM_ENABLE_CCU", "1"))

    ext_modules = [NpuExtension(
        name="custom_comm._C",
        sources=[
            "torch_ext/csrc/ops_registration.cpp",
            "torch_ext/csrc/allgather_batch.cpp",
        ] + _agb_sources,
        include_dirs=_inc + [
            os.path.join(os.path.dirname(__file__), "ops", "allgather_batch", "inc"),
            os.path.join(os.path.dirname(__file__), "ops", "allgather_batch", "src"),
        ],
        library_dirs=_lib,
        libraries=["hcomm", "ascendcl"] if _lib else [],
        extra_compile_args=["-std=c++17"],
        define_macros=[(k, v) for k, v in _extra_macros],
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
