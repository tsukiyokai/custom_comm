# FindCANN.cmake -- Locate CANN SDK for custom_comm
#
# SDK path priority:
#   1. CMake cache / -D flag
#   2. $ASCEND_CANN_PACKAGE_PATH
#   3. $ASCEND_HOME_PATH
#   4. /usr/local/Ascend/ascend-toolkit/latest
#
# Outputs:
#   CANN_FOUND            - TRUE if SDK headers located
#   CANN_INCLUDE_DIRS     - Public HCCL headers  (hccl_types.h, hccl_comm.h, ...)
#   CANN_CCU_INCLUDE_DIRS - Internal CCU headers  (ccu_kernel.h, hccl_inner.h, ...)
#   CANN_ACL_INCLUDE_DIRS - ACL runtime headers   (acl.h, acl_rt.h, ...)
#   CANN_LIBRARY_DIRS     - Library search paths
#   CANN_LIBRARIES        - Link targets (non-Apple only)

# ---- SDK root discovery ----
if(NOT ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
    elseif(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
    else()
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
endif()
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_CANN_PACKAGE_PATH}" CACHE PATH "CANN SDK root")
message(STATUS "CANN SDK root: ${ASCEND_CANN_PACKAGE_PATH}")

# ---- hcomm subpackage (double-nested: hcomm/hcomm/) ----
set(HCOMM_ROOT "${ASCEND_CANN_PACKAGE_PATH}/hcomm/hcomm")

set(CANN_INCLUDE_DIRS
    "${HCOMM_ROOT}/include"       # hccl/hccl_types.h, hccl_comm.h, hccl_res.h
    "${HCOMM_ROOT}/include/hccl"  # direct include without hccl/ prefix
)
set(CANN_CCU_INCLUDE_DIRS
    "${HCOMM_ROOT}/pkg_inc"       # hcomm/ccu/ccu_kernel.h, hccl/hccl_inner.h, hccl/hcom.h
)
set(CANN_ACL_INCLUDE_DIRS
    "${ASCEND_CANN_PACKAGE_PATH}/npu-runtime/runtime/include/external"  # acl/acl.h
    "${ASCEND_CANN_PACKAGE_PATH}/npu-runtime/include"                   # securec.h
)
set(CANN_LIBRARY_DIRS
    "${HCOMM_ROOT}/lib64"
    "${ASCEND_CANN_PACKAGE_PATH}/lib64"
)

# ---- Validate header existence ----
find_path(_HCCL_TYPES_H "hccl/hccl_types.h" PATHS ${CANN_INCLUDE_DIRS} NO_DEFAULT_PATH)
if(_HCCL_TYPES_H)
    set(CANN_FOUND TRUE)
    message(STATUS "CANN headers found at: ${_HCCL_TYPES_H}")
else()
    set(CANN_FOUND FALSE)
    message(WARNING "CANN SDK headers not found. C++ syntax-check will fail. "
                    "Set ASCEND_CANN_PACKAGE_PATH to SDK root.")
endif()

# ---- Libraries (skip on macOS -- syntax-check only) ----
if(NOT APPLE AND CANN_FOUND)
    find_library(HCOMM_LIB    hcomm    PATHS ${CANN_LIBRARY_DIRS} NO_DEFAULT_PATH)
    find_library(ASCENDCL_LIB ascendcl PATHS ${CANN_LIBRARY_DIRS} NO_DEFAULT_PATH)

    set(CANN_LIBRARIES "")
    if(HCOMM_LIB)
        list(APPEND CANN_LIBRARIES ${HCOMM_LIB})
    endif()
    if(ASCENDCL_LIB)
        list(APPEND CANN_LIBRARIES ${ASCENDCL_LIB})
    endif()
    message(STATUS "CANN libraries: ${CANN_LIBRARIES}")
else()
    set(CANN_LIBRARIES "")
    if(APPLE)
        message(STATUS "macOS detected -- skipping CANN library linking (syntax-check mode)")
    endif()
endif()
