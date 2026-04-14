# FindCANN.cmake -- locate CANN SDK headers and libraries for custom_comm.
#
# Supports two SDK layouts:
#   A. Installed toolkit (Docker / bare-metal):
#        ${SDK}/include/hccl/hccl_types.h
#        ${SDK}/pkg_inc/hccl/...
#        ${SDK}/include/acl/acl.h
#   B. Dev-tree (.sdk symlink, macOS syntax-check):
#        ${SDK}/hcomm/hcomm/include/hccl/hccl_types.h
#        ${SDK}/hcomm/hcomm/pkg_inc/...
#        ${SDK}/npu-runtime/include/external/acl/...
#
# SDK root search order:
#   1. -DASCEND_CANN_PACKAGE_PATH=...
#   2. $ENV{ASCEND_CANN_PACKAGE_PATH}
#   3. $ENV{ASCEND_HOME_PATH}
#   4. /usr/local/Ascend/ascend-toolkit/latest

# ---- Resolve SDK root ----
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

# ==== Detect SDK layout ====
# Layout A: installed toolkit  (include/hccl/hccl_types.h directly under SDK root)
# Layout B: dev-sdk tree       (hcomm/hcomm/include/hccl/hccl_types.h)
set(_SDK "${ASCEND_CANN_PACKAGE_PATH}")

if(EXISTS "${_SDK}/include/hccl/hccl_types.h")
    # Layout A: installed CANN toolkit
    set(CANN_INCLUDE_DIRS "${_SDK}/include" "${_SDK}/include/hccl")
    set(CANN_CCU_INCLUDE_DIRS "${_SDK}/pkg_inc")
    set(CANN_ACL_INCLUDE_DIRS "${_SDK}/include")
    set(CANN_LIBRARY_DIRS "${_SDK}/lib64")
    set(_LAYOUT "installed")
elseif(EXISTS "${_SDK}/hcomm/hcomm/include/hccl/hccl_types.h")
    # Layout B: local dev-sdk (macOS syntax-check)
    set(_HCOMM "${_SDK}/hcomm/hcomm")
    set(CANN_INCLUDE_DIRS "${_HCOMM}/include" "${_HCOMM}/include/hccl")
    set(CANN_CCU_INCLUDE_DIRS "${_HCOMM}/pkg_inc")
    set(CANN_ACL_INCLUDE_DIRS "${_SDK}/npu-runtime/runtime/include/external")
    set(CANN_LIBRARY_DIRS "${_SDK}/lib64")
    set(_LAYOUT "dev-sdk")
else()
    set(CANN_FOUND FALSE)
    message(WARNING "CANN SDK headers not found under ${_SDK}. "
                    "Set ASCEND_CANN_PACKAGE_PATH to the SDK root.")
    return()
endif()

set(CANN_FOUND TRUE)
message(STATUS "CANN SDK layout: ${_LAYOUT} (${_SDK})")
message(STATUS "  include:  ${CANN_INCLUDE_DIRS}")
message(STATUS "  pkg_inc:  ${CANN_CCU_INCLUDE_DIRS}")

# ---- Validate that hccl_types.h is reachable ----
find_path(_HCCL_TYPES_DIR hccl/hccl_types.h PATHS ${CANN_INCLUDE_DIRS} NO_DEFAULT_PATH)
if(NOT _HCCL_TYPES_DIR)
    set(CANN_FOUND FALSE)
    message(WARNING "hccl/hccl_types.h not found in ${CANN_INCLUDE_DIRS}")
    return()
endif()

# ---- Libraries (non-Apple only) ----
if(NOT APPLE)
    find_library(CANN_HCCL_LIB hccl PATHS ${CANN_LIBRARY_DIRS} NO_DEFAULT_PATH)
    if(CANN_HCCL_LIB)
        set(CANN_LIBRARIES ${CANN_HCCL_LIB})
    endif()
endif()
