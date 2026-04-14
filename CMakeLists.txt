cmake_minimum_required(VERSION 3.18)
project(custom_comm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- SDK discovery ----
include(cmake/FindCANN.cmake)

# ---- C API library ----
add_library(custom_comm_ops SHARED
    ops/allgather_batch/src/all_gather_batch.cc
    ops/allgather_batch/src/ccu_kernel_ag_batch_mesh1d.cc
    ops/allgather_batch/src/decomposed_strategy.cc
    ops/allgather_batch/src/engine_ctx.cc
)

target_include_directories(custom_comm_ops PRIVATE
    ops/allgather_batch/inc
    ${CANN_INCLUDE_DIRS}
    ${CANN_CCU_INCLUDE_DIRS}
    ${CANN_ACL_INCLUDE_DIRS}
)

if(CANN_LIBRARIES)
    target_link_libraries(custom_comm_ops PRIVATE ${CANN_LIBRARIES})
endif()

# On macOS, allow unresolved symbols (syntax-check only, no runtime)
if(APPLE)
    target_link_options(custom_comm_ops PRIVATE -undefined dynamic_lookup)
endif()
