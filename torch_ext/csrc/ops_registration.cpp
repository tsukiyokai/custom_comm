// Copyright (c) 2026 custom_comm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// TORCH_LIBRARY schema registration for custom_comm namespace.
// Defines the operator schema; implementations are in allgather_batch.cpp.

#include <torch/library.h>

TORCH_LIBRARY(custom_comm, m) {
    m.def("allgather_batch(Tensor[] inputs, str hcom, "
          "int world_size) -> Tensor[]");
}
