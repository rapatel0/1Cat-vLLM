// Standalone binding for the SM70 tile-runtime all-reduce experiments.
//
// The vLLM CustomAllreduce instance and IPC buffers are created by the normal
// extension.  This library accepts that instance's opaque pointer and launches
// a separately compiled experimental engine, keeping the production operator
// namespace and binary untouched.

#include <torch/extension.h>

#include "csrc/ops.h"

TORCH_LIBRARY(qwen38_tile_ar, m) {
  m.def(
      "tile_runtime_all_reduce(int fa, Tensor inp, Tensor! out, int "
      "reg_buffer, int reg_buffer_sz_bytes, int tile_numel, int "
      "engine_blocks, int compute_iters) -> ()");
  m.def(
      "tile_runtime_all_reduce_engine(int fa, Tensor inp, Tensor! out, int "
      "reg_buffer, int reg_buffer_sz_bytes, int tile_numel, int "
      "producer_blocks, int reducer_blocks, int compute_iters) -> ()");
  m.def(
      "tile_runtime_wait_reduce(int fa, Tensor staging, Tensor! out, "
      "int tile_numel, int reducer_blocks) -> ()");
}

TORCH_LIBRARY_IMPL(qwen38_tile_ar, CUDA, m) {
  m.impl("tile_runtime_all_reduce", &tile_runtime_all_reduce);
  m.impl("tile_runtime_all_reduce_engine",
         &tile_runtime_all_reduce_engine);
  m.impl("tile_runtime_wait_reduce", &tile_runtime_wait_reduce);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
