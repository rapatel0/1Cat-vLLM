// Standalone binding used to benchmark the SM70 fused all-reduce/Gemma
// RMSNorm prototype against the already-running vLLM custom communicator.
// This file is intentionally not part of the production extension target.

#include <torch/extension.h>

#include "csrc/ops.h"

TORCH_LIBRARY(qwen38_c2_ar, m) {
  m.def(
      "all_reduce_gemma_rms_norm_sm70(int fa, Tensor inp, Tensor residual, "
      "Tensor gamma, Tensor! norm_out, Tensor! residual_out, float epsilon, "
      "int reg_buffer, int reg_buffer_sz_bytes) -> ()");
  m.def(
      "all_reduce_gemma_rms_norm_sm70_cooperative(int fa, Tensor inp, "
      "Tensor residual, Tensor gamma, Tensor! norm_out, "
      "Tensor! residual_out, Tensor! row_sums, float epsilon, "
      "int reg_buffer, int reg_buffer_sz_bytes, int ctas_per_row, "
      "int threads) -> ()");
}

TORCH_LIBRARY_IMPL(qwen38_c2_ar, CUDA, m) {
  m.impl("all_reduce_gemma_rms_norm_sm70",
         &all_reduce_gemma_rms_norm_sm70);
  m.impl("all_reduce_gemma_rms_norm_sm70_cooperative",
         &all_reduce_gemma_rms_norm_sm70_cooperative);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
