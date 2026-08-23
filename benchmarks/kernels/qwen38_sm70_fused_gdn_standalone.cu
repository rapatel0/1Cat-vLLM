// Standalone SM70/FP16 adaptation of upstream vLLM's fused Qwen GDN MTP
// post-convolution recurrent update + gated RMSNorm kernel. This is a
// component benchmark, not a production extension target.

#include <torch/extension.h>

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kKeyHeads = 4;
constexpr int kValueHeads = 12;
constexpr int kDim = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxTokens = 4;
constexpr int kQkvDim = (2 * kKeyHeads + kValueHeads) * kDim;
constexpr int kConvWidth = 4;
constexpr int kConvStateLen = kConvWidth - 1 + kMaxTokens - 1;

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_fast(float x) {
  return x * sigmoid_fast(x);
}

__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ void conv4_channel(
    const half* __restrict__ raw_qkv, int64_t raw_stride0, int channel,
    const half* __restrict__ conv_state, int64_t conv_slot_stride,
    int64_t conv_channel_stride, int64_t conv_tap_stride, int source_slot,
    int selected,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias, bool has_conv_bias,
    half (&result)[kMaxTokens]) {
  const int offset = selected - 1;
  const int64_t state_base =
      static_cast<int64_t>(source_slot) * conv_slot_stride +
      static_cast<int64_t>(channel) * conv_channel_stride;
  half col0 = conv_state[state_base + offset * conv_tap_stride];
  half col1 = conv_state[state_base + (offset + 1) * conv_tap_stride];
  half col2 = conv_state[state_base + (offset + 2) * conv_tap_stride];
  const int64_t weight_base = static_cast<int64_t>(channel) * kConvWidth;
  const float w0 = __half2float(conv_weight[weight_base]);
  const float w1 = __half2float(conv_weight[weight_base + 1]);
  const float w2 = __half2float(conv_weight[weight_base + 2]);
  const float w3 = __half2float(conv_weight[weight_base + 3]);
  const float bias = has_conv_bias ? __half2float(conv_bias[channel]) : 0.0f;
#pragma unroll
  for (int token = 0; token < kMaxTokens; ++token) {
    const half x = raw_qkv[static_cast<int64_t>(token) * raw_stride0 + channel];
    float acc = bias;
    acc = fmaf(__half2float(col0), w0, acc);
    acc = fmaf(__half2float(col1), w1, acc);
    acc = fmaf(__half2float(col2), w2, acc);
    acc = fmaf(__half2float(x), w3, acc);
    result[token] = __float2half(acc / (1.0f + __expf(-acc)));
    col0 = col1;
    col1 = col2;
    col2 = x;
  }
}

__device__ __forceinline__ half conv4_channel_token(
    const half* __restrict__ raw_qkv, int64_t raw_stride0, int channel,
    int token, const half* __restrict__ conv_state,
    int64_t conv_slot_stride, int64_t conv_channel_stride,
    int64_t conv_tap_stride, int source_slot, int selected,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias, bool has_conv_bias) {
  const int offset = selected - 1;
  const int64_t state_base =
      static_cast<int64_t>(source_slot) * conv_slot_stride +
      static_cast<int64_t>(channel) * conv_channel_stride;
  const int64_t weight_base = static_cast<int64_t>(channel) * kConvWidth;
  float acc = has_conv_bias ? __half2float(conv_bias[channel]) : 0.0f;
#pragma unroll
  for (int tap = 0; tap < kConvWidth; ++tap) {
    const int sequence_index = token + tap;
    const half value =
        sequence_index < kConvWidth - 1
            ? conv_state[state_base +
                         (offset + sequence_index) * conv_tap_stride]
            : raw_qkv[static_cast<int64_t>(sequence_index -
                                           (kConvWidth - 1)) *
                              raw_stride0 +
                          channel];
    acc = fmaf(__half2float(value),
               __half2float(conv_weight[weight_base + tap]), acc);
  }
  return __float2half(acc / (1.0f + __expf(-acc)));
}

// Materialize the four causal-convolution outputs in the projection buffer
// before the row-split recurrence. One thread owns a channel, so it can retain
// all raw inputs until both the convolution and conv-state commit are done.
// This avoids recomputing every Q/K channel in each of the four row CTAs.
__global__ void fused_gdn_sm70_conv_prep_inplace_kernel(
    half* __restrict__ raw_qkv, int64_t raw_stride0,
    half* __restrict__ conv_state, int64_t conv_slot_stride,
    int64_t conv_channel_stride, int64_t conv_tap_stride,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias, bool has_conv_bias,
    const int* __restrict__ state_indices,
    const int* __restrict__ state_selector) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= kQkvDim) return;

  const int slot = state_indices[0];
  const int selected = state_selector[0];
  if (slot <= 0 || selected <= 0 || selected > kMaxTokens) return;

  const int offset = selected - 1;
  const int64_t state_base =
      static_cast<int64_t>(slot) * conv_slot_stride +
      static_cast<int64_t>(channel) * conv_channel_stride;
  const int64_t weight_base = static_cast<int64_t>(channel) * kConvWidth;

  const half history0 = conv_state[
      state_base + static_cast<int64_t>(offset) * conv_tap_stride];
  const half history1 = conv_state[
      state_base + static_cast<int64_t>(offset + 1) * conv_tap_stride];
  const half history2 = conv_state[
      state_base + static_cast<int64_t>(offset + 2) * conv_tap_stride];
  const half x0 = raw_qkv[channel];
  const half x1 = raw_qkv[raw_stride0 + channel];
  const half x2 = raw_qkv[2 * raw_stride0 + channel];
  const half x3 = raw_qkv[3 * raw_stride0 + channel];

  const float w0 = __half2float(conv_weight[weight_base]);
  const float w1 = __half2float(conv_weight[weight_base + 1]);
  const float w2 = __half2float(conv_weight[weight_base + 2]);
  const float w3 = __half2float(conv_weight[weight_base + 3]);
  const float bias =
      has_conv_bias ? __half2float(conv_bias[channel]) : 0.0f;
  const half inputs[kConvStateLen + 1] = {
      history0, history1, history2, x0, x1, x2, x3};

#pragma unroll
  for (int token = 0; token < kMaxTokens; ++token) {
    float acc = bias;
    acc = fmaf(__half2float(inputs[token]), w0, acc);
    acc = fmaf(__half2float(inputs[token + 1]), w1, acc);
    acc = fmaf(__half2float(inputs[token + 2]), w2, acc);
    acc = fmaf(__half2float(inputs[token + 3]), w3, acc);
    raw_qkv[static_cast<int64_t>(token) * raw_stride0 + channel] =
        __float2half(acc / (1.0f + __expf(-acc)));
  }

  // Preserve the established six-entry rolling layout exactly. The inputs
  // above are retained in registers, so overwriting raw_qkv cannot affect it.
  conv_state[state_base] = history1;
  conv_state[state_base + conv_tap_stride] = history2;
  conv_state[state_base + 2 * conv_tap_stride] = x0;
  conv_state[state_base + 3 * conv_tap_stride] = x1;
  conv_state[state_base + 4 * conv_tap_stride] = x2;
  conv_state[state_base + 5 * conv_tap_stride] = x3;
}

// Non-speculative q=1 variant. Ordinary decode owns exactly the causal-conv
// width-1 state: [h0, h1, h2] -> [h1, h2, x0]. The speculative executor may
// allocate extra columns, but this kernel neither requires nor touches them.
__global__ void fused_gdn_sm70_conv_prep_q1_inplace_kernel(
    half* __restrict__ raw_qkv,
    half* __restrict__ conv_state, int64_t conv_slot_stride,
    int64_t conv_channel_stride, int64_t conv_tap_stride,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias, bool has_conv_bias,
    const int* __restrict__ state_indices) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= kQkvDim) return;

  const int slot = state_indices[0];
  // PAD_SLOT_ID is -1. Slot zero is a valid live cache line.
  if (slot < 0) return;

  const int64_t state_base =
      static_cast<int64_t>(slot) * conv_slot_stride +
      static_cast<int64_t>(channel) * conv_channel_stride;
  const int64_t weight_base = static_cast<int64_t>(channel) * kConvWidth;
  const half history0 = conv_state[state_base];
  const half history1 = conv_state[state_base + conv_tap_stride];
  const half history2 = conv_state[state_base + 2 * conv_tap_stride];
  const half x0 = raw_qkv[channel];

  float acc = has_conv_bias ? __half2float(conv_bias[channel]) : 0.0f;
  acc = fmaf(__half2float(history0),
             __half2float(conv_weight[weight_base]), acc);
  acc = fmaf(__half2float(history1),
             __half2float(conv_weight[weight_base + 1]), acc);
  acc = fmaf(__half2float(history2),
             __half2float(conv_weight[weight_base + 2]), acc);
  acc = fmaf(__half2float(x0),
             __half2float(conv_weight[weight_base + 3]), acc);
  raw_qkv[channel] = __float2half(acc / (1.0f + __expf(-acc)));

  conv_state[state_base] = history1;
  conv_state[state_base + conv_tap_stride] = history2;
  conv_state[state_base + 2 * conv_tap_stride] = x0;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

__global__ __launch_bounds__(kThreads, 2) void fused_gdn_sm70_kernel(
    const half* __restrict__ mixed_qkv, const half* __restrict__ a,
    const half* __restrict__ b, const float* __restrict__ a_log,
    const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_selector, float* __restrict__ state,
    const half* __restrict__ output_gate,
    const void* __restrict__ norm_weight, half* __restrict__ out,
    int state_indices_width, int64_t state_slot_stride,
    bool dt_bias_is_bfloat16,
    bool norm_weight_is_float, float scale, float norm_eps) {
  const int value_head = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = query_start_loc[0];
  const int eos = query_start_loc[1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0 || num_tokens > kMaxTokens) return;

  const int selected = state_selector[0];
  const int source_slot =
      selected > 0 && selected <= state_indices_width
          ? state_indices[selected - 1]
          : 0;
  if (source_slot <= 0) {
    for (int linear = tid; linear < num_tokens * kDim; linear += kThreads) {
      const int token = bos + linear / kDim;
      const int value = linear % kDim;
      out[(token * kValueHeads + value_head) * kDim + value] =
          __float2half(0.0f);
    }
    return;
  }

  const int key_head = value_head / (kValueHeads / kKeyHeads);
  __shared__ float shared_q[kMaxTokens][kDim];
  __shared__ float shared_k[kMaxTokens][kDim];
  __shared__ half shared_v[kMaxTokens][kDim];
  __shared__ half shared_out[kMaxTokens][kDim];
  __shared__ float shared_decay[kMaxTokens];
  __shared__ float shared_beta[kMaxTokens];

  if (warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
    const int mixed_base = token * (2 * kKeyHeads + kValueHeads) * kDim;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] = __half2float(
          mixed_qkv[mixed_base + key_head * kDim + dim]);
      k_values[i] = __half2float(mixed_qkv[
          mixed_base + kKeyHeads * kDim + key_head * kDim + dim]);
      shared_v[t][dim] = mixed_qkv[
          mixed_base + 2 * kKeyHeads * kDim + value_head * kDim + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 sums = warp_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(
        0xffffffffu,
        lane == 0 ? rsqrtf(sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      shared_k[t][dim] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const float a_value = __half2float(a[token * kValueHeads + value_head]);
      const float b_value = __half2float(b[token * kValueHeads + value_head]);
      const float dt_bias_value =
          dt_bias_is_bfloat16
              ? __bfloat162float(
                    static_cast<const __nv_bfloat16*>(dt_bias)[value_head])
              : __half2float(static_cast<const half*>(dt_bias)[value_head]);
      const float g =
          -__expf(a_log[value_head]) *
          softplus_fast(a_value + dt_bias_value);
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    rows[row] = warp + row * kWarps;
  }

  const int64_t source_base =
      static_cast<int64_t>(source_slot) * state_slot_stride +
      static_cast<int64_t>(value_head) * kDim * kDim;
#pragma unroll
  for (int chunk = 0; chunk < kDim / kChunkV; ++chunk) {
    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row_i = 0; row_i < kRowsPerWarp; ++row_i) {
      // The warp owns four V rows. Across its 32 lanes each float4 load is a
      // contiguous 512-byte row transaction, so SM70 does not need the
      // SM80 kernel's cp.async-to-shared staging.
      const float4 value = *reinterpret_cast<const float4*>(
          state + source_base +
          (chunk * kChunkV + rows[row_i]) * kDim + k_base);
      h[row_i][0] = value.x;
      h[row_i][1] = value.y;
      h[row_i][2] = value.z;
      h[row_i][3] = value.w;
    }

    for (int t = 0; t < num_tokens; ++t) {
      const float4 q4 =
          *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
      const float4 k4 =
          *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};
      float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row_i = 0; row_i < kRowsPerWarp; ++row_i) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row_i][i] *= shared_decay[t];
          dot_hk[row_i] += h[row_i][i] * k_values[i];
        }
      }
      const Sum2 hk01 = warp_sum_pair(dot_hk[0], dot_hk[1]);
      const Sum2 hk23 = warp_sum_pair(dot_hk[2], dot_hk[3]);
      const float reduced_hk[kRowsPerWarp] = {hk01.x, hk01.y, hk23.x,
                                               hk23.y};
      float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row_i = 0; row_i < kRowsPerWarp; ++row_i) {
        const int value = chunk * kChunkV + rows[row_i];
        const float delta =
            (__half2float(shared_v[t][value]) - reduced_hk[row_i]) *
            shared_beta[t];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row_i][i] += k_values[i] * delta;
          dot_hq[row_i] += h[row_i][i] * q_values[i];
        }
      }
      const Sum2 hq01 = warp_sum_pair(dot_hq[0], dot_hq[1]);
      const Sum2 hq23 = warp_sum_pair(dot_hq[2], dot_hq[3]);
      if (lane == 0) {
        shared_out[t][chunk * kChunkV + rows[0]] = __float2half(hq01.x);
        shared_out[t][chunk * kChunkV + rows[1]] = __float2half(hq01.y);
        shared_out[t][chunk * kChunkV + rows[2]] = __float2half(hq23.x);
        shared_out[t][chunk * kChunkV + rows[3]] = __float2half(hq23.y);
      }

      const int destination_slot = state_indices[t];
      if (destination_slot > 0) {
        const int64_t destination =
            static_cast<int64_t>(destination_slot) * state_slot_stride +
            static_cast<int64_t>(value_head) * kDim * kDim +
            (chunk * kChunkV + rows[0]) * kDim + k_base;
#pragma unroll
        for (int row_i = 0; row_i < kRowsPerWarp; ++row_i) {
          const int64_t row_destination =
              destination + static_cast<int64_t>(row_i) * kWarps * kDim;
          *reinterpret_cast<float4*>(state + row_destination) =
              make_float4(h[row_i][0], h[row_i][1], h[row_i][2],
                          h[row_i][3]);
        }
      }
    }
  }
  __syncthreads();

  if (warp < num_tokens) {
    const int t = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      output_values[i] = __half2float(shared_out[t][value]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_sum(sum_square);
    const float rstd = rsqrtf(sum_square / static_cast<float>(kDim) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const int64_t offset =
          (static_cast<int64_t>(token) * kValueHeads + value_head) * kDim +
          value;
      const float gate = silu_fast(__half2float(output_gate[offset]));
      const float weight = norm_weight_is_float
                               ? static_cast<const float*>(norm_weight)[value]
                               : __half2float(
                                     static_cast<const half*>(norm_weight)[value]);
      out[offset] = __float2half(output_values[i] * rstd * weight * gate);
    }
  }
}

// V100 occupancy variant: split each value head's 128 recurrent rows across
// four independent CTAs. The first kernel writes the pre-normalization FP16
// result into `out`, which serves as graph-persistent scratch. The second
// kernel applies gated RMSNorm in place on the same stream.
__global__ __launch_bounds__(kThreads, 2)
void fused_gdn_sm70_48block_recurrence_kernel(
    const half* __restrict__ mixed_qkv, int64_t mixed_stride0,
    const half* __restrict__ conv_state, int64_t conv_slot_stride,
    int64_t conv_channel_stride, int64_t conv_tap_stride,
    const half* __restrict__ conv_weight,
    const half* __restrict__ conv_bias, bool has_conv_bias, bool apply_conv,
    const half* __restrict__ a, int64_t a_stride0,
    const half* __restrict__ b, int64_t b_stride0,
    const float* __restrict__ a_log,
    const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_selector, bool direct_state_slot,
    float* __restrict__ state,
    half* __restrict__ scratch, int state_indices_width,
    int64_t state_slot_stride, bool dt_bias_is_bfloat16, float scale) {
  const int value_head = blockIdx.x;
  const int chunk = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = query_start_loc[0];
  const int eos = query_start_loc[1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0 || num_tokens > kMaxTokens) return;

  const int selected = direct_state_slot ? 1 : state_selector[0];
  const int source_slot = direct_state_slot
                              ? state_indices[0]
                              : (selected > 0 && selected <= state_indices_width
                                     ? state_indices[selected - 1]
                                     : 0);
  if ((direct_state_slot && source_slot < 0) ||
      (!direct_state_slot && source_slot <= 0)) {
    for (int linear = tid; linear < num_tokens * kChunkV;
         linear += kThreads) {
      const int t = linear / kChunkV;
      const int value = chunk * kChunkV + linear % kChunkV;
      const int token = bos + t;
      scratch[(token * kValueHeads + value_head) * kDim + value] =
          __float2half(0.0f);
    }
    return;
  }

  const int key_head = value_head / (kValueHeads / kKeyHeads);
  const int conv_source_slot = state_indices[0];
  __shared__ float shared_q[kMaxTokens][kDim];
  __shared__ float shared_k[kMaxTokens][kDim];
  __shared__ half shared_v[kMaxTokens][kChunkV];
  __shared__ float shared_decay[kMaxTokens];
  __shared__ float shared_beta[kMaxTokens];

  if (warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * mixed_stride0;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      const int q_channel = key_head * kDim + dim;
      const int k_channel = kKeyHeads * kDim + key_head * kDim + dim;
      if (apply_conv) {
        q_values[i] = __half2float(conv4_channel_token(
            mixed_qkv, mixed_stride0, q_channel, t, conv_state,
            conv_slot_stride, conv_channel_stride, conv_tap_stride,
            conv_source_slot, selected, conv_weight, conv_bias,
            has_conv_bias));
        k_values[i] = __half2float(conv4_channel_token(
            mixed_qkv, mixed_stride0, k_channel, t, conv_state,
            conv_slot_stride, conv_channel_stride, conv_tap_stride,
            conv_source_slot, selected, conv_weight, conv_bias,
            has_conv_bias));
      } else {
        q_values[i] = __half2float(mixed_qkv[mixed_base + q_channel]);
        k_values[i] = __half2float(mixed_qkv[mixed_base + k_channel]);
      }
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const int v_channel = 2 * kKeyHeads * kDim + value_head * kDim +
                          chunk * kChunkV + lane;
    if (apply_conv) {
      shared_v[t][lane] = conv4_channel_token(
          mixed_qkv, mixed_stride0, v_channel, t, conv_state,
          conv_slot_stride, conv_channel_stride, conv_tap_stride,
          conv_source_slot, selected, conv_weight, conv_bias, has_conv_bias);
    } else {
      shared_v[t][lane] = mixed_qkv[mixed_base + v_channel];
    }
    const Sum2 sums = warp_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(
        0xffffffffu,
        lane == 0 ? rsqrtf(sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      shared_k[t][dim] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const float a_value =
          __half2float(a[static_cast<int64_t>(token) * a_stride0 + value_head]);
      const float b_value =
          __half2float(b[static_cast<int64_t>(token) * b_stride0 + value_head]);
      const float dt_bias_value =
          dt_bias_is_bfloat16
              ? __bfloat162float(
                    static_cast<const __nv_bfloat16*>(dt_bias)[value_head])
              : __half2float(static_cast<const half*>(dt_bias)[value_head]);
      const float g =
          -__expf(a_log[value_head]) *
          softplus_fast(a_value + dt_bias_value);
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    rows[row] = warp + row * kWarps;
  }
  const int64_t source_base =
      static_cast<int64_t>(source_slot) * state_slot_stride +
      static_cast<int64_t>(value_head) * kDim * kDim;
  float h[kRowsPerWarp][4];
#pragma unroll
  for (int row_i = 0; row_i < kRowsPerWarp; ++row_i) {
    const float4 value = *reinterpret_cast<const float4*>(
        state + source_base +
        (chunk * kChunkV + rows[row_i]) * kDim + k_base);
    h[row_i][0] = value.x;
    h[row_i][1] = value.y;
    h[row_i][2] = value.z;
    h[row_i][3] = value.w;
  }

  for (int t = 0; t < num_tokens; ++t) {
    const float4 q4 =
        *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
    const float4 k4 =
        *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
    const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
    const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};
    float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int row_i = 0; row_i < kRowsPerWarp; ++row_i) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        h[row_i][i] *= shared_decay[t];
        dot_hk[row_i] += h[row_i][i] * k_values[i];
      }
    }
    const Sum2 hk01 = warp_sum_pair(dot_hk[0], dot_hk[1]);
    const Sum2 hk23 = warp_sum_pair(dot_hk[2], dot_hk[3]);
    const float reduced_hk[kRowsPerWarp] = {hk01.x, hk01.y, hk23.x,
                                             hk23.y};
    float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int row_i = 0; row_i < kRowsPerWarp; ++row_i) {
      const float delta =
          (__half2float(shared_v[t][rows[row_i]]) - reduced_hk[row_i]) *
          shared_beta[t];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        h[row_i][i] += k_values[i] * delta;
        dot_hq[row_i] += h[row_i][i] * q_values[i];
      }
    }
    const Sum2 hq01 = warp_sum_pair(dot_hq[0], dot_hq[1]);
    const Sum2 hq23 = warp_sum_pair(dot_hq[2], dot_hq[3]);
    if (lane == 0) {
      const int token = bos + t;
      const int64_t scratch_base =
          (static_cast<int64_t>(token) * kValueHeads + value_head) * kDim +
          chunk * kChunkV;
      scratch[scratch_base + rows[0]] = __float2half(hq01.x);
      scratch[scratch_base + rows[1]] = __float2half(hq01.y);
      scratch[scratch_base + rows[2]] = __float2half(hq23.x);
      scratch[scratch_base + rows[3]] = __float2half(hq23.y);
    }

    const int destination_slot = state_indices[t];
    const bool destination_is_valid =
        direct_state_slot ? destination_slot >= 0 : destination_slot > 0;
    if (destination_is_valid) {
      const int64_t destination =
          static_cast<int64_t>(destination_slot) * state_slot_stride +
          static_cast<int64_t>(value_head) * kDim * kDim +
          (chunk * kChunkV + rows[0]) * kDim + k_base;
#pragma unroll
      for (int row_i = 0; row_i < kRowsPerWarp; ++row_i) {
        const int64_t row_destination =
            destination + static_cast<int64_t>(row_i) * kWarps * kDim;
        *reinterpret_cast<float4*>(state + row_destination) =
            make_float4(h[row_i][0], h[row_i][1], h[row_i][2],
                        h[row_i][3]);
      }
    }
  }
}

__global__ __launch_bounds__(kThreads, 2)
void fused_gdn_sm70_48block_norm_kernel(
    half* __restrict__ scratch, const int* __restrict__ query_start_loc,
    const half* __restrict__ output_gate, int64_t gate_stride0,
    int64_t gate_stride1,
    const void* __restrict__ norm_weight, bool norm_weight_is_float,
    float norm_eps) {
  const int value_head = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int bos = query_start_loc[0];
  const int num_tokens = query_start_loc[1] - bos;
  if (warp >= num_tokens || num_tokens > kMaxTokens) return;

  const int t = warp;
  const int token = bos + t;
  float output_values[4];
  float sum_square = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int value = lane + i * 32;
    const int64_t offset =
        (static_cast<int64_t>(token) * kValueHeads + value_head) * kDim +
        value;
    output_values[i] = __half2float(scratch[offset]);
    sum_square += output_values[i] * output_values[i];
  }
  sum_square = warp_sum(sum_square);
  const float rstd = rsqrtf(sum_square / static_cast<float>(kDim) + norm_eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int value = lane + i * 32;
    const int64_t offset =
        (static_cast<int64_t>(token) * kValueHeads + value_head) * kDim +
        value;
    const int64_t gate_offset = static_cast<int64_t>(token) * gate_stride0 +
                                value_head * gate_stride1 + value;
    const float gate = silu_fast(__half2float(output_gate[gate_offset]));
    const float weight = norm_weight_is_float
                             ? static_cast<const float*>(norm_weight)[value]
                             : __half2float(
                                   static_cast<const half*>(norm_weight)[value]);
    scratch[offset] =
        __float2half(output_values[i] * rstd * weight * gate);
  }
}

__global__ void fused_gdn_sm70_conv_state_commit_kernel(
    const half* __restrict__ raw_qkv, int64_t raw_stride0,
    half* __restrict__ conv_state, int64_t conv_slot_stride,
    int64_t conv_channel_stride, int64_t conv_tap_stride,
    const int* __restrict__ state_indices,
    const int* __restrict__ state_selector) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= kQkvDim) return;
  const int slot = state_indices[0];
  const int selected = state_selector[0];
  if (slot <= 0 || selected <= 0 || selected > kMaxTokens) return;
  const int offset = selected - 1;
  const int64_t base = static_cast<int64_t>(slot) * conv_slot_stride +
                       static_cast<int64_t>(channel) * conv_channel_stride;
  const half history0 = conv_state[base + (offset + 1) * conv_tap_stride];
  const half history1 = conv_state[base + (offset + 2) * conv_tap_stride];
  const half x0 = raw_qkv[channel];
  const half x1 = raw_qkv[raw_stride0 + channel];
  const half x2 = raw_qkv[2 * raw_stride0 + channel];
  const half x3 = raw_qkv[3 * raw_stride0 + channel];
  conv_state[base] = history0;
  conv_state[base + conv_tap_stride] = history1;
  conv_state[base + 2 * conv_tap_stride] = x0;
  conv_state[base + 3 * conv_tap_stride] = x1;
  conv_state[base + 4 * conv_tap_stride] = x2;
  conv_state[base + 5 * conv_tap_stride] = x3;
}

void fused_gdn_sm70(torch::Tensor mixed_qkv, torch::Tensor a,
                    torch::Tensor b, torch::Tensor a_log,
                    torch::Tensor dt_bias, torch::Tensor state_indices,
                    torch::Tensor query_start_loc,
                    torch::Tensor state_selector, torch::Tensor state,
                    torch::Tensor output_gate, torch::Tensor norm_weight,
                    torch::Tensor out, double scale, double norm_eps) {
  TORCH_CHECK(mixed_qkv.is_cuda() && mixed_qkv.scalar_type() == at::kHalf);
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kHalf);
  TORCH_CHECK(b.is_cuda() && b.scalar_type() == at::kHalf);
  TORCH_CHECK(a_log.is_cuda() && a_log.scalar_type() == at::kFloat);
  TORCH_CHECK(dt_bias.is_cuda() &&
              (dt_bias.scalar_type() == at::kHalf ||
               dt_bias.scalar_type() == at::kBFloat16));
  TORCH_CHECK(state_indices.is_cuda() && state_indices.scalar_type() == at::kInt);
  TORCH_CHECK(query_start_loc.is_cuda() &&
              query_start_loc.scalar_type() == at::kInt);
  TORCH_CHECK(state_selector.is_cuda() &&
              state_selector.scalar_type() == at::kInt);
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kFloat);
  TORCH_CHECK(output_gate.is_cuda() && output_gate.scalar_type() == at::kHalf);
  TORCH_CHECK(norm_weight.is_cuda() &&
              (norm_weight.scalar_type() == at::kHalf ||
               norm_weight.scalar_type() == at::kFloat));
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf);
  TORCH_CHECK(mixed_qkv.dim() == 2 && mixed_qkv.size(1) == 2560);
  TORCH_CHECK(mixed_qkv.size(0) >= 1 && mixed_qkv.size(0) <= kMaxTokens);
  TORCH_CHECK(a.sizes() == b.sizes() && a.size(1) == kValueHeads);
  TORCH_CHECK(state_indices.dim() == 2 && state_indices.size(0) == 1 &&
              state_indices.size(1) >= mixed_qkv.size(0));
  TORCH_CHECK(query_start_loc.numel() == 2 && state_selector.numel() == 1);
  TORCH_CHECK(state.dim() == 4 && state.size(1) == kValueHeads &&
              state.size(2) == kDim && state.size(3) == kDim);
  TORCH_CHECK(state.stride(1) == kDim * kDim &&
              state.stride(2) == kDim && state.stride(3) == 1);
  TORCH_CHECK(output_gate.sizes() == out.sizes() && out.dim() == 3 &&
              out.size(1) == kValueHeads && out.size(2) == kDim);
  TORCH_CHECK(norm_weight.numel() == kDim);
  TORCH_CHECK(mixed_qkv.is_contiguous() && a.is_contiguous() &&
              b.is_contiguous() && state_indices.is_contiguous() &&
              query_start_loc.is_contiguous() && state_selector.is_contiguous() &&
              output_gate.is_contiguous() &&
              norm_weight.is_contiguous() && out.is_contiguous());

  const at::cuda::OptionalCUDAGuard guard(at::device_of(mixed_qkv));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  fused_gdn_sm70_kernel<<<kValueHeads, kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(mixed_qkv.data_ptr()),
      reinterpret_cast<const half*>(a.data_ptr()),
      reinterpret_cast<const half*>(b.data_ptr()),
      a_log.data_ptr<float>(),
      dt_bias.data_ptr(),
      state_indices.data_ptr<int>(), query_start_loc.data_ptr<int>(),
      state_selector.data_ptr<int>(), state.data_ptr<float>(),
      reinterpret_cast<const half*>(output_gate.data_ptr()),
      norm_weight.data_ptr(), reinterpret_cast<half*>(out.data_ptr()),
      state_indices.size(1), state.stride(0),
      dt_bias.scalar_type() == at::kBFloat16,
      norm_weight.scalar_type() == at::kFloat,
      static_cast<float>(scale), static_cast<float>(norm_eps));
  AT_CUDA_CHECK(cudaGetLastError());
}

void fused_gdn_sm70_48block(
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
    torch::Tensor a_log, torch::Tensor dt_bias, torch::Tensor state_indices,
    torch::Tensor query_start_loc, torch::Tensor state_selector,
    torch::Tensor state, torch::Tensor output_gate,
    torch::Tensor norm_weight, torch::Tensor out, double scale,
    double norm_eps) {
  TORCH_CHECK(mixed_qkv.is_cuda() && mixed_qkv.scalar_type() == at::kHalf);
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kHalf);
  TORCH_CHECK(b.is_cuda() && b.scalar_type() == at::kHalf);
  TORCH_CHECK(a_log.is_cuda() && a_log.scalar_type() == at::kFloat);
  TORCH_CHECK(dt_bias.is_cuda() &&
              (dt_bias.scalar_type() == at::kHalf ||
               dt_bias.scalar_type() == at::kBFloat16));
  TORCH_CHECK(state_indices.is_cuda() &&
              state_indices.scalar_type() == at::kInt);
  TORCH_CHECK(query_start_loc.is_cuda() &&
              query_start_loc.scalar_type() == at::kInt);
  TORCH_CHECK(state_selector.is_cuda() &&
              state_selector.scalar_type() == at::kInt);
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kFloat);
  TORCH_CHECK(output_gate.is_cuda() && output_gate.scalar_type() == at::kHalf);
  TORCH_CHECK(norm_weight.is_cuda() &&
              (norm_weight.scalar_type() == at::kHalf ||
               norm_weight.scalar_type() == at::kFloat));
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf);
  TORCH_CHECK(mixed_qkv.dim() == 2 && mixed_qkv.size(1) == 2560);
  TORCH_CHECK(mixed_qkv.size(0) >= 1 && mixed_qkv.size(0) <= kMaxTokens);
  TORCH_CHECK(a.sizes() == b.sizes() && a.size(1) == kValueHeads);
  TORCH_CHECK(state_indices.dim() == 2 && state_indices.size(0) == 1 &&
              state_indices.size(1) >= mixed_qkv.size(0));
  TORCH_CHECK(query_start_loc.numel() == 2 && state_selector.numel() == 1);
  TORCH_CHECK(state.dim() == 4 && state.size(1) == kValueHeads &&
              state.size(2) == kDim && state.size(3) == kDim);
  TORCH_CHECK(state.stride(1) == kDim * kDim &&
              state.stride(2) == kDim && state.stride(3) == 1);
  TORCH_CHECK(output_gate.sizes() == out.sizes() && out.dim() == 3 &&
              out.size(1) == kValueHeads && out.size(2) == kDim);
  TORCH_CHECK(norm_weight.numel() == kDim);
  TORCH_CHECK(mixed_qkv.is_contiguous() && a.is_contiguous() &&
              b.is_contiguous() && state_indices.is_contiguous() &&
              query_start_loc.is_contiguous() &&
              state_selector.is_contiguous() && output_gate.is_contiguous() &&
              norm_weight.is_contiguous() && out.is_contiguous());

  const at::cuda::OptionalCUDAGuard guard(at::device_of(mixed_qkv));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const dim3 recurrence_grid(kValueHeads, kDim / kChunkV);
  fused_gdn_sm70_48block_recurrence_kernel
      <<<recurrence_grid, kThreads, 0, stream>>>(
          reinterpret_cast<const half*>(mixed_qkv.data_ptr()),
          mixed_qkv.stride(0), nullptr, 0, 0, 0, nullptr, nullptr, false,
          false,
          reinterpret_cast<const half*>(a.data_ptr()), a.stride(0),
          reinterpret_cast<const half*>(b.data_ptr()), b.stride(0),
          a_log.data_ptr<float>(),
          dt_bias.data_ptr(), state_indices.data_ptr<int>(),
          query_start_loc.data_ptr<int>(), state_selector.data_ptr<int>(),
          false,
          state.data_ptr<float>(), reinterpret_cast<half*>(out.data_ptr()),
          state_indices.size(1), state.stride(0),
          dt_bias.scalar_type() == at::kBFloat16,
          static_cast<float>(scale));
  fused_gdn_sm70_48block_norm_kernel<<<kValueHeads, kThreads, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      query_start_loc.data_ptr<int>(),
      reinterpret_cast<const half*>(output_gate.data_ptr()),
      output_gate.stride(0), output_gate.stride(1),
      norm_weight.data_ptr(), norm_weight.scalar_type() == at::kFloat,
      static_cast<float>(norm_eps));
  AT_CUDA_CHECK(cudaGetLastError());
}

void fused_gdn_sm70_48block_full(
    torch::Tensor raw_qkv, torch::Tensor conv_state,
    torch::Tensor conv_weight, torch::Tensor conv_bias, torch::Tensor a,
    torch::Tensor b, torch::Tensor a_log, torch::Tensor dt_bias,
    torch::Tensor state_indices, torch::Tensor query_start_loc,
    torch::Tensor state_selector, torch::Tensor state,
    torch::Tensor output_gate, torch::Tensor norm_weight, torch::Tensor out,
    double scale, double norm_eps) {
  TORCH_CHECK(raw_qkv.is_cuda() && raw_qkv.scalar_type() == at::kHalf);
  TORCH_CHECK(raw_qkv.dim() == 2 && raw_qkv.size(0) == kMaxTokens &&
              raw_qkv.size(1) == kQkvDim && raw_qkv.stride(1) == 1);
  TORCH_CHECK(conv_state.is_cuda() &&
              conv_state.scalar_type() == at::kHalf &&
              conv_state.dim() == 3 && conv_state.size(1) == kQkvDim &&
              conv_state.size(2) == kConvStateLen &&
              conv_state.stride(1) == 1);
  TORCH_CHECK(conv_weight.is_cuda() &&
              conv_weight.scalar_type() == at::kHalf &&
              conv_weight.sizes() == torch::IntArrayRef({kQkvDim, kConvWidth}) &&
              conv_weight.is_contiguous());
  TORCH_CHECK(conv_bias.is_cuda() &&
              conv_bias.scalar_type() == at::kHalf &&
              (conv_bias.numel() == 1 || conv_bias.numel() == kQkvDim) &&
              conv_bias.is_contiguous());
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kHalf &&
              a.sizes() == torch::IntArrayRef({kMaxTokens, kValueHeads}) &&
              a.stride(1) == 1);
  TORCH_CHECK(b.is_cuda() && b.scalar_type() == at::kHalf &&
              b.sizes() == a.sizes() && b.stride(1) == 1);
  TORCH_CHECK(a_log.is_cuda() && a_log.scalar_type() == at::kFloat &&
              a_log.numel() == kValueHeads && a_log.is_contiguous());
  TORCH_CHECK(dt_bias.is_cuda() &&
              (dt_bias.scalar_type() == at::kHalf ||
               dt_bias.scalar_type() == at::kBFloat16) &&
              dt_bias.numel() == kValueHeads && dt_bias.is_contiguous());
  TORCH_CHECK(state_indices.is_cuda() &&
              state_indices.scalar_type() == at::kInt &&
              state_indices.dim() == 2 && state_indices.size(0) == 1 &&
              state_indices.size(1) >= kMaxTokens &&
              state_indices.is_contiguous());
  TORCH_CHECK(query_start_loc.is_cuda() &&
              query_start_loc.scalar_type() == at::kInt &&
              query_start_loc.numel() == 2 && query_start_loc.is_contiguous());
  TORCH_CHECK(state_selector.is_cuda() &&
              state_selector.scalar_type() == at::kInt &&
              state_selector.numel() == 1 && state_selector.is_contiguous());
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kFloat &&
              state.dim() == 4 && state.size(1) == kValueHeads &&
              state.size(2) == kDim && state.size(3) == kDim &&
              state.stride(1) == kDim * kDim &&
              state.stride(2) == kDim && state.stride(3) == 1);
  TORCH_CHECK(output_gate.is_cuda() &&
              output_gate.scalar_type() == at::kHalf &&
              output_gate.sizes() == torch::IntArrayRef(
                  {kMaxTokens, kValueHeads, kDim}) &&
              output_gate.stride(2) == 1 &&
              output_gate.stride(1) >= kDim);
  TORCH_CHECK(norm_weight.is_cuda() &&
              (norm_weight.scalar_type() == at::kHalf ||
               norm_weight.scalar_type() == at::kFloat) &&
              norm_weight.numel() == kDim && norm_weight.is_contiguous());
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf &&
              out.sizes() == output_gate.sizes() && out.is_contiguous());

  const at::cuda::OptionalCUDAGuard guard(at::device_of(raw_qkv));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int kPrepThreads = 256;
  fused_gdn_sm70_conv_prep_inplace_kernel
      <<<(kQkvDim + kPrepThreads - 1) / kPrepThreads, kPrepThreads, 0,
         stream>>>(reinterpret_cast<half*>(raw_qkv.data_ptr()),
                   raw_qkv.stride(0),
                   reinterpret_cast<half*>(conv_state.data_ptr()),
                   conv_state.stride(0), conv_state.stride(1),
                   conv_state.stride(2),
                   reinterpret_cast<const half*>(conv_weight.data_ptr()),
                   reinterpret_cast<const half*>(conv_bias.data_ptr()),
                   conv_bias.numel() == kQkvDim,
                   state_indices.data_ptr<int>(),
                   state_selector.data_ptr<int>());
  const dim3 recurrence_grid(kValueHeads, kDim / kChunkV);
  fused_gdn_sm70_48block_recurrence_kernel
      <<<recurrence_grid, kThreads, 0, stream>>>(
          reinterpret_cast<const half*>(raw_qkv.data_ptr()),
          raw_qkv.stride(0), nullptr, 0, 0, 0, nullptr, nullptr, false,
          false,
          reinterpret_cast<const half*>(a.data_ptr()), a.stride(0),
          reinterpret_cast<const half*>(b.data_ptr()), b.stride(0),
          a_log.data_ptr<float>(),
          dt_bias.data_ptr(), state_indices.data_ptr<int>(),
          query_start_loc.data_ptr<int>(), state_selector.data_ptr<int>(),
          false,
          state.data_ptr<float>(), reinterpret_cast<half*>(out.data_ptr()),
          state_indices.size(1), state.stride(0),
          dt_bias.scalar_type() == at::kBFloat16,
          static_cast<float>(scale));
  fused_gdn_sm70_48block_norm_kernel<<<kValueHeads, kThreads, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      query_start_loc.data_ptr<int>(),
      reinterpret_cast<const half*>(output_gate.data_ptr()),
      output_gate.stride(0), output_gate.stride(1),
      norm_weight.data_ptr(), norm_weight.scalar_type() == at::kFloat,
      static_cast<float>(norm_eps));
  AT_CUDA_CHECK(cudaGetLastError());
}

void fused_gdn_sm70_48block_full_q1(
    torch::Tensor raw_qkv, torch::Tensor conv_state,
    torch::Tensor conv_weight, torch::Tensor conv_bias, torch::Tensor a,
    torch::Tensor b, torch::Tensor a_log, torch::Tensor dt_bias,
    torch::Tensor state_indices, torch::Tensor query_start_loc,
    torch::Tensor state, torch::Tensor output_gate,
    torch::Tensor norm_weight, torch::Tensor out, double scale,
    double norm_eps) {
  TORCH_CHECK(raw_qkv.is_cuda() && raw_qkv.scalar_type() == at::kHalf);
  TORCH_CHECK(raw_qkv.dim() == 2 && raw_qkv.size(0) == 1 &&
              raw_qkv.size(1) == kQkvDim && raw_qkv.stride(1) == 1);
  TORCH_CHECK(conv_state.is_cuda() &&
              conv_state.scalar_type() == at::kHalf &&
              conv_state.dim() == 3 && conv_state.size(1) == kQkvDim &&
              conv_state.size(2) >= kConvWidth - 1 &&
              conv_state.stride(1) == 1);
  TORCH_CHECK(conv_weight.is_cuda() &&
              conv_weight.scalar_type() == at::kHalf &&
              conv_weight.sizes() == torch::IntArrayRef({kQkvDim, kConvWidth}) &&
              conv_weight.is_contiguous());
  TORCH_CHECK(conv_bias.is_cuda() &&
              conv_bias.scalar_type() == at::kHalf &&
              (conv_bias.numel() == 1 || conv_bias.numel() == kQkvDim) &&
              conv_bias.is_contiguous());
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kHalf &&
              a.sizes() == torch::IntArrayRef({1, kValueHeads}) &&
              a.stride(1) == 1);
  TORCH_CHECK(b.is_cuda() && b.scalar_type() == at::kHalf &&
              b.sizes() == a.sizes() && b.stride(1) == 1);
  TORCH_CHECK(a_log.is_cuda() && a_log.scalar_type() == at::kFloat &&
              a_log.numel() == kValueHeads && a_log.is_contiguous());
  TORCH_CHECK(dt_bias.is_cuda() &&
              (dt_bias.scalar_type() == at::kHalf ||
               dt_bias.scalar_type() == at::kBFloat16) &&
              dt_bias.numel() == kValueHeads && dt_bias.is_contiguous());
  TORCH_CHECK(state_indices.is_cuda() &&
              state_indices.scalar_type() == at::kInt &&
              state_indices.dim() == 1 && state_indices.numel() >= 1 &&
              state_indices.is_contiguous());
  TORCH_CHECK(query_start_loc.is_cuda() &&
              query_start_loc.scalar_type() == at::kInt &&
              query_start_loc.numel() == 2 && query_start_loc.is_contiguous());
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kFloat &&
              state.dim() == 4 && state.size(1) == kValueHeads &&
              state.size(2) == kDim && state.size(3) == kDim &&
              state.stride(1) == kDim * kDim &&
              state.stride(2) == kDim && state.stride(3) == 1);
  TORCH_CHECK(output_gate.is_cuda() &&
              output_gate.scalar_type() == at::kHalf &&
              output_gate.sizes() ==
                  torch::IntArrayRef({1, kValueHeads, kDim}) &&
              output_gate.stride(2) == 1 && output_gate.stride(1) >= kDim);
  TORCH_CHECK(norm_weight.is_cuda() &&
              (norm_weight.scalar_type() == at::kHalf ||
               norm_weight.scalar_type() == at::kFloat) &&
              norm_weight.numel() == kDim && norm_weight.is_contiguous());
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf &&
              out.sizes() == output_gate.sizes() && out.is_contiguous());

  const at::cuda::OptionalCUDAGuard guard(at::device_of(raw_qkv));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int kPrepThreads = 256;
  fused_gdn_sm70_conv_prep_q1_inplace_kernel
      <<<(kQkvDim + kPrepThreads - 1) / kPrepThreads, kPrepThreads, 0,
         stream>>>(reinterpret_cast<half*>(raw_qkv.data_ptr()),
                   reinterpret_cast<half*>(conv_state.data_ptr()),
                   conv_state.stride(0), conv_state.stride(1),
                   conv_state.stride(2),
                   reinterpret_cast<const half*>(conv_weight.data_ptr()),
                   reinterpret_cast<const half*>(conv_bias.data_ptr()),
                   conv_bias.numel() == kQkvDim,
                   state_indices.data_ptr<int>());
  const dim3 recurrence_grid(kValueHeads, kDim / kChunkV);
  fused_gdn_sm70_48block_recurrence_kernel
      <<<recurrence_grid, kThreads, 0, stream>>>(
          reinterpret_cast<const half*>(raw_qkv.data_ptr()),
          raw_qkv.stride(0), nullptr, 0, 0, 0, nullptr, nullptr, false,
          false,
          reinterpret_cast<const half*>(a.data_ptr()), a.stride(0),
          reinterpret_cast<const half*>(b.data_ptr()), b.stride(0),
          a_log.data_ptr<float>(), dt_bias.data_ptr(),
          state_indices.data_ptr<int>(), query_start_loc.data_ptr<int>(),
          nullptr, true, state.data_ptr<float>(),
          reinterpret_cast<half*>(out.data_ptr()), state_indices.numel(),
          state.stride(0), dt_bias.scalar_type() == at::kBFloat16,
          static_cast<float>(scale));
  fused_gdn_sm70_48block_norm_kernel<<<kValueHeads, kThreads, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      query_start_loc.data_ptr<int>(),
      reinterpret_cast<const half*>(output_gate.data_ptr()),
      output_gate.stride(0), output_gate.stride(1), norm_weight.data_ptr(),
      norm_weight.scalar_type() == at::kFloat,
      static_cast<float>(norm_eps));
  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

TORCH_LIBRARY(qwen38_u1, m) {
  m.def(
      "fused_gdn_sm70(Tensor mixed_qkv, Tensor a, Tensor b, Tensor a_log, "
      "Tensor dt_bias, Tensor state_indices, Tensor query_start_loc, "
      "Tensor state_selector, Tensor! state, Tensor output_gate, "
      "Tensor norm_weight, Tensor! out, float scale, float norm_eps) -> ()");
  m.def(
      "fused_gdn_sm70_48block(Tensor mixed_qkv, Tensor a, Tensor b, "
      "Tensor a_log, Tensor dt_bias, Tensor state_indices, "
      "Tensor query_start_loc, Tensor state_selector, Tensor! state, "
      "Tensor output_gate, Tensor norm_weight, Tensor! out, float scale, "
      "float norm_eps) -> ()");
  m.def(
      "fused_gdn_sm70_48block_full(Tensor! raw_qkv, Tensor! conv_state, "
      "Tensor conv_weight, Tensor conv_bias, Tensor a, Tensor b, "
      "Tensor a_log, Tensor dt_bias, Tensor state_indices, "
      "Tensor query_start_loc, Tensor state_selector, Tensor! state, "
      "Tensor output_gate, Tensor norm_weight, Tensor! out, float scale, "
      "float norm_eps) -> ()");
  m.def(
      "fused_gdn_sm70_48block_full_q1(Tensor! raw_qkv, Tensor! conv_state, "
      "Tensor conv_weight, Tensor conv_bias, Tensor a, Tensor b, "
      "Tensor a_log, Tensor dt_bias, Tensor state_indices, "
      "Tensor query_start_loc, Tensor! state, Tensor output_gate, "
      "Tensor norm_weight, Tensor! out, float scale, float norm_eps) -> ()");
}

TORCH_LIBRARY_IMPL(qwen38_u1, CUDA, m) {
  m.impl("fused_gdn_sm70", &fused_gdn_sm70);
  m.impl("fused_gdn_sm70_48block", &fused_gdn_sm70_48block);
  m.impl("fused_gdn_sm70_48block_full", &fused_gdn_sm70_48block_full);
  m.impl("fused_gdn_sm70_48block_full_q1",
         &fused_gdn_sm70_48block_full_q1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
