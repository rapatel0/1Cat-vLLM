// Fused Prism Q2_0 x FP16 GEMM for Volta Tensor Cores.
//
// The lane mappings mirror TurboMind's SM70 HMMA 8x8x4 atom. Each warp
// computes an 8x32 output atom. Packed Q2 bytes are expanded in registers
// directly into the B operand fragment, so no dense FP16 weight tensor is
// materialized in global or shared memory.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#if !defined(USE_ROCM)

namespace vllm::gguf::q2_0_sm70 {

constexpr int kMmaM = 8;
constexpr int kMmaN = 32;
constexpr int kMmaK = 8;
constexpr int kWarpThreads = 32;
constexpr int kDecodeTileN = kMmaN;
constexpr int kDecodeSplitWarps = 4;
constexpr int kDecodeThreads = kDecodeSplitWarps * kWarpThreads;
constexpr int kDecodeAccumulatorGroups = 4;
constexpr int kPrefillWarps = 4;
constexpr int kPrefillThreads = kPrefillWarps * kWarpThreads;
constexpr int kPrefillTileN = kPrefillWarps * kMmaN;
constexpr int kShortPrefillTileM = 16;
constexpr int kLongPrefillTileM = 64;
constexpr int kLongPrefillThreshold = 128;

__device__ __forceinline__ void mma_m8n8k4_row_col_acc(float* c, const half* a,
                                                       const half* b) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  const uint32_t* a_regs = reinterpret_cast<const uint32_t*>(a);
  const uint32_t* b_regs = reinterpret_cast<const uint32_t*>(b);
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7},"
      "{%8, %9},"
      "{%10, %11},"
      "{%0, %1, %2, %3, %4, %5, %6, %7};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
        "+f"(c[6]), "+f"(c[7])
      : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_regs[0]), "r"(b_regs[1]));
  #endif
}

// Expand two packed bytes (eight Q2 values) and apply the block's FP16 scale.
// Code values map to {-1, 0, +1, +2}; code 3 is reserved by the ternary
// format, but decoding it as +2 matches the scalar and DP4A implementations.
__device__ __forceinline__ void expand_q2_0_mma_fragment(
    uint16_t packed, half scale, half (&fragment)[kMmaK]) {
  #pragma unroll
  for (int i = 0; i < kMmaK; ++i) {
    const int code = static_cast<int>((packed >> (2 * i)) & 0x3u);
    fragment[i] = __hmul(__int2half_rn(code - 1), scale);
  }
}

__device__ __forceinline__ void load_q2_0_mma_fragment(
    const uint8_t* weights, int64_t row_stride_bytes, int row, int k,
    half (&fragment)[kMmaK]) {
  const uint8_t* row_base =
      weights + static_cast<int64_t>(row) * row_stride_bytes;
  const block_q2_0* block =
      reinterpret_cast<const block_q2_0*>(row_base) + k / QK2_0;
  const int byte_index = (k % QK2_0) / 4;
  const uint16_t packed =
      *reinterpret_cast<const uint16_t*>(block->qs + byte_index);
  expand_q2_0_mma_fragment(packed, block->d, fragment);
}

template <int kAtomsM>
__device__ __forceinline__ void store_output_atoms(
    half* output, float (&accumulators)[kAtomsM][8], int tile_m, int tile_n,
    int warp_n, int lane, int m, int n) {
  const int lane_m = (lane & 1) + (lane / 16) * 4;
  const int lane_n = (lane & 2) + (lane & 12) * 2;

  #pragma unroll
  for (int atom_m = 0; atom_m < kAtomsM; ++atom_m) {
  #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const int global_m = tile_m + atom_m * kMmaM + lane_m + (pair & 1) * 2;
      const int global_n = tile_n + warp_n + lane_n + (pair / 2) * 4;
      if (global_m < m && global_n < n) {
        output[static_cast<int64_t>(global_m) * n + global_n] =
            __float2half_rn(accumulators[atom_m][2 * pair]);
      }
      if (global_m < m && global_n + 1 < n) {
        output[static_cast<int64_t>(global_m) * n + global_n + 1] =
            __float2half_rn(accumulators[atom_m][2 * pair + 1]);
      }
    }
  }
}

// Separate decode kernel for the four- and eight-slot serving shapes. The
// warps split K for the same output tile and reduce once after accumulation.
// The inactive rows in the batch-4 case are zero-padded inside the MMA atom.
__launch_bounds__(kDecodeThreads, 2) __global__
    void q2_0_mma_decode_kernel(const uint8_t* __restrict__ weights,
                                int64_t row_stride_bytes,
                                const half* __restrict__ input,
                                half* __restrict__ output, int m, int n,
                                int k) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  const int tid = threadIdx.x;
  const int split_warp = tid / kWarpThreads;
  const int lane = tid & (kWarpThreads - 1);
  const int tile_n = blockIdx.x * kDecodeTileN;
  const int weight_row =
      tile_n + (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
  const int activation_row = (lane / 16) * 4 + (lane % 4);

  const block_q2_0* weight_blocks =
      weight_row < n
          ? reinterpret_cast<const block_q2_0*>(
                weights + static_cast<int64_t>(weight_row) * row_stride_bytes)
          : nullptr;
  float accumulators[kDecodeAccumulatorGroups][8] = {};

  for (int block_k = split_warp; block_k < k / QK2_0;
       block_k += kDecodeSplitWarps) {
    const half scale = weight_blocks != nullptr ? weight_blocks[block_k].d
                                                : __float2half(0.0f);
    const uint8_t* packed_values =
        weight_blocks != nullptr ? weight_blocks[block_k].qs : nullptr;

    for (int block_offset = 0; block_offset < QK2_0;
         block_offset += kDecodeAccumulatorGroups * kMmaK) {
    #pragma unroll
      for (int group = 0; group < kDecodeAccumulatorGroups; ++group) {
        const int k0 = block_k * QK2_0 + block_offset + group * kMmaK;
        __align__(16) half a_fragment[kMmaK];
        if (activation_row < m) {
          *reinterpret_cast<uint4*>(a_fragment) =
              *reinterpret_cast<const uint4*>(
                  input + static_cast<int64_t>(activation_row) * k + k0);
        } else {
    #pragma unroll
          for (int i = 0; i < kMmaK; ++i) {
            a_fragment[i] = __float2half(0.0f);
          }
        }

        __align__(16) half b_fragment[kMmaK];
        const uint16_t packed =
            packed_values != nullptr
                ? *reinterpret_cast<const uint16_t*>(
                      packed_values + (block_offset + group * kMmaK) / 4)
                : 0;
        expand_q2_0_mma_fragment(packed, scale, b_fragment);

        mma_m8n8k4_row_col_acc(accumulators[group], a_fragment, b_fragment);
        mma_m8n8k4_row_col_acc(accumulators[group], a_fragment + 4,
                               b_fragment + 4);
      }
    }
  }

  __shared__ float partials[kDecodeSplitWarps][8][kWarpThreads];
    #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float partial = accumulators[0][i];
    #pragma unroll
    for (int group = 1; group < kDecodeAccumulatorGroups; ++group) {
      partial += accumulators[group][i];
    }
    partials[split_warp][i][lane] = partial;
  }
  __syncthreads();

  if (split_warp != 0) {
    return;
  }
  float final_accumulators[1][8];
    #pragma unroll
  for (int i = 0; i < 8; ++i) {
    final_accumulators[0][i] = partials[0][i][lane];
    #pragma unroll
    for (int warp = 1; warp < kDecodeSplitWarps; ++warp) {
      final_accumulators[0][i] += partials[warp][i][lane];
    }
  }
  store_output_atoms(output, final_accumulators, 0, tile_n, 0, lane, m, n);
  #endif
}

// Wide-token prefill kernel. Four warps share each activation tile while each
// warp expands its own 32 output rows directly into MMA fragments. A 16-row
// tile supplies enough blocks for short prefills; a 64-row tile amortizes
// dequantization for long prompts.
template <int kTileM>
__launch_bounds__(kPrefillThreads, 1) __global__
    void q2_0_mma_prefill_kernel(const uint8_t* __restrict__ weights,
                                 int64_t row_stride_bytes,
                                 const half* __restrict__ input,
                                 half* __restrict__ output, int m, int n,
                                 int k) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid & 31;
  const int tile_m = blockIdx.y * kTileM;
  const int tile_n = blockIdx.x * kPrefillTileN;
  const int warp_n = warp * kMmaN;
  const int weight_row =
      tile_n + warp_n + (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
  const int activation_row = (lane / 16) * 4 + (lane % 4);
  const block_q2_0* weight_blocks =
      weight_row < n
          ? reinterpret_cast<const block_q2_0*>(
                weights + static_cast<int64_t>(weight_row) * row_stride_bytes)
          : nullptr;

  __shared__ __align__(16) half input_tile[kTileM * kMmaK];
  float accumulators[kTileM / kMmaM][8] = {};

  for (int block_k = 0; block_k < k / QK2_0; ++block_k) {
    const half scale = weight_blocks != nullptr ? weight_blocks[block_k].d
                                                : __float2half(0.0f);
    const uint8_t* packed_values =
        weight_blocks != nullptr ? weight_blocks[block_k].qs : nullptr;

    for (int block_offset = 0; block_offset < QK2_0; block_offset += kMmaK) {
      const int k0 = block_k * QK2_0 + block_offset;
      for (int row = tid; row < kTileM; row += kPrefillThreads) {
        const int global_m = tile_m + row;
        uint4 values = {};
        if (global_m < m) {
          values = *reinterpret_cast<const uint4*>(
              input + static_cast<int64_t>(global_m) * k + k0);
        }
        *reinterpret_cast<uint4*>(input_tile + row * kMmaK) = values;
      }
      __syncthreads();

      __align__(16) half b_fragment[kMmaK];
      const uint16_t packed = packed_values != nullptr
                                  ? *reinterpret_cast<const uint16_t*>(
                                        packed_values + block_offset / 4)
                                  : 0;
      expand_q2_0_mma_fragment(packed, scale, b_fragment);

    #pragma unroll
      for (int atom_m = 0; atom_m < kTileM / kMmaM; ++atom_m) {
        __align__(16) half a_fragment[kMmaK];
        const half* source =
            input_tile + (atom_m * kMmaM + activation_row) * kMmaK;
        *reinterpret_cast<uint4*>(a_fragment) =
            *reinterpret_cast<const uint4*>(source);
        mma_m8n8k4_row_col_acc(accumulators[atom_m], a_fragment, b_fragment);
        mma_m8n8k4_row_col_acc(accumulators[atom_m], a_fragment + 4,
                               b_fragment + 4);
      }
      __syncthreads();
    }
  }

  store_output_atoms(output, accumulators, tile_m, tile_n, warp_n, lane, m, n);
  #endif
}

inline void mul_mat_q2_0_sm70_cuda(const void* weights,
                                   int64_t row_stride_bytes, const half* input,
                                   half* output, int m, int n, int k,
                                   cudaStream_t stream) {
  if (m == 4 || m == 8) {
    const dim3 block(kDecodeThreads);
    const dim3 grid((n + kDecodeTileN - 1) / kDecodeTileN);
    q2_0_mma_decode_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(weights), row_stride_bytes, input, output,
        m, n, k);
  } else {
    const dim3 block(kPrefillThreads);
    if (m < kLongPrefillThreshold) {
      const dim3 grid((n + kPrefillTileN - 1) / kPrefillTileN,
                      (m + kShortPrefillTileM - 1) / kShortPrefillTileM);
      q2_0_mma_prefill_kernel<kShortPrefillTileM><<<grid, block, 0, stream>>>(
          static_cast<const uint8_t*>(weights), row_stride_bytes, input, output,
          m, n, k);
    } else {
      const dim3 grid((n + kPrefillTileN - 1) / kPrefillTileN,
                      (m + kLongPrefillTileM - 1) / kLongPrefillTileM);
      q2_0_mma_prefill_kernel<kLongPrefillTileM><<<grid, block, 0, stream>>>(
          static_cast<const uint8_t*>(weights), row_stride_bytes, input, output,
          m, n, k);
    }
  }
}

}  // namespace vllm::gguf::q2_0_sm70

#endif  // !defined(USE_ROCM)
