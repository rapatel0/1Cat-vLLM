#include "core/registration.h"

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/mma_sm70.h"
#include "src/turbomind/kernels/gemm/arch/operand_sm70_s884.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/iterator_sm70.h"
#include "src/turbomind/kernels/gemm/mainloop_sm70.h"
#include "src/turbomind/kernels/gemm/scheduler_sm70.cuh"
#include "src/turbomind/kernels/gemm/thread_group_map.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "custom_all_reduce.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cstdint>
#include <tuple>
#include <type_traits>

namespace {

namespace wmma = nvcuda::wmma;
namespace cg = cooperative_groups;
namespace tm = turbomind::gemm;

constexpr int kTileM = 16;
constexpr int kTileN = 128;
constexpr int kTileK = 16;
constexpr int kHadamard = 128;
constexpr int kWarps = 8;
constexpr int kWarpSize = 32;
constexpr float kHadamardScale = 0.088388347648f;

#ifndef EXL3_SM70_K_SPLITS
#define EXL3_SM70_K_SPLITS 8
#endif
#ifndef EXL3_SM70_DIRECT_REG_FRAGMENTS
#define EXL3_SM70_DIRECT_REG_FRAGMENTS 0
#endif

// K4/K5 lane-state records already round up to four 32-bit words.  Keeping
// each independent state chain in its own word removes cross-word extracts
// without increasing the repacked weight footprint or changing the warp's
// word-plane-major global/shared-memory accesses.  Leave this opt-in until the
// real-shape V100 gate proves the instruction reduction translates to latency.
#ifndef EXL3_TM_K45_ALIGNED_STATE
#define EXL3_TM_K45_ALIGNED_STATE 0
#endif
#ifndef EXL3_TM_B_SHARED
// The compact shared-B experiment is retained for isolated A/B tests, but it
// raises the K5/K6 kernel from 96 to 128 registers and loses badly on every
// production Qwen3.8 shape.  The measured default is the just-in-time
// register decode; callers may still compile with EXL3_TM_B_SHARED=1.
#define EXL3_TM_B_SHARED 0
#endif

// Decode the shared compact trellis from the destination MMA lane instead of
// decoding both N16 source tiles and gathering fragments with warp shuffles.
// This is useful only with the raw-B shared pipeline and remains independently
// gated until its register count and real-shape latency are qualified on V100.
#ifndef EXL3_TM_DESTINATION_SHARED_DECODE
#define EXL3_TM_DESTINATION_SHARED_DECODE 0
#endif

// Split destination-local raw-trellis reconstruction around the independent
// current HMMA.  The compact shared words are reloaded for the high K8 half,
// trading cheap shared reads for shorter B-fragment and decode-temporary live
// ranges.  Keep it separate from the destination mapping gate for attribution.
#ifndef EXL3_TM_INTERLEAVE_DESTINATION_DECODE
#define EXL3_TM_INTERLEAVE_DESTINATION_DECODE 0
#endif

// The compact raw trellis is small enough that Volta can copy it directly into
// the inactive shared stage.  Avoiding a full register-resident B prefetch
// lowers the 128-register destination-decoder kernel toward the INT8 path's
// occupancy.  This deliberately trades some load overlap for residency and is
// independently gated for real-shape measurement.
#ifndef EXL3_TM_RAW_DIRECT_TO_SHARED
#define EXL3_TM_RAW_DIRECT_TO_SHARED 0
#endif

// The direct-to-shared raw path must retain some global-memory-level
// parallelism.  Loading and immediately storing one word at a time drops the
// raw K5 kernel below half of the register-prefetch throughput on V100.  Batch
// a small compile-time number of independent coalesced planes before their
// shared stores; two is the first residency-preserving experiment.
#ifndef EXL3_TM_RAW_DIRECT_GROUP
#define EXL3_TM_RAW_DIRECT_GROUP 1
#endif

// Pipeline the two five-word halves of a K5 raw prefetch across the current
// K64 MMA tile.  Unlike the direct-to-shared experiment, this preserves the
// load/compute overlap; unlike the original TurboMind-style prefetch, it never
// keeps all ten compressed words live at once.
#ifndef EXL3_TM_RAW_STAGGERED_K5_PREFETCH
#define EXL3_TM_RAW_STAGGERED_K5_PREFETCH 0
#endif
#ifndef EXL3_TM_STATE_LUT
// Experimental exact MCG decode path.  The 128 KiB table is initialized on
// the active device during the one-time state repack, then read through the
// read-only cache by the GEMM.  Set to zero to retain procedural 3INST decode.
#define EXL3_TM_STATE_LUT 0
#endif
#ifndef EXL3_TM_STATE_SHARED
// The qualified state layout stages each coalesced K64xN128 tile through
// shared memory so global fetch overlaps the current HMMA tile.  Keep a
// direct-global variant available for isolated occupancy/traffic A/B tests.
#define EXL3_TM_STATE_SHARED 1
#endif
#ifndef EXL3_TM_K6_DENSE_STATE
// Experimental K6-only layout: 32 lane records x 136 useful bits are packed
// into 136 uint32 words instead of five padded 32-word planes (160 words).
#define EXL3_TM_K6_DENSE_STATE 0
#endif
constexpr int kKSplits = EXL3_SM70_K_SPLITS;
static_assert(kKSplits == 4 || kKSplits == 8 || kKSplits == 16,
              "EXL3_SM70_K_SPLITS must be 4, 8, or 16");

constexpr int sm70_exl3_mcg_decode_splits(int bits, int64_t n) {
  if (bits == 5 && n != 1024 && n != 3072 && n <= 2560) {
    return 16;
  }
  if (bits == 6 && n >= 32768) {
    return 4;
  }
  return 8;
}

union Half2Bits {
  half2 h2;
  uint32_t u32;
};

struct FragB {
  half2 x[2];
};

struct alignas(8) Half4 {
  half2 x;
  half2 y;
};

struct Sm70Exl3Shared {
  // One 128-wide input-Hadamard block for each padded WMMA M row.
  half a_had[kTileM][kHadamard];
  union {
    struct {
      // Maximum packed size is K6: 16 * 6 uint16 values per 16x16 tile.
      alignas(8) uint16_t packed[kWarps][16 * 6];
      alignas(16) half b[kWarps][kTileK][kTileK];
    } mainloop;
    // The post-Hadamard consumes all eight warp output tiles together.
    alignas(16) float c[kTileM][kTileN];
  } workspace;
};

template <int Splits>
union Sm70Exl3DecodeShared {
  struct {
    // Each warp owns a disjoint K range for the same N=16 output tile. The
    // decoded FP16 tile exists only long enough to feed WMMA.
    alignas(16) half b[Splits][kTileK][kTileK];
  } mainloop;
  // Reuse the mainloop storage after all split-K warps finish.
  alignas(16) float partial[Splits][kTileM][kTileK];
};

template <int Splits>
union Sm70Exl3NativeDecodeShared {
  struct {
    alignas(16) half b[Splits][32][kTileK];
  } mainloop;
  // The native atom produces only eight padded M rows instead of WMMA's 16.
  alignas(16) float partial[Splits][8][32];
};

template <int Splits>
union Sm70Exl3NativeN128Shared {
  struct {
    // The four N32 tiles are reconstructed sequentially, so each split-K
    // warp needs only one decoded operand tile at a time.
    alignas(16) half b[Splits][32][kTileK];
  } mainloop;
  // After the mainloop, split 0 is reused for the reduced N128 tile that is
  // consumed directly by the warp-level output Hadamard.
  alignas(16) float partial[Splits][8][kHadamard];
};

__device__ __forceinline__ uint32_t exl3_fshift(uint32_t b, uint32_t a,
                                                int shift) {
  uint64_t merged = (static_cast<uint64_t>(a) << 32) | b;
  return static_cast<uint32_t>(merged >> shift);
}

template <int Bits>
__device__ __forceinline__ int wrap_trellis_word(int idx) {
  constexpr int kWords = Bits * 256 / 32;
  // dq4 can address at most the second copy of the circular word array.
  return idx >= kWords ? idx - kWords : idx;
}

constexpr uint32_t kMcgMul = 0xCBAC1FEDu;

// LOP3 + pair-add only.  The caller must already have issued the independent
// MCG multiplies so ptxas can hide IMUL latency behind later LOP3/HADD2.
__device__ __forceinline__ half2 decode_mcg_hashed_pair(uint32_t x0,
                                                         uint32_t x1) {
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
  Half2Bits u0;
  Half2Bits u1;
  u0.u32 = x0;
  u1.u32 = x1;
  half2 lo = __halves2half2(__low2half(u0.h2), __low2half(u1.h2));
  half2 hi = __halves2half2(__high2half(u0.h2), __high2half(u1.h2));
  return __hadd2(lo, hi);
}

__device__ __forceinline__ half2 decode_mcg_pair(uint32_t x0,
                                                 uint32_t x1) {
  x0 *= kMcgMul;
  x1 *= kMcgMul;
  return decode_mcg_hashed_pair(x0, x1);
}

// Four independent states: issue all four IMULs before any LOP3/PRMT/HADD2.
__device__ __forceinline__ void decode_mcg_quad(
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, half2& low,
    half2& high) {
  s0 *= kMcgMul;
  s1 *= kMcgMul;
  s2 *= kMcgMul;
  s3 *= kMcgMul;
  low = decode_mcg_hashed_pair(s0, s1);
  high = decode_mcg_hashed_pair(s2, s3);
}

#if EXL3_TM_STATE_LUT
__device__ __align__(128) uint16_t sm70_exl3_mcg_decode_lut[65536];

__global__ void sm70_exl3_init_mcg_decode_lut_kernel() {
  int const pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= 32768) return;
  Half2Bits decoded;
  decoded.h2 = decode_mcg_pair(static_cast<uint32_t>(2 * pair),
                                static_cast<uint32_t>(2 * pair + 1));
  reinterpret_cast<uint32_t*>(sm70_exl3_mcg_decode_lut)[pair] = decoded.u32;
}

__device__ __forceinline__ half2 decode_mcg_state_pair(uint32_t x0,
                                                        uint32_t x1) {
  uint32_t const low =
      __ldg(sm70_exl3_mcg_decode_lut + (x0 & 0xffffu));
  uint32_t const high =
      __ldg(sm70_exl3_mcg_decode_lut + (x1 & 0xffffu));
  Half2Bits result;
  result.u32 = low | (high << 16);
  return result.h2;
}
#else
__device__ __forceinline__ half2 decode_mcg_state_pair(uint32_t x0,
                                                        uint32_t x1) {
  return decode_mcg_pair(x0, x1);
}
#endif

template <int Bits>
__device__ __forceinline__ FragB decode_mcg_four(const uint32_t* packed,
                                                 int offset) {
  int const first_bit = (offset + 257) * Bits - 16;
  int const last_start_bit = first_bit + 3 * Bits;
  int const last_end_bit = last_start_bit + 16;
  int const first_word = first_bit / 32;
  int const last_word = (last_end_bit - 1) / 32;
  int const shift = (last_word + 1) * 32 - last_end_bit;

  uint32_t const a = packed[wrap_trellis_word<Bits>(first_word)];
  uint32_t const b = packed[wrap_trellis_word<Bits>(last_word)];
  uint32_t const w3 = exl3_fshift(b, a, shift) & 0xffffu;
  uint32_t const w2 = exl3_fshift(b, a, shift + Bits) & 0xffffu;
  uint32_t const w1 = exl3_fshift(b, a, shift + 2 * Bits) & 0xffffu;
  uint32_t const w0 = exl3_fshift(b, a, shift + 3 * Bits) & 0xffffu;
  half2 low;
  half2 high;
  decode_mcg_quad(w0, w1, w2, w3, low, high);
  return FragB{{low, high}};
}

__device__ __forceinline__ uint32_t warp_packed_word(uint32_t word0,
                                                     uint32_t word1,
                                                     int index) {
  int const source_lane = index & (kWarpSize - 1);
  // Both shuffles must execute in every lane because the full-warp mask is
  // used. Branching around either shuffle corrupts the K5/K6 second bank.
  uint32_t const low =
      __shfl_sync(0xffffffffu, word0, source_lane);
  uint32_t const high =
      __shfl_sync(0xffffffffu, word1, source_lane);
  return index < kWarpSize ? low : high;
}

template <int Bits>
__device__ __forceinline__ FragB decode_mcg_four_from_regs(
    uint32_t word0, uint32_t word1, int offset) {
  int const first_bit = (offset + 257) * Bits - 16;
  int const last_start_bit = first_bit + 3 * Bits;
  int const last_end_bit = last_start_bit + 16;
  int const first_raw_word = first_bit / 32;
  int const last_raw_word = (last_end_bit - 1) / 32;
  int const shift = (last_raw_word + 1) * 32 - last_end_bit;
  int const first_word = wrap_trellis_word<Bits>(first_raw_word);
  int const last_word = wrap_trellis_word<Bits>(last_raw_word);

  uint32_t const a = warp_packed_word(word0, word1, first_word);
  uint32_t const b = warp_packed_word(word0, word1, last_word);
  uint32_t const w3 = exl3_fshift(b, a, shift) & 0xffffu;
  uint32_t const w2 = exl3_fshift(b, a, shift + Bits) & 0xffffu;
  uint32_t const w1 = exl3_fshift(b, a, shift + 2 * Bits) & 0xffffu;
  uint32_t const w0 = exl3_fshift(b, a, shift + 3 * Bits) & 0xffffu;
  half2 low;
  half2 high;
  decode_mcg_quad(w0, w1, w2, w3, low, high);
  return FragB{{low, high}};
}

template <int Bits>
__device__ __forceinline__ void reconstruct_mcg_tile_to_global(
    const uint32_t* packed, half* dense_b, int stride, int lane) {
  // Keep the entire EXL3 decode in registers.  The packed tile is read
  // directly from global memory; decoded fragments are permuted with warp
  // shuffles and written once to the temporary dense FP16 matrix consumed by
  // the native tensor-core GEMM.  No shared-memory staging is used here.
  FragB const f0 = decode_mcg_four<Bits>(packed, lane * 8);
  FragB const f1 = decode_mcg_four<Bits>(packed, lane * 8 + 4);

  half2 const n0 = __shfl_down_sync(0xffffffffu, f0.x[0], 4);
  half2 const n1 = __shfl_down_sync(0xffffffffu, f0.x[1], 4);
  half2 const n2 = __shfl_down_sync(0xffffffffu, f1.x[0], 4);
  half2 const n3 = __shfl_down_sync(0xffffffffu, f1.x[1], 4);

  if ((lane & 4) == 0) {
    half2 const values[8] = {
        __halves2half2(__low2half(f0.x[0]), __low2half(n0)),
        __halves2half2(__high2half(f0.x[0]), __high2half(n0)),
        __halves2half2(__low2half(f0.x[1]), __low2half(n1)),
        __halves2half2(__high2half(f0.x[1]), __high2half(n1)),
        __halves2half2(__low2half(f1.x[0]), __low2half(n2)),
        __halves2half2(__high2half(f1.x[0]), __high2half(n2)),
        __halves2half2(__low2half(f1.x[1]), __low2half(n3)),
        __halves2half2(__high2half(f1.x[1]), __high2half(n3)),
    };
    int const r0 = (lane & 3) * 2;
    int const rows[8] = {r0, r0 + 1, r0 + 8, r0 + 9,
                         r0, r0 + 1, r0 + 8, r0 + 9};
    int const c0 = lane / 8;
    int const cols[8] = {c0, c0, c0, c0, c0 + 4, c0 + 4, c0 + 4, c0 + 4};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      *reinterpret_cast<half2*>(dense_b + rows[i] * stride + cols[i] * 2) =
          values[i];
    }
  }
}

template <int Bits>
__device__ __forceinline__ void reconstruct_mcg_tile(
    const uint32_t* packed, half tile[kTileK][kTileK], int lane) {
  FragB const f0 = decode_mcg_four<Bits>(packed, lane * 8);
  FragB const f1 = decode_mcg_four<Bits>(packed, lane * 8 + 4);

  half2 const n0 = __shfl_down_sync(0xffffffffu, f0.x[0], 4);
  half2 const n1 = __shfl_down_sync(0xffffffffu, f0.x[1], 4);
  half2 const n2 = __shfl_down_sync(0xffffffffu, f1.x[0], 4);
  half2 const n3 = __shfl_down_sync(0xffffffffu, f1.x[1], 4);

  // This is the inverse fragment permutation used by ExLlamaV3's
  // reconstruct.cu. It emits a logical row-major B[K][N] tile.
  if ((lane & 4) == 0) {
    half2 const values[8] = {
        __halves2half2(__low2half(f0.x[0]), __low2half(n0)),
        __halves2half2(__high2half(f0.x[0]), __high2half(n0)),
        __halves2half2(__low2half(f0.x[1]), __low2half(n1)),
        __halves2half2(__high2half(f0.x[1]), __high2half(n1)),
        __halves2half2(__low2half(f1.x[0]), __low2half(n2)),
        __halves2half2(__high2half(f1.x[0]), __high2half(n2)),
        __halves2half2(__low2half(f1.x[1]), __low2half(n3)),
        __halves2half2(__high2half(f1.x[1]), __high2half(n3)),
    };
    int const r0 = (lane & 3) * 2;
    int const rows[8] = {r0, r0 + 1, r0 + 8, r0 + 9,
                         r0, r0 + 1, r0 + 8, r0 + 9};
    int const c0 = lane / 8;
    int const cols[8] = {c0, c0, c0, c0, c0 + 4, c0 + 4, c0 + 4, c0 + 4};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      *reinterpret_cast<half2*>(&tile[rows[i]][cols[i] * 2]) = values[i];
    }
  }
}

template <int Bits>
__device__ __forceinline__ void reconstruct_mcg_tile_strided(
    const uint32_t* packed, half* tile, int column_offset, int lane) {
  FragB const f0 = decode_mcg_four<Bits>(packed, lane * 8);
  FragB const f1 = decode_mcg_four<Bits>(packed, lane * 8 + 4);

  half2 const n0 = __shfl_down_sync(0xffffffffu, f0.x[0], 4);
  half2 const n1 = __shfl_down_sync(0xffffffffu, f0.x[1], 4);
  half2 const n2 = __shfl_down_sync(0xffffffffu, f1.x[0], 4);
  half2 const n3 = __shfl_down_sync(0xffffffffu, f1.x[1], 4);

  if ((lane & 4) == 0) {
    half2 const values[8] = {
        __halves2half2(__low2half(f0.x[0]), __low2half(n0)),
        __halves2half2(__high2half(f0.x[0]), __high2half(n0)),
        __halves2half2(__low2half(f0.x[1]), __low2half(n1)),
        __halves2half2(__high2half(f0.x[1]), __high2half(n1)),
        __halves2half2(__low2half(f1.x[0]), __low2half(n2)),
        __halves2half2(__high2half(f1.x[0]), __high2half(n2)),
        __halves2half2(__low2half(f1.x[1]), __low2half(n3)),
        __halves2half2(__high2half(f1.x[1]), __high2half(n3)),
    };
    int const k0 = (lane & 3) * 2;
    int const n0 = column_offset + (lane / 8) * 2;
    // Transpose while the values are still in registers.  This keeps the
    // original eight half2 stores but gives the MMA atom contiguous K8 loads.
    *reinterpret_cast<half2*>(tile + n0 * kTileK + k0) =
        __halves2half2(__low2half(values[0]), __low2half(values[1]));
    *reinterpret_cast<half2*>(tile + (n0 + 1) * kTileK + k0) =
        __halves2half2(__high2half(values[0]), __high2half(values[1]));
    *reinterpret_cast<half2*>(tile + n0 * kTileK + k0 + 8) =
        __halves2half2(__low2half(values[2]), __low2half(values[3]));
    *reinterpret_cast<half2*>(tile + (n0 + 1) * kTileK + k0 + 8) =
        __halves2half2(__high2half(values[2]), __high2half(values[3]));
    *reinterpret_cast<half2*>(tile + (n0 + 8) * kTileK + k0) =
        __halves2half2(__low2half(values[4]), __low2half(values[5]));
    *reinterpret_cast<half2*>(tile + (n0 + 9) * kTileK + k0) =
        __halves2half2(__high2half(values[4]), __high2half(values[5]));
    *reinterpret_cast<half2*>(tile + (n0 + 8) * kTileK + k0 + 8) =
        __halves2half2(__low2half(values[6]), __low2half(values[7]));
    *reinterpret_cast<half2*>(tile + (n0 + 9) * kTileK + k0 + 8) =
        __halves2half2(__high2half(values[6]), __high2half(values[7]));
  }
}

__device__ __forceinline__ void mma_sm70_m8n8k4(
    float (&acc)[8], Half4 const& a, Half4 const& b) {
  Half2Bits a0;
  Half2Bits a1;
  Half2Bits b0;
  Half2Bits b1;
  a0.h2 = a.x;
  a1.h2 = a.y;
  b0.h2 = b.x;
  b1.h2 = b.y;
  float out[8];
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7},"
      "{%8,%9},{%10,%11},"
      "{%12,%13,%14,%15,%16,%17,%18,%19};"
      : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
        "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
      : "r"(a0.u32), "r"(a1.u32), "r"(b0.u32), "r"(b1.u32),
        "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]),
        "f"(acc[4]), "f"(acc[5]), "f"(acc[6]), "f"(acc[7]));
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = out[i];
}

template <int Bits>
__device__ __forceinline__ void reconstruct_mcg_native_n32_block(
    const uint16_t* trellis, half* b_tile, int k_block,
    int packed_blocks_n, int packed_n_block, int lane) {
  uint32_t const* packed0 = reinterpret_cast<const uint32_t*>(
      trellis + (k_block * packed_blocks_n + packed_n_block) *
                    (16 * Bits));
  uint32_t const* packed1 = reinterpret_cast<const uint32_t*>(
      trellis + (k_block * packed_blocks_n + packed_n_block + 1) *
                    (16 * Bits));
  constexpr int kPackedWords = Bits * 8;
  uint32_t const packed0_word0 = packed0[lane];
  uint32_t const packed0_word1 =
      lane + kWarpSize < kPackedWords ? packed0[lane + kWarpSize] : 0u;
  uint32_t const packed1_word0 = packed1[lane];
  uint32_t const packed1_word1 =
      lane + kWarpSize < kPackedWords ? packed1[lane + kWarpSize] : 0u;

  FragB const t0f0 =
      decode_mcg_four_from_regs<Bits>(packed0_word0, packed0_word1, lane * 8);
  FragB const t0f1 = decode_mcg_four_from_regs<Bits>(
      packed0_word0, packed0_word1, lane * 8 + 4);
  FragB const t1f0 =
      decode_mcg_four_from_regs<Bits>(packed1_word0, packed1_word1, lane * 8);
  FragB const t1f1 = decode_mcg_four_from_regs<Bits>(
      packed1_word0, packed1_word1, lane * 8 + 4);

  FragB const fragments[4] = {t0f0, t0f1, t1f0, t1f1};
#pragma unroll
  for (int fragment = 0; fragment < 4; fragment += 2) {
    FragB const f0 = fragments[fragment];
    FragB const f1 = fragments[fragment + 1];
    half2 const n0 = __shfl_down_sync(0xffffffffu, f0.x[0], 4);
    half2 const n1 = __shfl_down_sync(0xffffffffu, f0.x[1], 4);
    half2 const n2 = __shfl_down_sync(0xffffffffu, f1.x[0], 4);
    half2 const n3 = __shfl_down_sync(0xffffffffu, f1.x[1], 4);

    if ((lane & 4) == 0) {
      half2 const values[8] = {
          __halves2half2(__low2half(f0.x[0]), __low2half(n0)),
          __halves2half2(__high2half(f0.x[0]), __high2half(n0)),
          __halves2half2(__low2half(f0.x[1]), __low2half(n1)),
          __halves2half2(__high2half(f0.x[1]), __high2half(n1)),
          __halves2half2(__low2half(f1.x[0]), __low2half(n2)),
          __halves2half2(__high2half(f1.x[0]), __high2half(n2)),
          __halves2half2(__low2half(f1.x[1]), __low2half(n3)),
          __halves2half2(__high2half(f1.x[1]), __high2half(n3)),
      };
      int const k0 = (lane & 3) * 2;
      int const column_offset = fragment * 8;
      int const output_n0 = column_offset + (lane / 8) * 2;
      *reinterpret_cast<half2*>(b_tile + output_n0 * kTileK + k0) =
          __halves2half2(__low2half(values[0]), __low2half(values[1]));
      *reinterpret_cast<half2*>(b_tile + (output_n0 + 1) * kTileK + k0) =
          __halves2half2(__high2half(values[0]), __high2half(values[1]));
      *reinterpret_cast<half2*>(b_tile + output_n0 * kTileK + k0 + 8) =
          __halves2half2(__low2half(values[2]), __low2half(values[3]));
      *reinterpret_cast<half2*>(b_tile + (output_n0 + 1) * kTileK + k0 + 8) =
          __halves2half2(__high2half(values[2]), __high2half(values[3]));
      *reinterpret_cast<half2*>(b_tile + (output_n0 + 8) * kTileK + k0) =
          __halves2half2(__low2half(values[4]), __low2half(values[5]));
      *reinterpret_cast<half2*>(b_tile + (output_n0 + 9) * kTileK + k0) =
          __halves2half2(__high2half(values[4]), __high2half(values[5]));
      *reinterpret_cast<half2*>(b_tile + (output_n0 + 8) * kTileK + k0 + 8) =
          __halves2half2(__low2half(values[6]), __low2half(values[7]));
      *reinterpret_cast<half2*>(
          b_tile + (output_n0 + 9) * kTileK + k0 + 8) =
          __halves2half2(__high2half(values[6]), __high2half(values[7]));
    }
  }
}

__device__ __forceinline__ void
gather_mcg_native_n32_fragments_scalar(
    FragB const& t0f0, FragB const& t0f1, FragB const& t1f0,
    FragB const& t1f1,
    Half4& b_frag0, Half4& b_frag1, Half4& b_frag2, Half4& b_frag3,
    int lane, int effective_n) {
  // Choose the packed tile and N half only after the warp exchange.  A
  // source lane serves destinations from both packed tiles; selecting t0/t1
  // in the source lane before __shfl_sync makes every second N16 tile read
  // the wrong register bank.
  bool const second_tile = effective_n >= 16;
  int const local_n = effective_n & 15;
  int const source = ((local_n & 7) >> 1) * 8 + (local_n & 1) * 4;
  bool const high_n = local_n >= 8;
  auto gather_group = [&](auto group_tag, Half4& output) {
    constexpr int group = decltype(group_tag)::value;
    constexpr bool high_k = group >= 2;
    constexpr int source_offset = (group & 1) * 2;
    half2 const t0_nlo = high_k ? t0f0.x[1] : t0f0.x[0];
    half2 const t0_nhi = high_k ? t0f1.x[1] : t0f1.x[0];
    half2 const t1_nlo = high_k ? t1f0.x[1] : t1f0.x[0];
    half2 const t1_nhi = high_k ? t1f1.x[1] : t1f1.x[0];
    int const source0 = source + source_offset;
    int const source1 = source0 + 1;
    half2 const t0_lo0 = __shfl_sync(0xffffffffu, t0_nlo, source0);
    half2 const t0_hi0 = __shfl_sync(0xffffffffu, t0_nhi, source0);
    half2 const t0_lo1 = __shfl_sync(0xffffffffu, t0_nlo, source1);
    half2 const t0_hi1 = __shfl_sync(0xffffffffu, t0_nhi, source1);
    half2 const t1_lo0 = __shfl_sync(0xffffffffu, t1_nlo, source0);
    half2 const t1_hi0 = __shfl_sync(0xffffffffu, t1_nhi, source0);
    half2 const t1_lo1 = __shfl_sync(0xffffffffu, t1_nlo, source1);
    half2 const t1_hi1 = __shfl_sync(0xffffffffu, t1_nhi, source1);
    Half4 const tile0{
        high_n ? t0_hi0 : t0_lo0, high_n ? t0_hi1 : t0_lo1};
    Half4 const tile1{high_n ? t1_hi0 : t1_lo0,
                      high_n ? t1_hi1 : t1_lo1};
    output = second_tile ? tile1 : tile0;
  };
  gather_group(std::integral_constant<int, 0>{}, b_frag0);
  gather_group(std::integral_constant<int, 1>{}, b_frag1);
  gather_group(std::integral_constant<int, 2>{}, b_frag2);
  gather_group(std::integral_constant<int, 3>{}, b_frag3);
}

template <int Bits>
__device__ __forceinline__ void
reconstruct_mcg_native_n32_fragments_from_words_scalar(
    uint32_t packed0_word0, uint32_t packed0_word1,
    uint32_t packed1_word0, uint32_t packed1_word1,
    Half4& b_frag0, Half4& b_frag1, Half4& b_frag2, Half4& b_frag3,
    int lane, int effective_n) {
  FragB const t0f0 =
      decode_mcg_four_from_regs<Bits>(packed0_word0, packed0_word1, lane * 8);
  FragB const t0f1 = decode_mcg_four_from_regs<Bits>(
      packed0_word0, packed0_word1, lane * 8 + 4);
  FragB const t1f0 =
      decode_mcg_four_from_regs<Bits>(packed1_word0, packed1_word1, lane * 8);
  FragB const t1f1 = decode_mcg_four_from_regs<Bits>(
      packed1_word0, packed1_word1, lane * 8 + 4);
  gather_mcg_native_n32_fragments_scalar(
      t0f0, t0f1, t1f0, t1f1, b_frag0, b_frag1, b_frag2, b_frag3, lane,
      effective_n);
}

template <int Bits>
__device__ __forceinline__ void
reconstruct_mcg_native_n32_fragments_from_shared_scalar(
    uint32_t const* packed0, uint32_t const* packed1,
    Half4& b_frag0, Half4& b_frag1, Half4& b_frag2, Half4& b_frag3,
    int lane, int effective_n) {
  FragB const t0f0 = decode_mcg_four<Bits>(packed0, lane * 8);
  FragB const t0f1 = decode_mcg_four<Bits>(packed0, lane * 8 + 4);
  FragB const t1f0 = decode_mcg_four<Bits>(packed1, lane * 8);
  FragB const t1f1 = decode_mcg_four<Bits>(packed1, lane * 8 + 4);
  gather_mcg_native_n32_fragments_scalar(
      t0f0, t0f1, t1f0, t1f1, b_frag0, b_frag1, b_frag2, b_frag3, lane,
      effective_n);
}

template <int Bits>
__device__ __forceinline__ void
reconstruct_mcg_native_n32_fragments_from_shared_destination(
    uint32_t const* packed0, uint32_t const* packed1,
    Half4& b_frag0, Half4& b_frag1, Half4& b_frag2, Half4& b_frag3,
    int effective_n) {
  // The legacy source-lane path decodes both N16 tiles in every lane and then
  // gathers four fragment groups with 32 warp shuffles.  The destination MMA
  // mapping is static: each lane consumes four source offsets from exactly one
  // N16 tile.  Read those compact words directly from shared memory and decode
  // only the 16 weights this lane will feed to HMMA.
  bool const second_tile = effective_n >= 16;
  int const local_n = effective_n & 15;
  int const source = ((local_n & 7) >> 1) * 8 + (local_n & 1) * 4;
  int const n_half_offset = local_n >= 8 ? 4 : 0;
  uint32_t const* packed = second_tile ? packed1 : packed0;

  // Reuse one two-register decoded chain instead of keeping all four live
  // until the final aggregate assignments.  This matters because the raw-B
  // two-stage prefetch already puts the kernel near a V100 residency boundary.
  FragB chain =
      decode_mcg_four<Bits>(packed, (source + 0) * 8 + n_half_offset);
  b_frag0.x = chain.x[0];
  b_frag2.x = chain.x[1];
  chain = decode_mcg_four<Bits>(packed, (source + 1) * 8 + n_half_offset);
  b_frag0.y = chain.x[0];
  b_frag2.y = chain.x[1];
  chain = decode_mcg_four<Bits>(packed, (source + 2) * 8 + n_half_offset);
  b_frag1.x = chain.x[0];
  b_frag3.x = chain.x[1];
  chain = decode_mcg_four<Bits>(packed, (source + 3) * 8 + n_half_offset);
  b_frag1.y = chain.x[0];
  b_frag3.y = chain.x[1];
}

template <int Bits, bool HighPair>
__device__ __forceinline__ half2 decode_mcg_pair_from_shared(
    uint32_t const* packed, int offset) {
  int const first_bit = (offset + 257) * Bits - 16;
  int const last_start_bit = first_bit + 3 * Bits;
  int const last_end_bit = last_start_bit + 16;
  int const first_word = first_bit / 32;
  int const last_word = (last_end_bit - 1) / 32;
  int const shift = (last_word + 1) * 32 - last_end_bit;
  uint32_t const a = packed[wrap_trellis_word<Bits>(first_word)];
  uint32_t const b = packed[wrap_trellis_word<Bits>(last_word)];
  if constexpr (HighPair) {
    uint32_t const w3 = exl3_fshift(b, a, shift) & 0xffffu;
    uint32_t const w2 = exl3_fshift(b, a, shift + Bits) & 0xffffu;
    return decode_mcg_pair(w2, w3);
  } else {
    uint32_t const w1 = exl3_fshift(b, a, shift + 2 * Bits) & 0xffffu;
    uint32_t const w0 = exl3_fshift(b, a, shift + 3 * Bits) & 0xffffu;
    return decode_mcg_pair(w0, w1);
  }
}

template <int Bits, bool HighPair>
__device__ __forceinline__ void
reconstruct_mcg_native_n32_destination_half_from_shared(
    uint32_t const* packed0, uint32_t const* packed1,
    Half4& b_frag0, Half4& b_frag1, int effective_n) {
  bool const second_tile = effective_n >= 16;
  int const local_n = effective_n & 15;
  int const source = ((local_n & 7) >> 1) * 8 + (local_n & 1) * 4;
  int const n_half_offset = local_n >= 8 ? 4 : 0;
  uint32_t const* packed = second_tile ? packed1 : packed0;

  b_frag0.x = decode_mcg_pair_from_shared<Bits, HighPair>(
      packed, (source + 0) * 8 + n_half_offset);
  b_frag0.y = decode_mcg_pair_from_shared<Bits, HighPair>(
      packed, (source + 1) * 8 + n_half_offset);
  b_frag1.x = decode_mcg_pair_from_shared<Bits, HighPair>(
      packed, (source + 2) * 8 + n_half_offset);
  b_frag1.y = decode_mcg_pair_from_shared<Bits, HighPair>(
      packed, (source + 3) * 8 + n_half_offset);
}

template <int Bits>
__device__ __forceinline__ void
reconstruct_mcg_native_n32_fragments_from_words(
    uint32_t packed0_word0, uint32_t packed0_word1,
    uint32_t packed1_word0, uint32_t packed1_word1,
    Half4 (&b_frag)[4], int lane, int effective_n) {
  reconstruct_mcg_native_n32_fragments_from_words_scalar<Bits>(
      packed0_word0, packed0_word1, packed1_word0, packed1_word1,
      b_frag[0], b_frag[1], b_frag[2], b_frag[3], lane, effective_n);
}

template <int Bits>
__device__ __forceinline__ void reconstruct_mcg_native_n32_fragments(
    const uint16_t* trellis, Half4 (&b_frag)[4], int k_block,
    int packed_blocks_n, int packed_n_block, int lane, int effective_n) {
  uint32_t const* packed0 = reinterpret_cast<const uint32_t*>(
      trellis + (k_block * packed_blocks_n + packed_n_block) *
                    (16 * Bits));
  uint32_t const* packed1 = reinterpret_cast<const uint32_t*>(
      trellis + (k_block * packed_blocks_n + packed_n_block + 1) *
                    (16 * Bits));
  constexpr int kPackedWords = Bits * 8;
  uint32_t const packed0_word0 = packed0[lane];
  uint32_t const packed0_word1 =
      lane + kWarpSize < kPackedWords ? packed0[lane + kWarpSize] : 0u;
  uint32_t const packed1_word0 = packed1[lane];
  uint32_t const packed1_word1 =
      lane + kWarpSize < kPackedWords ? packed1[lane + kWarpSize] : 0u;
  reconstruct_mcg_native_n32_fragments_from_words<Bits>(
      packed0_word0, packed0_word1, packed1_word0, packed1_word1,
      b_frag, lane, effective_n);
}

struct McgStatePair {
  uint32_t first;
  uint32_t second;
};

// Extract the two consecutive 16-bit trellis states that produce one half2
// in the Volta B fragment.  Unlike decode_mcg_four_from_regs, this helper is
// used only by the one-time weight repack, so its warp shuffles never appear
// in the decode hot path.
template <int Bits>
__device__ __forceinline__ McgStatePair extract_mcg_state_pair_from_regs(
    uint32_t word0, uint32_t word1, int offset, bool high_pair) {
  int const first_bit = (offset + 257) * Bits - 16;
  int const last_start_bit = first_bit + 3 * Bits;
  int const last_end_bit = last_start_bit + 16;
  int const first_raw_word = first_bit / 32;
  int const last_raw_word = (last_end_bit - 1) / 32;
  int const shift = (last_raw_word + 1) * 32 - last_end_bit;
  int const first_word = wrap_trellis_word<Bits>(first_raw_word);
  int const last_word = wrap_trellis_word<Bits>(last_raw_word);
  uint32_t const a = warp_packed_word(word0, word1, first_word);
  uint32_t const b = warp_packed_word(word0, word1, last_word);
  uint32_t const w3 = exl3_fshift(b, a, shift) & 0xffffu;
  uint32_t const w2 = exl3_fshift(b, a, shift + Bits) & 0xffffu;
  uint32_t const w1 = exl3_fshift(b, a, shift + 2 * Bits) & 0xffffu;
  uint32_t const w0 = exl3_fshift(b, a, shift + 3 * Bits) & 0xffffu;
  return high_pair ? McgStatePair{w2, w3} : McgStatePair{w0, w1};
}

template <int Bits>
__device__ __forceinline__ uint64_t pack_mcg_state_chain(
    McgStatePair const& low, McgStatePair const& high) {
  constexpr uint32_t kTransitionMask = (1u << Bits) - 1u;
  // A Volta destination lane consumes four independent output-column
  // chains.  Its low and high half2 pairs are consecutive states in the
  // same chain, so preserve one seed plus the three new transition fields.
  // This is exact and costs 16+3*Bits per four reconstructed weights.
  return static_cast<uint64_t>(low.first) |
         (static_cast<uint64_t>(low.second & kTransitionMask) << 16) |
         (static_cast<uint64_t>(high.first & kTransitionMask)
          << (16 + Bits)) |
         (static_cast<uint64_t>(high.second & kTransitionMask)
          << (16 + 2 * Bits));
}

template <int Bits>
__global__ void sm70_exl3_repack_tm_state_kernel(
    const uint16_t* __restrict__ trellis,
    uint32_t* __restrict__ lane_state, int packed_blocks_n) {
  constexpr int kChainBits = 16 + 3 * Bits;
  constexpr int kStateWords = (4 * kChainBits + 31) / 32;
  int const lane = threadIdx.x;
  int const n32_block = blockIdx.x;
  int const k16_block = blockIdx.y;
  int const packed_n_block = n32_block * 2;
  constexpr int kPackedWords = Bits * 8;
  uint32_t const* packed0 = reinterpret_cast<const uint32_t*>(
      trellis + (k16_block * packed_blocks_n + packed_n_block) *
                    (16 * Bits));
  uint32_t const* packed1 = packed0 + kPackedWords;
  uint32_t const packed0_word0 = packed0[lane];
  uint32_t const packed0_word1 =
      lane + kWarpSize < kPackedWords ? packed0[lane + kWarpSize] : 0u;
  uint32_t const packed1_word0 = packed1[lane];
  uint32_t const packed1_word1 =
      lane + kWarpSize < kPackedWords ? packed1[lane + kWarpSize] : 0u;

  int const effective_n =
      (lane / 16) * 4 + (lane & 12) * 2 + lane % 4;
  bool const second_tile = effective_n >= 16;
  int const local_n = effective_n & 15;
  int const source = ((local_n & 7) >> 1) * 8 + (local_n & 1) * 4;
  bool const high_n = local_n >= 8;
  McgStatePair pairs[8];
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    bool const high_pair = group >= 2;
    int const source0 = source + (group & 1) * 2;
    int const source1 = source0 + 1;
    int const offset0 = source0 * 8 + (high_n ? 4 : 0);
    int const offset1 = source1 * 8 + (high_n ? 4 : 0);
    McgStatePair const tile0_pair0 =
        extract_mcg_state_pair_from_regs<Bits>(
            packed0_word0, packed0_word1, offset0, high_pair);
    McgStatePair const tile0_pair1 =
        extract_mcg_state_pair_from_regs<Bits>(
            packed0_word0, packed0_word1, offset1, high_pair);
    McgStatePair const tile1_pair0 =
        extract_mcg_state_pair_from_regs<Bits>(
            packed1_word0, packed1_word1, offset0, high_pair);
    McgStatePair const tile1_pair1 =
        extract_mcg_state_pair_from_regs<Bits>(
            packed1_word0, packed1_word1, offset1, high_pair);
    pairs[group * 2] = second_tile ? tile1_pair0 : tile0_pair0;
    pairs[group * 2 + 1] = second_tile ? tile1_pair1 : tile0_pair1;
  }

  uint32_t packed[kStateWords] = {};
#pragma unroll
  for (int chain = 0; chain < 4; ++chain) {
    // pairs[0..3] feed the first K8 fragment and pairs[4..7] feed the
    // second.  Matching indices are the low/high halves of one chain.
    uint64_t const chain_bits =
        pack_mcg_state_chain<Bits>(pairs[chain], pairs[chain + 4]);
#if EXL3_TM_K45_ALIGNED_STATE
    if constexpr (Bits == 4 || Bits == 5) {
      // K4 and K5 need 28 and 31 bits per chain respectively.  Their existing
      // lane records both occupy four uint32s after alignment, so dedicating
      // one word per chain is storage-neutral and avoids every cross-word
      // field extraction in the hot GEMM.
      packed[chain] = static_cast<uint32_t>(chain_bits);
      continue;
    }
#endif
    int const bit = chain * kChainBits;
    int const word = bit / 32;
    int const shift = bit % 32;
    uint64_t const positioned = chain_bits << shift;
    packed[word] |= static_cast<uint32_t>(positioned);
    if (word + 1 < kStateWords) {
      packed[word + 1] |= static_cast<uint32_t>(positioned >> 32);
    }
  }

  // Word-plane-major storage turns each warp load into one contiguous
  // transaction.  K6 can optionally remove the 24 padding bits per lane by
  // concatenating all 32 lane records into one 136-word tile.  Repacking is a
  // one-time startup operation, so favor a simple shared-memory transpose.
  uint32_t* tile = lane_state +
      (k16_block * gridDim.x + n32_block) *
          ((Bits == 6 && EXL3_TM_K6_DENSE_STATE != 0)
               ? 136
               : kStateWords * kWarpSize);
#if EXL3_TM_K6_DENSE_STATE
  if constexpr (Bits == 6) {
    __shared__ uint32_t lane_words[kWarpSize][5];
#pragma unroll
    for (int word = 0; word < 5; ++word) {
      lane_words[lane][word] = packed[word];
    }
    __syncwarp();
    constexpr int kDenseTileWords = 136;
#pragma unroll
    for (int output_word = lane; output_word < kDenseTileWords;
         output_word += kWarpSize) {
      int const source_bit = output_word * 32;
      int const source_lane = source_bit / 136;
      int const lane_bit = source_bit - source_lane * 136;
      int const source_word = lane_bit / 32;
      int const shift = lane_bit & 31;
      uint32_t const low = lane_words[source_lane][source_word];
      int const bits_in_lane = 136 - lane_bit;
      if (bits_in_lane >= 32) {
        uint32_t const high =
            source_word + 1 < 5
                ? lane_words[source_lane][source_word + 1]
                : 0u;
        tile[output_word] = exl3_fshift(low, high, shift);
      } else {
        // The virtual 136-bit record ends inside packed[4], not at its
        // physical 32-bit boundary.  Fill the rest of this output word from
        // the next lane's record instead of copying packed[4]'s padding.
        uint32_t current = low >> shift;
        if (source_word + 1 < 5 && shift != 0) {
          current |= lane_words[source_lane][source_word + 1]
                     << (32 - shift);
        }
        uint32_t const mask = (1u << bits_in_lane) - 1u;
        current &= mask;
        uint32_t const next =
            source_lane + 1 < kWarpSize
                ? lane_words[source_lane + 1][0]
                : 0u;
        tile[output_word] = current | (next << bits_in_lane);
      }
    }
    return;
  }
#endif
#pragma unroll
  for (int word = 0; word < kStateWords; ++word) {
    tile[word * kWarpSize + lane] = packed[word];
  }
}

__device__ __forceinline__ uint32_t tm_pack_int8x4(
    int8_t v0, int8_t v1, int8_t v2, int8_t v3) {
  return static_cast<uint8_t>(v0) |
         (static_cast<uint32_t>(static_cast<uint8_t>(v1)) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(v2)) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(v3)) << 24);
}

__device__ __forceinline__ uint32_t tm_pack_int6x4(
    int v0, int v1, int v2, int v3) {
  return static_cast<uint32_t>(v0 & 0x3f) |
         (static_cast<uint32_t>(v1 & 0x3f) << 6) |
         (static_cast<uint32_t>(v2 & 0x3f) << 12) |
         (static_cast<uint32_t>(v3 & 0x3f) << 18);
}

// Sixteen signed 10-bit fields fill five coalesced uint32 planes (160 bits).
// The 10-bit two's-complement payload is the entire FP16 mantissa, so the
// hot cvt is OR + sub.f16x2 rather than PRMT.
__device__ __forceinline__ void tm_pack_int10x16(
    int const q[16], uint32_t words[5]) {
#pragma unroll
  for (int word = 0; word < 5; ++word) {
    words[word] = 0u;
  }
#pragma unroll
  for (int index = 0; index < 16; ++index) {
    int const bit = index * 10;
    int const word = bit / 32;
    int const shift = bit % 32;
    uint32_t const field = static_cast<uint32_t>(q[index]) & 0x3ffu;
    words[word] |= field << shift;
    if (shift + 10 > 32) {
      words[word + 1] |= field >> (32 - shift);
    }
  }
}

__device__ __forceinline__ uint32_t tm_extract_int10_field(
    uint32_t const words[5], int index) {
  int const bit = index * 10;
  int const word = bit / 32;
  int const shift = bit % 32;
  uint32_t field = words[word] >> shift;
  if (shift + 10 > 32) {
    field |= words[word + 1] << (32 - shift);
  }
  return field & 0x3ffu;
}

template <int Bits>
__global__ void sm70_exl3_repack_tm_int8_kernel(
    const uint16_t* __restrict__ trellis,
    int8_t* __restrict__ packed_lane,
    half* __restrict__ tile_scales, int packed_blocks_n) {
  int const lane = threadIdx.x;
  int const n32_block = blockIdx.x;
  int const k16_block = blockIdx.y;
  int const effective_n =
      (lane / 16) * 4 + (lane & 12) * 2 + lane % 4;

  Half4 b_frag[4];
  reconstruct_mcg_native_n32_fragments<Bits>(
      trellis, b_frag, k16_block, packed_blocks_n, n32_block * 2,
      lane, effective_n);
  half const* values = reinterpret_cast<half const*>(&b_frag[0]);
  float max_abs = 0.0f;
#pragma unroll
  for (int index = 0; index < 16; ++index) {
    max_abs = fmaxf(max_abs, fabsf(__half2float(values[index])));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(
        max_abs, __shfl_down_sync(0xffffffffu, max_abs, offset));
  }
  max_abs = __shfl_sync(0xffffffffu, max_abs, 0);
  float const scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
  float const inverse_scale = 1.0f / scale;
  if (lane == 0) {
    tile_scales[k16_block * gridDim.x + n32_block] =
        __float2half_rn(scale);
  }

  uint32_t quantized[4];
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    int8_t q[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float const value = __half2float(values[group * 4 + item]);
      int const rounded = __float2int_rn(value * inverse_scale);
      q[item] = static_cast<int8_t>(max(-127, min(127, rounded)));
    }
    quantized[group] = tm_pack_int8x4(q[0], q[1], q[2], q[3]);
  }
  int8_t* destination =
      packed_lane +
      ((k16_block * gridDim.x + n32_block) * kWarpSize + lane) * 16;
  *reinterpret_cast<int4*>(destination) =
      make_int4(static_cast<int>(quantized[0]),
                static_cast<int>(quantized[1]),
                static_cast<int>(quantized[2]),
                static_cast<int>(quantized[3]));
}

// Transcode the exact EXL3 reconstruction once at model load into a compact
// lane-native INT6 stream.  A N32xK16 tile contains 512 weights.  The hot
// kernel reads three coalesced uint32 word planes per lane (384 bytes/tile)
// plus eight FP16 scales (one scale per N4xK16 group), for 6.25 bits/weight.
// Word-plane-major storage is intentional: a dense 12-byte lane record makes
// each warp issue strided global transactions, erasing much of the byte win.
template <int Bits>
__global__ void sm70_exl3_repack_tm_int6_kernel(
    const uint16_t* __restrict__ trellis,
    uint32_t* __restrict__ packed_words,
    half* __restrict__ group_scales, int packed_blocks_n) {
  int const lane = threadIdx.x;
  int const n32_block = blockIdx.x;
  int const k16_block = blockIdx.y;
  int const effective_n =
      (lane / 16) * 4 + (lane & 12) * 2 + lane % 4;

  Half4 b_frag[4];
  reconstruct_mcg_native_n32_fragments<Bits>(
      trellis, b_frag, k16_block, packed_blocks_n, n32_block * 2,
      lane, effective_n);
  half const* values = reinterpret_cast<half const*>(&b_frag[0]);
  float max_abs = 0.0f;
#pragma unroll
  for (int index = 0; index < 16; ++index) {
    max_abs = fmaxf(max_abs, fabsf(__half2float(values[index])));
  }
  // The Volta fragment permutation maps every contiguous lane quad to four
  // neighboring N columns.  Reduce those 64 weights to one scale without
  // crossing a scale group.
  max_abs = fmaxf(max_abs,
                  __shfl_xor_sync(0xffffffffu, max_abs, 1));
  max_abs = fmaxf(max_abs,
                  __shfl_xor_sync(0xffffffffu, max_abs, 2));
  // A max-abs scale is cheap but wastes too much of a six-bit codebook on a
  // single group outlier.  Repack runs only once at model load, so choose the
  // MSE-minimizing clipped scale from a small deterministic grid.  Quantize
  // against the FP16-rounded scale that the hot GEMM will actually consume.
  float scale = 1.0f;
  if (max_abs > 0.0f) {
    float best_error = 1.0e30f;
#pragma unroll
    for (int candidate = 0; candidate < 7; ++candidate) {
      float const clip = 0.70f + 0.05f * static_cast<float>(candidate);
      half const candidate_half = __float2half_rn(max_abs * clip / 31.0f);
      float const candidate_scale = __half2float(candidate_half);
      float const candidate_inverse = 1.0f / candidate_scale;
      float error = 0.0f;
#pragma unroll
      for (int index = 0; index < 16; ++index) {
        float const value = __half2float(values[index]);
        int const rounded = max(
            -31, min(31, __float2int_rn(value * candidate_inverse)));
        float const delta =
            value - static_cast<float>(rounded) * candidate_scale;
        error = fmaf(delta, delta, error);
      }
      error += __shfl_xor_sync(0xffffffffu, error, 1);
      error += __shfl_xor_sync(0xffffffffu, error, 2);
      if (error < best_error) {
        best_error = error;
        scale = candidate_scale;
      }
    }
  }
  float const inverse_scale = 1.0f / scale;
  int const scale_group = effective_n / 4;
  int const tile = k16_block * gridDim.x + n32_block;
  if ((lane & 3) == 0) {
    group_scales[tile * 8 + scale_group] = __float2half_rn(scale);
  }

  uint32_t groups[4];
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    int q[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float const value =
          __half2float(values[group * 4 + item]) * inverse_scale;
      q[item] = max(-31, min(31, __float2int_rn(value)));
    }
    groups[group] = tm_pack_int6x4(q[0], q[1], q[2], q[3]);
  }

  uint32_t const word0 = groups[0] | (groups[1] << 24);
  uint32_t const word1 = (groups[1] >> 8) | (groups[2] << 16);
  uint32_t const word2 = (groups[2] >> 16) | (groups[3] << 8);
  uint32_t* tile_words = packed_words + tile * 3 * kWarpSize;
  tile_words[0 * kWarpSize + lane] = word0;
  tile_words[1 * kWarpSize + lane] = word1;
  tile_words[2 * kWarpSize + lane] = word2;
}

// One-time exact MCG reconstruct → signed INT10 + per N4xK16 FP16 scale.
// 16 values × 10 bits = five coalesced uint32 planes/lane (640 bytes/tile)
// plus eight FP16 scales (16 bytes) = 10.25 bits/weight.  The extra two bits
// versus INT8 exist only to cut second-quant noise; decode stays INT8-shaped.
template <int Bits>
__global__ void sm70_exl3_repack_tm_int10_kernel(
    const uint16_t* __restrict__ trellis,
    uint32_t* __restrict__ packed_words,
    half* __restrict__ group_scales, int packed_blocks_n) {
  int const lane = threadIdx.x;
  int const n32_block = blockIdx.x;
  int const k16_block = blockIdx.y;
  int const effective_n =
      (lane / 16) * 4 + (lane & 12) * 2 + lane % 4;

  Half4 b_frag[4];
  reconstruct_mcg_native_n32_fragments<Bits>(
      trellis, b_frag, k16_block, packed_blocks_n, n32_block * 2,
      lane, effective_n);
  half const* values = reinterpret_cast<half const*>(&b_frag[0]);
  float max_abs = 0.0f;
#pragma unroll
  for (int index = 0; index < 16; ++index) {
    max_abs = fmaxf(max_abs, fabsf(__half2float(values[index])));
  }
  max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffffu, max_abs, 1));
  max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffffu, max_abs, 2));
  // Symmetric 10-bit two's complement avoids the -512 endpoint so the
  // 0x6400|biased / sub-1536 cvt stays inside exact FP16 integers.
  float const scale = max_abs > 0.0f ? max_abs / 511.0f : 1.0f;
  float const inverse_scale = 1.0f / scale;
  int const scale_group = effective_n / 4;
  int const tile = k16_block * gridDim.x + n32_block;
  if ((lane & 3) == 0) {
    group_scales[tile * 8 + scale_group] = __float2half_rn(scale);
  }

  int q[16];
#pragma unroll
  for (int index = 0; index < 16; ++index) {
    float const value = __half2float(values[index]) * inverse_scale;
    q[index] = max(-511, min(511, __float2int_rn(value)));
  }
  uint32_t words[5];
  tm_pack_int10x16(q, words);
  uint32_t* tile_words = packed_words + tile * 5 * kWarpSize;
#pragma unroll
  for (int word = 0; word < 5; ++word) {
    tile_words[word * kWarpSize + lane] = words[word];
  }
}

template <int Bits>
__global__ void sm70_exl3_repack_tm_e4m3_kernel(
    const uint16_t* __restrict__ trellis,
    uint8_t* __restrict__ packed_lane,
    half* __restrict__ tile_scales, int packed_blocks_n) {
  int const lane = threadIdx.x;
  int const n32_block = blockIdx.x;
  int const k16_block = blockIdx.y;
  int const effective_n =
      (lane / 16) * 4 + (lane & 12) * 2 + lane % 4;

  Half4 b_frag[4];
  reconstruct_mcg_native_n32_fragments<Bits>(
      trellis, b_frag, k16_block, packed_blocks_n, n32_block * 2,
      lane, effective_n);
  half const* values = reinterpret_cast<half const*>(&b_frag[0]);
  float max_abs = 0.0f;
#pragma unroll
  for (int index = 0; index < 16; ++index) {
    max_abs = fmaxf(max_abs, fabsf(__half2float(values[index])));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(
        max_abs, __shfl_down_sync(0xffffffffu, max_abs, offset));
  }
  max_abs = __shfl_sync(0xffffffffu, max_abs, 0);
  // CUDA E4M3 uses finite values through +/-448.  Quantize from float at
  // model load so the device constructor supplies the architecture-neutral
  // saturation and round-to-nearest behavior; decode remains a byte-only
  // register transform in the hot GEMM.
  float const scale = max_abs > 0.0f ? max_abs / 448.0f : 1.0f;
  float const inverse_scale = 1.0f / scale;
  if (lane == 0) {
    tile_scales[k16_block * gridDim.x + n32_block] =
        __float2half_rn(scale);
  }

  uint32_t quantized[4];
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    uint8_t bytes[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float const value =
          __half2float(values[group * 4 + item]) * inverse_scale;
      __nv_fp8_e4m3 const encoded(value);
      bytes[item] = static_cast<uint8_t>(encoded.__x);
    }
    // TurboMind's SM70 cvt_f16x4_e4m3 emits the packed byte order
    // [0, 2, 1, 3] in its two half2 words.  Swap the middle bytes once while
    // repacking so the hot-path conversion lands directly in the HMMA
    // fragment order without an extra PRMT.
    quantized[group] = static_cast<uint32_t>(bytes[0]) |
                       (static_cast<uint32_t>(bytes[2]) << 8) |
                       (static_cast<uint32_t>(bytes[1]) << 16) |
                       (static_cast<uint32_t>(bytes[3]) << 24);
  }
  uint8_t* destination =
      packed_lane +
      ((k16_block * gridDim.x + n32_block) * kWarpSize + lane) * 16;
  *reinterpret_cast<uint4*>(destination) =
      make_uint4(quantized[0], quantized[1], quantized[2], quantized[3]);
}

struct TmExl3StateWords {
  uint32_t w0;
  uint32_t w1;
  uint32_t w2;
  uint32_t w3;
  uint32_t w4;
  uint32_t w5;

  template <int Index>
  __device__ __forceinline__ uint32_t get() const {
    static_assert(Index >= 0 && Index < 6);
    if constexpr (Index == 0) return w0;
    if constexpr (Index == 1) return w1;
    if constexpr (Index == 2) return w2;
    if constexpr (Index == 3) return w3;
    if constexpr (Index == 4) return w4;
    return w5;
  }
};

struct McgDecodedChain {
  half2 low;
  half2 high;
};

struct McgDecodedLowChain {
  half2 value;
  uint32_t state1;
};

struct McgContinuationStates {
  uint32_t state0;
  uint32_t state1;
  uint32_t state2;
  uint32_t state3;
};

template <bool ApplyScale>
__device__ __forceinline__ Half4 tm_int8_word_to_half4(
    uint32_t word, half2 scale) {
  // Map signed bytes to u8 with xor(0x80), inject each byte into the low
  // mantissa bits of FP16 1024, then subtract FP16 1152.  This produces the
  // exact signed integer while replacing four scalar int->half conversions
  // with two PRMTs and two packed-half subtracts.  It is the same magic-number
  // technique used by TurboMind's SM70 quantized operand transforms.
  uint32_t const biased = word ^ 0x80808080u;
  constexpr uint32_t kF16Magic = 0x64000000u;
  constexpr uint32_t kF16Bias = 0x64806480u;
  uint32_t low = __byte_perm(biased, kF16Magic, 0x7170);
  uint32_t high = __byte_perm(biased, kF16Magic, 0x7372);
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(low)
               : "r"(low), "r"(kF16Bias));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(high)
               : "r"(high), "r"(kF16Bias));
  Half4 result{*reinterpret_cast<half2*>(&low),
               *reinterpret_cast<half2*>(&high)};
  if constexpr (ApplyScale) {
    result.x = __hmul2(result.x, scale);
    result.y = __hmul2(result.y, scale);
  }
  return result;
}

__device__ __forceinline__ Half4 tm_int6_group_to_half4(
    uint32_t packed, half2 scale) {
  // Spread four contiguous signed 6-bit two's-complement values into bytes,
  // bias them to u6, then reuse the same FP16 magic-number conversion as the
  // TurboMind-derived INT8 path.  0x6420 is FP16 1056 = 1024 + 32.
  uint32_t bytes = packed & 0x3fu;
  bytes |= (packed << 2) & 0x00003f00u;
  bytes |= (packed << 4) & 0x003f0000u;
  bytes |= (packed << 6) & 0x3f000000u;
  uint32_t const biased = bytes ^ 0x20202020u;
  constexpr uint32_t kF16Magic = 0x64000000u;
  constexpr uint32_t kF16Bias = 0x64206420u;
  uint32_t low = __byte_perm(biased, kF16Magic, 0x7170);
  uint32_t high = __byte_perm(biased, kF16Magic, 0x7372);
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(low)
               : "r"(low), "r"(kF16Bias));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(high)
               : "r"(high), "r"(kF16Bias));
  Half4 result{*reinterpret_cast<half2*>(&low),
               *reinterpret_cast<half2*>(&high)};
  result.x = __hmul2(result.x, scale);
  result.y = __hmul2(result.y, scale);
  return result;
}

__device__ __forceinline__ void tm_int6_words_to_fragments(
    uint32_t word0, uint32_t word1, uint32_t word2, half2 scale,
    Half4& frag0, Half4& frag1, Half4& frag2, Half4& frag3) {
  constexpr uint32_t kPacked4Mask = 0x00ffffffu;
  frag0 = tm_int6_group_to_half4(word0 & kPacked4Mask, scale);
  frag1 = tm_int6_group_to_half4(
      ((word0 >> 24) | (word1 << 8)) & kPacked4Mask, scale);
  frag2 = tm_int6_group_to_half4(
      ((word1 >> 16) | (word2 << 16)) & kPacked4Mask, scale);
  frag3 = tm_int6_group_to_half4((word2 >> 8) & kPacked4Mask, scale);
}

__device__ __forceinline__ Half4 tm_int10_quad_to_half4(
    uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3, half2 scale) {
  // 10-bit two's-complement → unsigned via xor 512, then the field *is*
  // the FP16 mantissa of (1024 + biased).  Subtract 1536 to recover the
  // signed integer.  Every integer in [-511, 511] is an exact FP16.
  uint32_t lo = (0x6400u | (t0 ^ 0x200u)) |
                ((0x6400u | (t1 ^ 0x200u)) << 16);
  uint32_t hi = (0x6400u | (t2 ^ 0x200u)) |
                ((0x6400u | (t3 ^ 0x200u)) << 16);
  constexpr uint32_t kF16Bias = 0x66006600u;
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(lo)
               : "r"(lo), "r"(kF16Bias));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(hi)
               : "r"(hi), "r"(kF16Bias));
  Half4 result{*reinterpret_cast<half2*>(&lo),
               *reinterpret_cast<half2*>(&hi)};
  result.x = __hmul2(result.x, scale);
  result.y = __hmul2(result.y, scale);
  return result;
}

__device__ __forceinline__ void tm_int10_words_to_fragments(
    uint32_t word0, uint32_t word1, uint32_t word2, uint32_t word3,
    uint32_t word4, half2 scale, Half4& frag0, Half4& frag1, Half4& frag2,
    Half4& frag3) {
  uint32_t const words[5] = {word0, word1, word2, word3, word4};
  frag0 = tm_int10_quad_to_half4(
      tm_extract_int10_field(words, 0), tm_extract_int10_field(words, 1),
      tm_extract_int10_field(words, 2), tm_extract_int10_field(words, 3),
      scale);
  frag1 = tm_int10_quad_to_half4(
      tm_extract_int10_field(words, 4), tm_extract_int10_field(words, 5),
      tm_extract_int10_field(words, 6), tm_extract_int10_field(words, 7),
      scale);
  frag2 = tm_int10_quad_to_half4(
      tm_extract_int10_field(words, 8), tm_extract_int10_field(words, 9),
      tm_extract_int10_field(words, 10), tm_extract_int10_field(words, 11),
      scale);
  frag3 = tm_int10_quad_to_half4(
      tm_extract_int10_field(words, 12), tm_extract_int10_field(words, 13),
      tm_extract_int10_field(words, 14), tm_extract_int10_field(words, 15),
      scale);
}

template <bool ApplyScale>
__device__ __forceinline__ Half4 tm_e4m3_word_to_half4(
    uint32_t word, half2 scale) {
  auto const converted = turbomind::cvt_f16x4_e4m3(
      *reinterpret_cast<const turbomind::Array<
          turbomind::fp8_e4m3_t, 4>*>(&word));
  Half4 result = *reinterpret_cast<const Half4*>(&converted);
  if constexpr (ApplyScale) {
    result.x = __hmul2(result.x, scale);
    result.y = __hmul2(result.y, scale);
  }
  return result;
}

template <int Bit, int Width>
__device__ __forceinline__ uint32_t extract_mcg_lane_state_field(
    TmExl3StateWords const& words) {
  static_assert(Bit >= 0 && Width > 0 && Width <= 16);
  constexpr int kWord = Bit / 32;
  constexpr int kShift = Bit % 32;
  constexpr uint32_t kMask = (1u << Width) - 1u;
  uint32_t value = words.template get<kWord>() >> kShift;
  if constexpr (kShift + Width > 32) {
    value |= words.template get<kWord + 1>() << (32 - kShift);
  }
  return value & kMask;
}

template <int Bits, int Chain, int ChainBit, int Width>
__device__ __forceinline__ uint32_t extract_mcg_lane_state_chain_field(
    TmExl3StateWords const& words) {
  static_assert(Bits == 4 || Bits == 5 || Bits == 6);
  static_assert(Chain >= 0 && Chain < 4);
  static_assert(ChainBit >= 0 && Width > 0);
  constexpr int kChainBits = 16 + 3 * Bits;
  static_assert(ChainBit + Width <= kChainBits);
#if EXL3_TM_K45_ALIGNED_STATE
  if constexpr (Bits == 4 || Bits == 5) {
    constexpr uint32_t kMask = (1u << Width) - 1u;
    return (words.template get<Chain>() >> ChainBit) & kMask;
  }
#endif
  return extract_mcg_lane_state_field<Chain * kChainBits + ChainBit, Width>(
      words);
}

template <int Bits, int Chain>
__device__ __forceinline__ McgDecodedChain decode_mcg_lane_state_chain(
    TmExl3StateWords const& words) {
  constexpr uint32_t kTransitionMask = (1u << Bits) - 1u;
  uint32_t const state0 =
      extract_mcg_lane_state_chain_field<Bits, Chain, 0, 16>(words);
  uint32_t const transition1 =
      extract_mcg_lane_state_chain_field<Bits, Chain, 16, Bits>(words);
  uint32_t const transition2 =
      extract_mcg_lane_state_chain_field<Bits, Chain, 16 + Bits, Bits>(words);
  uint32_t const transition3 =
      extract_mcg_lane_state_chain_field<Bits, Chain, 16 + 2 * Bits, Bits>(
          words);
  uint32_t const state1 =
      ((state0 << Bits) | (transition1 & kTransitionMask)) &
      0xffffu;
  uint32_t const state2 =
      ((state1 << Bits) | (transition2 & kTransitionMask)) &
      0xffffu;
  uint32_t const state3 =
      ((state2 << Bits) | (transition3 & kTransitionMask)) &
      0xffffu;
  half2 low;
  half2 high;
  decode_mcg_quad(state0, state1, state2, state3, low, high);
  return McgDecodedChain{low, high};
}

template <int Bits, int Chain>
__device__ __forceinline__ McgDecodedLowChain
decode_mcg_lane_state_chain_low(TmExl3StateWords const& words) {
  constexpr uint32_t kTransitionMask = (1u << Bits) - 1u;
  uint32_t const state0 =
      extract_mcg_lane_state_chain_field<Bits, Chain, 0, 16>(words);
  uint32_t const transition1 =
      extract_mcg_lane_state_chain_field<Bits, Chain, 16, Bits>(words);
  uint32_t const state1 =
      ((state0 << Bits) | (transition1 & kTransitionMask)) & 0xffffu;
  return McgDecodedLowChain{decode_mcg_state_pair(state0, state1), state1};
}

template <int Bits, int Chain>
__device__ __forceinline__ half2 decode_mcg_lane_state_chain_high(
    TmExl3StateWords const& words, uint32_t state1) {
  constexpr uint32_t kTransitionMask = (1u << Bits) - 1u;
  uint32_t const transition2 =
      extract_mcg_lane_state_chain_field<Bits, Chain, 16 + Bits, Bits>(words);
  uint32_t const transition3 =
      extract_mcg_lane_state_chain_field<Bits, Chain, 16 + 2 * Bits, Bits>(
          words);
  uint32_t const state2 =
      ((state1 << Bits) | (transition2 & kTransitionMask)) & 0xffffu;
  uint32_t const state3 =
      ((state2 << Bits) | (transition3 & kTransitionMask)) & 0xffffu;
  return decode_mcg_state_pair(state2, state3);
}

template <int Bits>
__device__ __forceinline__ McgContinuationStates
reconstruct_mcg_lane_state_fragments_low(
    TmExl3StateWords const& words, Half4& b_frag0, Half4& b_frag1) {
  McgDecodedLowChain const chain0 =
      decode_mcg_lane_state_chain_low<Bits, 0>(words);
  McgDecodedLowChain const chain1 =
      decode_mcg_lane_state_chain_low<Bits, 1>(words);
  McgDecodedLowChain const chain2 =
      decode_mcg_lane_state_chain_low<Bits, 2>(words);
  McgDecodedLowChain const chain3 =
      decode_mcg_lane_state_chain_low<Bits, 3>(words);
  b_frag0 = Half4{chain0.value, chain1.value};
  b_frag1 = Half4{chain2.value, chain3.value};
  return McgContinuationStates{chain0.state1, chain1.state1,
                               chain2.state1, chain3.state1};
}

template <int Bits>
__device__ __forceinline__ void reconstruct_mcg_lane_state_fragments_high(
    TmExl3StateWords const& words, McgContinuationStates const& states,
    Half4& b_frag2, Half4& b_frag3) {
  half2 const chain0 =
      decode_mcg_lane_state_chain_high<Bits, 0>(words, states.state0);
  half2 const chain1 =
      decode_mcg_lane_state_chain_high<Bits, 1>(words, states.state1);
  half2 const chain2 =
      decode_mcg_lane_state_chain_high<Bits, 2>(words, states.state2);
  half2 const chain3 =
      decode_mcg_lane_state_chain_high<Bits, 3>(words, states.state3);
  b_frag2 = Half4{chain0, chain1};
  b_frag3 = Half4{chain2, chain3};
}

template <int Bits>
__device__ __forceinline__ void reconstruct_mcg_lane_state_fragments(
    TmExl3StateWords const& words,
    Half4& b_frag0, Half4& b_frag1, Half4& b_frag2, Half4& b_frag3) {
  constexpr uint32_t kTransitionMask = (1u << Bits) - 1u;
  uint32_t s[16];
#pragma unroll
  for (int chain = 0; chain < 4; ++chain) {
    uint32_t state0;
    uint32_t t1;
    uint32_t t2;
    uint32_t t3;
    if (chain == 0) {
      state0 = extract_mcg_lane_state_chain_field<Bits, 0, 0, 16>(words);
      t1 = extract_mcg_lane_state_chain_field<Bits, 0, 16, Bits>(words);
      t2 = extract_mcg_lane_state_chain_field<Bits, 0, 16 + Bits, Bits>(words);
      t3 = extract_mcg_lane_state_chain_field<Bits, 0, 16 + 2 * Bits, Bits>(
          words);
    } else if (chain == 1) {
      state0 = extract_mcg_lane_state_chain_field<Bits, 1, 0, 16>(words);
      t1 = extract_mcg_lane_state_chain_field<Bits, 1, 16, Bits>(words);
      t2 = extract_mcg_lane_state_chain_field<Bits, 1, 16 + Bits, Bits>(words);
      t3 = extract_mcg_lane_state_chain_field<Bits, 1, 16 + 2 * Bits, Bits>(
          words);
    } else if (chain == 2) {
      state0 = extract_mcg_lane_state_chain_field<Bits, 2, 0, 16>(words);
      t1 = extract_mcg_lane_state_chain_field<Bits, 2, 16, Bits>(words);
      t2 = extract_mcg_lane_state_chain_field<Bits, 2, 16 + Bits, Bits>(words);
      t3 = extract_mcg_lane_state_chain_field<Bits, 2, 16 + 2 * Bits, Bits>(
          words);
    } else {
      state0 = extract_mcg_lane_state_chain_field<Bits, 3, 0, 16>(words);
      t1 = extract_mcg_lane_state_chain_field<Bits, 3, 16, Bits>(words);
      t2 = extract_mcg_lane_state_chain_field<Bits, 3, 16 + Bits, Bits>(words);
      t3 = extract_mcg_lane_state_chain_field<Bits, 3, 16 + 2 * Bits, Bits>(
          words);
    }
    uint32_t const state1 =
        ((state0 << Bits) | (t1 & kTransitionMask)) & 0xffffu;
    uint32_t const state2 =
        ((state1 << Bits) | (t2 & kTransitionMask)) & 0xffffu;
    uint32_t const state3 =
        ((state2 << Bits) | (t3 & kTransitionMask)) & 0xffffu;
    s[chain * 4 + 0] = state0;
    s[chain * 4 + 1] = state1;
    s[chain * 4 + 2] = state2;
    s[chain * 4 + 3] = state3;
  }
  // Four seed multiplies first (one per independent chain), then the rest.
  s[0] *= kMcgMul;
  s[4] *= kMcgMul;
  s[8] *= kMcgMul;
  s[12] *= kMcgMul;
  s[1] *= kMcgMul;
  s[5] *= kMcgMul;
  s[9] *= kMcgMul;
  s[13] *= kMcgMul;
  s[2] *= kMcgMul;
  s[6] *= kMcgMul;
  s[10] *= kMcgMul;
  s[14] *= kMcgMul;
  s[3] *= kMcgMul;
  s[7] *= kMcgMul;
  s[11] *= kMcgMul;
  s[15] *= kMcgMul;
  half2 const c0l = decode_mcg_hashed_pair(s[0], s[1]);
  half2 const c1l = decode_mcg_hashed_pair(s[4], s[5]);
  half2 const c2l = decode_mcg_hashed_pair(s[8], s[9]);
  half2 const c3l = decode_mcg_hashed_pair(s[12], s[13]);
  half2 const c0h = decode_mcg_hashed_pair(s[2], s[3]);
  half2 const c1h = decode_mcg_hashed_pair(s[6], s[7]);
  half2 const c2h = decode_mcg_hashed_pair(s[10], s[11]);
  half2 const c3h = decode_mcg_hashed_pair(s[14], s[15]);
  b_frag0 = Half4{c0l, c1l};
  b_frag1 = Half4{c2l, c3l};
  b_frag2 = Half4{c0h, c1h};
  b_frag3 = Half4{c2h, c3h};
}

// TurboMind's SM70 FP8 path is the scheduling and tensor-core oracle for this
// kernel.  Keep its A-side iterator, two-stage register/shared pipeline, MMA
// map, scheduler, and epilogue.  Only B is specialized: EXL3's K16xN16
// trellis tiles use the same two-stage gmem/register/shared pipeline as the
// SM70 NVFP4 kernel, then reconstruct directly from compact shared storage
// into FP16 m8n8k4 registers.  EXL3_TM_B_SHARED=0 retains the just-in-time
// global-register experiment for isolated comparison.
template <int Bits, int CtaN, int CtaK, int Threads>
struct TmExl3GmemIteratorB {
  static_assert(Bits == 4 || Bits == 5 || Bits == 6);
  static_assert(CtaN % 16 == 0 && CtaK % 16 == 0);

  static constexpr tm::Striding kMode = tm::Striding::kBlocked;
  static constexpr int kN16 = CtaN / 16;
  static constexpr int kK16 = CtaK / 16;
  static constexpr int kWarps = Threads / kWarpSize;
  static constexpr int kN16PerWarp = kN16 / kWarps;
  static constexpr int kPackedWords = Bits * 8;
  static constexpr int kStageWords = kK16 * kN16PerWarp * kPackedWords;
  static constexpr int kWordsPerThread = kStageWords / kWarpSize;
  static_assert(Threads == 128 && (CtaN == 128 || CtaN == 256) &&
                (CtaK == 32 || CtaK == 64));
  static_assert(kN16 % kWarps == 0 && kN16PerWarp % 2 == 0);
  static_assert(kStageWords % kWarpSize == 0);

  struct Fragments {
    uint32_t words[kWordsPerThread];
  };

  const uint32_t* src = nullptr;
  int packed_n16 = 0;
  int n16_begin = 0;
  int k16_begin = 0;
  bool g_mask = true;

  __device__ TmExl3GmemIteratorB() {}

  __device__ TmExl3GmemIteratorB(const tm::MatrixData& mat, int2 offset,
                                 int2 /*extent*/)
      : src(reinterpret_cast<const uint32_t*>(mat.ptr.ptr)),
        packed_n16(mat.ptr.stride),
        n16_begin(offset.x / 16),
        k16_begin(offset.y / 16) {}

  __device__ __forceinline__ void Fetch(Fragments& fragments,
                                         bool tile_mask) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
#pragma unroll
    for (int i = 0; i < kWordsPerThread; ++i) {
      int const linear = lane + i * kWarpSize;
      int const local_block = linear / kPackedWords;
      int const word = linear % kPackedWords;
      int const local_k16 = local_block / kN16PerWarp;
      int const local_n16 = local_block % kN16PerWarp;
      uint32_t const* packed =
          src + ((k16_begin + local_k16) * packed_n16 +
                 n16_begin + warp * kN16PerWarp + local_n16) *
                    kPackedWords;
      fragments.words[i] = tile_mask && g_mask ? packed[word] : 0u;
    }
  }

  __device__ __forceinline__ void Store(Fragments const& fragments,
                                         uint32_t* stage_words) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    uint32_t* warp_words = stage_words + warp * kStageWords;
#pragma unroll
    for (int i = 0; i < kWordsPerThread; ++i) {
      warp_words[lane + i * kWarpSize] = fragments.words[i];
    }
  }

  __device__ __forceinline__ void FetchToShared(uint32_t* stage_words,
                                                 bool tile_mask) const {
    static_assert(EXL3_TM_RAW_DIRECT_GROUP >= 1);
    constexpr int kDirectGroup =
        kWordsPerThread % EXL3_TM_RAW_DIRECT_GROUP == 0
            ? EXL3_TM_RAW_DIRECT_GROUP
            : 1;
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    uint32_t* warp_words = stage_words + warp * kStageWords;
    uint32_t pending[kDirectGroup];
#pragma unroll
    for (int base = 0; base < kWordsPerThread;
         base += kDirectGroup) {
#pragma unroll
      for (int item = 0; item < kDirectGroup; ++item) {
        int const i = base + item;
        int const linear = lane + i * kWarpSize;
        int const local_block = linear / kPackedWords;
        int const word = linear % kPackedWords;
        int const local_k16 = local_block / kN16PerWarp;
        int const local_n16 = local_block % kN16PerWarp;
        uint32_t const* packed =
            src + ((k16_begin + local_k16) * packed_n16 +
                   n16_begin + warp * kN16PerWarp + local_n16) *
                      kPackedWords;
        pending[item] = tile_mask && g_mask ? packed[word] : 0u;
      }
#pragma unroll
      for (int item = 0; item < kDirectGroup; ++item) {
        int const linear = lane + (base + item) * kWarpSize;
        warp_words[linear] = pending[item];
      }
    }
  }

  template <int Begin, int Count>
  __device__ __forceinline__ void FetchRange(
      uint32_t (&fragments)[Count], bool tile_mask) const {
    static_assert(Begin >= 0 && Count > 0 &&
                  Begin + Count <= kWordsPerThread);
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
#pragma unroll
    for (int item = 0; item < Count; ++item) {
      int const i = Begin + item;
      int const linear = lane + i * kWarpSize;
      int const local_block = linear / kPackedWords;
      int const word = linear % kPackedWords;
      int const local_k16 = local_block / kN16PerWarp;
      int const local_n16 = local_block % kN16PerWarp;
      uint32_t const* packed =
          src + ((k16_begin + local_k16) * packed_n16 +
                 n16_begin + warp * kN16PerWarp + local_n16) *
                    kPackedWords;
      fragments[item] = tile_mask && g_mask ? packed[word] : 0u;
    }
  }

  template <int Begin, int Count>
  __device__ __forceinline__ void StoreRange(
      uint32_t const (&fragments)[Count], uint32_t* stage_words) const {
    static_assert(Begin >= 0 && Count > 0 &&
                  Begin + Count <= kWordsPerThread);
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    uint32_t* warp_words = stage_words + warp * kStageWords;
#pragma unroll
    for (int item = 0; item < Count; ++item) {
      int const linear = lane + (Begin + item) * kWarpSize;
      warp_words[linear] = fragments[item];
    }
  }

  __device__ __forceinline__ void LoadK16(int local_k16, int local_n32,
                                          uint32_t& word0, uint32_t& word1,
                                          uint32_t& word2,
                                          uint32_t& word3) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    uint32_t const* packed0 =
        src + ((k16_begin + local_k16) * packed_n16 +
               n16_begin + warp * kN16PerWarp + local_n32 * 2) *
                  kPackedWords;
    uint32_t const* packed1 = packed0 + kPackedWords;
    word0 = packed0[lane];
    word1 = lane + kWarpSize < kPackedWords
                ? packed0[lane + kWarpSize]
                : 0u;
    word2 = packed1[lane];
    word3 = lane + kWarpSize < kPackedWords
                ? packed1[lane + kWarpSize]
                : 0u;
  }

  __device__ void Advance() { k16_begin += kK16; }
};

template <int Bits, int CtaN, int CtaK, int Threads>
struct TmExl3StateGmemIteratorB {
  static constexpr tm::Striding kMode = tm::Striding::kBlocked;
  static constexpr int kK16 = CtaK / 16;
  static constexpr int kWarps = Threads / kWarpSize;
  static constexpr int kN32 = CtaN / 32;
  static constexpr int kN32PerWarp = kN32 / kWarps;
  static constexpr int kStateWords =
      (4 * (16 + 3 * Bits) + 31) / 32;
  static constexpr bool kDenseK6 =
      Bits == 6 && EXL3_TM_K6_DENSE_STATE != 0;
  static constexpr int kStateTileWords =
      kDenseK6 ? 136 : kStateWords * kWarpSize;
  static constexpr int kStageWords =
      kK16 * kN32PerWarp * kStateTileWords;
  static constexpr int kWordsPerThread = kStageWords / kWarpSize;
  static_assert(Threads == 128 && (CtaN == 128 || CtaN == 256) &&
                (CtaK == 32 || CtaK == 64));
  static_assert(kN32 % kWarps == 0);
  static_assert(kStageWords % kWarpSize == 0);

  // Match TurboMind's FP8 gmem -> register -> shared pipeline.  Each lane
  // owns all state words for its four K16 blocks.  K4/K5 use 16 registers
  // and K6 uses 20, comparable to the stock FP8 B prefetch fragment.
  struct Fragments {
    uint32_t words[kWordsPerThread];
  };

  const uint32_t* src = nullptr;
  int packed_n32 = 0;
  int n32_begin = 0;
  int k16_begin = 0;
  bool g_mask = true;

  __device__ TmExl3StateGmemIteratorB() {}

  __device__ TmExl3StateGmemIteratorB(const tm::MatrixData& mat,
                                      int2 offset, int2 /*extent*/)
      : src(reinterpret_cast<const uint32_t*>(mat.ptr.ptr)),
        packed_n32(mat.ptr.stride),
        n32_begin(offset.x / 32),
        k16_begin(offset.y / 16) {}

  __device__ __forceinline__ void Fetch(Fragments& fragments,
                                         bool tile_mask) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    if constexpr (kDenseK6) {
#pragma unroll
      for (int item = 0; item < kWordsPerThread; ++item) {
        int const linear = lane + item * kWarpSize;
        int const local_tile = linear / kStateTileWords;
        int const local_k16 = local_tile / kN32PerWarp;
        int const local_n32 = local_tile % kN32PerWarp;
        int const word = linear - local_tile * kStateTileWords;
        uint32_t const* source =
            src + ((k16_begin + local_k16) * packed_n32 +
                   n32_begin + warp * kN32PerWarp + local_n32) *
                      kStateTileWords +
                  word;
        fragments.words[item] =
            tile_mask && g_mask ? __ldg(source) : 0u;
      }
      return;
    }
#pragma unroll
    for (int local_k16 = 0; local_k16 < kK16; ++local_k16) {
#pragma unroll
      for (int local_n32 = 0; local_n32 < kN32PerWarp; ++local_n32) {
#pragma unroll
        for (int word = 0; word < kStateWords; ++word) {
          uint32_t const* source =
              src + ((k16_begin + local_k16) * packed_n32 +
                     n32_begin + warp * kN32PerWarp + local_n32) *
                        kStateWords * kWarpSize +
                    word * kWarpSize + lane;
          fragments.words[(local_k16 * kN32PerWarp + local_n32) *
                              kStateWords +
                          word] =
              tile_mask && g_mask ? __ldg(source) : 0u;
        }
      }
    }
  }

  __device__ __forceinline__ void Store(
      Fragments const& fragments, uint32_t* stage_words) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    if constexpr (kDenseK6) {
      uint32_t* destination = stage_words + warp * kStageWords;
#pragma unroll
      for (int item = 0; item < kWordsPerThread; ++item) {
        destination[lane + item * kWarpSize] = fragments.words[item];
      }
      return;
    }
    uint32_t* destination = stage_words + warp * kStageWords + lane;
#pragma unroll
    for (int local_k16 = 0; local_k16 < kK16; ++local_k16) {
#pragma unroll
      for (int local_n32 = 0; local_n32 < kN32PerWarp; ++local_n32) {
#pragma unroll
        for (int word = 0; word < kStateWords; ++word) {
          int const item =
              (local_k16 * kN32PerWarp + local_n32) * kStateWords + word;
          destination[item * kWarpSize] = fragments.words[item];
        }
      }
    }
  }

  __device__ __forceinline__ void LoadK16FromShared(
      uint32_t const* stage_words, int local_k16, int local_n32,
      TmExl3StateWords& words) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    if constexpr (kDenseK6) {
      uint32_t const* dense = stage_words + warp * kStageWords +
                              (local_k16 * kN32PerWarp + local_n32) *
                                  kStateTileWords;
      int const lane_bit = lane * 136;
      int const base_word = lane_bit / 32;
      int const shift = lane_bit & 31;
      uint32_t const s0 = dense[base_word + 0];
      uint32_t const s1 = dense[base_word + 1];
      uint32_t const s2 = dense[base_word + 2];
      uint32_t const s3 = dense[base_word + 3];
      uint32_t const s4 = dense[base_word + 4];
      words.w0 = exl3_fshift(s0, s1, shift);
      words.w1 = exl3_fshift(s1, s2, shift);
      words.w2 = exl3_fshift(s2, s3, shift);
      words.w3 = exl3_fshift(s3, s4, shift);
      // Only the low eight bits of w4 belong to this 136-bit record.  The
      // decoder never observes its high padding bits, and avoiding s5 keeps
      // lane 31 strictly inside the 136-word tile.
      words.w4 = s4 >> shift;
      words.w5 = 0u;
      return;
    }
    uint32_t const* tile =
        stage_words + warp * kStageWords +
        (local_k16 * kN32PerWarp + local_n32) * kStateWords * kWarpSize +
        lane;
    words.w0 = tile[0 * kWarpSize];
    words.w1 = tile[1 * kWarpSize];
    words.w2 = tile[2 * kWarpSize];
    words.w3 = tile[3 * kWarpSize];
    if constexpr (kStateWords >= 5) {
      words.w4 = tile[4 * kWarpSize];
    } else {
      words.w4 = 0u;
    }
    words.w5 = 0u;
  }

  __device__ __forceinline__ void LoadK16(
      int local_k16, int local_n32, TmExl3StateWords& words) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    uint32_t const* tile =
        src + ((k16_begin + local_k16) * packed_n32 + n32_begin +
               warp * kN32PerWarp + local_n32) *
                  kStateWords * kWarpSize +
              lane;
    words.w0 = __ldg(tile + 0 * kWarpSize);
    words.w1 = __ldg(tile + 1 * kWarpSize);
    words.w2 = __ldg(tile + 2 * kWarpSize);
    words.w3 = __ldg(tile + 3 * kWarpSize);
    if constexpr (kStateWords >= 5) {
      words.w4 = __ldg(tile + 4 * kWarpSize);
    } else {
      words.w4 = 0u;
    }
    words.w5 = 0u;
  }

  __device__ void Advance() { k16_begin += kK16; }
};

template <int Bits, int CtaN, int CtaK, int Threads, bool E4m3 = false>
struct TmExl3Int8GmemIteratorB {
  static constexpr tm::Striding kMode = tm::Striding::kIndexed;
  static constexpr int kK16 = CtaK / 16;
  static constexpr int kWarps = Threads / kWarpSize;
  static constexpr int kN32 = CtaN / 32;
  static constexpr int kN32PerWarp = kN32 / kWarps;
  static constexpr int kWordsPerLane = 4;
  static constexpr int kDataStageWords =
      kK16 * kN32PerWarp * kWarpSize * kWordsPerLane;
  static constexpr int kScaleStageWords =
      (kK16 * kN32PerWarp + 1) / 2;
  static constexpr int kStageWords =
      (kDataStageWords + kScaleStageWords + 3) & ~3;
  static constexpr int kWordsPerThread =
      kK16 * kN32PerWarp * kWordsPerLane;
  static_assert(Threads == 128 && CtaN == 128 && CtaK == 64);
  static_assert(kN32PerWarp == 1);

  struct Fragments {
    uint32_t words[kWordsPerThread];
    half scale;
  };

  using Byte = std::conditional_t<E4m3, uint8_t, int8_t>;

  const Byte* src = nullptr;
  const half* scales = nullptr;
  int packed_n32 = 0;
  int n32_begin = 0;
  int k16_begin = 0;
  bool g_mask = true;

  __device__ TmExl3Int8GmemIteratorB() {}

  __device__ TmExl3Int8GmemIteratorB(const tm::MatrixData& mat,
                                     int2 offset, int2 /*extent*/)
      : src(reinterpret_cast<const Byte*>(mat.ptr.ptr)),
        scales(reinterpret_cast<const half*>(mat.idxs)),
        packed_n32(mat.ptr.stride),
        n32_begin(offset.x / 32),
        k16_begin(offset.y / 16) {}

  __device__ __forceinline__ void Fetch(Fragments& fragments,
                                         bool tile_mask) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
#pragma unroll
    for (int local_k16 = 0; local_k16 < kK16; ++local_k16) {
      int const n32 = n32_begin + warp;
      Byte const* lane_values =
          src + (((k16_begin + local_k16) * packed_n32 + n32) *
                     kWarpSize +
                 lane) *
                    16;
      int4 quantized = tile_mask && g_mask
                           ? *reinterpret_cast<int4 const*>(lane_values)
                           : make_int4(0, 0, 0, 0);
      int const base = local_k16 * kWordsPerLane;
      fragments.words[base + 0] = static_cast<uint32_t>(quantized.x);
      fragments.words[base + 1] = static_cast<uint32_t>(quantized.y);
      fragments.words[base + 2] = static_cast<uint32_t>(quantized.z);
      fragments.words[base + 3] = static_cast<uint32_t>(quantized.w);
    }
    fragments.scale = lane < kK16 && tile_mask && g_mask
                          ? __ldg(scales + (k16_begin + lane) * packed_n32 +
                                  n32_begin + warp)
                          : __float2half(1.0f);
  }

  __device__ __forceinline__ void Store(
      Fragments const& fragments, uint32_t* stage_words) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    uint32_t* warp_words = stage_words + warp * kStageWords;
#pragma unroll
    for (int local_k16 = 0; local_k16 < kK16; ++local_k16) {
      int const source = local_k16 * kWordsPerLane;
      int const destination =
          (local_k16 * kWarpSize + lane) * kWordsPerLane;
      *reinterpret_cast<uint4*>(warp_words + destination) =
          make_uint4(fragments.words[source + 0],
                     fragments.words[source + 1],
                     fragments.words[source + 2],
                     fragments.words[source + 3]);
    }
    if (lane < kK16) {
      reinterpret_cast<half*>(warp_words + kDataStageWords)[lane] =
          fragments.scale;
    }
  }

  __device__ __forceinline__ void LoadK16FromShared(
      uint32_t const* stage_words, int local_k16, int4& quantized,
      half& scale) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    uint32_t const* warp_words = stage_words + warp * kStageWords;
    int const source =
        (local_k16 * kWarpSize + lane) * kWordsPerLane;
    quantized = *reinterpret_cast<int4 const*>(warp_words + source);
    scale = reinterpret_cast<half const*>(
        warp_words + kDataStageWords)[local_k16];
  }

  __device__ __forceinline__ void LoadK16(
      int local_k16, int4& quantized, half& scale) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    int const k16 = k16_begin + local_k16;
    int const n32 = n32_begin + warp;
    const Byte* lane_values =
        src + ((k16 * packed_n32 + n32) * kWarpSize + lane) * 16;
    quantized = *reinterpret_cast<const int4*>(lane_values);
    // Every lane uses the same address.  NVCC emits a uniform cached load;
    // the one-time repack keeps the 512-byte lane records separately aligned.
    scale = __ldg(scales + k16 * packed_n32 + n32);
  }

  __device__ void Advance() { k16_begin += kK16; }
};

template <int Bits, int CtaN, int CtaK, int Threads>
struct TmExl3Int6GmemIteratorB {
  static constexpr tm::Striding kMode = tm::Striding::kIndexed;
  static constexpr int kK16 = CtaK / 16;
  static constexpr int kWarps = Threads / kWarpSize;
  static constexpr int kN32 = CtaN / 32;
  static constexpr int kN32PerWarp = kN32 / kWarps;
  static constexpr int kWordsPerLane = 3;
  static constexpr int kScaleGroups = 8;
  static constexpr int kDataStageWords =
      kK16 * kN32PerWarp * kWarpSize * kWordsPerLane;
  static constexpr int kScaleStageWords =
      (kK16 * kN32PerWarp * kScaleGroups + 1) / 2;
  static constexpr int kStageWords =
      (kDataStageWords + kScaleStageWords + 3) & ~3;
  static constexpr int kWordsPerThread =
      kK16 * kN32PerWarp * kWordsPerLane;
  static_assert(Threads == 128 && CtaN == 128 && CtaK == 64);
  static_assert(kN32PerWarp == 1);

  struct Fragments {
    uint32_t words[kWordsPerThread];
    half scale;
  };

  const uint32_t* src = nullptr;
  const half* scales = nullptr;
  int packed_n32 = 0;
  int n32_begin = 0;
  int k16_begin = 0;
  bool g_mask = true;

  __device__ TmExl3Int6GmemIteratorB() {}

  __device__ TmExl3Int6GmemIteratorB(const tm::MatrixData& mat,
                                     int2 offset, int2 /*extent*/)
      : src(reinterpret_cast<const uint32_t*>(mat.ptr.ptr)),
        scales(reinterpret_cast<const half*>(mat.idxs)),
        packed_n32(mat.ptr.stride),
        n32_begin(offset.x / 32),
        k16_begin(offset.y / 16) {}

  __device__ __forceinline__ void Fetch(Fragments& fragments,
                                         bool tile_mask) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
#pragma unroll
    for (int local_k16 = 0; local_k16 < kK16; ++local_k16) {
      int const n32 = n32_begin + warp;
      uint32_t const* tile =
          src + ((k16_begin + local_k16) * packed_n32 + n32) *
                    kWordsPerLane * kWarpSize;
      int const base = local_k16 * kWordsPerLane;
#pragma unroll
      for (int word = 0; word < kWordsPerLane; ++word) {
        fragments.words[base + word] =
            tile_mask && g_mask ? tile[word * kWarpSize + lane] : 0u;
      }
    }
    int const scale_k16 = lane / kScaleGroups;
    int const scale_group = lane % kScaleGroups;
    fragments.scale = tile_mask && g_mask
                          ? __ldg(scales +
                                  ((k16_begin + scale_k16) * packed_n32 +
                                   n32_begin + warp) *
                                      kScaleGroups +
                                  scale_group)
                          : __float2half(1.0f);
  }

  __device__ __forceinline__ void Store(
      Fragments const& fragments, uint32_t* stage_words) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    uint32_t* warp_words = stage_words + warp * kStageWords;
#pragma unroll
    for (int local_k16 = 0; local_k16 < kK16; ++local_k16) {
      int const source = local_k16 * kWordsPerLane;
#pragma unroll
      for (int word = 0; word < kWordsPerLane; ++word) {
        warp_words[(local_k16 * kWordsPerLane + word) * kWarpSize + lane] =
            fragments.words[source + word];
      }
    }
    reinterpret_cast<half*>(warp_words + kDataStageWords)[lane] =
        fragments.scale;
  }

  __device__ __forceinline__ int scale_group_for_lane() const {
    int const lane = threadIdx.x % kWarpSize;
    int const effective_n =
        (lane / 16) * 4 + (lane & 12) * 2 + lane % 4;
    return effective_n / 4;
  }

  __device__ __forceinline__ void LoadK16FromShared(
      uint32_t const* stage_words, int local_k16,
      uint32_t& word0, uint32_t& word1, uint32_t& word2,
      half& scale) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    uint32_t const* warp_words = stage_words + warp * kStageWords;
    int const base = local_k16 * kWordsPerLane;
    word0 = warp_words[(base + 0) * kWarpSize + lane];
    word1 = warp_words[(base + 1) * kWarpSize + lane];
    word2 = warp_words[(base + 2) * kWarpSize + lane];
    scale = reinterpret_cast<half const*>(
        warp_words + kDataStageWords)[local_k16 * kScaleGroups +
                                      scale_group_for_lane()];
  }

  __device__ __forceinline__ void LoadK16(
      int local_k16, uint32_t& word0, uint32_t& word1, uint32_t& word2,
      half& scale) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    int const k16 = k16_begin + local_k16;
    int const n32 = n32_begin + warp;
    uint32_t const* tile =
        src + (k16 * packed_n32 + n32) * kWordsPerLane * kWarpSize;
    word0 = tile[0 * kWarpSize + lane];
    word1 = tile[1 * kWarpSize + lane];
    word2 = tile[2 * kWarpSize + lane];
    scale = __ldg(scales + (k16 * packed_n32 + n32) * kScaleGroups +
                            scale_group_for_lane());
  }

  __device__ void Advance() { k16_begin += kK16; }
};

template <int Bits, int CtaN, int CtaK, int Threads>
struct TmExl3OperandB {
  using Dtype = uint16_t;
  using GmemIter = TmExl3GmemIteratorB<Bits, CtaN, CtaK, Threads>;

  static constexpr tm::Pack kPack = 0;
  static constexpr tm::Order kOrder = tm::kRowMajor;
  static constexpr int kGroupSize = 1;
};

template <int Bits, int CtaN, int CtaK, int Threads>
struct TmExl3StateOperandB {
  using Dtype = uint32_t;
  using GmemIter =
      TmExl3StateGmemIteratorB<Bits, CtaN, CtaK, Threads>;

  static constexpr tm::Pack kPack = 0;
  static constexpr tm::Order kOrder = tm::kRowMajor;
  static constexpr int kGroupSize = 1;
};

template <int Bits, int CtaN, int CtaK, int Threads>
struct TmExl3Int8OperandB {
  using Dtype = int8_t;
  using GmemIter =
      TmExl3Int8GmemIteratorB<Bits, CtaN, CtaK, Threads>;

  static constexpr tm::Pack kPack = 0;
  static constexpr tm::Order kOrder = tm::kRowMajor;
  static constexpr int kGroupSize = 1;
};

template <int Bits, int CtaN, int CtaK, int Threads>
struct TmExl3E4m3OperandB {
  using Dtype = uint8_t;
  using GmemIter =
      TmExl3Int8GmemIteratorB<Bits, CtaN, CtaK, Threads, true>;

  static constexpr tm::Pack kPack = 0;
  static constexpr tm::Order kOrder = tm::kRowMajor;
  static constexpr int kGroupSize = 1;
};

template <int Bits, int CtaN, int CtaK, int Threads>
struct TmExl3Int6OperandB {
  using Dtype = uint32_t;
  using GmemIter =
      TmExl3Int6GmemIteratorB<Bits, CtaN, CtaK, Threads>;

  static constexpr tm::Pack kPack = 0;
  static constexpr tm::Order kOrder = tm::kRowMajor;
  static constexpr int kGroupSize = 1;
};

template <int Bits, int CtaN, int CtaK, int Threads>
struct TmExl3Int10GmemIteratorB {
  static constexpr tm::Striding kMode = tm::Striding::kIndexed;
  static constexpr int kK16 = CtaK / 16;
  static constexpr int kWarps = Threads / kWarpSize;
  static constexpr int kN32 = CtaN / 32;
  static constexpr int kN32PerWarp = kN32 / kWarps;
  static constexpr int kWordsPerLane = 5;
  static constexpr int kScaleGroups = 8;
  static constexpr int kDataStageWords =
      kK16 * kN32PerWarp * kWarpSize * kWordsPerLane;
  static constexpr int kScaleStageWords =
      (kK16 * kN32PerWarp * kScaleGroups + 1) / 2;
  static constexpr int kStageWords =
      (kDataStageWords + kScaleStageWords + 3) & ~3;
  static constexpr int kWordsPerThread =
      kK16 * kN32PerWarp * kWordsPerLane;
  static_assert(Threads == 128 && CtaN == 128 && CtaK == 64);
  static_assert(kN32PerWarp == 1);

  struct Fragments {
    uint32_t words[kWordsPerThread];
    half scale;
  };

  const uint32_t* src = nullptr;
  const half* scales = nullptr;
  int packed_n32 = 0;
  int n32_begin = 0;
  int k16_begin = 0;
  bool g_mask = true;

  __device__ TmExl3Int10GmemIteratorB() {}

  __device__ TmExl3Int10GmemIteratorB(const tm::MatrixData& mat,
                                      int2 offset, int2 /*extent*/)
      : src(reinterpret_cast<const uint32_t*>(mat.ptr.ptr)),
        scales(reinterpret_cast<const half*>(mat.idxs)),
        packed_n32(mat.ptr.stride),
        n32_begin(offset.x / 32),
        k16_begin(offset.y / 16) {}

  __device__ __forceinline__ void Fetch(Fragments& fragments,
                                         bool tile_mask) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
#pragma unroll
    for (int local_k16 = 0; local_k16 < kK16; ++local_k16) {
      int const n32 = n32_begin + warp;
      uint32_t const* tile =
          src + ((k16_begin + local_k16) * packed_n32 + n32) *
                    kWordsPerLane * kWarpSize;
      int const base = local_k16 * kWordsPerLane;
#pragma unroll
      for (int word = 0; word < kWordsPerLane; ++word) {
        fragments.words[base + word] =
            tile_mask && g_mask ? tile[word * kWarpSize + lane] : 0u;
      }
    }
    int const scale_k16 = lane / kScaleGroups;
    int const scale_group = lane % kScaleGroups;
    fragments.scale = tile_mask && g_mask
                          ? __ldg(scales +
                                  ((k16_begin + scale_k16) * packed_n32 +
                                   n32_begin + warp) *
                                      kScaleGroups +
                                  scale_group)
                          : __float2half(1.0f);
  }

  __device__ __forceinline__ void Store(
      Fragments const& fragments, uint32_t* stage_words) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    uint32_t* warp_words = stage_words + warp * kStageWords;
#pragma unroll
    for (int local_k16 = 0; local_k16 < kK16; ++local_k16) {
      int const source = local_k16 * kWordsPerLane;
#pragma unroll
      for (int word = 0; word < kWordsPerLane; ++word) {
        warp_words[(local_k16 * kWordsPerLane + word) * kWarpSize + lane] =
            fragments.words[source + word];
      }
    }
    reinterpret_cast<half*>(warp_words + kDataStageWords)[lane] =
        fragments.scale;
  }

  __device__ __forceinline__ int scale_group_for_lane() const {
    int const lane = threadIdx.x % kWarpSize;
    int const effective_n =
        (lane / 16) * 4 + (lane & 12) * 2 + lane % 4;
    return effective_n / 4;
  }

  __device__ __forceinline__ void LoadK16FromShared(
      uint32_t const* stage_words, int local_k16, uint32_t& word0,
      uint32_t& word1, uint32_t& word2, uint32_t& word3, uint32_t& word4,
      half& scale) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    uint32_t const* warp_words = stage_words + warp * kStageWords;
    int const base = local_k16 * kWordsPerLane;
    word0 = warp_words[(base + 0) * kWarpSize + lane];
    word1 = warp_words[(base + 1) * kWarpSize + lane];
    word2 = warp_words[(base + 2) * kWarpSize + lane];
    word3 = warp_words[(base + 3) * kWarpSize + lane];
    word4 = warp_words[(base + 4) * kWarpSize + lane];
    scale = reinterpret_cast<half const*>(
        warp_words + kDataStageWords)[local_k16 * kScaleGroups +
                                      scale_group_for_lane()];
  }

  __device__ __forceinline__ void LoadK16(
      int local_k16, uint32_t& word0, uint32_t& word1, uint32_t& word2,
      uint32_t& word3, uint32_t& word4, half& scale) const {
    int const warp = threadIdx.x / kWarpSize;
    int const lane = threadIdx.x % kWarpSize;
    int const k16 = k16_begin + local_k16;
    int const n32 = n32_begin + warp;
    uint32_t const* tile =
        src + (k16 * packed_n32 + n32) * kWordsPerLane * kWarpSize;
    word0 = tile[0 * kWarpSize + lane];
    word1 = tile[1 * kWarpSize + lane];
    word2 = tile[2 * kWarpSize + lane];
    word3 = tile[3 * kWarpSize + lane];
    word4 = tile[4 * kWarpSize + lane];
    scale = __ldg(scales + (k16 * packed_n32 + n32) * kScaleGroups +
                            scale_group_for_lane());
  }

  __device__ void Advance() { k16_begin += kK16; }
};

template <int Bits, int CtaN, int CtaK, int Threads>
struct TmExl3Int10OperandB {
  using Dtype = uint32_t;
  using GmemIter =
      TmExl3Int10GmemIteratorB<Bits, CtaN, CtaK, Threads>;

  static constexpr tm::Pack kPack = 0;
  static constexpr tm::Order kOrder = tm::kRowMajor;
  static constexpr int kGroupSize = 1;
};

// NVCC 12 keeps TurboMind's K-fragment arrays in per-thread local memory when
// this loop is expressed with a run-time integer (384 bytes for A/data/B in
// the first prototype).  Force every K index into the type system so the
// arrays scalarize to registers just as they do in the qualified direct path.
template <int Index, int End, class Function>
__device__ __forceinline__ void exl3_tm_static_for(Function const& function) {
  if constexpr (Index < End) {
    function(std::integral_constant<int, Index>{});
    exl3_tm_static_for<Index + 1, End>(function);
  }
}

template <int Bits, int BMode, class MMA, class OperandA_,
          class IteratorA_, int Stages_, bool InterleaveStateDecode_>
struct TmExl3MainloopSm70 {
  using MMA_Atom = typename MMA::Atom;
  using MMA_Map = typename MMA::Map;
  using FragC = typename MMA_Atom::FragC[MMA::kMmaIterM][MMA::kMmaIterN];

  static constexpr int Stages = Stages_;
  static constexpr int CTA_M = MMA::M;
  static constexpr int CTA_N = MMA::N;
  static constexpr int CTA_K = MMA::K;
  static constexpr int WARPS = MMA::kThreadCount / kWarpSize;
  static constexpr auto kOpClass = MMA_Atom::kOpClass;
  static constexpr bool kPipelineStateB =
      BMode == 1 && EXL3_TM_STATE_SHARED != 0;
  static constexpr bool kPipelineRawB =
      BMode == 0 && EXL3_TM_B_SHARED != 0;
  static constexpr bool kStaggeredRawK5 =
      kPipelineRawB && Bits == 5 && CTA_N == 128 && CTA_K == 64 &&
      EXL3_TM_RAW_DIRECT_TO_SHARED != 0 &&
      EXL3_TM_RAW_STAGGERED_K5_PREFETCH != 0;
  static constexpr bool kPipelineInt8B =
      BMode == 2 && EXL3_TM_STATE_SHARED != 0;
  static constexpr bool kPipelineE4m3B =
      BMode == 3 && EXL3_TM_STATE_SHARED != 0;
  static constexpr bool kPipelineInt6B =
      BMode == 4 && EXL3_TM_STATE_SHARED != 0;
  static constexpr bool kPipelineInt10B =
      BMode == 5 && EXL3_TM_STATE_SHARED != 0;
  static constexpr bool kPipelineB =
      kPipelineStateB || kPipelineRawB || kPipelineInt8B ||
      kPipelineE4m3B || kPipelineInt6B || kPipelineInt10B;
  static constexpr bool kInterleaveStateDecode =
      kPipelineStateB && InterleaveStateDecode_;

  static_assert(Stages == 2, "EXL3 SM70 prototype uses two stages");
  static_assert(MMA::kAtomK == 1);
  static_assert(MMA::kMmaIterM == 1 &&
                    (MMA::kMmaIterN == 1 || MMA::kMmaIterN == 2),
                "EXL3 SM70 prototype expects one or two N fragments per "
                "warp");
  static_assert(MMA::kTileIterK == CTA_K / 8);
  static_assert(MMA::kTileIterK % 2 == 0,
                "EXL3 K16 decode requires paired K8 MMA iterations");

  using OperandA =
      tm::MakeOperand<OperandA_, IteratorA_, CTA_M, CTA_K, WARPS>;
  using OperandB = std::conditional_t<
      BMode == 1,
      TmExl3StateOperandB<Bits, CTA_N, CTA_K, MMA::kThreadCount>,
      std::conditional_t<
          BMode == 2,
          TmExl3Int8OperandB<Bits, CTA_N, CTA_K, MMA::kThreadCount>,
          std::conditional_t<
              BMode == 3,
              TmExl3E4m3OperandB<Bits, CTA_N, CTA_K,
                                  MMA::kThreadCount>,
              std::conditional_t<
                  BMode == 4,
                  TmExl3Int6OperandB<Bits, CTA_N, CTA_K,
                                     MMA::kThreadCount>,
                  std::conditional_t<
                      BMode == 5,
                      TmExl3Int10OperandB<Bits, CTA_N, CTA_K,
                                          MMA::kThreadCount>,
                      TmExl3OperandB<Bits, CTA_N, CTA_K,
                                      MMA::kThreadCount>>>>>>;
  using OperandU = tm::MakeOperand<tm::VoidOperand, IteratorA_, CTA_M, CTA_K,
                                   WARPS>;
  using OperandV = tm::MakeOperand<tm::VoidOperand, IteratorA_, CTA_N, CTA_K,
                                   WARPS, 128>;

  using Ta = typename OperandA::Dtype;
  using Tb = typename OperandB::Dtype;
  using Tu = typename OperandU::Dtype;
  using Tv = typename OperandV::Dtype;
  using SmemLayoutA = typename OperandA::SmemLayout;
  using SmemCopyA =
      tm::SmemCopy<OperandA, MMA_Map::kIterM, MMA_Map::kIterK,
                   MMA_Map::kDeltaM, MMA_Map::kDeltaK>;
  using GmemIterA = typename OperandA::GmemIter;
  using GmemIterB = typename OperandB::GmemIter;
  using RawGmemIterB =
      TmExl3GmemIteratorB<Bits, CTA_N, CTA_K, MMA::kThreadCount>;
  using GmemIterU = typename OperandU::GmemIter;
  using GmemIterV = typename OperandV::GmemIter;
  static constexpr int kSmemBStageWords =
      kPipelineB ? WARPS * GmemIterB::kStageWords : 1;

  struct SharedStorage {
    alignas(16) turbomind::Array<Ta, Stages * SmemLayoutA::kSize> A;
    alignas(16) turbomind::Array<
        uint32_t, Stages * kSmemBStageWords> B;
  };

  __device__ void operator()(GmemIterA& gmem_A, GmemIterB& gmem_B,
                             GmemIterU& /*gmem_U*/,
                             GmemIterV& /*gmem_V*/, FragC& frag_C,
                             int tile_iter, SharedStorage& storage) {
    typename MMA_Atom::FragA frag_A[MMA::kTileIterK][MMA::kMmaIterM];
    typename MMA_Atom::FragB frag_B[MMA::kTileIterK][MMA::kMmaIterN];
    typename SmemCopyA::Frag data_A[MMA::kTileIterK];
    typename GmemIterA::Fragments rmem_A;
    typename GmemIterB::Fragments rmem_B;
    uint32_t raw_pending[5];
    bool raw_fetch_mask = false;
    RawGmemIterB* raw_gmem_B =
        reinterpret_cast<RawGmemIterB*>(&gmem_B);

    tm::SmemIter<turbomind::get_pointer_type<Ta>, SmemLayoutA::kSize, Stages>
        smem_A{storage.A.data()};
    tm::SmemIter<uint32_t*, kSmemBStageWords, Stages> smem_B{
        storage.B.data()};
    uint32_t* smem_B_store = nullptr;

    auto advance_smem_stage = [&] {
      gmem_A.smem_data_ = smem_A.pointer;
      smem_A.Advance();
      if constexpr (kPipelineB) {
        smem_B_store = smem_B.pointer;
        smem_B.Advance();
      }
    };

    for (int stage = 0; stage < Stages; ++stage) {
      advance_smem_stage();
      gmem_A.ClearSmem();
    }
    __syncthreads();

    int gmem_iter = tile_iter;
    bool gmem_mask = true;
    auto fetch_stage = [&] {
      gmem_A.Fetch(rmem_A, gmem_mask);
      gmem_A.Advance();
      if constexpr (kPipelineB) {
#if EXL3_TM_RAW_DIRECT_TO_SHARED
        if constexpr (kPipelineRawB) {
          gmem_B.FetchToShared(smem_B_store, gmem_mask);
          gmem_B.Advance();
        } else {
          gmem_B.Fetch(rmem_B, gmem_mask);
          gmem_B.Advance();
        }
#else
        gmem_B.Fetch(rmem_B, gmem_mask);
        gmem_B.Advance();
#endif
      }
      if (--gmem_iter == 0) {
        gmem_mask = false;
      }
    };
    auto store_stage = [&] {
      gmem_A.Store(rmem_A);
      if constexpr (kPipelineB) {
#if EXL3_TM_RAW_DIRECT_TO_SHARED
        if constexpr (!kPipelineRawB) {
          gmem_B.Store(rmem_B, smem_B_store);
        }
#else
        gmem_B.Store(rmem_B, smem_B_store);
#endif
      }
    };
    auto advance_and_wait_smem_stage = [&] {
      __syncthreads();
      advance_smem_stage();
    };
    auto fetch_next_stage = [&] {
      if constexpr (kStaggeredRawK5) {
        gmem_A.Fetch(rmem_A, gmem_mask);
        gmem_A.Advance();
        raw_fetch_mask = gmem_mask;
        raw_gmem_B->template FetchRange<0, 5>(raw_pending, raw_fetch_mask);
        if (--gmem_iter == 0) {
          gmem_mask = false;
        }
      } else {
        fetch_stage();
      }
    };

    int3 const offset_mnk = MMA::get_offset(threadIdx.x);
    SmemCopyA smem_copy_A{{offset_mnk.x, offset_mnk.z}};
    int const lane = threadIdx.x % kWarpSize;
    int const effective_n =
        (lane / 16) * 4 + (lane & 12) * 2 + lane % 4;
    int transform_dummy = 0;

    auto preload_a = [&](int k_iter) {
      smem_copy_A(smem_A.pointer, data_A[k_iter], k_iter);
    };
    auto transform_a = [&](int k_iter) {
      tm::Transform_Default::apply(frag_A, k_iter, data_A,
                                   transform_dummy, 1);
    };
    auto preload_b_pair = [&](int k_iter) {
      if ((k_iter & 1) != 0) {
        return;
      }
      exl3_tm_static_for<0, MMA::kMmaIterN>([&](auto n_tag) {
        constexpr int n_iter = decltype(n_tag)::value;
        Half4& b_frag0 =
            *reinterpret_cast<Half4*>(&frag_B[k_iter][n_iter][0]);
        Half4& b_frag1 =
            *reinterpret_cast<Half4*>(&frag_B[k_iter][n_iter][4]);
        Half4& b_frag2 =
            *reinterpret_cast<Half4*>(&frag_B[k_iter + 1][n_iter][0]);
        Half4& b_frag3 =
            *reinterpret_cast<Half4*>(&frag_B[k_iter + 1][n_iter][4]);
        if constexpr (BMode == 1) {
          TmExl3StateWords words;
          if constexpr (kPipelineStateB) {
            gmem_B.LoadK16FromShared(smem_B.pointer, k_iter / 2, n_iter,
                                     words);
          } else {
            gmem_B.LoadK16(k_iter / 2, n_iter, words);
          }
          reconstruct_mcg_lane_state_fragments<Bits>(
              words, b_frag0, b_frag1, b_frag2, b_frag3);
        } else if constexpr (BMode == 2) {
          static_assert(MMA::kMmaIterN == 1,
                        "EXL3 int8 experiment supports only CTA_N=128");
          int4 quantized;
          half scale_value;
          if constexpr (kPipelineInt8B) {
            gmem_B.LoadK16FromShared(smem_B.pointer, k_iter / 2, quantized,
                                     scale_value);
          } else {
            gmem_B.LoadK16(k_iter / 2, quantized, scale_value);
          }
          half2 const scale =
              __halves2half2(scale_value, scale_value);
          b_frag0 = tm_int8_word_to_half4<true>(
              static_cast<uint32_t>(quantized.x), scale);
          b_frag1 = tm_int8_word_to_half4<true>(
              static_cast<uint32_t>(quantized.y), scale);
          b_frag2 = tm_int8_word_to_half4<true>(
              static_cast<uint32_t>(quantized.z), scale);
          b_frag3 = tm_int8_word_to_half4<true>(
              static_cast<uint32_t>(quantized.w), scale);
        } else if constexpr (BMode == 3) {
          static_assert(MMA::kMmaIterN == 1,
                        "EXL3 E4M3 experiment supports only CTA_N=128");
          int4 quantized;
          half scale_value;
          if constexpr (kPipelineE4m3B) {
            gmem_B.LoadK16FromShared(smem_B.pointer, k_iter / 2, quantized,
                                     scale_value);
          } else {
            gmem_B.LoadK16(k_iter / 2, quantized, scale_value);
          }
          half2 const scale =
              __halves2half2(scale_value, scale_value);
          b_frag0 = tm_e4m3_word_to_half4<true>(
              static_cast<uint32_t>(quantized.x), scale);
          b_frag1 = tm_e4m3_word_to_half4<true>(
              static_cast<uint32_t>(quantized.y), scale);
          b_frag2 = tm_e4m3_word_to_half4<true>(
              static_cast<uint32_t>(quantized.z), scale);
          b_frag3 = tm_e4m3_word_to_half4<true>(
              static_cast<uint32_t>(quantized.w), scale);
        } else if constexpr (BMode == 4) {
          static_assert(MMA::kMmaIterN == 1,
                        "EXL3 int6 experiment supports only CTA_N=128");
          uint32_t word0;
          uint32_t word1;
          uint32_t word2;
          half scale_value;
          if constexpr (kPipelineInt6B) {
            gmem_B.LoadK16FromShared(smem_B.pointer, k_iter / 2,
                                     word0, word1, word2, scale_value);
          } else {
            gmem_B.LoadK16(k_iter / 2, word0, word1, word2, scale_value);
          }
          half2 const scale =
              __halves2half2(scale_value, scale_value);
          tm_int6_words_to_fragments(word0, word1, word2, scale,
                                     b_frag0, b_frag1, b_frag2, b_frag3);
        } else if constexpr (BMode == 5) {
          static_assert(MMA::kMmaIterN == 1,
                        "EXL3 int10 experiment supports only CTA_N=128");
          uint32_t word0;
          uint32_t word1;
          uint32_t word2;
          uint32_t word3;
          uint32_t word4;
          half scale_value;
          if constexpr (kPipelineInt10B) {
            gmem_B.LoadK16FromShared(smem_B.pointer, k_iter / 2, word0,
                                     word1, word2, word3, word4,
                                     scale_value);
          } else {
            gmem_B.LoadK16(k_iter / 2, word0, word1, word2, word3, word4,
                           scale_value);
          }
          half2 const scale =
              __halves2half2(scale_value, scale_value);
          tm_int10_words_to_fragments(word0, word1, word2, word3, word4,
                                      scale, b_frag0, b_frag1, b_frag2,
                                      b_frag3);
        } else {
          if constexpr (kPipelineRawB) {
            int const warp = threadIdx.x / kWarpSize;
            uint32_t const* packed0 =
                smem_B.pointer + warp * GmemIterB::kStageWords +
                ((k_iter / 2) * MMA::kMmaIterN + n_iter) * 2 *
                    GmemIterB::kPackedWords;
            uint32_t const* packed1 = packed0 + GmemIterB::kPackedWords;
#if EXL3_TM_DESTINATION_SHARED_DECODE
            reconstruct_mcg_native_n32_fragments_from_shared_destination<Bits>(
                packed0, packed1, b_frag0, b_frag1, b_frag2, b_frag3,
                effective_n);
#else
            reconstruct_mcg_native_n32_fragments_from_shared_scalar<Bits>(
                packed0, packed1, b_frag0, b_frag1, b_frag2, b_frag3, lane,
                effective_n);
#endif
          } else {
            uint32_t word0;
            uint32_t word1;
            uint32_t word2;
            uint32_t word3;
            gmem_B.LoadK16(k_iter / 2, n_iter, word0, word1, word2, word3);
            reconstruct_mcg_native_n32_fragments_from_words_scalar<Bits>(
                word0, word1, word2, word3, b_frag0, b_frag1, b_frag2,
                b_frag3, lane, effective_n);
          }
        }
      });
    };
    auto mma_current = [&](int k_iter) {
      if constexpr (MMA::kMmaIterN == 1) {
        MMA_Atom::fma(frag_C[0][0], frag_A[k_iter][0],
                      frag_B[k_iter][0], frag_C[0][0]);
      } else {
        MMA::mma_k_iter(frag_C, frag_A[k_iter], frag_B[k_iter], frag_C);
      }
    };
    advance_smem_stage();
    fetch_stage();
    store_stage();
    advance_and_wait_smem_stage();

    preload_a(0);
    transform_a(0);
    preload_b_pair(0);

    for (; tile_iter > 0; --tile_iter) {
      constexpr int IterK = MMA::kTileIterK;
      exl3_tm_static_for<0, IterK>([&](auto k_tag) {
        constexpr int k_iter = decltype(k_tag)::value;
        constexpr int next = (k_iter + 1) % IterK;
        if constexpr (kStaggeredRawK5 && k_iter == 3) {
          raw_gmem_B->template StoreRange<0, 5>(raw_pending, smem_B_store);
          raw_gmem_B->template FetchRange<5, 5>(raw_pending,
                                                raw_fetch_mask);
          raw_gmem_B->Advance();
        }
        if constexpr (k_iter == IterK - 1) {
          if constexpr (kStaggeredRawK5) {
            raw_gmem_B->template StoreRange<5, 5>(raw_pending,
                                                  smem_B_store);
          }
          store_stage();
          advance_and_wait_smem_stage();
          if constexpr (!kPipelineB) {
            if (tile_iter > 1) {
              gmem_B.Advance();
            }
          }
        }

        preload_a(next);
        if constexpr (kPipelineStateB) {
          if constexpr (kInterleaveStateDecode &&
                        MMA::kMmaIterN == 1 && (next & 1) == 0) {
            // Decode the low K8 half of the next B tile, issue the independent
            // current HMMA, then finish the high half.  This preserves the
            // exact state chain and accumulation order while giving ptxas an
            // HMMA-latency window in which to schedule the second decode half.
            bool const next_pair_valid =
                k_iter != IterK - 1 || tile_iter > 1;
            TmExl3StateWords next_words;
            McgContinuationStates next_states;
            Half4& next_b0 =
                *reinterpret_cast<Half4*>(&frag_B[next][0][0]);
            Half4& next_b1 =
                *reinterpret_cast<Half4*>(&frag_B[next][0][4]);
            Half4& next_b2 =
                *reinterpret_cast<Half4*>(&frag_B[next + 1][0][0]);
            Half4& next_b3 =
                *reinterpret_cast<Half4*>(&frag_B[next + 1][0][4]);
            if (next_pair_valid) {
              gmem_B.LoadK16FromShared(
                  smem_B.pointer, next / 2, 0, next_words);
              next_states = reconstruct_mcg_lane_state_fragments_low<Bits>(
                  next_words, next_b0, next_b1);
            }
            if constexpr (k_iter == 0) {
              fetch_next_stage();
            }
            mma_current(k_iter);
            if (next_pair_valid) {
              reconstruct_mcg_lane_state_fragments_high<Bits>(
                  next_words, next_states, next_b2, next_b3);
            }
          } else {
            if constexpr (k_iter != IterK - 1) {
              preload_b_pair(next);
            } else if (tile_iter > 1) {
              preload_b_pair(0);
            }
            if constexpr (k_iter == 0) {
              fetch_stage();
            }
            mma_current(k_iter);
          }
        } else if constexpr (
            kPipelineRawB && EXL3_TM_DESTINATION_SHARED_DECODE != 0 &&
            EXL3_TM_INTERLEAVE_DESTINATION_DECODE != 0 &&
            MMA::kMmaIterN == 1 && (next & 1) == 0) {
          // As with the state decoder above, construct the next raw-trellis B
          // pair around the independent HMMA.  Re-reading the compact shared
          // words is cheaper than keeping all four decoded Half4 fragments
          // live across the current matrix multiply.
          bool const next_pair_valid =
              k_iter != IterK - 1 || tile_iter > 1;
          Half4& next_b0 =
              *reinterpret_cast<Half4*>(&frag_B[next][0][0]);
          Half4& next_b1 =
              *reinterpret_cast<Half4*>(&frag_B[next][0][4]);
          Half4& next_b2 =
              *reinterpret_cast<Half4*>(&frag_B[next + 1][0][0]);
          Half4& next_b3 =
              *reinterpret_cast<Half4*>(&frag_B[next + 1][0][4]);
          int const warp = threadIdx.x / kWarpSize;
          uint32_t const* packed0 =
              smem_B.pointer + warp * RawGmemIterB::kStageWords +
              (next / 2) * 2 * RawGmemIterB::kPackedWords;
          uint32_t const* packed1 = packed0 + RawGmemIterB::kPackedWords;
          if (next_pair_valid) {
            reconstruct_mcg_native_n32_destination_half_from_shared<Bits,
                                                                     false>(
                packed0, packed1, next_b0, next_b1, effective_n);
          }
          if constexpr (k_iter == 0) {
            fetch_next_stage();
          }
          mma_current(k_iter);
          if (next_pair_valid) {
            reconstruct_mcg_native_n32_destination_half_from_shared<Bits,
                                                                     true>(
                packed0, packed1, next_b2, next_b3, effective_n);
          }
        } else {
          if constexpr (k_iter != IterK - 1) {
            preload_b_pair(next);
          } else if (tile_iter > 1) {
            preload_b_pair(0);
          }
          if constexpr (k_iter == 0) {
            fetch_next_stage();
          }
          mma_current(k_iter);
        }
        transform_a(next);
      });
    }
    __syncthreads();
  }
};

// Instantiate the same decode tile used by TurboMind's selected M=1 FP8
// kernel.  The only intentional difference is the B operand: FP8+scale
// transform is replaced by direct EXL3 trellis prefetch and register decode.
using TmSm70Exl3MmaMap =
    tm::MMA_Map<8, 128, 64, 8, 32, 8,
                tm::Blocked<1, 4, tm::kColMajor>, 1>;
using TmSm70Exl3Mma =
    tm::Tiled_MMA_v2<tm::SM70_MMA_884, TmSm70Exl3MmaMap>;
using TmSm70Exl3IteratorA =
    tm::IteratorSm70<tm::Striding::kIndexed,
                     turbomind::cache_policy::Default>;

template <int Bits, int BMode = 0, bool InterleaveStateDecode = false>
using TmSm70Exl3Mainloop =
    TmExl3MainloopSm70<Bits, BMode, TmSm70Exl3Mma,
                       tm::sm70_s884::Operand_A<half>,
                       TmSm70Exl3IteratorA, 2, InterleaveStateDecode>;

using TmSm70Exl3Epilogue =
    tm::Epilogue_<half, 8, 128, 8, 128,
                  TmSm70Exl3Mma::kThreadCount,
                  tm::Rearrange<TmSm70Exl3Mma>,
                  tm::sm70_s884::Operand_C<float, tm::kRowMajor>,
                  tm::Striding::kBlocked, true>;
using TmSm70Exl3AccumEpilogue =
    tm::Epilogue_<float, 8, 128, 8, 128,
                  TmSm70Exl3Mma::kThreadCount,
                  tm::Rearrange<TmSm70Exl3Mma>,
                  tm::sm70_s884::Operand_C<float, tm::kRowMajor>,
                  tm::Striding::kBlocked, true>;
using TmSm70Exl3Scheduler =
    tm::SchedulerSm70<tm::kColMajor, 8, 128, 64, 128, true, 0>;
// Group along N so one copied TurboMind grid services the two independent
// gate/up matrices.  Scheduler offsets keep every tile branch-local while
// group_id selects that branch's A, B, output, partial, and scale storage.
using TmSm70Exl3GateUpScheduler =
    tm::SchedulerSm70<tm::kColMajor, 8, 128, 64, 128, true, 1>;

// TurboMind registers both N128 and N256 CTA shapes for its SM70 FP8
// decode kernel.  Keep the qualified N128 path as the production default and
// expose N256 independently so real EXL3 shards can decide whether halving
// the CTA count and reusing A across two N32 fragments offsets the additional
// B-state/register pressure.
using TmSm70Exl3MmaMapN256 =
    tm::MMA_Map<8, 256, 64, 8, 32, 8,
                tm::Blocked<1, 4, tm::kColMajor>, 1>;
using TmSm70Exl3MmaN256 =
    tm::Tiled_MMA_v2<tm::SM70_MMA_884, TmSm70Exl3MmaMapN256>;
template <int Bits>
using TmSm70Exl3MainloopN256 =
    TmExl3MainloopSm70<Bits, 1, TmSm70Exl3MmaN256,
                       tm::sm70_s884::Operand_A<half>,
                       TmSm70Exl3IteratorA, 2, false>;
using TmSm70Exl3EpilogueN256 =
    tm::Epilogue_<half, 8, 256, 8, 128,
                  TmSm70Exl3MmaN256::kThreadCount,
                  tm::Rearrange<TmSm70Exl3MmaN256>,
                  tm::sm70_s884::Operand_C<float, tm::kRowMajor>,
                  tm::Striding::kBlocked, true>;
using TmSm70Exl3SchedulerN256 =
    tm::SchedulerSm70<tm::kColMajor, 8, 256, 64, 128, true, 0>;

using TmSm70Exl3MmaMapK32 =
    tm::MMA_Map<8, 128, 32, 8, 32, 8,
                tm::Blocked<1, 4, tm::kColMajor>, 1>;
using TmSm70Exl3MmaK32 =
    tm::Tiled_MMA_v2<tm::SM70_MMA_884, TmSm70Exl3MmaMapK32>;
template <int Bits>
using TmSm70Exl3MainloopK32 =
    TmExl3MainloopSm70<Bits, 1, TmSm70Exl3MmaK32,
                       tm::sm70_s884::Operand_A<half>,
                       TmSm70Exl3IteratorA, 2, false>;
using TmSm70Exl3EpilogueK32 =
    tm::Epilogue_<half, 8, 128, 8, 128,
                  TmSm70Exl3MmaK32::kThreadCount,
                  tm::Rearrange<TmSm70Exl3MmaK32>,
                  tm::sm70_s884::Operand_C<float, tm::kRowMajor>,
                  tm::Striding::kBlocked, true>;
using TmSm70Exl3SchedulerK32 =
    tm::SchedulerSm70<tm::kColMajor, 8, 128, 32, 128, true, 0>;

template <int Bits, int BMode = 0, bool FloatOutput = false,
          bool InterleaveStateDecode = false>
using TmSm70Exl3Gemm =
    tm::GemmUniversal<
        tm::Sm70,
        TmSm70Exl3Mainloop<Bits, BMode, InterleaveStateDecode>,
                      std::conditional_t<FloatOutput,
                                         TmSm70Exl3AccumEpilogue,
                                         TmSm70Exl3Epilogue>,
                      TmSm70Exl3Scheduler>;

template <int Bits, int BMode = 0, bool FloatOutput = false,
          bool InterleaveStateDecode = false>
void launch_tm_sm70_exl3_core_out(torch::Tensor const& out,
                                  torch::Tensor const& x_had,
                                  torch::Tensor const& weights,
                                  torch::Tensor const& partials,
                                  torch::Tensor const& locks, int splits,
                                  int swizzle,
                                  torch::Tensor const& scales = {}) {
  using Gemm =
      TmSm70Exl3Gemm<Bits, BMode, FloatOutput, InterleaveStateDecode>;

  int64_t const m = x_had.size(0);
  int64_t const k = x_had.size(1);
  int64_t const n = weights.size(1) * (BMode == 0 ? 16 : 32);

  tm::MatrixParam b_param{weights.data_ptr(),
                          static_cast<int>(weights.size(1)), nullptr,
                          nullptr, nullptr};
  if constexpr (BMode == 2) {
    b_param.idxs = reinterpret_cast<int*>(scales.data_ptr<at::Half>());
  }

  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), static_cast<int>(x_had.stride(0)),
                      nullptr, nullptr, nullptr},
      b_param,
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  epilogue.c =
      tm::MatrixParam{out.data_ptr(), static_cast<int>(out.stride(0)),
                      nullptr, nullptr, nullptr};
  epilogue.partials =
      tm::MatrixParam{partials.defined() ? partials.data_ptr() : nullptr,
                      static_cast<int>(n), nullptr, nullptr, nullptr};
  epilogue.locks =
      locks.defined() ? locks.data_ptr<int32_t>() : nullptr;
  epilogue.combine_mat =
      tm::MatrixCombination_v3{tm::MatrixParam{}, 1.0f, 0.0f};

  TmSm70Exl3Scheduler scheduler{
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1},
      swizzle, splits};
  scheduler.offsets_ = nullptr;

  auto const grid = scheduler.get_grid_shape();
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3Scheduler>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Bits>
void launch_tm_sm70_exl3_state_core_out_n256(
    torch::Tensor const& out, torch::Tensor const& x_had,
    torch::Tensor const& state, torch::Tensor const& partials,
    torch::Tensor const& locks, int splits, int swizzle) {
  using Gemm = tm::GemmUniversal<tm::Sm70,
                                 TmSm70Exl3MainloopN256<Bits>,
                                 TmSm70Exl3EpilogueN256,
                                 TmSm70Exl3SchedulerN256>;

  int64_t const m = x_had.size(0);
  int64_t const k = x_had.size(1);
  int64_t const n = state.size(1) * 32;
  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), static_cast<int>(x_had.stride(0)),
                      nullptr, nullptr, nullptr},
      tm::MatrixParam{state.data_ptr(), static_cast<int>(state.size(1)),
                      nullptr, nullptr, nullptr},
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  epilogue.c =
      tm::MatrixParam{out.data_ptr(), static_cast<int>(out.stride(0)),
                      nullptr, nullptr, nullptr};
  epilogue.partials =
      tm::MatrixParam{partials.data_ptr(), static_cast<int>(n), nullptr,
                      nullptr, nullptr};
  epilogue.locks = locks.data_ptr<int32_t>();
  epilogue.combine_mat =
      tm::MatrixCombination_v3{tm::MatrixParam{}, 1.0f, 0.0f};

  TmSm70Exl3SchedulerN256 scheduler{
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1},
      swizzle, splits};
  scheduler.offsets_ = nullptr;
  auto const grid = scheduler.get_grid_shape();
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3SchedulerN256>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Bits>
void launch_tm_sm70_exl3_state_core_out_k32(
    torch::Tensor const& out, torch::Tensor const& x_had,
    torch::Tensor const& state, torch::Tensor const& partials,
    torch::Tensor const& locks, int splits, int swizzle) {
  using Gemm = tm::GemmUniversal<tm::Sm70,
                                 TmSm70Exl3MainloopK32<Bits>,
                                 TmSm70Exl3EpilogueK32,
                                 TmSm70Exl3SchedulerK32>;
  int64_t const m = x_had.size(0);
  int64_t const k = x_had.size(1);
  int64_t const n = state.size(1) * 32;
  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), static_cast<int>(x_had.stride(0)),
                      nullptr, nullptr, nullptr},
      tm::MatrixParam{state.data_ptr(), static_cast<int>(state.size(1)),
                      nullptr, nullptr, nullptr},
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  epilogue.c =
      tm::MatrixParam{out.data_ptr(), static_cast<int>(out.stride(0)),
                      nullptr, nullptr, nullptr};
  epilogue.partials =
      tm::MatrixParam{partials.data_ptr(), static_cast<int>(n), nullptr,
                      nullptr, nullptr};
  epilogue.locks = locks.data_ptr<int32_t>();
  epilogue.combine_mat =
      tm::MatrixCombination_v3{tm::MatrixParam{}, 1.0f, 0.0f};
  TmSm70Exl3SchedulerK32 scheduler{
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1},
      swizzle, splits};
  scheduler.offsets_ = nullptr;
  auto const grid = scheduler.get_grid_shape();
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3SchedulerK32>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Bits>
torch::Tensor launch_tm_sm70_exl3_core(torch::Tensor const& x_had,
                                       torch::Tensor const& trellis) {
  int64_t const m = x_had.size(0);
  int64_t const n = trellis.size(1) * 16;
  auto out = torch::empty({m, n},
                          x_had.options().dtype(at::ScalarType::Half));
  launch_tm_sm70_exl3_core_out<Bits>(out, x_had, trellis, torch::Tensor{},
                                     torch::Tensor{}, 1, 0);
  return out;
}

torch::Tensor exl3_sm70_tm_core(torch::Tensor const& x_had,
                                torch::Tensor const& trellis) {
  c10::cuda::CUDAGuard device_guard(x_had.device());
  TORCH_CHECK(x_had.is_cuda() && trellis.is_cuda(),
              "TurboMind EXL3 core tensors must be CUDA tensors");
  TORCH_CHECK(x_had.scalar_type() == at::ScalarType::Half &&
                  x_had.dim() == 2 && x_had.is_contiguous(),
              "TurboMind EXL3 core input must be contiguous rank-2 float16");
  TORCH_CHECK(trellis.scalar_type() == at::ScalarType::Short &&
                  trellis.dim() == 3 && trellis.is_contiguous(),
              "TurboMind EXL3 core trellis must be contiguous rank-3 int16");

  int64_t const m = x_had.size(0);
  int64_t const k = x_had.size(1);
  int64_t const n = trellis.size(1) * 16;
  int64_t const bits = trellis.size(2) / 16;
  TORCH_CHECK(m > 0 && m <= 8,
              "TurboMind EXL3 core requires 1 <= M <= 8, got ", m);
  TORCH_CHECK(k == trellis.size(0) * 16 && k % 64 == 0 && n % 128 == 0,
              "TurboMind EXL3 core requires compatible K64/N128 dimensions");
  TORCH_CHECK(trellis.size(2) == bits * 16,
              "TurboMind EXL3 core has an invalid trellis bit dimension");

  switch (bits) {
    case 4:
      return launch_tm_sm70_exl3_core<4>(x_had, trellis);
    case 5:
      return launch_tm_sm70_exl3_core<5>(x_had, trellis);
    case 6:
      return launch_tm_sm70_exl3_core<6>(x_had, trellis);
    default:
      TORCH_CHECK(false, "TurboMind EXL3 core supports K4/K5/K6, got K",
                  bits);
  }
}

void exl3_sm70_tm_core_out(torch::Tensor const& out,
                           torch::Tensor const& x_had,
                           torch::Tensor const& trellis,
                           torch::Tensor const& partials,
                           torch::Tensor const& locks, int64_t splits,
                           int64_t swizzle) {
  c10::cuda::CUDAGuard device_guard(x_had.device());
  int64_t const m = x_had.size(0);
  int64_t const k = x_had.size(1);
  int64_t const n = trellis.size(1) * 16;
  int64_t const bits = trellis.size(2) / 16;
  int64_t const tiles = ((m + 7) / 8) * (n / 128);
  TORCH_CHECK(x_had.is_cuda() && trellis.is_cuda() && out.is_cuda() &&
                  partials.is_cuda() && locks.is_cuda(),
              "TurboMind EXL3 split core tensors must be CUDA tensors");
  TORCH_CHECK(x_had.scalar_type() == at::ScalarType::Half &&
                  out.scalar_type() == at::ScalarType::Half &&
                  trellis.scalar_type() == at::ScalarType::Short &&
                  partials.scalar_type() == at::ScalarType::Float &&
                  locks.scalar_type() == at::ScalarType::Int,
              "TurboMind EXL3 split core tensor dtypes disagree");
  TORCH_CHECK(x_had.is_contiguous() && trellis.is_contiguous() &&
                  out.is_contiguous() && partials.is_contiguous() &&
                  locks.is_contiguous(),
              "TurboMind EXL3 split core tensors must be contiguous");
  TORCH_CHECK(m > 0 && m <= 8 && out.size(0) == m && out.size(1) == n &&
                  k == trellis.size(0) * 16 && k % 128 == 0 &&
                  n % 128 == 0,
              "TurboMind EXL3 split core tensor shapes disagree");
  TORCH_CHECK(partials.numel() >= 8 * n && locks.numel() >= tiles,
              "TurboMind EXL3 split core workspace is too small");
  TORCH_CHECK(splits >= 1 && splits <= k / 128 && swizzle >= 0 &&
                  swizzle <= 5,
              "TurboMind EXL3 split/swizzle is out of range");

  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_core_out<4>(out, x_had, trellis, partials,
                                       locks, splits, swizzle);
      return;
    case 5:
      launch_tm_sm70_exl3_core_out<5>(out, x_had, trellis, partials,
                                       locks, splits, swizzle);
      return;
    case 6:
      launch_tm_sm70_exl3_core_out<6>(out, x_had, trellis, partials,
                                       locks, splits, swizzle);
      return;
    default:
      TORCH_CHECK(false, "TurboMind EXL3 split core supports K4/K5/K6");
  }
}

template <int Bits>
torch::Tensor launch_tm_sm70_exl3_state_repack(
    torch::Tensor const& trellis) {
  constexpr int kStateWords =
      (4 * (16 + 3 * Bits) + 31) / 32;
  int64_t const k16_blocks = trellis.size(0);
  int64_t const packed_n16 = trellis.size(1);
  int64_t const n32_blocks = packed_n16 / 2;
  auto state =
      Bits == 6 && EXL3_TM_K6_DENSE_STATE != 0
          ? torch::empty({k16_blocks, n32_blocks, 136, 1},
                         trellis.options().dtype(at::ScalarType::Int))
          : torch::empty({k16_blocks, n32_blocks, kStateWords, kWarpSize},
                         trellis.options().dtype(at::ScalarType::Int));
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(trellis.get_device()).stream();
#if EXL3_TM_STATE_LUT
  sm70_exl3_init_mcg_decode_lut_kernel<<<128, 256, 0, stream>>>();
#endif
  dim3 const grid(static_cast<unsigned>(n32_blocks),
                  static_cast<unsigned>(k16_blocks));
  sm70_exl3_repack_tm_state_kernel<Bits><<<grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
      reinterpret_cast<uint32_t*>(state.data_ptr<int32_t>()),
      static_cast<int>(packed_n16));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return state;
}

torch::Tensor exl3_sm70_tm_state_repack(torch::Tensor const& trellis) {
  c10::cuda::CUDAGuard device_guard(trellis.device());
  TORCH_CHECK(trellis.is_cuda() &&
                  trellis.scalar_type() == at::ScalarType::Short &&
                  trellis.dim() == 3 && trellis.is_contiguous(),
              "TurboMind EXL3 state repack expects contiguous CUDA int16");
  int64_t const bits = trellis.size(2) / 16;
  TORCH_CHECK(trellis.size(0) > 0 && trellis.size(1) > 0 &&
                  trellis.size(1) % 2 == 0 &&
                  trellis.size(2) == bits * 16,
              "TurboMind EXL3 state repack tensor shape is invalid");
  switch (bits) {
    case 4:
      return launch_tm_sm70_exl3_state_repack<4>(trellis);
    case 5:
      return launch_tm_sm70_exl3_state_repack<5>(trellis);
    case 6:
      return launch_tm_sm70_exl3_state_repack<6>(trellis);
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 state repack supports K4/K5/K6");
  }
}

bool sm70_exl3_tm_state_shape_valid(torch::Tensor const& state,
                                    int64_t bits) {
  int64_t const state_words = (4 * (16 + 3 * bits) + 31) / 32;
  int64_t const expected_tile_words =
      bits == 6 && EXL3_TM_K6_DENSE_STATE != 0
          ? 136
          : state_words * kWarpSize;
  return state.dim() == 4 &&
         state.size(2) * state.size(3) == expected_tile_words;
}

__global__ void sm70_exl3_init_gate_up_metadata_kernel(
    tm::StridedPtr* table, void* gate_state, int gate_state_stride,
    void* up_state, int up_state_stride, void* gate_svh, int gate_svh_stride,
    void* up_svh, int up_svh_stride) {
  if (threadIdx.x != 0) {
    return;
  }
  table[0] = tm::StridedPtr{gate_state, gate_state_stride};
  table[1] = tm::StridedPtr{up_state, up_state_stride};
  table[2] = tm::StridedPtr{gate_svh, gate_svh_stride};
  table[3] = tm::StridedPtr{up_svh, up_svh_stride};
}

torch::Tensor exl3_sm70_tm_gate_up_metadata(
    torch::Tensor const& gate_state, torch::Tensor const& up_state,
    torch::Tensor const& gate_svh, torch::Tensor const& up_svh) {
  c10::cuda::CUDAGuard device_guard(gate_state.device());
  TORCH_CHECK(gate_state.is_cuda() && up_state.is_cuda() &&
                  gate_svh.is_cuda() && up_svh.is_cuda() &&
                  gate_state.device() == up_state.device() &&
                  gate_state.device() == gate_svh.device() &&
                  gate_state.device() == up_svh.device(),
              "TurboMind EXL3 paired metadata tensors must share one CUDA "
              "device");
  TORCH_CHECK(gate_state.scalar_type() == at::ScalarType::Int &&
                  up_state.scalar_type() == at::ScalarType::Int &&
                  gate_svh.scalar_type() == at::ScalarType::Half &&
                  up_svh.scalar_type() == at::ScalarType::Half &&
                  gate_state.dim() == 4 && up_state.dim() == 4 &&
                  gate_state.sizes() == up_state.sizes() &&
                  gate_svh.dim() == 1 && up_svh.dim() == 1 &&
                  gate_svh.numel() == up_svh.numel() &&
                  gate_state.size(1) * 32 == gate_svh.numel(),
              "TurboMind EXL3 paired metadata shapes disagree");

  auto metadata = torch::empty(
      {static_cast<int64_t>(4 * sizeof(tm::StridedPtr))},
      gate_state.options().dtype(at::ScalarType::Byte));
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(gate_state.get_device()).stream();
  sm70_exl3_init_gate_up_metadata_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<tm::StridedPtr*>(metadata.data_ptr<uint8_t>()),
      gate_state.data_ptr(), static_cast<int>(gate_state.size(1)),
      up_state.data_ptr(), static_cast<int>(up_state.size(1)),
      gate_svh.data_ptr(), static_cast<int>(gate_svh.numel()),
      up_svh.data_ptr(), static_cast<int>(up_svh.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return metadata;
}

torch::Tensor exl3_sm70_tm_raw_pair_metadata(
    torch::Tensor const& trellis0, torch::Tensor const& trellis1,
    torch::Tensor const& svh0, torch::Tensor const& svh1) {
  c10::cuda::CUDAGuard device_guard(trellis0.device());
  TORCH_CHECK(
      trellis0.is_cuda() && trellis1.is_cuda() && svh0.is_cuda() &&
          svh1.is_cuda() && trellis0.device() == trellis1.device() &&
          trellis0.device() == svh0.device() &&
          trellis0.device() == svh1.device(),
      "TurboMind EXL3 raw pair metadata tensors must share one CUDA device");
  TORCH_CHECK(
      trellis0.scalar_type() == at::ScalarType::Short &&
          trellis1.scalar_type() == at::ScalarType::Short &&
          svh0.scalar_type() == at::ScalarType::Half &&
          svh1.scalar_type() == at::ScalarType::Half &&
          trellis0.dim() == 3 && trellis0.sizes() == trellis1.sizes() &&
          trellis0.is_contiguous() && trellis1.is_contiguous() &&
          svh0.dim() == 1 && svh1.dim() == 1 &&
          svh0.numel() == svh1.numel() &&
          trellis0.size(1) * 16 == svh0.numel(),
      "TurboMind EXL3 raw pair metadata shapes disagree");

  auto metadata = torch::empty(
      {static_cast<int64_t>(4 * sizeof(tm::StridedPtr))},
      trellis0.options().dtype(at::ScalarType::Byte));
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(trellis0.get_device()).stream();
  sm70_exl3_init_gate_up_metadata_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<tm::StridedPtr*>(metadata.data_ptr<uint8_t>()),
      trellis0.data_ptr(), static_cast<int>(trellis0.size(1)),
      trellis1.data_ptr(), static_cast<int>(trellis1.size(1)), svh0.data_ptr(),
      static_cast<int>(svh0.numel()), svh1.data_ptr(),
      static_cast<int>(svh1.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return metadata;
}

__global__ void sm70_exl3_init_int8_pair_metadata_kernel(
    tm::StridedPtr* table, void* packed0, int packed0_stride, void* scales0,
    void* packed1, int packed1_stride, void* scales1, void* svh0,
    int svh0_stride, void* svh1, int svh1_stride) {
  if (threadIdx.x != 0) {
    return;
  }
  table[0] = tm::StridedPtr{packed0, packed0_stride};
  table[1] = tm::StridedPtr{packed1, packed1_stride};
  table[2] = tm::StridedPtr{svh0, svh0_stride};
  table[3] = tm::StridedPtr{svh1, svh1_stride};
  void** scale_table = reinterpret_cast<void**>(table + 4);
  scale_table[0] = scales0;
  scale_table[1] = scales1;
}

torch::Tensor exl3_sm70_tm_int8_pair_metadata(
    torch::Tensor const& packed0, torch::Tensor const& scales0,
    torch::Tensor const& packed1, torch::Tensor const& scales1,
    torch::Tensor const& svh0, torch::Tensor const& svh1) {
  c10::cuda::CUDAGuard device_guard(packed0.device());
  TORCH_CHECK(
      packed0.is_cuda() && scales0.is_cuda() && packed1.is_cuda() &&
          scales1.is_cuda() && svh0.is_cuda() && svh1.is_cuda() &&
          packed0.device() == scales0.device() &&
          packed0.device() == packed1.device() &&
          packed0.device() == scales1.device() &&
          packed0.device() == svh0.device() &&
          packed0.device() == svh1.device(),
      "TurboMind EXL3 INT8 pair metadata tensors must share one CUDA device");
  TORCH_CHECK(
      packed0.scalar_type() == at::ScalarType::Char &&
          packed1.scalar_type() == at::ScalarType::Char &&
          scales0.scalar_type() == at::ScalarType::Half &&
          scales1.scalar_type() == at::ScalarType::Half &&
          svh0.scalar_type() == at::ScalarType::Half &&
          svh1.scalar_type() == at::ScalarType::Half && packed0.dim() == 4 &&
          packed0.sizes() == packed1.sizes() && scales0.dim() == 2 &&
          scales0.sizes() == scales1.sizes() &&
          scales0.size(0) == packed0.size(0) &&
          scales0.size(1) == packed0.size(1) &&
          packed0.size(2) == kWarpSize && packed0.size(3) == 16 &&
          svh0.dim() == 1 && svh1.dim() == 1 &&
          svh0.numel() == svh1.numel() &&
          packed0.size(1) * 32 == svh0.numel(),
      "TurboMind EXL3 INT8 pair metadata shapes disagree");

  constexpr int64_t kMetadataBytes =
      4 * sizeof(tm::StridedPtr) + 2 * sizeof(void*);
  auto metadata =
      torch::empty({kMetadataBytes}, packed0.options().dtype(at::ScalarType::Byte));
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(packed0.get_device()).stream();
  sm70_exl3_init_int8_pair_metadata_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<tm::StridedPtr*>(metadata.data_ptr<uint8_t>()),
      packed0.data_ptr(), static_cast<int>(packed0.size(1)),
      scales0.data_ptr(), packed1.data_ptr(),
      static_cast<int>(packed1.size(1)), scales1.data_ptr(), svh0.data_ptr(),
      static_cast<int>(svh0.numel()), svh1.data_ptr(),
      static_cast<int>(svh1.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return metadata;
}

template <int Bits>
std::tuple<torch::Tensor, torch::Tensor>
launch_tm_sm70_exl3_int8_repack(torch::Tensor const& trellis) {
  int64_t const k16_blocks = trellis.size(0);
  int64_t const packed_n16 = trellis.size(1);
  int64_t const n32_blocks = packed_n16 / 2;
  auto packed_lane = torch::empty(
      {k16_blocks, n32_blocks, kWarpSize, 16},
      trellis.options().dtype(at::ScalarType::Char));
  auto tile_scales = torch::empty(
      {k16_blocks, n32_blocks},
      trellis.options().dtype(at::ScalarType::Half));
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(trellis.get_device()).stream();
  dim3 const grid(static_cast<unsigned>(n32_blocks),
                  static_cast<unsigned>(k16_blocks));
  sm70_exl3_repack_tm_int8_kernel<Bits><<<grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
      packed_lane.data_ptr<int8_t>(),
      reinterpret_cast<half*>(tile_scales.data_ptr<at::Half>()),
      static_cast<int>(packed_n16));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed_lane, tile_scales};
}

std::tuple<torch::Tensor, torch::Tensor> exl3_sm70_tm_int8_repack(
    torch::Tensor const& trellis) {
  c10::cuda::CUDAGuard device_guard(trellis.device());
  TORCH_CHECK(trellis.is_cuda() &&
                  trellis.scalar_type() == at::ScalarType::Short &&
                  trellis.dim() == 3 && trellis.is_contiguous(),
              "TurboMind EXL3 int8 repack expects contiguous CUDA int16");
  int64_t const bits = trellis.size(2) / 16;
  TORCH_CHECK(trellis.size(0) > 0 && trellis.size(1) > 0 &&
                  trellis.size(1) % 2 == 0 &&
                  trellis.size(2) == bits * 16,
              "TurboMind EXL3 int8 repack tensor shape is invalid");
  switch (bits) {
    case 4:
      return launch_tm_sm70_exl3_int8_repack<4>(trellis);
    case 5:
      return launch_tm_sm70_exl3_int8_repack<5>(trellis);
    case 6:
      return launch_tm_sm70_exl3_int8_repack<6>(trellis);
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 int8 repack supports K4/K5/K6");
  }
}

template <int Bits>
std::tuple<torch::Tensor, torch::Tensor>
launch_tm_sm70_exl3_int6_repack(torch::Tensor const& trellis) {
  int64_t const k16 = trellis.size(0);
  int64_t const n32 = trellis.size(1) / 2;
  auto packed_words = torch::empty(
      {k16, n32, 3, kWarpSize},
      trellis.options().dtype(at::ScalarType::Int));
  auto group_scales = torch::empty(
      {k16, n32, 8}, trellis.options().dtype(at::ScalarType::Half));
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(trellis.get_device()).stream();
  dim3 const grid(static_cast<unsigned>(n32),
                  static_cast<unsigned>(k16));
  sm70_exl3_repack_tm_int6_kernel<Bits><<<grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
      reinterpret_cast<uint32_t*>(packed_words.data_ptr<int32_t>()),
      reinterpret_cast<half*>(group_scales.data_ptr<at::Half>()),
      static_cast<int>(trellis.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed_words, group_scales};
}

std::tuple<torch::Tensor, torch::Tensor> exl3_sm70_tm_int6_repack(
    torch::Tensor const& trellis) {
  c10::cuda::CUDAGuard device_guard(trellis.device());
  TORCH_CHECK(trellis.is_cuda() &&
                  trellis.scalar_type() == at::ScalarType::Short &&
                  trellis.is_contiguous(),
              "TurboMind EXL3 int6 repack expects contiguous CUDA int16");
  int64_t const bits = trellis.size(2) / 16;
  TORCH_CHECK(trellis.dim() == 3 && trellis.size(0) > 0 &&
                  trellis.size(1) > 0 && trellis.size(1) % 2 == 0 &&
                  trellis.size(2) == bits * 16,
              "TurboMind EXL3 int6 repack tensor shape is invalid");
  switch (bits) {
    case 4:
      return launch_tm_sm70_exl3_int6_repack<4>(trellis);
    case 5:
      return launch_tm_sm70_exl3_int6_repack<5>(trellis);
    case 6:
      return launch_tm_sm70_exl3_int6_repack<6>(trellis);
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 int6 repack supports K4/K5/K6");
  }
}

template <int Bits>
std::tuple<torch::Tensor, torch::Tensor>
launch_tm_sm70_exl3_int10_repack(torch::Tensor const& trellis) {
  int64_t const k16 = trellis.size(0);
  int64_t const n32 = trellis.size(1) / 2;
  auto packed_words = torch::empty(
      {k16, n32, 5, kWarpSize},
      trellis.options().dtype(at::ScalarType::Int));
  auto group_scales = torch::empty(
      {k16, n32, 8}, trellis.options().dtype(at::ScalarType::Half));
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(trellis.get_device()).stream();
  dim3 const grid(static_cast<unsigned>(n32),
                  static_cast<unsigned>(k16));
  sm70_exl3_repack_tm_int10_kernel<Bits><<<grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
      reinterpret_cast<uint32_t*>(packed_words.data_ptr<int32_t>()),
      reinterpret_cast<half*>(group_scales.data_ptr<at::Half>()),
      static_cast<int>(trellis.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed_words, group_scales};
}

std::tuple<torch::Tensor, torch::Tensor> exl3_sm70_tm_int10_repack(
    torch::Tensor const& trellis) {
  c10::cuda::CUDAGuard device_guard(trellis.device());
  TORCH_CHECK(trellis.is_cuda() &&
                  trellis.scalar_type() == at::ScalarType::Short &&
                  trellis.is_contiguous(),
              "TurboMind EXL3 int10 repack expects contiguous CUDA int16");
  int64_t const bits = trellis.size(2) / 16;
  TORCH_CHECK(trellis.dim() == 3 && trellis.size(0) > 0 &&
                  trellis.size(1) > 0 && trellis.size(1) % 2 == 0 &&
                  trellis.size(2) == bits * 16,
              "TurboMind EXL3 int10 repack tensor shape is invalid");
  switch (bits) {
    case 4:
      return launch_tm_sm70_exl3_int10_repack<4>(trellis);
    case 5:
      return launch_tm_sm70_exl3_int10_repack<5>(trellis);
    case 6:
      return launch_tm_sm70_exl3_int10_repack<6>(trellis);
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 int10 repack supports K4/K5/K6");
  }
}

template <int Bits>
std::tuple<torch::Tensor, torch::Tensor>
launch_tm_sm70_exl3_e4m3_repack(torch::Tensor const& trellis) {
  int64_t const k16_blocks = trellis.size(0);
  int64_t const packed_n16 = trellis.size(1);
  int64_t const n32_blocks = packed_n16 / 2;
  auto packed_lane = torch::empty(
      {k16_blocks, n32_blocks, kWarpSize, 16},
      trellis.options().dtype(at::ScalarType::Byte));
  auto tile_scales = torch::empty(
      {k16_blocks, n32_blocks},
      trellis.options().dtype(at::ScalarType::Half));
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(trellis.get_device()).stream();
  dim3 const grid(static_cast<unsigned>(n32_blocks),
                  static_cast<unsigned>(k16_blocks));
  sm70_exl3_repack_tm_e4m3_kernel<Bits><<<grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
      packed_lane.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(tile_scales.data_ptr<at::Half>()),
      static_cast<int>(packed_n16));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed_lane, tile_scales};
}

std::tuple<torch::Tensor, torch::Tensor> exl3_sm70_tm_e4m3_repack(
    torch::Tensor const& trellis) {
  c10::cuda::CUDAGuard device_guard(trellis.device());
  TORCH_CHECK(trellis.is_cuda() &&
                  trellis.scalar_type() == at::ScalarType::Short &&
                  trellis.dim() == 3 && trellis.is_contiguous(),
              "TurboMind EXL3 E4M3 repack expects contiguous CUDA int16");
  int64_t const bits = trellis.size(2) / 16;
  TORCH_CHECK(trellis.size(0) > 0 && trellis.size(1) > 0 &&
                  trellis.size(1) % 2 == 0 &&
                  trellis.size(2) == bits * 16,
              "TurboMind EXL3 E4M3 repack tensor shape is invalid");
  switch (bits) {
    case 4:
      return launch_tm_sm70_exl3_e4m3_repack<4>(trellis);
    case 5:
      return launch_tm_sm70_exl3_e4m3_repack<5>(trellis);
    case 6:
      return launch_tm_sm70_exl3_e4m3_repack<6>(trellis);
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 E4M3 repack supports K4/K5/K6");
  }
}

void exl3_sm70_tm_state_core_out(
    torch::Tensor const& out, torch::Tensor const& x_had,
    torch::Tensor const& state, torch::Tensor const& partials,
    torch::Tensor const& locks, int64_t bits, int64_t splits,
    int64_t swizzle) {
  c10::cuda::CUDAGuard device_guard(x_had.device());
  int64_t const m = x_had.size(0);
  int64_t const k = state.size(0) * 16;
  int64_t const n = state.size(1) * 32;
  int64_t const tiles = ((m + 7) / 8) * (n / 128);
  TORCH_CHECK(x_had.is_cuda() && state.is_cuda() && out.is_cuda() &&
                  partials.is_cuda() && locks.is_cuda(),
              "TurboMind EXL3 state core tensors must be CUDA tensors");
  TORCH_CHECK(x_had.scalar_type() == at::ScalarType::Half &&
                  out.scalar_type() == at::ScalarType::Half &&
                  state.scalar_type() == at::ScalarType::Int &&
                  partials.scalar_type() == at::ScalarType::Float &&
                  locks.scalar_type() == at::ScalarType::Int,
              "TurboMind EXL3 state core tensor dtypes disagree");
  TORCH_CHECK(x_had.is_contiguous() && state.is_contiguous() &&
                  out.is_contiguous() && partials.is_contiguous() &&
                  locks.is_contiguous(),
              "TurboMind EXL3 state core tensors must be contiguous");
  TORCH_CHECK(sm70_exl3_tm_state_shape_valid(state, bits) &&
                  m > 0 && m <= 8 &&
                  x_had.dim() == 2 && x_had.size(1) == k &&
                  out.size(0) == m && out.size(1) == n &&
                  k % 128 == 0 && n % 128 == 0,
              "TurboMind EXL3 state core tensor shapes disagree");
  TORCH_CHECK(partials.numel() >= 8 * n && locks.numel() >= tiles,
              "TurboMind EXL3 state core workspace is too small");
  TORCH_CHECK(splits >= 1 && splits <= k / 128 && swizzle >= 0 &&
                  swizzle <= 5,
              "TurboMind EXL3 state split/swizzle is out of range");

  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_core_out<4, true>(
          out, x_had, state, partials, locks, splits, swizzle);
      return;
    case 5:
      launch_tm_sm70_exl3_core_out<5, true>(
          out, x_had, state, partials, locks, splits, swizzle);
      return;
    case 6:
      launch_tm_sm70_exl3_core_out<6, true>(
          out, x_had, state, partials, locks, splits, swizzle);
      return;
    default:
      TORCH_CHECK(false, "TurboMind EXL3 state core supports K4/K5/K6");
  }
}

void exl3_sm70_tm_state_core_out_n256(
    torch::Tensor const& out, torch::Tensor const& x_had,
    torch::Tensor const& state, torch::Tensor const& partials,
    torch::Tensor const& locks, int64_t bits, int64_t splits,
    int64_t swizzle) {
  c10::cuda::CUDAGuard device_guard(x_had.device());
  int64_t const m = x_had.size(0);
  int64_t const k = state.size(0) * 16;
  int64_t const n = state.size(1) * 32;
  int64_t const tiles = ((m + 7) / 8) * (n / 256);
  TORCH_CHECK(x_had.is_cuda() && state.is_cuda() && out.is_cuda() &&
                  partials.is_cuda() && locks.is_cuda() &&
                  x_had.scalar_type() == at::ScalarType::Half &&
                  out.scalar_type() == at::ScalarType::Half &&
                  state.scalar_type() == at::ScalarType::Int &&
                  partials.scalar_type() == at::ScalarType::Float &&
                  locks.scalar_type() == at::ScalarType::Int &&
                  x_had.is_contiguous() && state.is_contiguous() &&
                  out.is_contiguous() && partials.is_contiguous() &&
                  locks.is_contiguous(),
              "TurboMind EXL3 N256 state core tensors are invalid");
  TORCH_CHECK(sm70_exl3_tm_state_shape_valid(state, bits) && m > 0 &&
                  m <= 8 && x_had.dim() == 2 && x_had.size(1) == k &&
                  out.size(0) == m && out.size(1) == n && k % 128 == 0 &&
                  n % 256 == 0 && partials.numel() >= 8 * n &&
                  locks.numel() >= tiles && splits >= 1 &&
                  splits <= k / 128 && swizzle >= 0 && swizzle <= 5,
              "TurboMind EXL3 N256 state core shape/policy is invalid");
  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_state_core_out_n256<4>(
          out, x_had, state, partials, locks, splits, swizzle);
      return;
    case 5:
      launch_tm_sm70_exl3_state_core_out_n256<5>(
          out, x_had, state, partials, locks, splits, swizzle);
      return;
    case 6:
      launch_tm_sm70_exl3_state_core_out_n256<6>(
          out, x_had, state, partials, locks, splits, swizzle);
      return;
    default:
      TORCH_CHECK(false, "TurboMind EXL3 N256 core supports K4/K5/K6");
  }
}

void exl3_sm70_tm_state_core_out_k32(
    torch::Tensor const& out, torch::Tensor const& x_had,
    torch::Tensor const& state, torch::Tensor const& partials,
    torch::Tensor const& locks, int64_t bits, int64_t splits,
    int64_t swizzle) {
  c10::cuda::CUDAGuard device_guard(x_had.device());
  int64_t const m = x_had.size(0);
  int64_t const k = state.size(0) * 16;
  int64_t const n = state.size(1) * 32;
  int64_t const tiles = ((m + 7) / 8) * (n / 128);
  TORCH_CHECK(x_had.is_cuda() && state.is_cuda() && out.is_cuda() &&
                  partials.is_cuda() && locks.is_cuda() &&
                  x_had.scalar_type() == at::ScalarType::Half &&
                  out.scalar_type() == at::ScalarType::Half &&
                  state.scalar_type() == at::ScalarType::Int &&
                  partials.scalar_type() == at::ScalarType::Float &&
                  locks.scalar_type() == at::ScalarType::Int &&
                  x_had.is_contiguous() && state.is_contiguous() &&
                  out.is_contiguous() && partials.is_contiguous() &&
                  locks.is_contiguous(),
              "TurboMind EXL3 K32 state core tensors are invalid");
  TORCH_CHECK(sm70_exl3_tm_state_shape_valid(state, bits) && m > 0 &&
                  m <= 8 && x_had.dim() == 2 && x_had.size(1) == k &&
                  out.size(0) == m && out.size(1) == n && k % 128 == 0 &&
                  n % 128 == 0 && partials.numel() >= 8 * n &&
                  locks.numel() >= tiles && splits >= 1 &&
                  splits <= k / 128 && swizzle >= 0 && swizzle <= 5,
              "TurboMind EXL3 K32 state core shape/policy is invalid");
  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_state_core_out_k32<4>(
          out, x_had, state, partials, locks, splits, swizzle);
      return;
    case 5:
      launch_tm_sm70_exl3_state_core_out_k32<5>(
          out, x_had, state, partials, locks, splits, swizzle);
      return;
    case 6:
      launch_tm_sm70_exl3_state_core_out_k32<6>(
          out, x_had, state, partials, locks, splits, swizzle);
      return;
    default:
      TORCH_CHECK(false, "TurboMind EXL3 K32 core supports K4/K5/K6");
  }
}

template <int Bits>
torch::Tensor launch_tm_sm70_exl3_reconstruct(torch::Tensor const& trellis);

torch::Tensor exl3_sm70_tm_reconstruct(torch::Tensor const& trellis);

// Unreachable direct-fragment research helper retained only for isolated
// microbenchmarks; no registered operator dispatches to this experiment.
template <int Bits>
__device__ __forceinline__ half2 shuffle_mcg_fragment_value(
    FragB const& f0, FragB const& f1, int source_lane,
    int fragment_register) {
  // A Volta shuffle exposes one register per source lane.  Every destination
  // lane needs a different one of the four decoded half2 registers, so keep
  // all four shuffles converged and select after the exchange.
  half2 const f00 = __shfl_sync(0xffffffffu, f0.x[0], source_lane);
  half2 const f01 = __shfl_sync(0xffffffffu, f0.x[1], source_lane);
  half2 const f10 = __shfl_sync(0xffffffffu, f1.x[0], source_lane);
  half2 const f11 = __shfl_sync(0xffffffffu, f1.x[1], source_lane);
  return fragment_register == 0 ? f00
       : fragment_register == 1 ? f01
       : fragment_register == 2 ? f10
                                : f11;
}

template <int Bits>
__device__ __forceinline__ void reconstruct_mcg_b_fragment(
    const uint32_t* packed,
    wmma::fragment<wmma::matrix_b, kTileM, kTileK, kTileK, half,
                   wmma::row_major>& b_frag,
    int lane) {
  FragB const f0 = decode_mcg_four<Bits>(packed, lane * 8);
  FragB const f1 = decode_mcg_four<Bits>(packed, lane * 8 + 4);

  // Empirically derived on SM70 from wmma::load_matrix_sync.  For a row-major
  // B[16][16] tile, each lane owns four adjacent N values from K row k0 and
  // four from K row k0 + 4.  Lanes 8..15 and 24..31 duplicate the fragment
  // data needed by lanes 0..7 and 16..23 respectively.
  int const q = lane & 7;
  int const p = q >> 2;
  int const fragment_register = ((lane >> 4) << 1) | (q & 1);
  bool const take_high = ((q >> 1) & 1) != 0;

  half2 const r0n01 = shuffle_mcg_fragment_value<Bits>(
      f0, f1, p, fragment_register);
  half2 const r0n01_partner = shuffle_mcg_fragment_value<Bits>(
      f0, f1, p + 4, fragment_register);
  half2 const r0n23 = shuffle_mcg_fragment_value<Bits>(
      f0, f1, p + 8, fragment_register);
  half2 const r0n23_partner = shuffle_mcg_fragment_value<Bits>(
      f0, f1, p + 12, fragment_register);
  half2 const r1n01 = shuffle_mcg_fragment_value<Bits>(
      f0, f1, p + 2, fragment_register);
  half2 const r1n01_partner = shuffle_mcg_fragment_value<Bits>(
      f0, f1, p + 6, fragment_register);
  half2 const r1n23 = shuffle_mcg_fragment_value<Bits>(
      f0, f1, p + 10, fragment_register);
  half2 const r1n23_partner = shuffle_mcg_fragment_value<Bits>(
      f0, f1, p + 14, fragment_register);

  half2 const row0_lo = take_high
                            ? __halves2half2(__high2half(r0n01),
                                             __high2half(r0n01_partner))
                            : __halves2half2(__low2half(r0n01),
                                             __low2half(r0n01_partner));
  half2 const row0_hi = take_high
                            ? __halves2half2(__high2half(r0n23),
                                             __high2half(r0n23_partner))
                            : __halves2half2(__low2half(r0n23),
                                             __low2half(r0n23_partner));
  half2 const row1_lo = take_high
                            ? __halves2half2(__high2half(r1n01),
                                             __high2half(r1n01_partner))
                            : __halves2half2(__low2half(r1n01),
                                             __low2half(r1n01_partner));
  half2 const row1_hi = take_high
                            ? __halves2half2(__high2half(r1n23),
                                             __high2half(r1n23_partner))
                            : __halves2half2(__low2half(r1n23),
                                             __low2half(r1n23_partner));

  b_frag.x[0] = __low2half(row0_lo);
  b_frag.x[1] = __high2half(row0_lo);
  b_frag.x[2] = __low2half(row0_hi);
  b_frag.x[3] = __high2half(row0_hi);
  b_frag.x[4] = __low2half(row1_lo);
  b_frag.x[5] = __high2half(row1_lo);
  b_frag.x[6] = __low2half(row1_hi);
  b_frag.x[7] = __high2half(row1_hi);
}

template <int Bits>
__device__ __forceinline__ void reconstruct_mcg_tile_from_regs(
    uint32_t word0, uint32_t word1, half tile[kTileK][kTileK], int lane) {
  FragB const f0 =
      decode_mcg_four_from_regs<Bits>(word0, word1, lane * 8);
  FragB const f1 =
      decode_mcg_four_from_regs<Bits>(word0, word1, lane * 8 + 4);

  half2 const n0 = __shfl_down_sync(0xffffffffu, f0.x[0], 4);
  half2 const n1 = __shfl_down_sync(0xffffffffu, f0.x[1], 4);
  half2 const n2 = __shfl_down_sync(0xffffffffu, f1.x[0], 4);
  half2 const n3 = __shfl_down_sync(0xffffffffu, f1.x[1], 4);

  if ((lane & 4) == 0) {
    half2 const values[8] = {
        __halves2half2(__low2half(f0.x[0]), __low2half(n0)),
        __halves2half2(__high2half(f0.x[0]), __high2half(n0)),
        __halves2half2(__low2half(f0.x[1]), __low2half(n1)),
        __halves2half2(__high2half(f0.x[1]), __high2half(n1)),
        __halves2half2(__low2half(f1.x[0]), __low2half(n2)),
        __halves2half2(__high2half(f1.x[0]), __high2half(n2)),
        __halves2half2(__low2half(f1.x[1]), __low2half(n3)),
        __halves2half2(__high2half(f1.x[1]), __high2half(n3)),
    };
    int const r0 = (lane & 3) * 2;
    int const rows[8] = {r0, r0 + 1, r0 + 8, r0 + 9,
                         r0, r0 + 1, r0 + 8, r0 + 9};
    int const c0 = lane / 8;
    int const cols[8] = {c0, c0, c0, c0, c0 + 4, c0 + 4, c0 + 4, c0 + 4};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      *reinterpret_cast<half2*>(&tile[rows[i]][cols[i] * 2]) = values[i];
    }
  }
}

__device__ __forceinline__ void hadamard_shuffle4(float& h0, float& h1,
                                                   float& h2, float& h3,
                                                   int lane) {
#pragma unroll
  for (int mask = 1; mask < kWarpSize; mask <<= 1) {
    float const p0 = __shfl_xor_sync(0xffffffffu, h0, mask);
    float const p1 = __shfl_xor_sync(0xffffffffu, h1, mask);
    float const p2 = __shfl_xor_sync(0xffffffffu, h2, mask);
    float const p3 = __shfl_xor_sync(0xffffffffu, h3, mask);
    float const sign = (lane & mask) ? -1.0f : 1.0f;
    h0 = sign * h0 + p0;
    h1 = sign * h1 + p1;
    h2 = sign * h2 + p2;
    h3 = sign * h3 + p3;
  }
}

__device__ __forceinline__ void input_hadamard_128(
    const half* x, const half* scale, half* out, int lane) {
  Half4 const v = *reinterpret_cast<const Half4*>(x + lane * 4);
  Half4 const s = *reinterpret_cast<const Half4*>(scale + lane * 4);
  half2 const vx = __hmul2(v.x, s.x);
  half2 const vy = __hmul2(v.y, s.y);
  float const v0 = __half2float(__low2half(vx));
  float const v1 = __half2float(__high2half(vx));
  float const v2 = __half2float(__low2half(vy));
  float const v3 = __half2float(__high2half(vy));
  float const s0 = v0 + v1;
  float const d0 = v0 - v1;
  float const s1 = v2 + v3;
  float const d1 = v2 - v3;
  float h0 = s0 + s1;
  float h1 = d0 + d1;
  float h2 = s0 - s1;
  float h3 = d0 - d1;
  hadamard_shuffle4(h0, h1, h2, h3, lane);
  Half4 result{
      __floats2half2_rn(h0 * kHadamardScale, h1 * kHadamardScale),
      __floats2half2_rn(h2 * kHadamardScale, h3 * kHadamardScale),
  };
  *reinterpret_cast<Half4*>(out + lane * 4) = result;
}

__device__ __forceinline__ void output_hadamard_128(
    const float* in, half* out, const half* scale, int lane) {
  float4 v = *reinterpret_cast<const float4*>(in + lane * 4);
  float const s0 = v.x + v.y;
  float const d0 = v.x - v.y;
  float const s1 = v.z + v.w;
  float const d1 = v.z - v.w;
  float h0 = s0 + s1;
  float h1 = d0 + d1;
  float h2 = s0 - s1;
  float h3 = d0 - d1;
  hadamard_shuffle4(h0, h1, h2, h3, lane);

  Half4 const post = *reinterpret_cast<const Half4*>(scale + lane * 4);
  h0 *= kHadamardScale * __half2float(__low2half(post.x));
  h1 *= kHadamardScale * __half2float(__high2half(post.x));
  h2 *= kHadamardScale * __half2float(__low2half(post.y));
  h3 *= kHadamardScale * __half2float(__high2half(post.y));
  Half4 const result{__floats2half2_rn(h0, h1),
                     __floats2half2_rn(h2, h3)};
  *reinterpret_cast<Half4*>(out + lane * 4) = result;
}

__global__ void sm70_exl3_input_hadamard_kernel(
    const half* __restrict__ x, const half* __restrict__ suh,
    half* __restrict__ x_had, int m, int k) {
  int const row = blockIdx.y;
  int const k0 = blockIdx.x * kHadamard;
  half* dst = x_had + row * k + k0;
  if (row < m) {
    input_hadamard_128(x + row * k + k0, suh + k0, dst, threadIdx.x);
  } else {
    *reinterpret_cast<Half4*>(dst + threadIdx.x * 4) =
        Half4{__float2half2_rn(0.0f), __float2half2_rn(0.0f)};
  }
}

// Gate and up own distinct input-Hadamard vectors.  Materialize both fixed
// M<=8 branch buffers in one launch so TurboMind's grouped scheduler can
// consume them as two GEMMs without a host-visible launch boundary.
__global__ void sm70_exl3_input_hadamard_pair_kernel(
    const half* __restrict__ x, const half* __restrict__ gate_suh,
    const half* __restrict__ up_suh, half* __restrict__ x_had, int m,
    int k) {
  int const branch = blockIdx.z;
  int const row = blockIdx.y;
  int const k0 = blockIdx.x * kHadamard;
  half const* scale = branch == 0 ? gate_suh : up_suh;
  half* dst = x_had + (branch * 8 + row) * k + k0;
  input_hadamard_128(x + row * k + k0, scale + k0, dst, threadIdx.x);
}

__global__ void sm70_exl3_output_hadamard_kernel(
    const float* __restrict__ accum, half* __restrict__ out,
    const half* __restrict__ svh, int m, int n) {
  int const row = blockIdx.y;
  int const n0 = blockIdx.x * kHadamard;
  output_hadamard_128(accum + row * n + n0, out + row * n + n0, svh + n0,
                      threadIdx.x);
}

__global__ void sm70_exl3_output_hadamard_pair_kernel(
    const float* __restrict__ accum, half* __restrict__ out,
    const half* __restrict__ svh0, const half* __restrict__ svh1, int m,
    int n) {
  int const branch = blockIdx.z;
  int const row = blockIdx.y;
  int const n0 = blockIdx.x * kHadamard;
  const half* svh = branch == 0 ? svh0 : svh1;
  output_hadamard_128(accum + (branch * 8 + row) * n + n0,
                      out + row * (2 * n) + branch * n + n0, svh + n0,
                      threadIdx.x);
}

// The M8xN128 TurboMind epilogue map already gives each warp one output row
// at a time and each lane four consecutive FP32 columns.  That is exactly the
// register layout consumed by output_hadamard_128: a local H4 followed by an
// H32 warp exchange produces the complete H128 transform.  Fuse that work
// into the final split-K epilogue so the FP32 accumulator is never written to
// and reread from global memory.
struct TmSm70Exl3OutputHadamardEpilogue : TmSm70Exl3Epilogue {
  using Base = TmSm70Exl3Epilogue;
  using Dtype = typename Base::Dtype;
  using Tc = typename Base::Tc;
  using SharedStorage = typename Base::SharedStorage;

  static_assert(std::is_same_v<Dtype, float>);
  static_assert(std::is_same_v<Tc, half>);
  static_assert(Base::TM == 8 && Base::TN == 128);
  static_assert(Base::S == 2 && Base::C == 1 && Base::kAccess == 4);
  static_assert(Base::Map::kDeltaS == 4);

  template <class FragC>
  __device__ void operator()(FragC& frag_C, const int4& tile_offset,
                             const int2& extents, int splits, int tile_id,
                             bool is_last, const tm::EpilogueParam& param,
                             SharedStorage& storage) {
    int2 const cta_cs =
        tm::mk2cs<Base::kOrder>(tile_offset.x * 8, tile_offset.y * 128);
    int2 const end_cs = tm::mk2cs<Base::kOrder>(extents);

    typename Base::template OutputC<Dtype> tmp_C[Base::S][Base::C];
    this->Rearrange(frag_C, storage, tmp_C);

    tm::Predicate<Base::S, Base::C, false, false> pred{};
    int2 const thr_cs = Base::Map::get_offset(
        threadIdx.x / kWarpSize, threadIdx.x % kWarpSize);
    int2 const cs0 = {cta_cs.x + thr_cs.x, cta_cs.y + thr_cs.y};
#pragma unroll
    for (int s = 0; s < Base::S; ++s) {
#pragma unroll
      for (int c = 0; c < Base::C; ++c) {
        int const ss = thr_cs.y + s * Base::Map::kDeltaS;
        int const cc = thr_cs.x + c * Base::Map::kDeltaC;
        if (ss < end_cs.y && cc < end_cs.x) {
          pred.set(s, c);
        }
      }
    }

    // Preserve TurboMind's ordered split-K protocol.  Non-final split CTAs
    // publish their FP32 partial and return.  Only the final CTA reaches the
    // transform, after it has consumed every preceding partial, and it resets
    // the graph-persistent semaphore to zero before continuing.
    if (splits > 1) {
      int* barrier = &param.locks[tile_id];
      turbomind::sem_wait(barrier, tile_offset.z, threadIdx.x == 0);
      tm::MatrixData const partial =
          tm::resolve<Dtype, Base::kMode>(param.partials, tile_offset.w);
      this->Reduce(tmp_C, partial, tile_offset.z == 0, is_last, cs0, pred);
      int const post_id = is_last ? 0 : tile_offset.z + 1;
      turbomind::sem_post(barrier, post_id, threadIdx.x == 0);
      if (!is_last) {
        return;
      }
    }

    // This dedicated epilogue is instantiated only by the exact EXL3
    // projection launcher below.  Its combine operation is identity; the
    // otherwise-unused combine source pointer carries the immutable svh
    // vector without changing TurboMind's shared EpilogueParam ABI.
    // A normal projection supplies one flat scale vector.  The paired
    // gate/up path supplies a two-entry StridedPtr table and uses the
    // scheduler group id to select the correct immutable vector.
    tm::MatrixData const scale = tm::resolve<half, Base::kMode>(
        param.combine_mat.param_c, tile_offset.w);
    half const* svh = reinterpret_cast<half const*>(scale.ptr.ptr);
    // The ordinary path resolves one independent output matrix per group.
    // A same-shape Q/K or V/Z pair instead carries cumulative column offsets
    // in group_idxs and writes directly into one concatenated [M, 2N]
    // result.  That avoids a post-GEMM pack launch while leaving every MMA,
    // split-K partial, and FP16 rounding operation unchanged.
    bool const concat_groups = param.c.group_idxs != nullptr;
    tm::MatrixData output{};
    int output_column_offset = 0;
    if (concat_groups) {
      output.ptr = tm::StridedPtr{param.c.ptr, param.c.stride};
      output.idxs = nullptr;
      output_column_offset = __ldg(param.c.group_idxs + tile_offset.w);
    } else {
      output = tm::resolve<Tc, Base::kMode>(param.c, tile_offset.w);
    }
    int const lane = threadIdx.x % kWarpSize;

#pragma unroll
    for (int s = 0; s < Base::S; ++s) {
      if (!pred(s, 0)) {
        continue;
      }
      auto const& values = tmp_C[s][0];
      float const sum01 = values[0] + values[1];
      float const diff01 = values[0] - values[1];
      float const sum23 = values[2] + values[3];
      float const diff23 = values[2] - values[3];
      float h0 = sum01 + sum23;
      float h1 = diff01 + diff23;
      float h2 = sum01 - sum23;
      float h3 = diff01 - diff23;
      hadamard_shuffle4(h0, h1, h2, h3, lane);

      Half4 const post = *reinterpret_cast<Half4 const*>(svh + cs0.x);
      h0 *= kHadamardScale * __half2float(__low2half(post.x));
      h1 *= kHadamardScale * __half2float(__high2half(post.x));
      h2 *= kHadamardScale * __half2float(__low2half(post.y));
      h3 *= kHadamardScale * __half2float(__high2half(post.y));
      Half4 const result{__floats2half2_rn(h0, h1),
                         __floats2half2_rn(h2, h3)};
      half* destination = reinterpret_cast<half*>(output.ptr.ptr) +
                          (cs0.y + s * Base::Map::kDeltaS) *
                              output.ptr.stride +
                          output_column_offset + cs0.x;
      *reinterpret_cast<Half4*>(destination) = result;
    }

    // The final split-K CTA owns every output row in this N128 tile.  Publish
    // only after all fused-Hadamard FP16 stores are system-visible.  One
    // signal per N tile is sufficient for M<=4 because CTA_M is eight.
    if (param.tile_allreduce && is_last && tile_offset.x == 0 &&
        tile_offset.w == 0) {
      auto const& tile_ar = param.tile_allreduce_param;
      int const signal_id = tile_offset.y;
      if ((tile_ar.world_size == 2 || tile_ar.world_size == 4) &&
          signal_id >= 0 &&
          signal_id < vllm::sm70_tile_runtime::kMaxBlocks &&
          tile_ar.self_signal != nullptr) {
        __threadfence_system();
        __syncthreads();
        auto* self_signal =
            reinterpret_cast<vllm::sm70_tile_runtime::Signal*>(
                tile_ar.self_signal);
        auto const flag = self_signal->_flag[signal_id] + 1;
        if (threadIdx.x < tile_ar.world_size) {
          auto* peer_signal =
              reinterpret_cast<vllm::sm70_tile_runtime::Signal*>(
                  tile_ar.signals[threadIdx.x]);
          auto* peer_flag =
              &peer_signal->start[signal_id][tile_ar.rank];
          vllm::sm70_tile_runtime::store_flag_sys_visible(peer_flag, flag);
        }
      }
    }
  }
};

template <int Bits>
void launch_tm_sm70_exl3_raw_hadamard_core_out(
    torch::Tensor const& out, torch::Tensor const& x_had,
    torch::Tensor const& trellis, torch::Tensor const& svh,
    torch::Tensor const& partials, torch::Tensor const& locks, int splits,
    int swizzle) {
  // Keep TurboMind's SM70 FP16 mainloop, scheduler, split-K protocol, and
  // fused output-Hadamard epilogue intact.  BMode=0 changes only the operand-B
  // iterator/transform: compact EXL3 trellis tiles are copied to shared memory
  // and reconstructed directly into the HMMA fragments consumed by each lane.
  using Gemm = tm::GemmUniversal<
      tm::Sm70, TmSm70Exl3Mainloop<Bits, 0, false>,
      TmSm70Exl3OutputHadamardEpilogue, TmSm70Exl3Scheduler>;

  int64_t const m = x_had.size(0);
  int64_t const k = x_had.size(1);
  int64_t const n = trellis.size(1) * 16;
  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), static_cast<int>(x_had.stride(0)),
                      nullptr, nullptr, nullptr},
      tm::MatrixParam{trellis.data_ptr(), static_cast<int>(trellis.size(1)),
                      nullptr, nullptr, nullptr},
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  epilogue.c =
      tm::MatrixParam{out.data_ptr(), static_cast<int>(out.stride(0)),
                      nullptr, nullptr, nullptr};
  epilogue.partials =
      tm::MatrixParam{partials.data_ptr(), static_cast<int>(n), nullptr,
                      nullptr, nullptr};
  epilogue.locks = locks.data_ptr<int32_t>();
  epilogue.combine_mat = tm::MatrixCombination_v3{
      tm::MatrixParam{svh.data_ptr(), static_cast<int>(n), nullptr, nullptr,
                      nullptr},
      1.0f, 0.0f};

  TmSm70Exl3Scheduler scheduler{
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1},
      swizzle, splits};
  scheduler.offsets_ = nullptr;
  auto const grid = scheduler.get_grid_shape();
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3Scheduler>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Bits, bool InterleaveStateDecode = false>
void launch_tm_sm70_exl3_state_hadamard_core_out(
    torch::Tensor const& out, torch::Tensor const& x_had,
    torch::Tensor const& state, torch::Tensor const& svh,
    torch::Tensor const& partials, torch::Tensor const& locks, int splits,
    int swizzle,
    tm::TileAllReduceParam const* tile_reduce = nullptr) {
  using Gemm = tm::GemmUniversal<
      tm::Sm70, TmSm70Exl3Mainloop<Bits, 1, InterleaveStateDecode>,
      TmSm70Exl3OutputHadamardEpilogue, TmSm70Exl3Scheduler>;

  int64_t const m = x_had.size(0);
  int64_t const k = x_had.size(1);
  int64_t const n = state.size(1) * 32;

  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), static_cast<int>(x_had.stride(0)),
                      nullptr, nullptr, nullptr},
      tm::MatrixParam{state.data_ptr(), static_cast<int>(state.size(1)),
                      nullptr, nullptr, nullptr},
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  epilogue.c =
      tm::MatrixParam{out.data_ptr(), static_cast<int>(out.stride(0)),
                      nullptr, nullptr, nullptr};
  epilogue.partials =
      tm::MatrixParam{partials.data_ptr(), static_cast<int>(n), nullptr,
                      nullptr, nullptr};
  epilogue.locks = locks.data_ptr<int32_t>();
  epilogue.combine_mat =
      tm::MatrixCombination_v3{
          tm::MatrixParam{svh.data_ptr(), static_cast<int>(n), nullptr,
                          nullptr, nullptr},
          1.0f, 0.0f};
  if (tile_reduce != nullptr) {
    epilogue.tile_allreduce = true;
    epilogue.tile_allreduce_param = *tile_reduce;
  }

  TmSm70Exl3Scheduler scheduler{
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1},
      swizzle, splits};
  scheduler.offsets_ = nullptr;

  auto grid = scheduler.get_grid_shape();
  if (tile_reduce != nullptr) {
    auto& tile_ar = epilogue.tile_allreduce_param;
    tile_ar.producer_grid_x = static_cast<int>(grid.x);
    tile_ar.producer_grid_y = static_cast<int>(grid.y);
    tile_ar.producer_grid_z = static_cast<int>(grid.z);
    TORCH_CHECK(tile_ar.kernel_reducer_blocks > 0 &&
                    tile_ar.kernel_reducer_blocks <=
                        tile_ar.producer_grid_x * tile_ar.producer_grid_y &&
                    grid.z < 65535,
                "TurboMind EXL3 tile reducer grid is invalid");
    grid.z += 1;
  }
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3Scheduler>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Bits, int BMode = 2>
void launch_tm_sm70_exl3_int8_hadamard_core_out(
    torch::Tensor const& out, torch::Tensor const& x_had,
    torch::Tensor const& packed_lane, torch::Tensor const& tile_scales,
    torch::Tensor const& svh, torch::Tensor const& partials,
    torch::Tensor const& locks, int splits, int swizzle) {
  using Gemm = tm::GemmUniversal<
      tm::Sm70, TmSm70Exl3Mainloop<Bits, BMode, false>,
      TmSm70Exl3OutputHadamardEpilogue, TmSm70Exl3Scheduler>;

  int64_t const m = x_had.size(0);
  int64_t const k = x_had.size(1);
  int64_t const n = packed_lane.size(1) * 32;
  tm::MatrixParam b_param{
      packed_lane.data_ptr(), static_cast<int>(packed_lane.size(1)),
      nullptr, reinterpret_cast<int*>(tile_scales.data_ptr<at::Half>()),
      nullptr};
  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), static_cast<int>(x_had.stride(0)),
                      nullptr, nullptr, nullptr},
      b_param,
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  epilogue.c =
      tm::MatrixParam{out.data_ptr(), static_cast<int>(out.stride(0)),
                      nullptr, nullptr, nullptr};
  epilogue.partials =
      tm::MatrixParam{partials.data_ptr(), static_cast<int>(n), nullptr,
                      nullptr, nullptr};
  epilogue.locks = locks.data_ptr<int32_t>();
  epilogue.combine_mat = tm::MatrixCombination_v3{
      tm::MatrixParam{svh.data_ptr(), static_cast<int>(n), nullptr, nullptr,
                      nullptr},
      1.0f, 0.0f};

  TmSm70Exl3Scheduler scheduler{
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1},
      swizzle, splits};
  scheduler.offsets_ = nullptr;
  auto const grid = scheduler.get_grid_shape();
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3Scheduler>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Bits>
void launch_tm_sm70_exl3_e4m3_hadamard_core_out(
    torch::Tensor const& out, torch::Tensor const& x_had,
    torch::Tensor const& packed_lane, torch::Tensor const& tile_scales,
    torch::Tensor const& svh, torch::Tensor const& partials,
    torch::Tensor const& locks, int splits, int swizzle) {
  using Gemm = tm::GemmUniversal<
      tm::Sm70, TmSm70Exl3Mainloop<Bits, 3, false>,
      TmSm70Exl3OutputHadamardEpilogue, TmSm70Exl3Scheduler>;

  int64_t const m = x_had.size(0);
  int64_t const k = x_had.size(1);
  int64_t const n = packed_lane.size(1) * 32;
  tm::MatrixParam b_param{
      packed_lane.data_ptr(), static_cast<int>(packed_lane.size(1)),
      nullptr, reinterpret_cast<int*>(tile_scales.data_ptr<at::Half>()),
      nullptr};
  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), static_cast<int>(x_had.stride(0)),
                      nullptr, nullptr, nullptr},
      b_param,
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  epilogue.c =
      tm::MatrixParam{out.data_ptr(), static_cast<int>(out.stride(0)),
                      nullptr, nullptr, nullptr};
  epilogue.partials =
      tm::MatrixParam{partials.data_ptr(), static_cast<int>(n), nullptr,
                      nullptr, nullptr};
  epilogue.locks = locks.data_ptr<int32_t>();
  epilogue.combine_mat = tm::MatrixCombination_v3{
      tm::MatrixParam{svh.data_ptr(), static_cast<int>(n), nullptr, nullptr,
                      nullptr},
      1.0f, 0.0f};

  TmSm70Exl3Scheduler scheduler{
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1},
      swizzle, splits};
  scheduler.offsets_ = nullptr;
  auto const grid = scheduler.get_grid_shape();
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3Scheduler>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Bits, bool InterleaveStateDecode = false>
void launch_tm_sm70_exl3_state_gate_up_core_out(
    torch::Tensor const& projected, torch::Tensor const& x_had,
    torch::Tensor const& metadata, torch::Tensor const& offsets,
    torch::Tensor const& partials, torch::Tensor const& locks, int m, int k,
    int n, int splits, int swizzle) {
  using Gemm = tm::GemmUniversal<
      tm::Sm70, TmSm70Exl3Mainloop<Bits, 1, InterleaveStateDecode>,
      TmSm70Exl3OutputHadamardEpilogue, TmSm70Exl3GateUpScheduler>;

  int* n_offsets = offsets.data_ptr<int32_t>();
  int* row_offsets = n_offsets + 3;
  char* metadata_ptr = reinterpret_cast<char*>(metadata.data_ptr<uint8_t>());
  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), k, row_offsets, nullptr, nullptr},
      // A zero stride tells TurboMind resolve() that ptr is a two-entry
      // StridedPtr table indexed by scheduler group_id.
      tm::MatrixParam{metadata_ptr, 0, nullptr, nullptr, nullptr},
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  epilogue.c = tm::MatrixParam{projected.data_ptr(), n, row_offsets, nullptr,
                               nullptr};
  epilogue.partials = tm::MatrixParam{partials.data_ptr(), n, row_offsets,
                                      nullptr, nullptr};
  epilogue.locks = locks.data_ptr<int32_t>();
  epilogue.combine_mat = tm::MatrixCombination_v3{
      tm::MatrixParam{metadata_ptr + 2 * sizeof(tm::StridedPtr), 0, nullptr,
                      nullptr, nullptr},
      1.0f, 0.0f};

  TmSm70Exl3GateUpScheduler scheduler{
      // The constructor needs the total grouped extent to size the grid.
      // offsets_ replaces it with each branch's local N inside init().
      {m, 2 * n, k, 2}, swizzle, splits};
  scheduler.offsets_ = n_offsets;

  auto const grid = scheduler.get_grid_shape();
  TORCH_CHECK(locks.numel() >=
                  static_cast<int64_t>(grid.x) *
                      static_cast<int64_t>(grid.y),
              "TurboMind EXL3 paired gate/up lock workspace is too small");
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3GateUpScheduler>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Bits>
void launch_tm_sm70_exl3_raw_gate_up_core_out(
    torch::Tensor const& projected, torch::Tensor const& x_had,
    torch::Tensor const& metadata, torch::Tensor const& offsets,
    torch::Tensor const& partials, torch::Tensor const& locks, int m, int k,
    int n, int splits, int swizzle) {
  using Gemm = tm::GemmUniversal<
      tm::Sm70, TmSm70Exl3Mainloop<Bits, 0, false>,
      TmSm70Exl3OutputHadamardEpilogue, TmSm70Exl3GateUpScheduler>;

  int* n_offsets = offsets.data_ptr<int32_t>();
  int* row_offsets = n_offsets + 3;
  char* metadata_ptr = reinterpret_cast<char*>(metadata.data_ptr<uint8_t>());
  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), k, row_offsets, nullptr, nullptr},
      // resolve<kBlocked>() loads the branch-specific raw-trellis pointer and
      // packed-N16 stride from this two-entry StridedPtr table.
      tm::MatrixParam{metadata_ptr, 0, nullptr, nullptr, nullptr},
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  epilogue.c = tm::MatrixParam{projected.data_ptr(), n, row_offsets, nullptr,
                               nullptr};
  epilogue.partials = tm::MatrixParam{partials.data_ptr(), n, row_offsets,
                                      nullptr, nullptr};
  epilogue.locks = locks.data_ptr<int32_t>();
  epilogue.combine_mat = tm::MatrixCombination_v3{
      tm::MatrixParam{metadata_ptr + 2 * sizeof(tm::StridedPtr), 0, nullptr,
                      nullptr, nullptr},
      1.0f, 0.0f};

  TmSm70Exl3GateUpScheduler scheduler{{m, 2 * n, k, 2}, swizzle, splits};
  scheduler.offsets_ = n_offsets;
  auto const grid = scheduler.get_grid_shape();
  TORCH_CHECK(locks.numel() >=
                  static_cast<int64_t>(grid.x) *
                      static_cast<int64_t>(grid.y),
              "TurboMind EXL3 raw gate/up lock workspace is too small");
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3GateUpScheduler>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// INT8-repacked counterpart of the exact gate/up launcher above.  The
// scheduler and epilogue deliberately retain the gate/up row-offset layout:
// the fused SiLU boundary consumes two fixed eight-row workspaces rather than
// the adjacent-column layout used by ordinary QKV pairs.  Only the B operand
// resolver changes, selecting the persistent lane-packed weights and their
// per-tile scale pointer for each scheduler group.
template <int Bits>
void launch_tm_sm70_exl3_int8_gate_up_core_out(
    torch::Tensor const& projected, torch::Tensor const& x_had,
    torch::Tensor const& metadata, torch::Tensor const& offsets,
    torch::Tensor const& partials, torch::Tensor const& locks, int m, int k,
    int n, int splits, int swizzle) {
  using Gemm = tm::GemmUniversal<
      tm::Sm70, TmSm70Exl3Mainloop<Bits, 2, false>,
      TmSm70Exl3OutputHadamardEpilogue, TmSm70Exl3GateUpScheduler>;

  int* n_offsets = offsets.data_ptr<int32_t>();
  int* row_offsets = n_offsets + 3;
  char* metadata_ptr = reinterpret_cast<char*>(metadata.data_ptr<uint8_t>());
  int* scale_pointer_table = reinterpret_cast<int*>(
      metadata_ptr + 4 * sizeof(tm::StridedPtr));
  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), k, row_offsets, nullptr, nullptr},
      tm::MatrixParam{metadata_ptr, 0, nullptr, scale_pointer_table, nullptr},
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  epilogue.c = tm::MatrixParam{projected.data_ptr(), n, row_offsets, nullptr,
                               nullptr};
  epilogue.partials = tm::MatrixParam{partials.data_ptr(), n, row_offsets,
                                      nullptr, nullptr};
  epilogue.locks = locks.data_ptr<int32_t>();
  epilogue.combine_mat = tm::MatrixCombination_v3{
      tm::MatrixParam{metadata_ptr + 2 * sizeof(tm::StridedPtr), 0, nullptr,
                      nullptr, nullptr},
      1.0f, 0.0f};

  TmSm70Exl3GateUpScheduler scheduler{{m, 2 * n, k, 2}, swizzle, splits};
  scheduler.offsets_ = n_offsets;
  auto const grid = scheduler.get_grid_shape();
  TORCH_CHECK(locks.numel() >=
                  static_cast<int64_t>(grid.x) *
                      static_cast<int64_t>(grid.y),
              "TurboMind EXL3 INT8 gate/up lock workspace is too small");
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3GateUpScheduler>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Execute two equal-shape projections in one TurboMind scheduler grid and
// write them directly as adjacent column ranges.  The two branches retain
// their established split/swizzle policy, independent state/SUH/SVH data,
// and split-K semaphore ranges.  This is the exact building block for Q/K,
// V/Z, and K/V pairs in Qwen3.8's packed projection modules.
template <int Bits, bool InterleaveStateDecode = false>
void launch_tm_sm70_exl3_state_pair_core_out(
    torch::Tensor const& out, torch::Tensor const& x_had,
    torch::Tensor const& metadata, torch::Tensor const& offsets,
    torch::Tensor const& partials, torch::Tensor const& locks, int m, int k,
    int n, int splits, int swizzle) {
  using Gemm = tm::GemmUniversal<
      tm::Sm70, TmSm70Exl3Mainloop<Bits, 1, InterleaveStateDecode>,
      TmSm70Exl3OutputHadamardEpilogue, TmSm70Exl3GateUpScheduler>;

  int* n_offsets = offsets.data_ptr<int32_t>();
  int* input_row_offsets = n_offsets + 3;
  int* partial_row_offsets = input_row_offsets + 3;
  char* metadata_ptr = reinterpret_cast<char*>(metadata.data_ptr<uint8_t>());
  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), k, input_row_offsets, nullptr,
                      nullptr},
      tm::MatrixParam{metadata_ptr, 0, nullptr, nullptr, nullptr},
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  // group_idxs is intentionally repurposed as a read-only cumulative column
  // table by TmSm70Exl3OutputHadamardEpilogue.  Other launchers leave it null.
  epilogue.c = tm::MatrixParam{out.data_ptr(), 2 * n, nullptr, nullptr,
                               n_offsets};
  epilogue.partials = tm::MatrixParam{partials.data_ptr(), n,
                                      partial_row_offsets, nullptr, nullptr};
  epilogue.locks = locks.data_ptr<int32_t>();
  epilogue.combine_mat = tm::MatrixCombination_v3{
      tm::MatrixParam{metadata_ptr + 2 * sizeof(tm::StridedPtr), 0, nullptr,
                      nullptr, nullptr},
      1.0f, 0.0f};

  TmSm70Exl3GateUpScheduler scheduler{{m, 2 * n, k, 2}, swizzle, splits};
  scheduler.offsets_ = n_offsets;

  auto const grid = scheduler.get_grid_shape();
  TORCH_CHECK(locks.numel() >=
                  static_cast<int64_t>(grid.x) *
                      static_cast<int64_t>(grid.y),
              "TurboMind EXL3 paired projection lock workspace is too small");
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3GateUpScheduler>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Bits>
void launch_tm_sm70_exl3_raw_pair_core_out(
    torch::Tensor const& out, torch::Tensor const& x_had,
    torch::Tensor const& metadata, torch::Tensor const& offsets,
    torch::Tensor const& partials, torch::Tensor const& locks, int m, int k,
    int n, int splits, int swizzle) {
  using Gemm = tm::GemmUniversal<
      tm::Sm70, TmSm70Exl3Mainloop<Bits, 0, false>,
      TmSm70Exl3OutputHadamardEpilogue, TmSm70Exl3GateUpScheduler>;

  int* n_offsets = offsets.data_ptr<int32_t>();
  int* input_row_offsets = n_offsets + 3;
  int* partial_row_offsets = input_row_offsets + 3;
  char* metadata_ptr = reinterpret_cast<char*>(metadata.data_ptr<uint8_t>());
  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), k, input_row_offsets, nullptr,
                      nullptr},
      tm::MatrixParam{metadata_ptr, 0, nullptr, nullptr, nullptr},
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  epilogue.c = tm::MatrixParam{out.data_ptr(), 2 * n, nullptr, nullptr,
                               n_offsets};
  epilogue.partials = tm::MatrixParam{partials.data_ptr(), n,
                                      partial_row_offsets, nullptr, nullptr};
  epilogue.locks = locks.data_ptr<int32_t>();
  epilogue.combine_mat = tm::MatrixCombination_v3{
      tm::MatrixParam{metadata_ptr + 2 * sizeof(tm::StridedPtr), 0, nullptr,
                      nullptr, nullptr},
      1.0f, 0.0f};

  TmSm70Exl3GateUpScheduler scheduler{{m, 2 * n, k, 2}, swizzle, splits};
  scheduler.offsets_ = n_offsets;
  auto const grid = scheduler.get_grid_shape();
  TORCH_CHECK(locks.numel() >=
                  static_cast<int64_t>(grid.x) *
                      static_cast<int64_t>(grid.y),
              "TurboMind EXL3 raw paired-projection lock workspace is too "
              "small");
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3GateUpScheduler>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Bits, bool PairLayout>
void launch_tm_sm70_exl3_raw_grouped_accum_core_out(
    torch::Tensor const& accum, torch::Tensor const& x_had,
    torch::Tensor const& metadata, torch::Tensor const& offsets,
    torch::Tensor const& partials, torch::Tensor const& locks, int m, int k,
    int n, int splits, int swizzle) {
  using Gemm = tm::GemmUniversal<
      tm::Sm70, TmSm70Exl3Mainloop<Bits, 0, false>,
      TmSm70Exl3AccumEpilogue, TmSm70Exl3GateUpScheduler>;

  int* n_offsets = offsets.data_ptr<int32_t>();
  int* input_row_offsets = n_offsets + 3;
  int* output_row_offsets = n_offsets + (PairLayout ? 6 : 3);
  char* metadata_ptr = reinterpret_cast<char*>(metadata.data_ptr<uint8_t>());
  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), k, input_row_offsets, nullptr,
                      nullptr},
      tm::MatrixParam{metadata_ptr, 0, nullptr, nullptr, nullptr},
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  epilogue.c = tm::MatrixParam{accum.data_ptr(), n, output_row_offsets,
                               nullptr, nullptr};
  epilogue.partials = tm::MatrixParam{partials.data_ptr(), n,
                                      output_row_offsets, nullptr, nullptr};
  epilogue.locks = locks.data_ptr<int32_t>();
  epilogue.combine_mat =
      tm::MatrixCombination_v3{tm::MatrixParam{}, 1.0f, 0.0f};

  TmSm70Exl3GateUpScheduler scheduler{{m, 2 * n, k, 2}, swizzle, splits};
  scheduler.offsets_ = n_offsets;
  auto const grid = scheduler.get_grid_shape();
  TORCH_CHECK(locks.numel() >=
                  static_cast<int64_t>(grid.x) *
                      static_cast<int64_t>(grid.y),
              "TurboMind EXL3 raw grouped-accum lock workspace is too small");
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3GateUpScheduler>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Bits>
void launch_tm_sm70_exl3_int8_pair_core_out(
    torch::Tensor const& out, torch::Tensor const& x_had,
    torch::Tensor const& metadata, torch::Tensor const& offsets,
    torch::Tensor const& partials, torch::Tensor const& locks, int m, int k,
    int n, int splits, int swizzle) {
  using Gemm = tm::GemmUniversal<
      tm::Sm70, TmSm70Exl3Mainloop<Bits, 2, false>,
      TmSm70Exl3OutputHadamardEpilogue, TmSm70Exl3GateUpScheduler>;

  int* n_offsets = offsets.data_ptr<int32_t>();
  int* input_row_offsets = n_offsets + 3;
  int* partial_row_offsets = input_row_offsets + 3;
  char* metadata_ptr = reinterpret_cast<char*>(metadata.data_ptr<uint8_t>());
  int* scale_pointer_table = reinterpret_cast<int*>(
      metadata_ptr + 4 * sizeof(tm::StridedPtr));
  tm::GemmParam param{
      tm::MatrixParam{x_had.data_ptr(), k, input_row_offsets, nullptr,
                      nullptr},
      tm::MatrixParam{metadata_ptr, 0, nullptr, scale_pointer_table, nullptr},
      tm::MatrixParam{},
      tm::MatrixParam{},
  };
  tm::EpilogueParam epilogue{};
  epilogue.c = tm::MatrixParam{out.data_ptr(), 2 * n, nullptr, nullptr,
                               n_offsets};
  epilogue.partials = tm::MatrixParam{partials.data_ptr(), n,
                                      partial_row_offsets, nullptr, nullptr};
  epilogue.locks = locks.data_ptr<int32_t>();
  epilogue.combine_mat = tm::MatrixCombination_v3{
      tm::MatrixParam{metadata_ptr + 2 * sizeof(tm::StridedPtr), 0, nullptr,
                      nullptr, nullptr},
      1.0f, 0.0f};

  TmSm70Exl3GateUpScheduler scheduler{{m, 2 * n, k, 2}, swizzle, splits};
  scheduler.offsets_ = n_offsets;
  auto const grid = scheduler.get_grid_shape();
  TORCH_CHECK(locks.numel() >=
                  static_cast<int64_t>(grid.x) *
                      static_cast<int64_t>(grid.y),
              "TurboMind EXL3 INT8 paired projection lock workspace is too "
              "small");
  constexpr int kThreads = Gemm::Impl::WARPS * kWarpSize;
  constexpr int kSmemBytes = sizeof(typename Gemm::SharedStorage);
  auto kernel = tm::gemm_kernel<Gemm, tm::GemmParam, tm::EpilogueParam,
                                TmSm70Exl3GateUpScheduler>;
  if constexpr (kSmemBytes > (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x_had.get_device()).stream();
  kernel<<<grid, kThreads, kSmemBytes, stream>>>(param, epilogue, scheduler);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ void output_hadamard_silu_mul_128(
    const float* gate_in, const float* up_in, half* out,
    const half* gate_scale, const half* up_scale, int lane) {
  float4 gate = *reinterpret_cast<const float4*>(gate_in + lane * 4);
  float4 up = *reinterpret_cast<const float4*>(up_in + lane * 4);

  float gate_h0 = (gate.x + gate.y) + (gate.z + gate.w);
  float gate_h1 = (gate.x - gate.y) + (gate.z - gate.w);
  float gate_h2 = (gate.x + gate.y) - (gate.z + gate.w);
  float gate_h3 = (gate.x - gate.y) - (gate.z - gate.w);
  float up_h0 = (up.x + up.y) + (up.z + up.w);
  float up_h1 = (up.x - up.y) + (up.z - up.w);
  float up_h2 = (up.x + up.y) - (up.z + up.w);
  float up_h3 = (up.x - up.y) - (up.z - up.w);
  hadamard_shuffle4(gate_h0, gate_h1, gate_h2, gate_h3, lane);
  hadamard_shuffle4(up_h0, up_h1, up_h2, up_h3, lane);

  Half4 const gate_post =
      *reinterpret_cast<const Half4*>(gate_scale + lane * 4);
  Half4 const up_post =
      *reinterpret_cast<const Half4*>(up_scale + lane * 4);
  float const transform_scale = kHadamardScale;
  gate_h0 *= transform_scale * __half2float(__low2half(gate_post.x));
  gate_h1 *= transform_scale * __half2float(__high2half(gate_post.x));
  gate_h2 *= transform_scale * __half2float(__low2half(gate_post.y));
  gate_h3 *= transform_scale * __half2float(__high2half(gate_post.y));
  up_h0 *= transform_scale * __half2float(__low2half(up_post.x));
  up_h1 *= transform_scale * __half2float(__high2half(up_post.x));
  up_h2 *= transform_scale * __half2float(__low2half(up_post.y));
  up_h3 *= transform_scale * __half2float(__high2half(up_post.y));

  // Preserve the existing numerical contract exactly: each projection first
  // rounds its output-Hadamard result to FP16, then the SM70 compatibility
  // SiluAndMul kernel rounds SiLU to FP16 before the FP16 multiply.
  Half4 const gate_half{__floats2half2_rn(gate_h0, gate_h1),
                        __floats2half2_rn(gate_h2, gate_h3)};
  Half4 const up_half{__floats2half2_rn(up_h0, up_h1),
                      __floats2half2_rn(up_h2, up_h3)};
  half const gate_values[4] = {
      __low2half(gate_half.x), __high2half(gate_half.x),
      __low2half(gate_half.y), __high2half(gate_half.y)};
  half const up_values[4] = {
      __low2half(up_half.x), __high2half(up_half.x),
      __low2half(up_half.y), __high2half(up_half.y)};
  half result[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float const gate_f = __half2float(gate_values[i]);
    half const silu =
        __float2half_rn(gate_f / (1.0f + expf(-gate_f)));
    result[i] = __hmul(silu, up_values[i]);
  }
  Half4 const packed_result{
      __halves2half2(result[0], result[1]),
      __halves2half2(result[2], result[3])};
  *reinterpret_cast<Half4*>(out + lane * 4) = packed_result;
}

__global__ void sm70_exl3_output_hadamard_silu_mul_kernel(
    const float* __restrict__ gate_accum,
    const float* __restrict__ up_accum, half* __restrict__ out,
    const half* __restrict__ gate_svh, const half* __restrict__ up_svh,
    int m, int n) {
  int const row = blockIdx.y;
  int const n0 = blockIdx.x * kHadamard;
  output_hadamard_silu_mul_128(
      gate_accum + row * n + n0, up_accum + row * n + n0,
      out + row * n + n0, gate_svh + n0, up_svh + n0, threadIdx.x);
}

__global__ void sm70_exl3_silu_mul_pair_kernel(
    const half* __restrict__ gate, const half* __restrict__ up,
    half* __restrict__ out, int elements) {
  int const index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  float const gate_f = __half2float(gate[index]);
  half const silu = __float2half_rn(gate_f / (1.0f + expf(-gate_f)));
  out[index] = __hmul(silu, up[index]);
}

// Preserve the established two-kernel arithmetic contract while removing the
// FP16 activation round trip before a down projection.  Each scalar is rounded
// after SiLU and again after the gate/up multiply, exactly as
// sm70_exl3_silu_mul_pair_kernel does.  The resulting Half4 then enters the
// unchanged input-Hadamard arithmetic (including its FP16 suh multiply).
__device__ __forceinline__ half sm70_exl3_silu_mul_half(half gate, half up) {
  float const gate_f = __half2float(gate);
  half const silu = __float2half_rn(gate_f / (1.0f + expf(-gate_f)));
  return __hmul(silu, up);
}

__global__ void sm70_exl3_silu_mul_input_hadamard_kernel(
    const half* __restrict__ gate, const half* __restrict__ up,
    const half* __restrict__ suh, half* __restrict__ x_had, int m, int k) {
  int const row = blockIdx.y;
  int const k0 = blockIdx.x * kHadamard;
  int const offset = row * k + k0 + threadIdx.x * 4;
  Half4 const gate4 = *reinterpret_cast<const Half4*>(gate + offset);
  Half4 const up4 = *reinterpret_cast<const Half4*>(up + offset);
  Half4 const activated{
      __halves2half2(
          sm70_exl3_silu_mul_half(__low2half(gate4.x), __low2half(up4.x)),
          sm70_exl3_silu_mul_half(__high2half(gate4.x), __high2half(up4.x))),
      __halves2half2(
          sm70_exl3_silu_mul_half(__low2half(gate4.y), __low2half(up4.y)),
          sm70_exl3_silu_mul_half(__high2half(gate4.y), __high2half(up4.y))),
  };
  Half4 const scale =
      *reinterpret_cast<const Half4*>(suh + k0 + threadIdx.x * 4);
  half2 const vx = __hmul2(activated.x, scale.x);
  half2 const vy = __hmul2(activated.y, scale.y);
  float const v0 = __half2float(__low2half(vx));
  float const v1 = __half2float(__high2half(vx));
  float const v2 = __half2float(__low2half(vy));
  float const v3 = __half2float(__high2half(vy));
  float const s0 = v0 + v1;
  float const d0 = v0 - v1;
  float const s1 = v2 + v3;
  float const d1 = v2 - v3;
  float h0 = s0 + s1;
  float h1 = d0 + d1;
  float h2 = s0 - s1;
  float h3 = d0 - d1;
  hadamard_shuffle4(h0, h1, h2, h3, threadIdx.x);
  Half4 const result{
      __floats2half2_rn(h0 * kHadamardScale, h1 * kHadamardScale),
      __floats2half2_rn(h2 * kHadamardScale, h3 * kHadamardScale),
  };
  *reinterpret_cast<Half4*>(x_had + offset) = result;
}

torch::Tensor exl3_sm70_silu_mul_input_hadamard(
    torch::Tensor const& gate, torch::Tensor const& up,
    torch::Tensor const& suh) {
  c10::cuda::CUDAGuard device_guard(gate.device());
  TORCH_CHECK(gate.is_cuda() && up.is_cuda() && suh.is_cuda(),
              "SM70 fused activation/Hadamard tensors must be CUDA tensors");
  TORCH_CHECK(gate.scalar_type() == at::ScalarType::Half &&
                  up.scalar_type() == at::ScalarType::Half &&
                  suh.scalar_type() == at::ScalarType::Half,
              "SM70 fused activation/Hadamard tensors must be float16");
  TORCH_CHECK(gate.dim() == 2 && gate.sizes() == up.sizes() && suh.dim() == 1 &&
                  suh.numel() == gate.size(1) && gate.size(0) > 0 &&
                  gate.size(0) <= 8 && gate.size(1) % kHadamard == 0,
              "SM70 fused activation/Hadamard tensor shapes disagree");
  TORCH_CHECK(gate.is_contiguous() && up.is_contiguous() && suh.is_contiguous(),
              "SM70 fused activation/Hadamard tensors must be contiguous");
  auto x_had = torch::empty_like(gate);
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(gate.get_device()).stream();
  dim3 const grid(static_cast<unsigned>(gate.size(1) / kHadamard),
                  static_cast<unsigned>(gate.size(0)));
  sm70_exl3_silu_mul_input_hadamard_kernel<<<grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(gate.size(0)), static_cast<int>(gate.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return x_had;
}

// Test/oracle entry point for the already-qualified standalone transform.
// Keeping this wrapper separate lets the component test prove bitwise parity
// with the exact production sequence instead of reimplementing H128 in Python.
torch::Tensor exl3_sm70_input_hadamard(torch::Tensor const& x,
                                      torch::Tensor const& suh) {
  c10::cuda::CUDAGuard device_guard(x.device());
  TORCH_CHECK(x.is_cuda() && suh.is_cuda() &&
                  x.scalar_type() == at::ScalarType::Half &&
                  suh.scalar_type() == at::ScalarType::Half && x.dim() == 2 &&
                  suh.dim() == 1 && suh.numel() == x.size(1) && x.size(0) > 0 &&
                  x.size(0) <= 8 && x.size(1) % kHadamard == 0 &&
                  x.is_contiguous() && suh.is_contiguous(),
              "SM70 input-Hadamard tensors are invalid");
  auto x_had = torch::empty_like(x);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  dim3 const grid(static_cast<unsigned>(x.size(1) / kHadamard),
                  static_cast<unsigned>(x.size(0)));
  sm70_exl3_input_hadamard_kernel<<<grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(x.size(0)), static_cast<int>(x.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return x_had;
}

torch::Tensor exl3_sm70_silu_mul_input_hadamard_baseline(
    torch::Tensor const& gate, torch::Tensor const& up,
    torch::Tensor const& suh) {
  c10::cuda::CUDAGuard device_guard(gate.device());
  TORCH_CHECK(gate.is_cuda() && up.is_cuda() && suh.is_cuda() &&
                  gate.scalar_type() == at::ScalarType::Half &&
                  up.scalar_type() == at::ScalarType::Half &&
                  suh.scalar_type() == at::ScalarType::Half && gate.dim() == 2 &&
                  gate.sizes() == up.sizes() && suh.dim() == 1 &&
                  suh.numel() == gate.size(1) && gate.size(0) > 0 &&
                  gate.size(0) <= 8 && gate.size(1) % kHadamard == 0 &&
                  gate.is_contiguous() && up.is_contiguous() &&
                  suh.is_contiguous(),
              "SM70 baseline activation/Hadamard tensors are invalid");
  auto activated = torch::empty_like(gate);
  auto x_had = torch::empty_like(gate);
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(gate.get_device()).stream();
  int64_t const elements = gate.numel();
  constexpr int kThreads = 256;
  int const blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
  sm70_exl3_silu_mul_pair_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
      reinterpret_cast<half*>(activated.data_ptr<at::Half>()),
      static_cast<int>(elements));
  dim3 const grid(static_cast<unsigned>(gate.size(1) / kHadamard),
                  static_cast<unsigned>(gate.size(0)));
  sm70_exl3_input_hadamard_kernel<<<grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const half*>(activated.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(gate.size(0)), static_cast<int>(gate.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return x_had;
}

bool sm70_exl3_tm_use_interleaved_state_decode(int64_t bits, int64_t k,
                                                int64_t n) {
  // Bracketed real-shard measurements show that splitting the state decode
  // around the current HMMA helps these projection shapes, but regresses the
  // GDN-z, self-Q, and vocabulary kernels.  Compile both instruction schedules
  // and preserve the established path for every unmeasured shape.
  if (bits == 5) {
    return (k == 5120 && (n == 1024 || n == 2560 || n == 4352)) ||
           (k == 1536 && n == 5120);
  }
  return bits == 6 && k == 4352 && n == 5120;
}

void exl3_sm70_tm_state_gemm_out(
    torch::Tensor const& out, torch::Tensor const& x,
    torch::Tensor const& state, torch::Tensor const& suh,
    torch::Tensor const& svh, torch::Tensor const& x_had,
    torch::Tensor const& accum, torch::Tensor const& partials,
    torch::Tensor const& locks, int64_t bits, int64_t splits,
    int64_t swizzle) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);
  int64_t const k = state.size(0) * 16;
  int64_t const n = state.size(1) * 32;
  int64_t const tiles = ((m + 7) / 8) * (n / 128);
  TORCH_CHECK(out.is_cuda() && x.is_cuda() && state.is_cuda() &&
                  suh.is_cuda() && svh.is_cuda() && x_had.is_cuda() &&
                  accum.is_cuda() && partials.is_cuda() && locks.is_cuda(),
              "TurboMind EXL3 state GEMM tensors must be CUDA tensors");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Half &&
                  x.scalar_type() == at::ScalarType::Half &&
                  state.scalar_type() == at::ScalarType::Int &&
                  suh.scalar_type() == at::ScalarType::Half &&
                  svh.scalar_type() == at::ScalarType::Half &&
                  x_had.scalar_type() == at::ScalarType::Half &&
                  accum.scalar_type() == at::ScalarType::Float &&
                  partials.scalar_type() == at::ScalarType::Float &&
                  locks.scalar_type() == at::ScalarType::Int,
              "TurboMind EXL3 state GEMM tensor dtypes disagree");
  TORCH_CHECK(out.is_contiguous() && x.is_contiguous() &&
                  state.is_contiguous() && suh.is_contiguous() &&
                  svh.is_contiguous() && x_had.is_contiguous() &&
                  accum.is_contiguous() && partials.is_contiguous() &&
                  locks.is_contiguous(),
              "TurboMind EXL3 state GEMM tensors must be contiguous");
  TORCH_CHECK(m > 0 && m <= 8 && x.dim() == 2 && x.size(1) == k &&
                  sm70_exl3_tm_state_shape_valid(state, bits) &&
                  suh.numel() == k &&
                  svh.numel() == n && x_had.sizes() == x.sizes() &&
                  out.dim() == 2 && out.size(0) == m && out.size(1) == n &&
                  accum.dim() == 2 && accum.size(0) == m &&
                  accum.size(1) == n && k % kHadamard == 0 &&
                  n % kHadamard == 0,
              "TurboMind EXL3 state GEMM tensor shapes disagree");
  TORCH_CHECK(partials.numel() >= 8 * n && locks.numel() >= tiles &&
                  splits >= 1 && splits <= k / 128 && swizzle >= 0 &&
                  swizzle <= 5,
              "TurboMind EXL3 state GEMM workspace/policy is invalid");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  dim3 const had_block(kWarpSize);
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                        static_cast<unsigned>(m));
  sm70_exl3_input_hadamard_kernel<<<input_grid, had_block, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_core_out<4, true, true>(
          accum, x_had, state, partials, locks, splits, swizzle);
      break;
    case 5:
      launch_tm_sm70_exl3_core_out<5, true, true>(
          accum, x_had, state, partials, locks, splits, swizzle);
      break;
    case 6:
      launch_tm_sm70_exl3_core_out<6, true, true>(
          accum, x_had, state, partials, locks, splits, swizzle);
      break;
    default:
      TORCH_CHECK(false, "TurboMind EXL3 state GEMM supports K4/K5/K6");
  }

  dim3 const output_grid(static_cast<unsigned>(n / kHadamard),
                         static_cast<unsigned>(m));
  sm70_exl3_output_hadamard_kernel<<<output_grid, had_block, 0, stream>>>(
      accum.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(svh.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3_sm70_tm_state_gemm_hadamard_out_impl(
    torch::Tensor const& out, torch::Tensor const& x,
    torch::Tensor const& state, torch::Tensor const& suh,
    torch::Tensor const& svh, torch::Tensor const& x_had,
    torch::Tensor const& partials, torch::Tensor const& locks, int64_t bits,
    int64_t splits, int64_t swizzle,
    tm::TileAllReduceParam const* tile_reduce) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);
  int64_t const k = state.size(0) * 16;
  int64_t const n = state.size(1) * 32;
  int64_t const tiles = ((m + 7) / 8) * (n / 128);
  TORCH_CHECK(out.is_cuda() && x.is_cuda() && state.is_cuda() &&
                  suh.is_cuda() && svh.is_cuda() && x_had.is_cuda() &&
                  partials.is_cuda() && locks.is_cuda(),
              "TurboMind EXL3 fused-Hadamard tensors must be CUDA tensors");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Half &&
                  x.scalar_type() == at::ScalarType::Half &&
                  state.scalar_type() == at::ScalarType::Int &&
                  suh.scalar_type() == at::ScalarType::Half &&
                  svh.scalar_type() == at::ScalarType::Half &&
                  x_had.scalar_type() == at::ScalarType::Half &&
                  partials.scalar_type() == at::ScalarType::Float &&
                  locks.scalar_type() == at::ScalarType::Int,
              "TurboMind EXL3 fused-Hadamard tensor dtypes disagree");
  TORCH_CHECK(out.is_contiguous() && x.is_contiguous() &&
                  state.is_contiguous() && suh.is_contiguous() &&
                  svh.is_contiguous() && x_had.is_contiguous() &&
                  partials.is_contiguous() && locks.is_contiguous(),
              "TurboMind EXL3 fused-Hadamard tensors must be contiguous");
  TORCH_CHECK(m > 0 && m <= 8 && x.dim() == 2 && x.size(1) == k &&
                  sm70_exl3_tm_state_shape_valid(state, bits) &&
                  suh.numel() == k &&
                  svh.numel() == n && x_had.sizes() == x.sizes() &&
                  out.dim() == 2 && out.size(0) == m && out.size(1) == n &&
                  k % kHadamard == 0 && n % kHadamard == 0,
              "TurboMind EXL3 fused-Hadamard tensor shapes disagree");
  TORCH_CHECK(partials.numel() >= 8 * n && locks.numel() >= tiles &&
                  splits >= 1 && splits <= k / 128 && swizzle >= 0 &&
                  swizzle <= 5,
              "TurboMind EXL3 fused-Hadamard workspace/policy is invalid");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  dim3 const had_block(kWarpSize);
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                        static_cast<unsigned>(m));
  sm70_exl3_input_hadamard_kernel<<<input_grid, had_block, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  bool const interleave =
      sm70_exl3_tm_use_interleaved_state_decode(bits, k, n);
  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_state_hadamard_core_out<4>(
          out, x_had, state, svh, partials, locks, splits, swizzle,
          tile_reduce);
      return;
    case 5:
      if (interleave) {
        launch_tm_sm70_exl3_state_hadamard_core_out<5, true>(
            out, x_had, state, svh, partials, locks, splits, swizzle,
            tile_reduce);
      } else {
        launch_tm_sm70_exl3_state_hadamard_core_out<5>(
            out, x_had, state, svh, partials, locks, splits, swizzle,
            tile_reduce);
      }
      return;
    case 6:
      if (interleave) {
        launch_tm_sm70_exl3_state_hadamard_core_out<6, true>(
            out, x_had, state, svh, partials, locks, splits, swizzle,
            tile_reduce);
      } else {
        launch_tm_sm70_exl3_state_hadamard_core_out<6>(
            out, x_had, state, svh, partials, locks, splits, swizzle,
            tile_reduce);
      }
      return;
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 fused-Hadamard GEMM supports K4/K5/K6");
  }
}

void exl3_sm70_tm_state_gemm_hadamard_out(
    torch::Tensor const& out, torch::Tensor const& x,
    torch::Tensor const& state, torch::Tensor const& suh,
    torch::Tensor const& svh, torch::Tensor const& x_had,
    torch::Tensor const& partials, torch::Tensor const& locks, int64_t bits,
    int64_t splits, int64_t swizzle) {
  exl3_sm70_tm_state_gemm_hadamard_out_impl(
      out, x, state, suh, svh, x_had, partials, locks, bits, splits,
      swizzle, nullptr);
}

void exl3_sm70_tm_state_gemm_hadamard_tile_reduce_out(
    torch::Tensor const& reduced, torch::Tensor const& staging,
    torch::Tensor const& x, torch::Tensor const& state,
    torch::Tensor const& suh, torch::Tensor const& svh,
    torch::Tensor const& x_had, torch::Tensor const& partials,
    torch::Tensor const& locks, int64_t bits, int64_t splits,
    int64_t swizzle, int64_t fa_ptr, int64_t reducer_blocks) {
  TORCH_CHECK(reduced.is_cuda() && staging.is_cuda() &&
                  reduced.scalar_type() == at::ScalarType::Half &&
                  staging.scalar_type() == at::ScalarType::Half &&
                  reduced.is_contiguous() && staging.is_contiguous() &&
                  reduced.sizes() == staging.sizes(),
              "TurboMind EXL3 tile-reduce outputs are invalid");
  TORCH_CHECK(staging.dim() == 2 &&
                  (staging.size(0) == 1 || staging.size(0) == 4) &&
                  staging.size(1) % kHadamard == 0 &&
                  staging.size(1) / kHadamard <=
                      vllm::sm70_tile_runtime::kMaxBlocks,
              "TurboMind EXL3 tile-reduce supports M=1|4 and N128 tiles");
  TORCH_CHECK(fa_ptr != 0 && reducer_blocks > 0,
              "TurboMind EXL3 tile-reduce communicator/config is invalid");

  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(fa_ptr);
  TORCH_CHECK(fa->world_size_ == 4 && fa->fully_connected_,
              "TurboMind EXL3 tile-reduce requires fully-connected TP4");
  c10::cuda::CUDAGuard device_guard(x.device());
  auto current_stream = at::cuda::getCurrentCUDAStream(x.get_device());
  cudaStreamCaptureStatus capture_status;
  C10_CUDA_CHECK(cudaStreamIsCapturing(current_stream.stream(),
                                       &capture_status));
  TORCH_CHECK(capture_status == cudaStreamCaptureStatusActive,
              "TurboMind EXL3 tile-reduce requires CUDA graph capture");

  tm::TileAllReduceParam tile_reduce{};
  for (int rank = 0; rank < fa->world_size_; ++rank) {
    tile_reduce.signals[rank] = fa->sg_.signals[rank];
  }
  tile_reduce.self_signal = fa->self_sg_;
  tile_reduce.rank_data = fa->rank_data_for_buffer(
      current_stream.stream(), staging.data_ptr(),
      "exl3_sm70_tm_state_gemm_hadamard_tile_reduce_out");
  tile_reduce.output = reduced.data_ptr();
  tile_reduce.rank = fa->rank_;
  tile_reduce.world_size = fa->world_size_;
  tile_reduce.rows = static_cast<int>(staging.size(0));
  tile_reduce.row_stride = static_cast<int>(staging.size(1));
  tile_reduce.tile_columns = kHadamard;
  tile_reduce.tile_numel = tile_reduce.rows * tile_reduce.tile_columns;
  tile_reduce.output_numel = static_cast<int>(staging.numel());
  tile_reduce.kernel_reducer_blocks = static_cast<int>(reducer_blocks);

  exl3_sm70_tm_state_gemm_hadamard_out_impl(
      staging, x, state, suh, svh, x_had, partials, locks, bits, splits,
      swizzle, &tile_reduce);
}

torch::Tensor exl3_sm70_tm_state_gemm(
    torch::Tensor const& x, torch::Tensor const& state,
    torch::Tensor const& suh, torch::Tensor const& svh, int64_t bits,
    int64_t splits, int64_t swizzle) {
  int64_t const m = x.size(0);
  int64_t const n = state.size(1) * 32;
  auto out = torch::empty({m, n}, x.options().dtype(at::ScalarType::Half));
  auto x_had = torch::empty_like(x);
  auto accum = torch::empty(
      {m, n}, x.options().dtype(at::ScalarType::Float));
  auto partials = torch::empty(
      {8, n}, x.options().dtype(at::ScalarType::Float));
  auto locks = torch::zeros(
      {((m + 7) / 8) * (n / 128)},
      x.options().dtype(at::ScalarType::Int));
  exl3_sm70_tm_state_gemm_out(out, x, state, suh, svh, x_had, accum,
                               partials, locks, bits, splits, swizzle);
  return out;
}

bool sm70_exl3_tm_use_fused_output_hadamard(int64_t bits, int64_t k,
                                             int64_t n, int64_t m) {
  // Verifier-width projections consistently win with the fused epilogue.
  // M=1 is launch/occupancy sensitive, so retain the standalone transform for
  // the three measured regressors: GDN z, full-attention q, and lm_head.
  if (m > 1) {
    return true;
  }
  if (bits == 5 && k == 5120) {
    return n == 256 || n == 512 || n == 1024 || n == 2560 || n == 3072 ||
           n == 4352;
  }
  return (bits == 5 && k == 1536 && n == 5120) ||
         (bits == 6 && k == 4352 && n == 5120);
}

torch::Tensor exl3_sm70_tm_state_gemm_persistent_locks(
    torch::Tensor const& x, torch::Tensor const& state,
    torch::Tensor const& suh, torch::Tensor const& svh,
    torch::Tensor const& locks, int64_t bits, int64_t splits,
    int64_t swizzle) {
  int64_t const m = x.size(0);
  int64_t const n = state.size(1) * 32;
  auto out = torch::empty({m, n}, x.options().dtype(at::ScalarType::Half));
  auto x_had = torch::empty_like(x);
  auto partials = torch::empty(
      {8, n}, x.options().dtype(at::ScalarType::Float));
  if (sm70_exl3_tm_use_fused_output_hadamard(bits, x.size(1), n, m)) {
    exl3_sm70_tm_state_gemm_hadamard_out(
        out, x, state, suh, svh, x_had, partials, locks, bits, splits,
        swizzle);
    return out;
  }
  auto accum = torch::empty(
      {m, n}, x.options().dtype(at::ScalarType::Float));
  exl3_sm70_tm_state_gemm_out(out, x, state, suh, svh, x_had, accum,
                               partials, locks, bits, splits, swizzle);
  return out;
}

void exl3_sm70_tm_int8_gemm_hadamard_out(
    torch::Tensor const& out, torch::Tensor const& x,
    torch::Tensor const& packed_lane, torch::Tensor const& tile_scales,
    torch::Tensor const& suh, torch::Tensor const& svh,
    torch::Tensor const& x_had, torch::Tensor const& partials,
    torch::Tensor const& locks, int64_t bits, int64_t splits,
    int64_t swizzle) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);
  int64_t const k = packed_lane.size(0) * 16;
  int64_t const n = packed_lane.size(1) * 32;
  int64_t const tiles = ((m + 7) / 8) * (n / 128);
  TORCH_CHECK(out.is_cuda() && x.is_cuda() && packed_lane.is_cuda() &&
                  tile_scales.is_cuda() && suh.is_cuda() && svh.is_cuda() &&
                  x_had.is_cuda() && partials.is_cuda() && locks.is_cuda(),
              "TurboMind EXL3 int8 fused-Hadamard tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Half &&
                  x.scalar_type() == at::ScalarType::Half &&
                  packed_lane.scalar_type() == at::ScalarType::Char &&
                  tile_scales.scalar_type() == at::ScalarType::Half &&
                  suh.scalar_type() == at::ScalarType::Half &&
                  svh.scalar_type() == at::ScalarType::Half &&
                  x_had.scalar_type() == at::ScalarType::Half &&
                  partials.scalar_type() == at::ScalarType::Float &&
                  locks.scalar_type() == at::ScalarType::Int,
              "TurboMind EXL3 int8 fused-Hadamard dtypes disagree");
  TORCH_CHECK(out.is_contiguous() && x.is_contiguous() &&
                  packed_lane.is_contiguous() &&
                  tile_scales.is_contiguous() && suh.is_contiguous() &&
                  svh.is_contiguous() && x_had.is_contiguous() &&
                  partials.is_contiguous() && locks.is_contiguous(),
              "TurboMind EXL3 int8 fused-Hadamard tensors must be contiguous");
  TORCH_CHECK(m > 0 && m <= 8 && x.dim() == 2 && x.size(1) == k &&
                  packed_lane.dim() == 4 &&
                  packed_lane.size(2) == kWarpSize &&
                  packed_lane.size(3) == 16 && tile_scales.dim() == 2 &&
                  tile_scales.size(0) == packed_lane.size(0) &&
                  tile_scales.size(1) == packed_lane.size(1) &&
                  suh.numel() == k && svh.numel() == n &&
                  x_had.sizes() == x.sizes() && out.dim() == 2 &&
                  out.size(0) == m && out.size(1) == n &&
                  k % kHadamard == 0 && n % kHadamard == 0,
              "TurboMind EXL3 int8 fused-Hadamard shapes disagree");
  TORCH_CHECK(partials.numel() >= 8 * n && locks.numel() >= tiles &&
                  splits >= 1 && splits <= k / 128 && swizzle >= 0 &&
                  swizzle <= 5,
              "TurboMind EXL3 int8 fused-Hadamard policy is invalid");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  dim3 const had_block(kWarpSize);
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                        static_cast<unsigned>(m));
  sm70_exl3_input_hadamard_kernel<<<input_grid, had_block, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_int8_hadamard_core_out<4>(
          out, x_had, packed_lane, tile_scales, svh, partials, locks,
          splits, swizzle);
      return;
    case 5:
      launch_tm_sm70_exl3_int8_hadamard_core_out<5>(
          out, x_had, packed_lane, tile_scales, svh, partials, locks,
          splits, swizzle);
      return;
    case 6:
      launch_tm_sm70_exl3_int8_hadamard_core_out<6>(
          out, x_had, packed_lane, tile_scales, svh, partials, locks,
          splits, swizzle);
      return;
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 int8 fused-Hadamard supports K4/K5/K6");
  }
}

void exl3_sm70_tm_int6_gemm_hadamard_out(
    torch::Tensor const& out, torch::Tensor const& x,
    torch::Tensor const& packed_words, torch::Tensor const& group_scales,
    torch::Tensor const& suh, torch::Tensor const& svh,
    torch::Tensor const& x_had, torch::Tensor const& partials,
    torch::Tensor const& locks, int64_t bits, int64_t splits,
    int64_t swizzle) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);
  int64_t const k = packed_words.size(0) * 16;
  int64_t const n = packed_words.size(1) * 32;
  int64_t const tiles = ((m + 7) / 8) * (n / 128);
  TORCH_CHECK(out.is_cuda() && x.is_cuda() && packed_words.is_cuda() &&
                  group_scales.is_cuda() && suh.is_cuda() && svh.is_cuda() &&
                  x_had.is_cuda() && partials.is_cuda() && locks.is_cuda(),
              "TurboMind EXL3 int6 fused-Hadamard tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Half &&
                  x.scalar_type() == at::ScalarType::Half &&
                  packed_words.scalar_type() == at::ScalarType::Int &&
                  group_scales.scalar_type() == at::ScalarType::Half &&
                  suh.scalar_type() == at::ScalarType::Half &&
                  svh.scalar_type() == at::ScalarType::Half &&
                  x_had.scalar_type() == at::ScalarType::Half &&
                  partials.scalar_type() == at::ScalarType::Float &&
                  locks.scalar_type() == at::ScalarType::Int,
              "TurboMind EXL3 int6 fused-Hadamard dtypes disagree");
  TORCH_CHECK(out.is_contiguous() && x.is_contiguous() &&
                  packed_words.is_contiguous() &&
                  group_scales.is_contiguous() && suh.is_contiguous() &&
                  svh.is_contiguous() && x_had.is_contiguous() &&
                  partials.is_contiguous() && locks.is_contiguous(),
              "TurboMind EXL3 int6 fused-Hadamard tensors must be contiguous");
  TORCH_CHECK(m > 0 && m <= 8 && x.dim() == 2 && x.size(1) == k &&
                  packed_words.dim() == 4 && packed_words.size(2) == 3 &&
                  packed_words.size(3) == kWarpSize &&
                  group_scales.dim() == 3 &&
                  group_scales.size(0) == packed_words.size(0) &&
                  group_scales.size(1) == packed_words.size(1) &&
                  group_scales.size(2) == 8 && suh.numel() == k &&
                  svh.numel() == n && x_had.sizes() == x.sizes() &&
                  out.dim() == 2 && out.size(0) == m && out.size(1) == n &&
                  k % kHadamard == 0 && n % kHadamard == 0,
              "TurboMind EXL3 int6 fused-Hadamard shapes disagree");
  TORCH_CHECK(partials.numel() >= 8 * n && locks.numel() >= tiles &&
                  splits >= 1 && splits <= k / 128 && swizzle >= 0 &&
                  swizzle <= 5,
              "TurboMind EXL3 int6 fused-Hadamard policy is invalid");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                        static_cast<unsigned>(m));
  sm70_exl3_input_hadamard_kernel<<<input_grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_int8_hadamard_core_out<4, 4>(
          out, x_had, packed_words, group_scales, svh, partials, locks,
          splits, swizzle);
      return;
    case 5:
      launch_tm_sm70_exl3_int8_hadamard_core_out<5, 4>(
          out, x_had, packed_words, group_scales, svh, partials, locks,
          splits, swizzle);
      return;
    case 6:
      launch_tm_sm70_exl3_int8_hadamard_core_out<6, 4>(
          out, x_had, packed_words, group_scales, svh, partials, locks,
          splits, swizzle);
      return;
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 int6 fused-Hadamard supports K4/K5/K6");
  }
}

void exl3_sm70_tm_int10_gemm_hadamard_out(
    torch::Tensor const& out, torch::Tensor const& x,
    torch::Tensor const& packed_words, torch::Tensor const& group_scales,
    torch::Tensor const& suh, torch::Tensor const& svh,
    torch::Tensor const& x_had, torch::Tensor const& partials,
    torch::Tensor const& locks, int64_t bits, int64_t splits,
    int64_t swizzle) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);
  int64_t const k = packed_words.size(0) * 16;
  int64_t const n = packed_words.size(1) * 32;
  int64_t const tiles = ((m + 7) / 8) * (n / 128);
  TORCH_CHECK(out.is_cuda() && x.is_cuda() && packed_words.is_cuda() &&
                  group_scales.is_cuda() && suh.is_cuda() && svh.is_cuda() &&
                  x_had.is_cuda() && partials.is_cuda() && locks.is_cuda(),
              "TurboMind EXL3 int10 fused-Hadamard tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Half &&
                  x.scalar_type() == at::ScalarType::Half &&
                  packed_words.scalar_type() == at::ScalarType::Int &&
                  group_scales.scalar_type() == at::ScalarType::Half &&
                  suh.scalar_type() == at::ScalarType::Half &&
                  svh.scalar_type() == at::ScalarType::Half &&
                  x_had.scalar_type() == at::ScalarType::Half &&
                  partials.scalar_type() == at::ScalarType::Float &&
                  locks.scalar_type() == at::ScalarType::Int,
              "TurboMind EXL3 int10 fused-Hadamard dtypes disagree");
  TORCH_CHECK(out.is_contiguous() && x.is_contiguous() &&
                  packed_words.is_contiguous() &&
                  group_scales.is_contiguous() && suh.is_contiguous() &&
                  svh.is_contiguous() && x_had.is_contiguous() &&
                  partials.is_contiguous() && locks.is_contiguous(),
              "TurboMind EXL3 int10 fused-Hadamard tensors must be contiguous");
  TORCH_CHECK(m > 0 && m <= 8 && x.dim() == 2 && x.size(1) == k &&
                  packed_words.dim() == 4 && packed_words.size(2) == 5 &&
                  packed_words.size(3) == kWarpSize &&
                  group_scales.dim() == 3 &&
                  group_scales.size(0) == packed_words.size(0) &&
                  group_scales.size(1) == packed_words.size(1) &&
                  group_scales.size(2) == 8 && suh.numel() == k &&
                  svh.numel() == n && x_had.sizes() == x.sizes() &&
                  out.dim() == 2 && out.size(0) == m && out.size(1) == n &&
                  k % kHadamard == 0 && n % kHadamard == 0,
              "TurboMind EXL3 int10 fused-Hadamard shapes disagree");
  TORCH_CHECK(partials.numel() >= 8 * n && locks.numel() >= tiles &&
                  splits >= 1 && splits <= k / 128 && swizzle >= 0 &&
                  swizzle <= 5,
              "TurboMind EXL3 int10 fused-Hadamard policy is invalid");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                        static_cast<unsigned>(m));
  sm70_exl3_input_hadamard_kernel<<<input_grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_int8_hadamard_core_out<4, 5>(
          out, x_had, packed_words, group_scales, svh, partials, locks,
          splits, swizzle);
      return;
    case 5:
      launch_tm_sm70_exl3_int8_hadamard_core_out<5, 5>(
          out, x_had, packed_words, group_scales, svh, partials, locks,
          splits, swizzle);
      return;
    case 6:
      launch_tm_sm70_exl3_int8_hadamard_core_out<6, 5>(
          out, x_had, packed_words, group_scales, svh, partials, locks,
          splits, swizzle);
      return;
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 int10 fused-Hadamard supports K4/K5/K6");
  }
}

void exl3_sm70_tm_int8_gemm_out(
    torch::Tensor const& out, torch::Tensor const& x,
    torch::Tensor const& packed_lane, torch::Tensor const& tile_scales,
    torch::Tensor const& suh, torch::Tensor const& svh,
    torch::Tensor const& x_had, torch::Tensor const& accum,
    torch::Tensor const& partials, torch::Tensor const& locks,
    int64_t bits, int64_t splits, int64_t swizzle) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);
  int64_t const k = packed_lane.size(0) * 16;
  int64_t const n = packed_lane.size(1) * 32;
  int64_t const tiles = ((m + 7) / 8) * (n / 128);
  TORCH_CHECK(out.is_cuda() && x.is_cuda() && packed_lane.is_cuda() &&
                  tile_scales.is_cuda() && suh.is_cuda() && svh.is_cuda() &&
                  x_had.is_cuda() && accum.is_cuda() &&
                  partials.is_cuda() && locks.is_cuda(),
              "TurboMind EXL3 int8 GEMM tensors must be CUDA tensors");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Half &&
                  x.scalar_type() == at::ScalarType::Half &&
                  packed_lane.scalar_type() == at::ScalarType::Char &&
                  tile_scales.scalar_type() == at::ScalarType::Half &&
                  suh.scalar_type() == at::ScalarType::Half &&
                  svh.scalar_type() == at::ScalarType::Half &&
                  x_had.scalar_type() == at::ScalarType::Half &&
                  accum.scalar_type() == at::ScalarType::Float &&
                  partials.scalar_type() == at::ScalarType::Float &&
                  locks.scalar_type() == at::ScalarType::Int,
              "TurboMind EXL3 int8 GEMM tensor dtypes disagree");
  TORCH_CHECK(out.is_contiguous() && x.is_contiguous() &&
                  packed_lane.is_contiguous() &&
                  tile_scales.is_contiguous() && suh.is_contiguous() &&
                  svh.is_contiguous() && x_had.is_contiguous() &&
                  accum.is_contiguous() && partials.is_contiguous() &&
                  locks.is_contiguous(),
              "TurboMind EXL3 int8 GEMM tensors must be contiguous");
  TORCH_CHECK(m > 0 && m <= 8 && x.dim() == 2 && x.size(1) == k &&
                  packed_lane.dim() == 4 &&
                  packed_lane.size(2) == kWarpSize &&
                  packed_lane.size(3) == 16 && tile_scales.dim() == 2 &&
                  tile_scales.size(0) == packed_lane.size(0) &&
                  tile_scales.size(1) == packed_lane.size(1) &&
                  suh.numel() == k && svh.numel() == n &&
                  x_had.sizes() == x.sizes() && out.dim() == 2 &&
                  out.size(0) == m && out.size(1) == n &&
                  accum.dim() == 2 && accum.size(0) == m &&
                  accum.size(1) == n && k % kHadamard == 0 &&
                  n % kHadamard == 0,
              "TurboMind EXL3 int8 GEMM tensor shapes disagree");
  TORCH_CHECK(partials.numel() >= 8 * n && locks.numel() >= tiles &&
                  splits >= 1 && splits <= k / 128 && swizzle >= 0 &&
                  swizzle <= 5,
              "TurboMind EXL3 int8 GEMM workspace/policy is invalid");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  dim3 const had_block(kWarpSize);
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                        static_cast<unsigned>(m));
  sm70_exl3_input_hadamard_kernel<<<input_grid, had_block, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_core_out<4, 2, true>(
          accum, x_had, packed_lane, partials, locks, splits, swizzle,
          tile_scales);
      break;
    case 5:
      launch_tm_sm70_exl3_core_out<5, 2, true>(
          accum, x_had, packed_lane, partials, locks, splits, swizzle,
          tile_scales);
      break;
    case 6:
      launch_tm_sm70_exl3_core_out<6, 2, true>(
          accum, x_had, packed_lane, partials, locks, splits, swizzle,
          tile_scales);
      break;
    default:
      TORCH_CHECK(false, "TurboMind EXL3 int8 GEMM supports K4/K5/K6");
  }

  dim3 const output_grid(static_cast<unsigned>(n / kHadamard),
                         static_cast<unsigned>(m));
  sm70_exl3_output_hadamard_kernel<<<output_grid, had_block, 0, stream>>>(
      accum.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(svh.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3_sm70_tm_e4m3_gemm_hadamard_out(
    torch::Tensor const& out, torch::Tensor const& x,
    torch::Tensor const& packed_lane, torch::Tensor const& tile_scales,
    torch::Tensor const& suh, torch::Tensor const& svh,
    torch::Tensor const& x_had, torch::Tensor const& partials,
    torch::Tensor const& locks, int64_t bits, int64_t splits,
    int64_t swizzle) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);
  int64_t const k = packed_lane.size(0) * 16;
  int64_t const n = packed_lane.size(1) * 32;
  int64_t const tiles = ((m + 7) / 8) * (n / 128);
  TORCH_CHECK(out.is_cuda() && x.is_cuda() && packed_lane.is_cuda() &&
                  tile_scales.is_cuda() && suh.is_cuda() && svh.is_cuda() &&
                  x_had.is_cuda() && partials.is_cuda() && locks.is_cuda(),
              "TurboMind EXL3 E4M3 fused-Hadamard tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Half &&
                  x.scalar_type() == at::ScalarType::Half &&
                  packed_lane.scalar_type() == at::ScalarType::Byte &&
                  tile_scales.scalar_type() == at::ScalarType::Half &&
                  suh.scalar_type() == at::ScalarType::Half &&
                  svh.scalar_type() == at::ScalarType::Half &&
                  x_had.scalar_type() == at::ScalarType::Half &&
                  partials.scalar_type() == at::ScalarType::Float &&
                  locks.scalar_type() == at::ScalarType::Int,
              "TurboMind EXL3 E4M3 fused-Hadamard dtypes disagree");
  TORCH_CHECK(out.is_contiguous() && x.is_contiguous() &&
                  packed_lane.is_contiguous() &&
                  tile_scales.is_contiguous() && suh.is_contiguous() &&
                  svh.is_contiguous() && x_had.is_contiguous() &&
                  partials.is_contiguous() && locks.is_contiguous(),
              "TurboMind EXL3 E4M3 fused-Hadamard tensors must be contiguous");
  TORCH_CHECK(m > 0 && m <= 8 && x.dim() == 2 && x.size(1) == k &&
                  packed_lane.dim() == 4 &&
                  packed_lane.size(2) == kWarpSize &&
                  packed_lane.size(3) == 16 && tile_scales.dim() == 2 &&
                  tile_scales.size(0) == packed_lane.size(0) &&
                  tile_scales.size(1) == packed_lane.size(1) &&
                  suh.numel() == k && svh.numel() == n &&
                  x_had.sizes() == x.sizes() && out.dim() == 2 &&
                  out.size(0) == m && out.size(1) == n &&
                  k % kHadamard == 0 && n % kHadamard == 0,
              "TurboMind EXL3 E4M3 fused-Hadamard shapes disagree");
  TORCH_CHECK(partials.numel() >= 8 * n && locks.numel() >= tiles &&
                  splits >= 1 && splits <= k / 128 && swizzle >= 0 &&
                  swizzle <= 5,
              "TurboMind EXL3 E4M3 fused-Hadamard policy is invalid");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                        static_cast<unsigned>(m));
  sm70_exl3_input_hadamard_kernel<<<input_grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_e4m3_hadamard_core_out<4>(
          out, x_had, packed_lane, tile_scales, svh, partials, locks,
          splits, swizzle);
      return;
    case 5:
      launch_tm_sm70_exl3_e4m3_hadamard_core_out<5>(
          out, x_had, packed_lane, tile_scales, svh, partials, locks,
          splits, swizzle);
      return;
    case 6:
      launch_tm_sm70_exl3_e4m3_hadamard_core_out<6>(
          out, x_had, packed_lane, tile_scales, svh, partials, locks,
          splits, swizzle);
      return;
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 E4M3 fused-Hadamard supports K4/K5/K6");
  }
}

torch::Tensor exl3_sm70_tm_int8_gemm(
    torch::Tensor const& x, torch::Tensor const& packed_lane,
    torch::Tensor const& tile_scales, torch::Tensor const& suh,
    torch::Tensor const& svh, int64_t bits, int64_t splits,
    int64_t swizzle) {
  int64_t const m = x.size(0);
  int64_t const n = packed_lane.size(1) * 32;
  auto out = torch::empty({m, n}, x.options().dtype(at::ScalarType::Half));
  auto x_had = torch::empty_like(x);
  auto accum = torch::empty(
      {m, n}, x.options().dtype(at::ScalarType::Float));
  auto partials = torch::empty(
      {8, n}, x.options().dtype(at::ScalarType::Float));
  auto locks = torch::zeros(
      {((m + 7) / 8) * (n / 128)},
      x.options().dtype(at::ScalarType::Int));
  exl3_sm70_tm_int8_gemm_out(
      out, x, packed_lane, tile_scales, suh, svh, x_had, accum, partials,
      locks, bits, splits, swizzle);
  return out;
}

template <int Bits>
__global__ void sm70_exl3_reconstruct_mcg_kernel(
    const uint16_t* __restrict__ trellis, half* __restrict__ dense_b, int n) {
  int const lane = threadIdx.x;
  int const k_block = blockIdx.y;
  int const n_block = blockIdx.x;
  int const blocks_n = n / kTileK;
  uint16_t const* packed_global =
      trellis + (k_block * blocks_n + n_block) * (16 * Bits);
  half* tile = dense_b + k_block * kTileK * n + n_block * kTileK;
  reconstruct_mcg_tile_to_global<Bits>(
      reinterpret_cast<const uint32_t*>(packed_global), tile, n, lane);
}

template <int Bits>
torch::Tensor launch_tm_sm70_exl3_reconstruct(
    torch::Tensor const& trellis) {
  int64_t const k = trellis.size(0) * 16;
  int64_t const n = trellis.size(1) * 16;
  auto dense = torch::empty({k, n},
                            trellis.options().dtype(at::ScalarType::Half));
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(trellis.get_device()).stream();
  dim3 const grid(static_cast<unsigned>(trellis.size(1)),
                  static_cast<unsigned>(trellis.size(0)));
  sm70_exl3_reconstruct_mcg_kernel<Bits><<<grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
      reinterpret_cast<half*>(dense.data_ptr<at::Half>()),
      static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dense;
}

torch::Tensor exl3_sm70_tm_reconstruct(torch::Tensor const& trellis) {
  c10::cuda::CUDAGuard device_guard(trellis.device());
  TORCH_CHECK(trellis.is_cuda() &&
                  trellis.scalar_type() == at::ScalarType::Short &&
                  trellis.dim() == 3 && trellis.is_contiguous(),
              "TurboMind EXL3 reconstruct expects contiguous rank-3 int16");
  int64_t const bits = trellis.size(2) / 16;
  TORCH_CHECK(trellis.size(2) == bits * 16,
              "TurboMind EXL3 reconstruct has an invalid bit dimension");
  switch (bits) {
    case 4:
      return launch_tm_sm70_exl3_reconstruct<4>(trellis);
    case 5:
      return launch_tm_sm70_exl3_reconstruct<5>(trellis);
    case 6:
      return launch_tm_sm70_exl3_reconstruct<6>(trellis);
    default:
      TORCH_CHECK(false, "TurboMind EXL3 reconstruct supports K4/K5/K6");
  }
}

template <int Bits, int Splits>
__global__ __launch_bounds__(Splits * kWarpSize, 2)
    void sm70_exl3_mcg_decode_kernel(
        const half* __restrict__ x_had,
        const uint16_t* __restrict__ trellis,
        float* __restrict__ accum, int n, int k) {
  __shared__ Sm70Exl3DecodeShared<Splits> shared;

  int const warp = threadIdx.x / kWarpSize;
  int const lane = threadIdx.x % kWarpSize;
  int const blocks_n = n / kTileK;
  int const n_block = blockIdx.x;
  int const blocks_k = k / kTileK;
  int const blocks_per_split = (blocks_k + Splits - 1) / Splits;
  int const k_begin = warp * blocks_per_split;
  int const k_end = min(blocks_k, k_begin + blocks_per_split);

  wmma::fragment<wmma::accumulator, kTileM, kTileK, kTileK, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k_block = k_begin; k_block < k_end; ++k_block) {
    uint32_t const* packed_global = reinterpret_cast<const uint32_t*>(
        trellis + (k_block * blocks_n + n_block) * (16 * Bits));

    half (*b_tile)[kTileK] = shared.mainloop.b[warp];
    reconstruct_mcg_tile<Bits>(packed_global, b_tile, lane);
    __syncwarp();

    wmma::fragment<wmma::matrix_a, kTileM, kTileK, kTileK, half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kTileM, kTileK, kTileK, half,
                   wmma::row_major>
        b_frag;
    wmma::load_matrix_sync(a_frag, x_had + k_block * kTileK, k);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kTileK);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncwarp();
  }

  // All warps must leave the mainloop before its shared storage is reused for
  // partial accumulators.
  __syncthreads();
  wmma::store_matrix_sync(&shared.partial[warp][0][0], acc, kTileK,
                          wmma::mem_row_major);
  __syncthreads();

  if (warp == 0) {
    for (int index = lane; index < kTileM * kTileK;
         index += kWarpSize) {
      float sum = 0.0f;
#pragma unroll
      for (int split = 0; split < Splits; ++split) {
        sum += shared.partial[split][0][index];
      }
      int const row = index / kTileK;
      int const col = index % kTileK;
      accum[row * n + n_block * kTileK + col] = sum;
    }
  }
}

template <int Bits, int Splits>
__global__ __launch_bounds__(Splits * kWarpSize, 2)
    void sm70_exl3_mcg_decode_native_kernel(
        const half* __restrict__ x_had,
        const uint16_t* __restrict__ trellis,
        float* __restrict__ accum, int n, int k) {
  __shared__ Sm70Exl3NativeDecodeShared<Splits> shared;

  int const warp = threadIdx.x / kWarpSize;
  int const lane = threadIdx.x % kWarpSize;
  int const packed_blocks_n = n / kTileK;
  int const n_block = blockIdx.x;
  int const packed_n_block = n_block * 2;
  int const blocks_k = k / kTileK;
  int const blocks_per_split = (blocks_k + Splits - 1) / Splits;
  int const k_begin = warp * blocks_per_split;
  int const k_end = min(blocks_k, k_begin + blocks_per_split);

  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 0.0f, 0.0f};
  int const a_row = (lane / 16) * 4 + lane % 4;
  int const effective_n =
      (lane / 16) * 4 + (lane & 12) * 2 + lane % 4;

  for (int k_block = k_begin; k_block < k_end; ++k_block) {
#if EXL3_SM70_DIRECT_REG_FRAGMENTS
    Half4 b_frag[4];
    reconstruct_mcg_native_n32_fragments<Bits>(
        trellis, b_frag, k_block, packed_blocks_n, packed_n_block, lane,
        effective_n);
#else
    half* b_tile = &shared.mainloop.b[warp][0][0];
    uint32_t const* packed0 = reinterpret_cast<const uint32_t*>(
        trellis + (k_block * packed_blocks_n + packed_n_block) *
                      (16 * Bits));
    uint32_t const* packed1 = reinterpret_cast<const uint32_t*>(
        trellis + (k_block * packed_blocks_n + packed_n_block + 1) *
                      (16 * Bits));
    reconstruct_mcg_tile_strided<Bits>(packed0, b_tile, 0, lane);
    reconstruct_mcg_tile_strided<Bits>(packed1, b_tile, 16, lane);
    __syncwarp();
#endif

#pragma unroll
    for (int sub_k = 0; sub_k < 2; ++sub_k) {
      int const k0 = k_block * kTileK + sub_k * 8;
      Half4 const a0 =
          *reinterpret_cast<const Half4*>(x_had + a_row * k + k0);
      Half4 const a1 =
          *reinterpret_cast<const Half4*>(x_had + a_row * k + k0 + 4);
#if EXL3_SM70_DIRECT_REG_FRAGMENTS
      Half4 const bf0 = b_frag[sub_k * 2];
      Half4 const bf1 = b_frag[sub_k * 2 + 1];
#else
      Half4 const bf0 = *reinterpret_cast<const Half4*>(
          b_tile + effective_n * kTileK + sub_k * 8);
      Half4 const bf1 = *reinterpret_cast<const Half4*>(
          b_tile + effective_n * kTileK + sub_k * 8 + 4);
#endif
      mma_sm70_m8n8k4(acc, a0, bf0);
      mma_sm70_m8n8k4(acc, a1, bf1);
    }
#if !EXL3_SM70_DIRECT_REG_FRAGMENTS
    __syncwarp();
#endif
  }

  __syncthreads();
  int const c_row = (lane & 1) + (lane / 16) * 4;
  int const c_col = (lane & 2) + (lane & 12) * 2;
  shared.partial[warp][c_row][c_col] = acc[0];
  shared.partial[warp][c_row][c_col + 1] = acc[1];
  shared.partial[warp][c_row + 2][c_col] = acc[2];
  shared.partial[warp][c_row + 2][c_col + 1] = acc[3];
  shared.partial[warp][c_row][c_col + 4] = acc[4];
  shared.partial[warp][c_row][c_col + 5] = acc[5];
  shared.partial[warp][c_row + 2][c_col + 4] = acc[6];
  shared.partial[warp][c_row + 2][c_col + 5] = acc[7];
  __syncthreads();

  if (warp == 0) {
    for (int index = lane; index < 8 * 32; index += kWarpSize) {
      float sum = 0.0f;
#pragma unroll
      for (int split = 0; split < Splits; ++split) {
        sum += shared.partial[split][0][index];
      }
      int const row = index / 32;
      int const col = index % 32;
      accum[row * n + n_block * 32 + col] = sum;
    }
  }
}

template <int Bits, int Splits>
__global__ __launch_bounds__(Splits * kWarpSize, 1)
    void sm70_exl3_mcg_decode_native_n128_kernel(
        const half* __restrict__ x_had,
        const uint16_t* __restrict__ trellis, half* __restrict__ out,
        const half* __restrict__ svh, int m, int n, int k) {
  __shared__ Sm70Exl3NativeN128Shared<Splits> shared;

  int const warp = threadIdx.x / kWarpSize;
  int const lane = threadIdx.x % kWarpSize;
  int const packed_blocks_n = n / kTileK;
  int const n_group = blockIdx.x;
  int const blocks_k = k / kTileK;
  int const blocks_per_split = (blocks_k + Splits - 1) / Splits;
  int const k_begin = warp * blocks_per_split;
  int const k_end = min(blocks_k, k_begin + blocks_per_split);
  int const a_row = (lane / 16) * 4 + lane % 4;
  int const effective_n =
      (lane / 16) * 4 + (lane & 12) * 2 + lane % 4;

  float acc[4][8] = {};
  for (int k_block = k_begin; k_block < k_end; ++k_block) {
#pragma unroll
    for (int n_subtile = 0; n_subtile < 4; ++n_subtile) {
      int const packed_n_block = n_group * 8 + n_subtile * 2;
#if EXL3_SM70_DIRECT_REG_FRAGMENTS
      Half4 b_frag[4];
      reconstruct_mcg_native_n32_fragments<Bits>(
          trellis, b_frag, k_block, packed_blocks_n, packed_n_block, lane,
          effective_n);
#else
      half* b_tile = &shared.mainloop.b[warp][0][0];
      uint32_t const* packed0 = reinterpret_cast<const uint32_t*>(
          trellis + (k_block * packed_blocks_n + packed_n_block) *
                        (16 * Bits));
      uint32_t const* packed1 = reinterpret_cast<const uint32_t*>(
          trellis + (k_block * packed_blocks_n + packed_n_block + 1) *
                        (16 * Bits));
      reconstruct_mcg_tile_strided<Bits>(packed0, b_tile, 0, lane);
      reconstruct_mcg_tile_strided<Bits>(packed1, b_tile, 16, lane);
      __syncwarp();
#endif

#pragma unroll
      for (int sub_k = 0; sub_k < 2; ++sub_k) {
        int const k0 = k_block * kTileK + sub_k * 8;
        Half4 const a0 =
            *reinterpret_cast<const Half4*>(x_had + a_row * k + k0);
        Half4 const a1 =
            *reinterpret_cast<const Half4*>(x_had + a_row * k + k0 + 4);
#if EXL3_SM70_DIRECT_REG_FRAGMENTS
        mma_sm70_m8n8k4(acc[n_subtile], a0, b_frag[sub_k * 2]);
        mma_sm70_m8n8k4(acc[n_subtile], a1, b_frag[sub_k * 2 + 1]);
#else
        Half4 const bf0 = *reinterpret_cast<const Half4*>(
            b_tile + effective_n * kTileK + sub_k * 8);
        Half4 const bf1 = *reinterpret_cast<const Half4*>(
            b_tile + effective_n * kTileK + sub_k * 8 + 4);
        mma_sm70_m8n8k4(acc[n_subtile], a0, bf0);
        mma_sm70_m8n8k4(acc[n_subtile], a1, bf1);
#endif
      }
#if !EXL3_SM70_DIRECT_REG_FRAGMENTS
      __syncwarp();
#endif
    }
  }

  __syncthreads();
  int const c_row = (lane & 1) + (lane / 16) * 4;
  int const c_col = (lane & 2) + (lane & 12) * 2;
#pragma unroll
  for (int n_subtile = 0; n_subtile < 4; ++n_subtile) {
    int const n0 = n_subtile * 32;
    shared.partial[warp][c_row][n0 + c_col] = acc[n_subtile][0];
    shared.partial[warp][c_row][n0 + c_col + 1] = acc[n_subtile][1];
    shared.partial[warp][c_row + 2][n0 + c_col] = acc[n_subtile][2];
    shared.partial[warp][c_row + 2][n0 + c_col + 1] = acc[n_subtile][3];
    shared.partial[warp][c_row][n0 + c_col + 4] = acc[n_subtile][4];
    shared.partial[warp][c_row][n0 + c_col + 5] = acc[n_subtile][5];
    shared.partial[warp][c_row + 2][n0 + c_col + 4] = acc[n_subtile][6];
    shared.partial[warp][c_row + 2][n0 + c_col + 5] = acc[n_subtile][7];
  }
  __syncthreads();

  if (warp == 0) {
    for (int index = lane; index < 8 * kHadamard; index += kWarpSize) {
      float sum = 0.0f;
#pragma unroll
      for (int split = 0; split < Splits; ++split) {
        // Index through an explicitly flattened row-major view. Addressing
        // partial[split][0][index] for index >= 128 happens to linearize with
        // nvcc, but is formally outside the innermost array and gives the
        // optimizer unnecessary latitude.
        float const* split_partial = &shared.partial[split][0][0];
        sum += split_partial[index];
      }
      float* reduced_partial = &shared.partial[0][0][0];
      reduced_partial[index] = sum;
    }
  }
  __syncthreads();

  if (warp < m) {
    int const n0 = n_group * kHadamard;
    output_hadamard_128(&shared.partial[0][warp][0],
                        out + warp * n + n0, svh + n0, lane);
  }
}

template <int Bits, int Splits>
__global__ __launch_bounds__(Splits * kWarpSize, 2)
    void sm70_exl3_mcg_decode_native_cooperative_kernel(
        const half* __restrict__ x, const half* __restrict__ suh,
        half* __restrict__ x_had, const uint16_t* __restrict__ trellis,
        float* __restrict__ accum, half* __restrict__ out,
        const half* __restrict__ svh, int m, int n, int k) {
  __shared__ Sm70Exl3NativeDecodeShared<Splits> shared;

  cg::grid_group const grid = cg::this_grid();
  int const warp = threadIdx.x / kWarpSize;
  int const lane = threadIdx.x % kWarpSize;
  int const grid_warp = blockIdx.x * Splits + warp;
  int const grid_warps = gridDim.x * Splits;
  int const input_blocks = k / kHadamard;

  // Only eight padded rows are consumed by the native M8 atom. Distribute the
  // input transform across all resident warps, then make it visible to the
  // projection phase without returning to the host launch stream.
  for (int task = grid_warp; task < 8 * input_blocks; task += grid_warps) {
    int const row = task / input_blocks;
    int const k0 = (task % input_blocks) * kHadamard;
    half* dst = x_had + row * k + k0;
    if (row < m) {
      input_hadamard_128(x + row * k + k0, suh + k0, dst, lane);
    } else {
      *reinterpret_cast<Half4*>(dst + lane * 4) =
          Half4{__float2half2_rn(0.0f), __float2half2_rn(0.0f)};
    }
  }
  grid.sync();

  int const packed_blocks_n = n / kTileK;
  int const blocks_k = k / kTileK;
  int const blocks_per_split = (blocks_k + Splits - 1) / Splits;
  int const k_begin = warp * blocks_per_split;
  int const k_end = min(blocks_k, k_begin + blocks_per_split);
  int const a_row = (lane / 16) * 4 + lane % 4;
  int const effective_n =
      (lane / 16) * 4 + (lane & 12) * 2 + lane % 4;

  for (int n_block = blockIdx.x; n_block < n / 32;
       n_block += gridDim.x) {
    int const packed_n_block = n_block * 2;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_block = k_begin; k_block < k_end; ++k_block) {
      Half4 b_frag[4];
      reconstruct_mcg_native_n32_fragments<Bits>(
          trellis, b_frag, k_block, packed_blocks_n, packed_n_block, lane,
          effective_n);
#pragma unroll
      for (int sub_k = 0; sub_k < 2; ++sub_k) {
        int const k0 = k_block * kTileK + sub_k * 8;
        Half4 const a0 =
            *reinterpret_cast<const Half4*>(x_had + a_row * k + k0);
        Half4 const a1 =
            *reinterpret_cast<const Half4*>(x_had + a_row * k + k0 + 4);
        mma_sm70_m8n8k4(acc, a0, b_frag[sub_k * 2]);
        mma_sm70_m8n8k4(acc, a1, b_frag[sub_k * 2 + 1]);
      }
    }

    __syncthreads();
    int const c_row = (lane & 1) + (lane / 16) * 4;
    int const c_col = (lane & 2) + (lane & 12) * 2;
    shared.partial[warp][c_row][c_col] = acc[0];
    shared.partial[warp][c_row][c_col + 1] = acc[1];
    shared.partial[warp][c_row + 2][c_col] = acc[2];
    shared.partial[warp][c_row + 2][c_col + 1] = acc[3];
    shared.partial[warp][c_row][c_col + 4] = acc[4];
    shared.partial[warp][c_row][c_col + 5] = acc[5];
    shared.partial[warp][c_row + 2][c_col + 4] = acc[6];
    shared.partial[warp][c_row + 2][c_col + 5] = acc[7];
    __syncthreads();

    if (warp == 0) {
      for (int index = lane; index < 8 * 32; index += kWarpSize) {
        float sum = 0.0f;
#pragma unroll
        for (int split = 0; split < Splits; ++split) {
          sum += shared.partial[split][0][index];
        }
        int const row = index / 32;
        int const col = index % 32;
        accum[row * n + n_block * 32 + col] = sum;
      }
    }
    __syncthreads();
  }
  grid.sync();

  int const output_blocks = n / kHadamard;
  for (int task = grid_warp; task < m * output_blocks; task += grid_warps) {
    int const row = task / output_blocks;
    int const n0 = (task % output_blocks) * kHadamard;
    output_hadamard_128(accum + row * n + n0, out + row * n + n0,
                        svh + n0, lane);
  }
}

template <int Bits, int Splits>
__global__ __launch_bounds__(Splits * kWarpSize, 4)
    void sm70_exl3_mcg_decode_cooperative_kernel(
        const half* __restrict__ x, const half* __restrict__ suh,
        half* __restrict__ x_had, const uint16_t* __restrict__ trellis,
        float* __restrict__ accum, half* __restrict__ out,
        const half* __restrict__ svh, int m, int n, int k) {
  __shared__ Sm70Exl3DecodeShared<Splits> shared;

  cg::grid_group const grid = cg::this_grid();
  int const warp = threadIdx.x / kWarpSize;
  int const lane = threadIdx.x % kWarpSize;
  int const grid_warp = blockIdx.x * Splits + warp;
  int const grid_warps = gridDim.x * Splits;
  int const input_blocks = k / kHadamard;

  // Phase 1: all resident warps cooperatively produce the padded input. The
  // padded rows preserve the fixed m16 WMMA shape for M=1..4.
  for (int task = grid_warp; task < kTileM * input_blocks;
       task += grid_warps) {
    int const row = task / input_blocks;
    int const k0 = (task % input_blocks) * kHadamard;
    half* dst = x_had + row * k + k0;
    if (row < m) {
      input_hadamard_128(x + row * k + k0, suh + k0, dst, lane);
    } else {
      *reinterpret_cast<Half4*>(dst + lane * 4) =
          Half4{__float2half2_rn(0.0f), __float2half2_rn(0.0f)};
    }
  }
  grid.sync();

  int const blocks_n = n / kTileK;
  int const blocks_k = k / kTileK;
  int const blocks_per_split = (blocks_k + Splits - 1) / Splits;
  int const k_begin = warp * blocks_per_split;
  int const k_end = min(blocks_k, k_begin + blocks_per_split);

  // Phase 2: a persistent resident block owns one N=16 tile at a time. Each
  // compressed K/N tile is consumed by exactly one split-K warp and decoded
  // only into the FP16 operand tile that immediately feeds WMMA.
  for (int n_block = blockIdx.x; n_block < blocks_n;
       n_block += gridDim.x) {
    wmma::fragment<wmma::accumulator, kTileM, kTileK, kTileK, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k_block = k_begin; k_block < k_end; ++k_block) {
      constexpr int kPackedWords = Bits * 8;
      uint32_t const* packed_global = reinterpret_cast<const uint32_t*>(
          trellis + (k_block * blocks_n + n_block) * (16 * Bits));
      uint32_t const word0 = packed_global[lane];
      uint32_t const word1 =
          lane + kWarpSize < kPackedWords
              ? packed_global[lane + kWarpSize]
              : 0u;

      half (*b_tile)[kTileK] = shared.mainloop.b[warp];
      reconstruct_mcg_tile_from_regs<Bits>(word0, word1, b_tile, lane);
      __syncwarp();

      wmma::fragment<wmma::matrix_a, kTileM, kTileK, kTileK, half,
                     wmma::row_major>
          a_frag;
      wmma::fragment<wmma::matrix_b, kTileM, kTileK, kTileK, half,
                     wmma::row_major>
          b_frag;
      wmma::load_matrix_sync(a_frag, x_had + k_block * kTileK, k);
      wmma::load_matrix_sync(b_frag, &b_tile[0][0], kTileK);
      wmma::mma_sync(acc, a_frag, b_frag, acc);
      __syncwarp();
    }

    __syncthreads();
    wmma::store_matrix_sync(&shared.partial[warp][0][0], acc, kTileK,
                            wmma::mem_row_major);
    __syncthreads();

    if (warp == 0) {
      for (int index = lane; index < kTileM * kTileK;
           index += kWarpSize) {
        float sum = 0.0f;
#pragma unroll
        for (int split = 0; split < Splits; ++split) {
          sum += shared.partial[split][0][index];
        }
        int const row = index / kTileK;
        int const col = index % kTileK;
        accum[row * n + n_block * kTileK + col] = sum;
      }
    }
    // The partial array aliases the next decoded-B tile.
    __syncthreads();
  }
  grid.sync();

  // Phase 3: only real rows are transformed and committed to the output.
  int const output_blocks = n / kHadamard;
  for (int task = grid_warp; task < m * output_blocks;
       task += grid_warps) {
    int const row = task / output_blocks;
    int const n0 = (task % output_blocks) * kHadamard;
    output_hadamard_128(accum + row * n + n0, out + row * n + n0, svh + n0,
                        lane);
  }
}

template <int Bits>
__global__ __launch_bounds__(kWarps * kWarpSize, 1) void sm70_exl3_mcg_kernel(
    const half* __restrict__ x, const uint16_t* __restrict__ trellis,
    const half* __restrict__ suh, const half* __restrict__ svh,
    half* __restrict__ out, int m, int n, int k) {
  __shared__ Sm70Exl3Shared shared;

  int const warp = threadIdx.x / kWarpSize;
  int const lane = threadIdx.x % kWarpSize;
  int const m0 = blockIdx.y * kTileM;
  int const valid_m = min(kTileM, m - m0);
  int const n0 = blockIdx.x * kTileN;
  int const n_block = n0 / kTileK + warp;
  int const blocks_n = n / kTileK;

  wmma::fragment<wmma::accumulator, kTileM, kTileK, kTileK, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k_group = 0; k_group < k; k_group += kHadamard) {
    // One warp owns each row transform. Invalid WMMA rows are explicitly zero.
    for (int row = warp; row < kTileM; row += kWarps) {
      half* dst = &shared.a_had[row][0];
      if (row < valid_m) {
        input_hadamard_128(x + (m0 + row) * k + k_group,
                           suh + k_group, dst, lane);
      } else {
        *reinterpret_cast<Half4*>(dst + lane * 4) =
            Half4{__float2half2_rn(0.0f), __float2half2_rn(0.0f)};
      }
    }
    __syncthreads();

#pragma unroll
    for (int sub_k = 0; sub_k < kHadamard / kTileK; ++sub_k) {
      int const k_block = k_group / kTileK + sub_k;
      uint16_t const* packed_global =
          trellis + (k_block * blocks_n + n_block) * (16 * Bits);
      uint16_t* packed_shared = shared.workspace.mainloop.packed[warp];
#pragma unroll
      for (int i = lane; i < 16 * Bits; i += kWarpSize) {
        packed_shared[i] = packed_global[i];
      }
      __syncwarp();

      half (*b_tile)[kTileK] = shared.workspace.mainloop.b[warp];
      reconstruct_mcg_tile<Bits>(
          reinterpret_cast<const uint32_t*>(packed_shared), b_tile, lane);
      __syncwarp();

      wmma::fragment<wmma::matrix_a, kTileM, kTileK, kTileK, half,
                     wmma::row_major>
          a_frag;
      wmma::fragment<wmma::matrix_b, kTileM, kTileK, kTileK, half,
                     wmma::row_major>
          b_frag;
      wmma::load_matrix_sync(a_frag, &shared.a_had[0][sub_k * kTileK],
                             kHadamard);
      wmma::load_matrix_sync(b_frag, &b_tile[0][0], kTileK);
      wmma::mma_sync(acc, a_frag, b_frag, acc);
      __syncwarp();
    }
    __syncthreads();
  }

  wmma::store_matrix_sync(&shared.workspace.c[0][warp * kTileK], acc,
                          kTileN, wmma::mem_row_major);
  __syncthreads();

  // One warp owns each row post-transform and applies the exact EXL3 svh scale.
  for (int row = warp; row < valid_m; row += kWarps) {
    output_hadamard_128(&shared.workspace.c[row][0],
                        out + (m0 + row) * n + n0, svh + n0, lane);
  }
}

template <int Bits>
torch::Tensor launch_sm70_exl3_mcg(torch::Tensor const& x,
                                   torch::Tensor const& trellis,
                                   torch::Tensor const& suh,
                                   torch::Tensor const& svh) {
  int64_t const m = x.size(0);
  int64_t const k = x.size(1);
  int64_t const n = trellis.size(1) * 16;
  auto out = torch::empty({m, n}, x.options().dtype(at::ScalarType::Half));
  dim3 const block(kWarps * kWarpSize);
  dim3 const grid(static_cast<unsigned>(n / kTileN),
                  static_cast<unsigned>((m + kTileM - 1) / kTileM));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  sm70_exl3_mcg_kernel<Bits><<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(svh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), static_cast<int>(m),
      static_cast<int>(n), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor launch_sm70_exl3_mcg_dispatch(
    torch::Tensor const& x, torch::Tensor const& trellis,
    torch::Tensor const& suh, torch::Tensor const& svh) {
  int64_t const bits = trellis.size(2) / 16;
  switch (bits) {
    case 4:
      return launch_sm70_exl3_mcg<4>(x, trellis, suh, svh);
    case 5:
      return launch_sm70_exl3_mcg<5>(x, trellis, suh, svh);
    case 6:
      return launch_sm70_exl3_mcg<6>(x, trellis, suh, svh);
    default:
      TORCH_CHECK(false, "SM70 EXL3 fused gate/up supports K4/K5/K6, got K",
                  bits);
  }
}

template <int Bits, int Splits = kKSplits>
torch::Tensor launch_sm70_exl3_mcg_decode(torch::Tensor const& x,
                                          torch::Tensor const& trellis,
                                          torch::Tensor const& suh,
                                          torch::Tensor const& svh) {
  int64_t const m = x.size(0);
  int64_t const k = x.size(1);
  int64_t const n = trellis.size(1) * 16;
  constexpr int kPaddedM = kTileM;
  auto x_had =
      torch::empty({kPaddedM, k}, x.options().dtype(at::ScalarType::Half));
  auto accum = torch::empty({kPaddedM, n},
                            x.options().dtype(at::ScalarType::Float));
  auto out = torch::empty({m, n}, x.options().dtype(at::ScalarType::Half));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();

  dim3 const had_block(kWarpSize);
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard), kPaddedM);
  sm70_exl3_input_hadamard_kernel<<<input_grid, had_block, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));

  dim3 const gemm_block(Splits * kWarpSize);
  if (n == 1024 || n == 3072) {
    dim3 const gemm_grid(static_cast<unsigned>(n / kTileK));
    sm70_exl3_mcg_decode_kernel<Bits, Splits>
        <<<gemm_grid, gemm_block, 0, stream>>>(
            reinterpret_cast<const half*>(x_had.data_ptr<at::Half>()),
            reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
            accum.data_ptr<float>(), static_cast<int>(n), static_cast<int>(k));
  } else {
    dim3 const gemm_grid(static_cast<unsigned>(n / 32));
    sm70_exl3_mcg_decode_native_kernel<Bits, Splits>
        <<<gemm_grid, gemm_block, 0, stream>>>(
            reinterpret_cast<const half*>(x_had.data_ptr<at::Half>()),
            reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
            accum.data_ptr<float>(), static_cast<int>(n), static_cast<int>(k));
  }

  dim3 const output_grid(static_cast<unsigned>(n / kHadamard),
                         static_cast<unsigned>(m));
  sm70_exl3_output_hadamard_kernel<<<output_grid, had_block, 0, stream>>>(
      accum.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(svh.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <int Bits, int Splits>
torch::Tensor launch_sm70_exl3_mcg_decode_accum(
    torch::Tensor const& x, torch::Tensor const& trellis,
    torch::Tensor const& suh) {
  int64_t const m = x.size(0);
  int64_t const k = x.size(1);
  int64_t const n = trellis.size(1) * 16;
  constexpr int kPaddedM = kTileM;
  auto x_had =
      torch::empty({kPaddedM, k}, x.options().dtype(at::ScalarType::Half));
  auto accum = torch::empty({kPaddedM, n},
                            x.options().dtype(at::ScalarType::Float));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();

  dim3 const had_block(kWarpSize);
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard), kPaddedM);
  sm70_exl3_input_hadamard_kernel<<<input_grid, had_block, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));

  dim3 const gemm_block(Splits * kWarpSize);
  if (n == 1024 || n == 3072) {
    dim3 const gemm_grid(static_cast<unsigned>(n / kTileK));
    sm70_exl3_mcg_decode_kernel<Bits, Splits>
        <<<gemm_grid, gemm_block, 0, stream>>>(
            reinterpret_cast<const half*>(x_had.data_ptr<at::Half>()),
            reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
            accum.data_ptr<float>(), static_cast<int>(n), static_cast<int>(k));
  } else {
    dim3 const gemm_grid(static_cast<unsigned>(n / 32));
    sm70_exl3_mcg_decode_native_kernel<Bits, Splits>
        <<<gemm_grid, gemm_block, 0, stream>>>(
            reinterpret_cast<const half*>(x_had.data_ptr<at::Half>()),
            reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
            accum.data_ptr<float>(), static_cast<int>(n), static_cast<int>(k));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return accum;
}

torch::Tensor launch_sm70_exl3_mcg_decode_accum_dispatch(
    torch::Tensor const& x, torch::Tensor const& trellis,
    torch::Tensor const& suh) {
  int64_t const bits = trellis.size(2) / 16;
  int64_t const n = trellis.size(1) * 16;
  int const splits = sm70_exl3_mcg_decode_splits(bits, n);
  switch (bits) {
    case 4:
      return launch_sm70_exl3_mcg_decode_accum<4, 8>(x, trellis, suh);
    case 5:
      return splits == 16
                 ? launch_sm70_exl3_mcg_decode_accum<5, 16>(x, trellis, suh)
                 : launch_sm70_exl3_mcg_decode_accum<5, 8>(x, trellis, suh);
    case 6:
      TORCH_CHECK(splits == 8,
                  "SM70 EXL3 fused gate/up excludes vocabulary projections");
      return launch_sm70_exl3_mcg_decode_accum<6, 8>(x, trellis, suh);
    default:
      TORCH_CHECK(false, "SM70 EXL3 fused gate/up supports K4/K5/K6, got K",
                  bits);
  }
}

template <int Bits, int Splits>
torch::Tensor launch_sm70_exl3_mcg_decode_native_n128(
    torch::Tensor const& x, torch::Tensor const& trellis,
    torch::Tensor const& suh, torch::Tensor const& svh) {
  int64_t const m = x.size(0);
  int64_t const k = x.size(1);
  int64_t const n = trellis.size(1) * 16;
  auto x_had = torch::empty({8, k}, x.options().dtype(at::ScalarType::Half));
  auto out = torch::empty({m, n}, x.options().dtype(at::ScalarType::Half));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();

  dim3 const had_block(kWarpSize);
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard), 8);
  sm70_exl3_input_hadamard_kernel<<<input_grid, had_block, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));

  dim3 const gemm_grid(static_cast<unsigned>(n / kHadamard));
  dim3 const gemm_block(Splits * kWarpSize);
  sm70_exl3_mcg_decode_native_n128_kernel<Bits, Splits>
      <<<gemm_grid, gemm_block, 0, stream>>>(
          reinterpret_cast<const half*>(x_had.data_ptr<at::Half>()),
          reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(svh.data_ptr<at::Half>()),
          static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// The cooperative and dense-materialize launchers below are intentionally
// unreachable from registered operators. They are rejected research variants,
// not qualified runtime fallbacks.
template <int Bits, int Splits>
torch::Tensor launch_sm70_exl3_mcg_decode_native_cooperative(
    torch::Tensor const& x, torch::Tensor const& trellis,
    torch::Tensor const& suh, torch::Tensor const& svh) {
  int64_t const m = x.size(0);
  int64_t const k = x.size(1);
  int64_t const n = trellis.size(1) * 16;
  auto x_had = torch::empty({8, k}, x.options().dtype(at::ScalarType::Half));
  auto accum =
      torch::empty({8, n}, x.options().dtype(at::ScalarType::Float));
  auto out = torch::empty({m, n}, x.options().dtype(at::ScalarType::Half));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();

  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm,
      sm70_exl3_mcg_decode_native_cooperative_kernel<Bits, Splits>,
      Splits * kWarpSize, 0));
  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, x.get_device()));
  int const grid_blocks =
      min(static_cast<int>(n / 32), blocks_per_sm * sm_count);
  TORCH_CHECK(grid_blocks > 0,
              "SM70 EXL3 native cooperative decode has no resident blocks");

  const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  const half* suh_ptr =
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>());
  half* x_had_ptr = reinterpret_cast<half*>(x_had.data_ptr<at::Half>());
  const uint16_t* trellis_ptr =
      reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>());
  float* accum_ptr = accum.data_ptr<float>();
  half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const half* svh_ptr =
      reinterpret_cast<const half*>(svh.data_ptr<at::Half>());
  int m_arg = static_cast<int>(m);
  int n_arg = static_cast<int>(n);
  int k_arg = static_cast<int>(k);
  void* args[] = {&x_ptr,       &suh_ptr, &x_had_ptr, &trellis_ptr,
                  &accum_ptr,   &out_ptr, &svh_ptr,   &m_arg,
                  &n_arg,       &k_arg};

  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(
          sm70_exl3_mcg_decode_native_cooperative_kernel<Bits, Splits>),
      dim3(static_cast<unsigned>(grid_blocks)),
      dim3(Splits * kWarpSize), args, 0, stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <int Bits>
torch::Tensor launch_sm70_exl3_mcg_decode_cooperative(
    torch::Tensor const& x, torch::Tensor const& trellis,
    torch::Tensor const& suh, torch::Tensor const& svh) {
  int64_t const m = x.size(0);
  int64_t const k = x.size(1);
  int64_t const n = trellis.size(1) * 16;
  constexpr int kPaddedM = kTileM;
  auto x_had =
      torch::empty({kPaddedM, k}, x.options().dtype(at::ScalarType::Half));
  auto accum = torch::empty({kPaddedM, n},
                            x.options().dtype(at::ScalarType::Float));
  auto out = torch::empty({m, n}, x.options().dtype(at::ScalarType::Half));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();

  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm,
      sm70_exl3_mcg_decode_cooperative_kernel<Bits, kKSplits>,
      kKSplits * kWarpSize, 0));
  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, x.get_device()));
  int const grid_blocks =
      min(static_cast<int>(n / kTileK), blocks_per_sm * sm_count);
  TORCH_CHECK(grid_blocks > 0,
              "SM70 EXL3 cooperative decode has no resident blocks");

  const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  const half* suh_ptr =
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>());
  half* x_had_ptr = reinterpret_cast<half*>(x_had.data_ptr<at::Half>());
  const uint16_t* trellis_ptr =
      reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>());
  float* accum_ptr = accum.data_ptr<float>();
  half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const half* svh_ptr =
      reinterpret_cast<const half*>(svh.data_ptr<at::Half>());
  int m_arg = static_cast<int>(m);
  int n_arg = static_cast<int>(n);
  int k_arg = static_cast<int>(k);
  void* args[] = {&x_ptr,       &suh_ptr, &x_had_ptr, &trellis_ptr,
                  &accum_ptr,   &out_ptr, &svh_ptr,   &m_arg,
                  &n_arg,       &k_arg};

  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(
          sm70_exl3_mcg_decode_cooperative_kernel<Bits, kKSplits>),
      dim3(static_cast<unsigned>(grid_blocks)),
      dim3(kKSplits * kWarpSize), args, 0, stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <int Bits>
torch::Tensor launch_sm70_exl3_mcg_large_m(torch::Tensor const& x,
                                           torch::Tensor const& trellis,
                                           torch::Tensor const& suh,
                                           torch::Tensor const& svh) {
  int64_t const m = x.size(0);
  int64_t const k = x.size(1);
  int64_t const n = trellis.size(1) * 16;
  auto x_had = torch::empty_like(x);
  auto dense_b = torch::empty({k, n}, x.options().dtype(at::ScalarType::Half));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();

  dim3 const had_block(kWarpSize);
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                        static_cast<unsigned>(m));
  sm70_exl3_input_hadamard_kernel<<<input_grid, had_block, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));

  dim3 const reconstruct_grid(static_cast<unsigned>(n / kTileN * kWarps),
                              static_cast<unsigned>(k / kTileK));
  sm70_exl3_reconstruct_mcg_kernel<Bits>
      <<<reconstruct_grid, had_block, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
          reinterpret_cast<half*>(dense_b.data_ptr<at::Half>()),
          static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Keep the GEMM result in FP32 until after the output Hadamard transform.
  // Returning FP16 from at::matmul here would introduce an extra rounding
  // point that the fused small-M WMMA path and ExLlama reconstruction do not
  // have.
  auto accum = torch::empty({m, n}, x.options().dtype(at::ScalarType::Float));
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
              "SM70 EXL3 failed to set the cuBLAS stream");
  float const alpha = 1.0f;
  float const beta = 0.0f;
  cublasStatus_t const status = cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n),
      static_cast<int>(m), static_cast<int>(k), &alpha,
      dense_b.data_ptr<at::Half>(), CUDA_R_16F, static_cast<int>(n),
      x_had.data_ptr<at::Half>(), CUDA_R_16F, static_cast<int>(k), &beta,
      accum.data_ptr<float>(), CUDA_R_32F, static_cast<int>(n),
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "SM70 EXL3 FP16xFP16->FP32 cuBLAS GEMM failed with status ",
              static_cast<int>(status));

  auto out = torch::empty({m, n}, x.options().dtype(at::ScalarType::Half));
  dim3 const output_grid(static_cast<unsigned>(n / kHadamard),
                         static_cast<unsigned>(m));
  sm70_exl3_output_hadamard_kernel<<<output_grid, had_block, 0, stream>>>(
      accum.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(svh.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor exl3_sm70_gemm(torch::Tensor const& x,
                             torch::Tensor const& trellis,
                             torch::Tensor const& suh,
                             torch::Tensor const& svh, bool mcg, bool mul1) {
  c10::cuda::CUDAGuard device_guard(x.device());
  TORCH_CHECK(x.is_cuda() && trellis.is_cuda() && suh.is_cuda() && svh.is_cuda(),
              "SM70 EXL3 tensors must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Half && x.dim() == 2,
              "SM70 EXL3 input must be rank-2 float16");
  TORCH_CHECK(trellis.scalar_type() == at::ScalarType::Short &&
                  trellis.dim() == 3,
              "SM70 EXL3 trellis must be rank-3 int16");
  TORCH_CHECK(suh.scalar_type() == at::ScalarType::Half && suh.dim() == 1 &&
                  svh.scalar_type() == at::ScalarType::Half && svh.dim() == 1,
              "SM70 EXL3 Hadamard scales must be rank-1 float16");
  TORCH_CHECK(x.is_contiguous() && trellis.is_contiguous() &&
                  suh.is_contiguous() && svh.is_contiguous(),
              "SM70 EXL3 tensors must be contiguous");
  TORCH_CHECK(mcg && !mul1,
              "SM70 EXL3 initial kernel supports only the MCG codebook");

  int64_t const m = x.size(0);
  int64_t const k = x.size(1);
  int64_t const n = trellis.size(1) * 16;
  int64_t const bits = trellis.size(2) / 16;
  int const splits = sm70_exl3_mcg_decode_splits(bits, n);
  TORCH_CHECK(m > 0, "SM70 EXL3 input must contain at least one row");
  TORCH_CHECK(k == trellis.size(0) * 16 && suh.numel() == k &&
                  svh.numel() == n,
              "SM70 EXL3 tensor dimensions disagree");
  TORCH_CHECK(k % kHadamard == 0 && n % kTileN == 0,
              "SM70 EXL3 requires K and N divisible by 128, got K=", k,
              ", N=", n);
  TORCH_CHECK(trellis.size(2) == bits * 16,
              "SM70 EXL3 invalid trellis bit dimension");

  switch (bits) {
    case 4:
      return m <= 4 ? launch_sm70_exl3_mcg_decode<4, 8>(x, trellis, suh, svh)
                    : launch_sm70_exl3_mcg<4>(x, trellis, suh, svh);
    case 5:
      if (m <= 4) {
        if (splits == 16) {
          return launch_sm70_exl3_mcg_decode<5, 16>(x, trellis, suh, svh);
        }
        return launch_sm70_exl3_mcg_decode<5, 8>(x, trellis, suh, svh);
      }
      return launch_sm70_exl3_mcg<5>(x, trellis, suh, svh);
    case 6:
      if (m <= 4) {
        // The TP4 vocabulary shard has thousands of N tiles, so split-K4
        // minimizes reduction overhead without sacrificing occupancy. It is
        // also the only measured shape where owning the full N128 Hadamard
        // group is faster: keep that fusion pinned to the exact large-K
        // vocabulary projection and leave every narrower K6 projection on
        // the qualified N32 path.
        if (splits == 4 && k >= 5120) {
          return launch_sm70_exl3_mcg_decode_native_n128<6, 4>(
              x, trellis, suh, svh);
        }
        return splits == 4
                   ? launch_sm70_exl3_mcg_decode<6, 4>(x, trellis, suh, svh)
                   : launch_sm70_exl3_mcg_decode<6, 8>(x, trellis, suh, svh);
      }
      return launch_sm70_exl3_mcg<6>(x, trellis, suh, svh);
    default:
      TORCH_CHECK(false, "SM70 EXL3 supports K4/K5/K6, got K", bits);
  }
}

torch::Tensor exl3_sm70_gemm_n128(torch::Tensor const& x,
                                  torch::Tensor const& trellis,
                                  torch::Tensor const& suh,
                                  torch::Tensor const& svh, bool mcg,
                                  bool mul1) {
  c10::cuda::CUDAGuard device_guard(x.device());
  TORCH_CHECK(x.is_cuda() && trellis.is_cuda() && suh.is_cuda() && svh.is_cuda(),
              "SM70 EXL3 N128 tensors must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Half && x.dim() == 2,
              "SM70 EXL3 N128 input must be rank-2 float16");
  TORCH_CHECK(trellis.scalar_type() == at::ScalarType::Short &&
                  trellis.dim() == 3,
              "SM70 EXL3 N128 trellis must be rank-3 int16");
  TORCH_CHECK(suh.scalar_type() == at::ScalarType::Half && suh.dim() == 1 &&
                  svh.scalar_type() == at::ScalarType::Half && svh.dim() == 1,
              "SM70 EXL3 N128 Hadamard scales must be rank-1 float16");
  TORCH_CHECK(x.is_contiguous() && trellis.is_contiguous() &&
                  suh.is_contiguous() && svh.is_contiguous(),
              "SM70 EXL3 N128 tensors must be contiguous");
  TORCH_CHECK(mcg && !mul1,
              "SM70 EXL3 N128 kernel supports only the MCG codebook");

  int64_t const m = x.size(0);
  int64_t const k = x.size(1);
  int64_t const n = trellis.size(1) * 16;
  int64_t const bits = trellis.size(2) / 16;
  TORCH_CHECK(m > 0 && m <= 4,
              "SM70 EXL3 N128 kernel requires 1 <= M <= 4, got ", m);
  TORCH_CHECK(k == trellis.size(0) * 16 && suh.numel() == k &&
                  svh.numel() == n,
              "SM70 EXL3 N128 tensor dimensions disagree");
  TORCH_CHECK(k % kHadamard == 0 && n % kHadamard == 0,
              "SM70 EXL3 N128 requires K and N divisible by 128, got K=", k,
              ", N=", n);
  TORCH_CHECK(trellis.size(2) == bits * 16,
              "SM70 EXL3 N128 invalid trellis bit dimension");

  switch (bits) {
    case 4:
      return launch_sm70_exl3_mcg_decode_native_n128<4, 8>(x, trellis, suh,
                                                            svh);
    case 5:
      // N128 ownership already reduces the grid by 4x. Keep split-K8 so the
      // CTA stays below the 48-KiB static shared-memory threshold on SM70;
      // narrow split-K16 projections remain on the qualified N32 path.
      return launch_sm70_exl3_mcg_decode_native_n128<5, 8>(x, trellis, suh,
                                                            svh);
    case 6:
      return n >= 32768
                 ? launch_sm70_exl3_mcg_decode_native_n128<6, 4>(
                       x, trellis, suh, svh)
                 : launch_sm70_exl3_mcg_decode_native_n128<6, 8>(
                       x, trellis, suh, svh);
    default:
      TORCH_CHECK(false, "SM70 EXL3 N128 supports K4/K5/K6, got K", bits);
  }
}

torch::Tensor exl3_sm70_gate_up_silu_mul(
    torch::Tensor const& x, torch::Tensor const& gate_trellis,
    torch::Tensor const& up_trellis, torch::Tensor const& gate_suh,
    torch::Tensor const& up_suh, torch::Tensor const& gate_svh,
    torch::Tensor const& up_svh, bool gate_mcg, bool up_mcg,
    bool gate_mul1, bool up_mul1) {
  c10::cuda::CUDAGuard device_guard(x.device());
  TORCH_CHECK(x.is_cuda() && gate_trellis.is_cuda() && up_trellis.is_cuda() &&
                  gate_suh.is_cuda() && up_suh.is_cuda() &&
                  gate_svh.is_cuda() && up_svh.is_cuda(),
              "SM70 EXL3 fused gate/up tensors must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Half && x.dim() == 2,
              "SM70 EXL3 fused gate/up input must be rank-2 float16");
  TORCH_CHECK(gate_trellis.scalar_type() == at::ScalarType::Short &&
                  up_trellis.scalar_type() == at::ScalarType::Short &&
                  gate_trellis.dim() == 3 && up_trellis.dim() == 3,
              "SM70 EXL3 fused gate/up trellises must be rank-3 int16");
  TORCH_CHECK(gate_suh.scalar_type() == at::ScalarType::Half &&
                  up_suh.scalar_type() == at::ScalarType::Half &&
                  gate_svh.scalar_type() == at::ScalarType::Half &&
                  up_svh.scalar_type() == at::ScalarType::Half &&
                  gate_suh.dim() == 1 && up_suh.dim() == 1 &&
                  gate_svh.dim() == 1 && up_svh.dim() == 1,
              "SM70 EXL3 fused gate/up Hadamard scales must be rank-1 float16");
  TORCH_CHECK(x.is_contiguous() && gate_trellis.is_contiguous() &&
                  up_trellis.is_contiguous() && gate_suh.is_contiguous() &&
                  up_suh.is_contiguous() && gate_svh.is_contiguous() &&
                  up_svh.is_contiguous(),
              "SM70 EXL3 fused gate/up tensors must be contiguous");
  TORCH_CHECK(gate_mcg && up_mcg && !gate_mul1 && !up_mul1,
              "SM70 EXL3 fused gate/up supports only the MCG codebook");

  int64_t const m = x.size(0);
  int64_t const k = x.size(1);
  int64_t const gate_n = gate_trellis.size(1) * 16;
  int64_t const up_n = up_trellis.size(1) * 16;
  int64_t const gate_bits = gate_trellis.size(2) / 16;
  int64_t const up_bits = up_trellis.size(2) / 16;
  TORCH_CHECK(m > 0,
              "SM70 EXL3 fused gate/up requires at least one row, got ", m);
  TORCH_CHECK(gate_n == up_n,
              "SM70 EXL3 fused gate/up requires equal packed N, got ", gate_n,
              " and ", up_n);
  TORCH_CHECK(k == gate_trellis.size(0) * 16 &&
                  k == up_trellis.size(0) * 16 && gate_suh.numel() == k &&
                  up_suh.numel() == k && gate_svh.numel() == gate_n &&
                  up_svh.numel() == up_n,
              "SM70 EXL3 fused gate/up tensor dimensions disagree");
  TORCH_CHECK(k % kHadamard == 0 && gate_n % kHadamard == 0,
              "SM70 EXL3 fused gate/up requires K and N divisible by 128");
  TORCH_CHECK(gate_trellis.size(2) == gate_bits * 16 &&
                  up_trellis.size(2) == up_bits * 16,
              "SM70 EXL3 fused gate/up invalid trellis bit dimension");
  TORCH_CHECK(gate_n < 32768,
              "SM70 EXL3 fused gate/up is only qualified for non-vocabulary "
              "N32 decode shapes");

  auto out = torch::empty({m, gate_n},
                          x.options().dtype(at::ScalarType::Half));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  if (m > 4) {
    // Preserve the established M>4 projection accumulation tree and remove
    // only the concatenation before SiLU/multiply.
    auto gate_out = launch_sm70_exl3_mcg_dispatch(
        x, gate_trellis, gate_suh, gate_svh);
    auto up_out = launch_sm70_exl3_mcg_dispatch(
        x, up_trellis, up_suh, up_svh);
    int64_t const elements = m * gate_n;
    constexpr int kThreads = 256;
    int const blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
    sm70_exl3_silu_mul_pair_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(gate_out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(up_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        static_cast<int>(elements));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }

  auto gate_accum = launch_sm70_exl3_mcg_decode_accum_dispatch(
      x, gate_trellis, gate_suh);
  auto up_accum = launch_sm70_exl3_mcg_decode_accum_dispatch(
      x, up_trellis, up_suh);
  dim3 const block(kWarpSize);
  dim3 const grid(static_cast<unsigned>(gate_n / kHadamard),
                  static_cast<unsigned>(m));
  sm70_exl3_output_hadamard_silu_mul_kernel<<<grid, block, 0, stream>>>(
      gate_accum.data_ptr<float>(), up_accum.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate_svh.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up_svh.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(gate_n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::pair<int, int> sm70_exl3_tm_state_policy(int64_t bits, int64_t k,
                                               int64_t n, int64_t m) {
  // The FP8-derived two-stage B pipeline changes the M=1 occupancy/split-K
  // optimum.  Keep M=4 on its independently measured policies until the
  // verifier path is retuned; non-MTP decode is the first promotion target.
  if (m == 1) {
    if (bits == 5 && k == 5120 && n == 2560) return {8, 2};
    if (bits == 5 && k == 5120 && n == 1536) return {6, 2};
    if (bits == 5 && k == 1536 && n == 5120) return {5, 0};
    if (bits == 5 && k == 5120 && n == 4352) return {7, 0};
    if (bits == 5 && k == 5120 && n == 3072) return {6, 2};
    if (bits == 5 && k == 5120 && n == 1024) return {9, 3};
    if (bits == 6 && k == 4352 && n == 5120) return {4, 0};
    if (bits == 6 && k == 5120 && n == 62080) return {3, 0};
  }
  // The exact-state two-stage B pipeline and fused output-Hadamard changed
  // the verifier-width optimum after the original M=4 policies were chosen.
  // These policies are isolated to the fixed Qwen3.8 MTP3 verifier width;
  // M=2/3/5-8 retain their established accumulation order.  The split-K
  // reassociation is quality-gated rather than bitwise: the real-weight sweep
  // measured max_abs <= 2.44e-4 and relative_l2 <= 3.1e-5.
  if (m == 4) {
    if (bits == 5 && k == 5120 && n == 512) return {11, 0};
    if (bits == 5 && k == 5120 && n == 1536) return {7, 0};
    if (bits == 5 && k == 1536 && n == 5120) return {5, 0};
    if (bits == 5 && k == 5120 && n == 3072) return {6, 0};
    if (bits == 5 && k == 5120 && n == 256) return {11, 0};
    // The single-matrix sweep favored split 7, but the live two-branch
    // gate/up scheduler regressed by 0.386 ms across 64 layers.  Preserve its
    // grouped-grid split 11 policy; the other seven policies won live.
    if (bits == 5 && k == 5120 && n == 4352) return {11, 1};
    if (bits == 6 && k == 4352 && n == 5120) return {6, 2};
    if (bits == 6 && k == 5120 && n == 62080) return {3, 2};
  }
  if (bits == 5 && k == 5120 && n == 2560) {
    return {11, 0};
  }
  if (bits == 6 && k == 5120 && n == 62080) {
    return {1, 0};
  }

  std::pair<int, int> policy{8, 0};
  if (bits == 5 && k == 5120 && n == 1536) {
    policy = {15, 0};
  } else if (bits == 5 && k == 5120 && n == 3072) {
    policy = {11, 0};
  } else if (bits == 5 && k == 5120 && n == 4352) {
    policy = {11, 1};
  } else if (bits == 5 && k == 5120 && n == 1024) {
    policy = {9, 0};
  } else if (bits == 6 && k == 4352 && n == 5120) {
    policy = {9, 2};
  }
  policy.first = std::min(
      policy.first, static_cast<int>(std::max<int64_t>(1, k / 128)));
  return policy;
}

std::pair<int, int> sm70_exl3_tm_raw_policy(int64_t bits, int64_t k,
                                             int64_t n, int64_t m) {
  // Real Qwen3.8 checkpoint sweep for the destination-local trellis decoder.
  // The raw operand has materially different register pressure from both the
  // expanded exact-state and INT8 paths, so it must not inherit their policy
  // tables merely because all three share the TurboMind scheduler.
  if (m == 1) {
    if (bits == 5 && k == 5120 && n == 512) return {7, 0};
    if (bits == 5 && k == 5120 && n == 256) return {7, 1};
    if (bits == 5 && k == 5120 && n == 2560) return {8, 2};
    if (bits == 5 && k == 5120 && n == 1536) return {6, 0};
    if (bits == 5 && k == 1536 && n == 5120) return {4, 1};
    if (bits == 5 && k == 5120 && n == 4352) return {7, 0};
    if (bits == 5 && k == 5120 && n == 3072) return {6, 0};
    if (bits == 5 && k == 5120 && n == 1024) return {9, 1};
    if (bits == 6 && k == 4352 && n == 5120) return {4, 1};
    if (bits == 6 && k == 5120 && n == 62080) return {4, 0};
  }
  if (m == 4) {
    if (bits == 5 && k == 5120 && n == 2560) return {8, 0};
    if (bits == 5 && k == 5120 && n == 1536) return {4, 0};
    if (bits == 5 && k == 1536 && n == 5120) return {4, 0};
    if (bits == 5 && k == 5120 && n == 4352) return {7, 0};
    if (bits == 5 && k == 5120 && n == 3072) return {6, 2};
    if (bits == 5 && k == 5120 && n == 1024) return {9, 3};
    if (bits == 6 && k == 4352 && n == 5120) return {4, 2};
  }
  return sm70_exl3_tm_state_policy(bits, k, n, m);
}

torch::Tensor exl3_sm70_tm_raw_pair_gemm(
    torch::Tensor const& x, torch::Tensor const& trellis0,
    torch::Tensor const& trellis1, torch::Tensor const& suh0,
    torch::Tensor const& suh1, torch::Tensor const& svh0,
    torch::Tensor const& svh1, torch::Tensor const& metadata,
    torch::Tensor const& offsets, torch::Tensor const& locks, int64_t bits,
    int64_t requested_splits, int64_t requested_swizzle,
    bool fused_output) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);
  int64_t const k = trellis0.size(0) * 16;
  int64_t const n = trellis0.size(1) * 16;
  TORCH_CHECK(
      x.is_cuda() && trellis0.is_cuda() && trellis1.is_cuda() &&
          suh0.is_cuda() && suh1.is_cuda() && svh0.is_cuda() &&
          svh1.is_cuda() && metadata.is_cuda() && offsets.is_cuda() &&
          locks.is_cuda(),
      "TurboMind EXL3 raw paired-projection tensors must be CUDA tensors");
  TORCH_CHECK(
      x.scalar_type() == at::ScalarType::Half &&
          trellis0.scalar_type() == at::ScalarType::Short &&
          trellis1.scalar_type() == at::ScalarType::Short &&
          suh0.scalar_type() == at::ScalarType::Half &&
          suh1.scalar_type() == at::ScalarType::Half &&
          svh0.scalar_type() == at::ScalarType::Half &&
          svh1.scalar_type() == at::ScalarType::Half &&
          metadata.scalar_type() == at::ScalarType::Byte &&
          offsets.scalar_type() == at::ScalarType::Int &&
          locks.scalar_type() == at::ScalarType::Int,
      "TurboMind EXL3 raw paired-projection tensor dtypes disagree");
  TORCH_CHECK(
      x.is_contiguous() && trellis0.is_contiguous() &&
          trellis1.is_contiguous() && suh0.is_contiguous() &&
          suh1.is_contiguous() && svh0.is_contiguous() &&
          svh1.is_contiguous() && metadata.is_contiguous() &&
          offsets.is_contiguous() && locks.is_contiguous(),
      "TurboMind EXL3 raw paired-projection tensors must be contiguous");
  TORCH_CHECK(
      m > 0 && x.dim() == 2 && x.size(1) == k && trellis0.dim() == 3 &&
          trellis0.sizes() == trellis1.sizes() &&
          trellis0.size(2) == bits * 16 && suh0.numel() == k &&
          suh1.numel() == k && svh0.numel() == n && svh1.numel() == n &&
          metadata.numel() >=
              static_cast<int64_t>(4 * sizeof(tm::StridedPtr)) &&
          offsets.numel() >= 9 && locks.numel() >= 2 * (n / kHadamard) &&
          k % kHadamard == 0 && n % kHadamard == 0,
      "TurboMind EXL3 raw paired-projection tensor shapes disagree");

  if (m > 8) {
    auto out0 = exl3_sm70_gemm(x, trellis0, suh0, svh0, true, false);
    auto out1 = exl3_sm70_gemm(x, trellis1, suh1, svh1, true, false);
    return at::cat({out0, out1}, 1);
  }

  auto out =
      torch::empty({m, 2 * n}, x.options().dtype(at::ScalarType::Half));
  auto x_had =
      torch::empty({16, k}, x.options().dtype(at::ScalarType::Half));
  auto partials =
      torch::empty({16, n}, x.options().dtype(at::ScalarType::Float));
  auto accum =
      torch::empty({16, n}, x.options().dtype(at::ScalarType::Float));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  bool const shared_input_transform = suh0.data_ptr() == suh1.data_ptr();
  if (shared_input_transform) {
    dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                          static_cast<unsigned>(m));
    sm70_exl3_input_hadamard_kernel<<<input_grid, kWarpSize, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(suh0.data_ptr<at::Half>()),
        reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
        static_cast<int>(m), static_cast<int>(k));
  } else {
    dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                          static_cast<unsigned>(m), 2);
    sm70_exl3_input_hadamard_pair_kernel<<<input_grid, kWarpSize, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(suh0.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(suh1.data_ptr<at::Half>()),
        reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
        static_cast<int>(m), static_cast<int>(k));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto const default_policy = sm70_exl3_tm_raw_policy(bits, k, n, m);
  int const splits = requested_splits > 0
                         ? static_cast<int>(requested_splits)
                         : default_policy.first;
  int const swizzle = requested_splits > 0
                          ? static_cast<int>(requested_swizzle)
                          : default_policy.second;
  TORCH_CHECK(splits >= 1 && splits <= k / 128 && swizzle >= 0 &&
                  swizzle <= 5,
              "TurboMind EXL3 raw pair split/swizzle is out of range");
  if (fused_output) {
    switch (bits) {
      case 4:
        launch_tm_sm70_exl3_raw_pair_core_out<4>(
            out, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
        break;
      case 5:
        launch_tm_sm70_exl3_raw_pair_core_out<5>(
            out, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
        break;
      case 6:
        launch_tm_sm70_exl3_raw_pair_core_out<6>(
            out, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
        break;
      default:
        TORCH_CHECK(false,
                    "TurboMind EXL3 raw pair supports K4/K5/K6");
    }
    return out;
  }

  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_raw_grouped_accum_core_out<4, true>(
          accum, x_had, metadata, offsets, partials, locks,
          static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
          splits, swizzle);
      break;
    case 5:
      launch_tm_sm70_exl3_raw_grouped_accum_core_out<5, true>(
          accum, x_had, metadata, offsets, partials, locks,
          static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
          splits, swizzle);
      break;
    case 6:
      launch_tm_sm70_exl3_raw_grouped_accum_core_out<6, true>(
          accum, x_had, metadata, offsets, partials, locks,
          static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
          splits, swizzle);
      break;
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 raw pair supports K4/K5/K6");
  }
  dim3 const output_grid(static_cast<unsigned>(n / kHadamard),
                         static_cast<unsigned>(m), 2);
  sm70_exl3_output_hadamard_pair_kernel<<<output_grid, kWarpSize, 0, stream>>>(
      accum.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(svh0.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(svh1.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor exl3_sm70_tm_state_pair_gemm(
    torch::Tensor const& x, torch::Tensor const& trellis0,
    torch::Tensor const& trellis1, torch::Tensor const& state0,
    torch::Tensor const& state1, torch::Tensor const& suh0,
    torch::Tensor const& suh1, torch::Tensor const& svh0,
    torch::Tensor const& svh1, torch::Tensor const& metadata,
    torch::Tensor const& offsets, torch::Tensor const& locks, int64_t bits) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);
  int64_t const k = state0.size(0) * 16;
  int64_t const n = state0.size(1) * 32;

  TORCH_CHECK(x.is_cuda() && trellis0.is_cuda() && trellis1.is_cuda() &&
                  state0.is_cuda() && state1.is_cuda() && suh0.is_cuda() &&
                  suh1.is_cuda() && svh0.is_cuda() && svh1.is_cuda() &&
                  metadata.is_cuda() && offsets.is_cuda() && locks.is_cuda(),
              "TurboMind EXL3 paired projection tensors must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Half &&
                  trellis0.scalar_type() == at::ScalarType::Short &&
                  trellis1.scalar_type() == at::ScalarType::Short &&
                  state0.scalar_type() == at::ScalarType::Int &&
                  state1.scalar_type() == at::ScalarType::Int &&
                  suh0.scalar_type() == at::ScalarType::Half &&
                  suh1.scalar_type() == at::ScalarType::Half &&
                  svh0.scalar_type() == at::ScalarType::Half &&
                  svh1.scalar_type() == at::ScalarType::Half &&
                  metadata.scalar_type() == at::ScalarType::Byte &&
                  offsets.scalar_type() == at::ScalarType::Int &&
                  locks.scalar_type() == at::ScalarType::Int,
              "TurboMind EXL3 paired projection tensor dtypes disagree");
  TORCH_CHECK(x.is_contiguous() && trellis0.is_contiguous() &&
                  trellis1.is_contiguous() && state0.is_contiguous() &&
                  state1.is_contiguous() && suh0.is_contiguous() &&
                  suh1.is_contiguous() && svh0.is_contiguous() &&
                  svh1.is_contiguous() && metadata.is_contiguous() &&
                  offsets.is_contiguous() && locks.is_contiguous(),
              "TurboMind EXL3 paired projection tensors must be contiguous");
  TORCH_CHECK(
      m > 0 && x.dim() == 2 && x.size(1) == k && trellis0.dim() == 3 &&
          trellis0.sizes() == trellis1.sizes() && state0.dim() == 4 &&
          state0.sizes() == state1.sizes() &&
          sm70_exl3_tm_state_shape_valid(state0, bits) &&
          trellis0.size(0) * 16 == k && trellis0.size(1) * 16 == n &&
          trellis0.size(2) == bits * 16 && suh0.numel() == k &&
          suh1.numel() == k && svh0.numel() == n && svh1.numel() == n &&
          metadata.numel() >=
              static_cast<int64_t>(4 * sizeof(tm::StridedPtr)) &&
          offsets.numel() >= 9 && k % kHadamard == 0 &&
          n % kHadamard == 0,
      "TurboMind EXL3 paired projection tensor shapes disagree");

  // A capture-time M<=8 graph may be reused for a large prefill shape.  Keep
  // the exact established trellis implementation as the runtime fallback.
  if (m > 8) {
    auto out0 = exl3_sm70_gemm(x, trellis0, suh0, svh0, true, false);
    auto out1 = exl3_sm70_gemm(x, trellis1, suh1, svh1, true, false);
    return at::cat({out0, out1}, 1);
  }

  auto out =
      torch::empty({m, 2 * n}, x.options().dtype(at::ScalarType::Half));
  auto x_had =
      torch::empty({16, k}, x.options().dtype(at::ScalarType::Half));
  auto partials =
      torch::empty({16, n}, x.options().dtype(at::ScalarType::Float));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  bool const shared_input_transform = suh0.data_ptr() == suh1.data_ptr();
  if (shared_input_transform) {
    dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                          static_cast<unsigned>(m));
    sm70_exl3_input_hadamard_kernel<<<input_grid, kWarpSize, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(suh0.data_ptr<at::Half>()),
        reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
        static_cast<int>(m), static_cast<int>(k));
  } else {
    dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                          static_cast<unsigned>(m), 2);
    sm70_exl3_input_hadamard_pair_kernel<<<input_grid, kWarpSize, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(suh0.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(suh1.data_ptr<at::Half>()),
        reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
        static_cast<int>(m), static_cast<int>(k));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto const [splits, swizzle] = sm70_exl3_tm_state_policy(bits, k, n, m);
  bool const interleave =
      sm70_exl3_tm_use_interleaved_state_decode(bits, k, n);
  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_state_pair_core_out<4>(
          out, x_had, metadata, offsets, partials, locks,
          static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
          splits, swizzle);
      break;
    case 5:
      if (interleave) {
        launch_tm_sm70_exl3_state_pair_core_out<5, true>(
            out, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
      } else {
        launch_tm_sm70_exl3_state_pair_core_out<5>(
            out, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
      }
      break;
    case 6:
      if (interleave) {
        launch_tm_sm70_exl3_state_pair_core_out<6, true>(
            out, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
      } else {
        launch_tm_sm70_exl3_state_pair_core_out<6>(
            out, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
      }
      break;
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 paired projection supports K4/K5/K6");
  }
  return out;
}

torch::Tensor exl3_sm70_tm_raw_gate_up_silu_mul(
    torch::Tensor const& x, torch::Tensor const& gate_trellis,
    torch::Tensor const& up_trellis, torch::Tensor const& gate_suh,
    torch::Tensor const& up_suh, torch::Tensor const& gate_svh,
    torch::Tensor const& up_svh, torch::Tensor const& metadata,
    torch::Tensor const& offsets, torch::Tensor const& locks, int64_t bits,
    int64_t requested_splits, int64_t requested_swizzle,
    bool fused_output) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);
  if (m > 8) {
    return exl3_sm70_gate_up_silu_mul(
        x, gate_trellis, up_trellis, gate_suh, up_suh, gate_svh, up_svh,
        true, true, false, false);
  }
  int64_t const k = gate_trellis.size(0) * 16;
  int64_t const n = gate_trellis.size(1) * 16;
  TORCH_CHECK(
      x.is_cuda() && gate_trellis.is_cuda() && up_trellis.is_cuda() &&
          gate_suh.is_cuda() && up_suh.is_cuda() && gate_svh.is_cuda() &&
          up_svh.is_cuda() && metadata.is_cuda() && offsets.is_cuda() &&
          locks.is_cuda(),
      "TurboMind EXL3 raw gate/up tensors must be CUDA tensors");
  TORCH_CHECK(
      x.scalar_type() == at::ScalarType::Half &&
          gate_trellis.scalar_type() == at::ScalarType::Short &&
          up_trellis.scalar_type() == at::ScalarType::Short &&
          gate_suh.scalar_type() == at::ScalarType::Half &&
          up_suh.scalar_type() == at::ScalarType::Half &&
          gate_svh.scalar_type() == at::ScalarType::Half &&
          up_svh.scalar_type() == at::ScalarType::Half &&
          metadata.scalar_type() == at::ScalarType::Byte &&
          offsets.scalar_type() == at::ScalarType::Int &&
          locks.scalar_type() == at::ScalarType::Int,
      "TurboMind EXL3 raw gate/up tensor dtypes disagree");
  TORCH_CHECK(
      x.is_contiguous() && gate_trellis.is_contiguous() &&
          up_trellis.is_contiguous() && gate_suh.is_contiguous() &&
          up_suh.is_contiguous() && gate_svh.is_contiguous() &&
          up_svh.is_contiguous() && metadata.is_contiguous() &&
          offsets.is_contiguous() && locks.is_contiguous(),
      "TurboMind EXL3 raw gate/up tensors must be contiguous");
  TORCH_CHECK(
      m > 0 && m <= 8 && x.dim() == 2 && x.size(1) == k &&
          gate_trellis.dim() == 3 &&
          gate_trellis.sizes() == up_trellis.sizes() &&
          gate_trellis.size(2) == bits * 16 && gate_suh.numel() == k &&
          up_suh.numel() == k && gate_svh.numel() == n &&
          up_svh.numel() == n && k % kHadamard == 0 &&
          n % kHadamard == 0 &&
          metadata.numel() >=
              static_cast<int64_t>(4 * sizeof(tm::StridedPtr)) &&
          offsets.numel() >= 6 && locks.numel() >= 2 * (n / kHadamard),
      "TurboMind EXL3 raw gate/up tensor shapes disagree");

  auto out = torch::empty({m, n}, x.options().dtype(at::ScalarType::Half));
  auto x_had = torch::empty({16, k}, x.options().dtype(at::ScalarType::Half));
  auto projected =
      torch::empty({16, n}, x.options().dtype(at::ScalarType::Half));
  auto partials =
      torch::empty({16, n}, x.options().dtype(at::ScalarType::Float));
  auto accum =
      torch::empty({16, n}, x.options().dtype(at::ScalarType::Float));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                        static_cast<unsigned>(m), 2);
  sm70_exl3_input_hadamard_pair_kernel<<<input_grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate_suh.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up_suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto const default_policy = sm70_exl3_tm_raw_policy(bits, k, n, m);
  int const splits = requested_splits > 0
                         ? static_cast<int>(requested_splits)
                         : default_policy.first;
  int const swizzle = requested_splits > 0
                          ? static_cast<int>(requested_swizzle)
                          : default_policy.second;
  TORCH_CHECK(splits >= 1 && splits <= k / 128 && swizzle >= 0 &&
                  swizzle <= 5,
              "TurboMind EXL3 raw gate/up split/swizzle is out of range");
  if (fused_output) {
    switch (bits) {
      case 4:
        launch_tm_sm70_exl3_raw_gate_up_core_out<4>(
            projected, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
        break;
      case 5:
        launch_tm_sm70_exl3_raw_gate_up_core_out<5>(
            projected, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
        break;
      case 6:
        launch_tm_sm70_exl3_raw_gate_up_core_out<6>(
            projected, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
        break;
      default:
        TORCH_CHECK(false,
                    "TurboMind EXL3 raw gate/up supports K4/K5/K6");
    }

    int64_t const elements = m * n;
    constexpr int kThreads = 256;
    int const blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
    half const* projected_ptr =
        reinterpret_cast<const half*>(projected.data_ptr<at::Half>());
    sm70_exl3_silu_mul_pair_kernel<<<blocks, kThreads, 0, stream>>>(
        projected_ptr, projected_ptr + 8 * n,
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        static_cast<int>(elements));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }

  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_raw_grouped_accum_core_out<4, false>(
          accum, x_had, metadata, offsets, partials, locks,
          static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
          splits, swizzle);
      break;
    case 5:
      launch_tm_sm70_exl3_raw_grouped_accum_core_out<5, false>(
          accum, x_had, metadata, offsets, partials, locks,
          static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
          splits, swizzle);
      break;
    case 6:
      launch_tm_sm70_exl3_raw_grouped_accum_core_out<6, false>(
          accum, x_had, metadata, offsets, partials, locks,
          static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
          splits, swizzle);
      break;
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 raw gate/up supports K4/K5/K6");
  }
  dim3 const output_grid(static_cast<unsigned>(n / kHadamard),
                         static_cast<unsigned>(m));
  sm70_exl3_output_hadamard_silu_mul_kernel<<<output_grid, kWarpSize, 0,
                                               stream>>>(
      accum.data_ptr<float>(), accum.data_ptr<float>() + 8 * n,
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate_svh.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up_svh.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor exl3_sm70_tm_state_gate_up_silu_mul(
    torch::Tensor const& x, torch::Tensor const& gate_trellis,
    torch::Tensor const& up_trellis, torch::Tensor const& gate_state,
    torch::Tensor const& up_state, torch::Tensor const& gate_suh,
    torch::Tensor const& up_suh, torch::Tensor const& gate_svh,
    torch::Tensor const& up_svh, torch::Tensor const& metadata,
    torch::Tensor const& offsets, torch::Tensor const& locks, int64_t bits,
    int64_t requested_splits, int64_t requested_swizzle) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);
  // Dynamo captures this custom op from an M<=8 decode example but reuses the
  // enclosing symbolic graph for prefill.  Keep the run-time boundary inside
  // C++ so a large-M call takes the already qualified exact trellis path.
  if (m > 8) {
    return exl3_sm70_gate_up_silu_mul(
        x, gate_trellis, up_trellis, gate_suh, up_suh, gate_svh, up_svh,
        true, true, false, false);
  }
  int64_t const k = gate_state.size(0) * 16;
  int64_t const n = gate_state.size(1) * 32;
  TORCH_CHECK(x.is_cuda() && gate_state.is_cuda() && up_state.is_cuda() &&
                  gate_suh.is_cuda() && up_suh.is_cuda() &&
                  gate_svh.is_cuda() && up_svh.is_cuda() &&
                  metadata.is_cuda() && offsets.is_cuda() && locks.is_cuda(),
              "TurboMind EXL3 paired gate/up tensors must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Half &&
                  gate_state.scalar_type() == at::ScalarType::Int &&
                  up_state.scalar_type() == at::ScalarType::Int &&
                  gate_suh.scalar_type() == at::ScalarType::Half &&
                  up_suh.scalar_type() == at::ScalarType::Half &&
                  gate_svh.scalar_type() == at::ScalarType::Half &&
                  up_svh.scalar_type() == at::ScalarType::Half &&
                  metadata.scalar_type() == at::ScalarType::Byte &&
                  offsets.scalar_type() == at::ScalarType::Int &&
                  locks.scalar_type() == at::ScalarType::Int,
              "TurboMind EXL3 paired gate/up tensor dtypes disagree");
  TORCH_CHECK(x.is_contiguous() && gate_state.is_contiguous() &&
                  up_state.is_contiguous() && gate_suh.is_contiguous() &&
                  up_suh.is_contiguous() && gate_svh.is_contiguous() &&
                  up_svh.is_contiguous() && metadata.is_contiguous() &&
                  offsets.is_contiguous() && locks.is_contiguous(),
              "TurboMind EXL3 paired gate/up tensors must be contiguous");
  TORCH_CHECK(m > 0 && m <= 8 && x.dim() == 2 && x.size(1) == k &&
                  gate_state.dim() == 4 &&
                  gate_state.sizes() == up_state.sizes() &&
                  sm70_exl3_tm_state_shape_valid(gate_state, bits) &&
                  gate_suh.numel() == k && up_suh.numel() == k &&
                  gate_svh.numel() == n && up_svh.numel() == n &&
                  k % kHadamard == 0 && n % kHadamard == 0 &&
                  metadata.numel() >=
                      static_cast<int64_t>(4 * sizeof(tm::StridedPtr)) &&
                  offsets.numel() >= 6,
              "TurboMind EXL3 paired gate/up tensor shapes disagree");

  auto out = torch::empty({m, n}, x.options().dtype(at::ScalarType::Half));
  // Fixed branch stride keeps the grouped MatrixParam offsets graph-static
  // for every supported M while the scheduler exposes only the live rows.
  auto x_had = torch::empty({16, k}, x.options().dtype(at::ScalarType::Half));
  auto projected =
      torch::empty({16, n}, x.options().dtype(at::ScalarType::Half));
  auto partials =
      torch::empty({16, n}, x.options().dtype(at::ScalarType::Float));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                        static_cast<unsigned>(m), 2);
  sm70_exl3_input_hadamard_pair_kernel<<<input_grid, kWarpSize, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate_suh.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up_suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto const default_policy = sm70_exl3_tm_state_policy(bits, k, n, m);
  int const splits = requested_splits > 0
                         ? static_cast<int>(requested_splits)
                         : default_policy.first;
  int const swizzle = requested_splits > 0
                          ? static_cast<int>(requested_swizzle)
                          : default_policy.second;
  TORCH_CHECK(splits >= 1 && splits <= k / 128 && swizzle >= 0 &&
                  swizzle <= 5,
              "TurboMind EXL3 paired gate/up split/swizzle is out of range");
  bool const interleave =
      sm70_exl3_tm_use_interleaved_state_decode(bits, k, n);
  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_state_gate_up_core_out<4>(
          projected, x_had, metadata, offsets, partials, locks,
          static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
          splits, swizzle);
      break;
    case 5:
      if (interleave) {
        launch_tm_sm70_exl3_state_gate_up_core_out<5, true>(
            projected, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
      } else {
        launch_tm_sm70_exl3_state_gate_up_core_out<5>(
            projected, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
      }
      break;
    case 6:
      if (interleave) {
        launch_tm_sm70_exl3_state_gate_up_core_out<6, true>(
            projected, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
      } else {
        launch_tm_sm70_exl3_state_gate_up_core_out<6>(
            projected, x_had, metadata, offsets, partials, locks,
            static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
            splits, swizzle);
      }
      break;
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 paired gate/up supports K4/K5/K6");
  }

  int64_t const elements = m * n;
  constexpr int kThreads = 256;
  int const blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
  half const* projected_ptr =
      reinterpret_cast<const half*>(projected.data_ptr<at::Half>());
  sm70_exl3_silu_mul_pair_kernel<<<blocks, kThreads, 0, stream>>>(
      projected_ptr, projected_ptr + 8 * n,
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(elements));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::pair<int, int> sm70_exl3_tm_int8_policy(int64_t bits, int64_t k,
                                              int64_t n, int64_t m);

torch::Tensor exl3_sm70_tm_state_mlp(
    torch::Tensor const& x, torch::Tensor const& gate_trellis,
    torch::Tensor const& up_trellis, torch::Tensor const& down_trellis,
    torch::Tensor const& gate_state, torch::Tensor const& up_state,
    torch::Tensor const& down_state, torch::Tensor const& gate_suh,
    torch::Tensor const& up_suh, torch::Tensor const& down_suh,
    torch::Tensor const& gate_svh, torch::Tensor const& up_svh,
    torch::Tensor const& down_svh, torch::Tensor const& down_packed_lane,
    torch::Tensor const& down_tile_scales,
    torch::Tensor const& gate_metadata, torch::Tensor const& gate_offsets,
    torch::Tensor const& gate_locks, torch::Tensor const& down_locks,
    int64_t gate_bits, int64_t down_bits, bool int8_gate, bool int8_down) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);

  // A decode-shaped custom op can be reused by Dynamo's symbolic prefill
  // graph.  Preserve the established exact large-M path inside C++ rather
  // than putting a data-dependent branch around the custom op in Python.
  if (m > 8) {
    auto activated = exl3_sm70_gate_up_silu_mul(
        x, gate_trellis, up_trellis, gate_suh, up_suh, gate_svh, up_svh,
        true, true, false, false);
    return exl3_sm70_gemm(activated, down_trellis, down_suh, down_svh, true,
                          false);
  }

  int64_t const gate_k = gate_trellis.size(0) * 16;
  int64_t const intermediate_n = gate_trellis.size(1) * 16;
  int64_t const down_k = down_trellis.size(0) * 16;
  int64_t const down_n = down_trellis.size(1) * 16;
  TORCH_CHECK(
      x.is_cuda() && gate_trellis.is_cuda() && up_trellis.is_cuda() &&
          down_trellis.is_cuda() && gate_state.is_cuda() &&
          up_state.is_cuda() && down_state.is_cuda() && gate_suh.is_cuda() &&
          up_suh.is_cuda() && down_suh.is_cuda() && gate_svh.is_cuda() &&
          up_svh.is_cuda() && down_svh.is_cuda() && gate_metadata.is_cuda() &&
          gate_offsets.is_cuda() && gate_locks.is_cuda() &&
          down_locks.is_cuda() &&
          (!int8_down ||
           (down_packed_lane.is_cuda() && down_tile_scales.is_cuda())),
      "TurboMind EXL3 fused-MLP tensors must be CUDA tensors");
  TORCH_CHECK(
      x.scalar_type() == at::ScalarType::Half &&
          gate_trellis.scalar_type() == at::ScalarType::Short &&
          up_trellis.scalar_type() == at::ScalarType::Short &&
          down_trellis.scalar_type() == at::ScalarType::Short &&
          gate_state.scalar_type() == at::ScalarType::Int &&
          up_state.scalar_type() == at::ScalarType::Int &&
          down_state.scalar_type() == at::ScalarType::Int &&
          gate_suh.scalar_type() == at::ScalarType::Half &&
          up_suh.scalar_type() == at::ScalarType::Half &&
          down_suh.scalar_type() == at::ScalarType::Half &&
          gate_svh.scalar_type() == at::ScalarType::Half &&
          up_svh.scalar_type() == at::ScalarType::Half &&
          down_svh.scalar_type() == at::ScalarType::Half &&
          gate_metadata.scalar_type() == at::ScalarType::Byte &&
          gate_offsets.scalar_type() == at::ScalarType::Int &&
          gate_locks.scalar_type() == at::ScalarType::Int &&
          down_locks.scalar_type() == at::ScalarType::Int &&
          (!int8_down ||
           (down_packed_lane.scalar_type() == at::ScalarType::Char &&
          down_tile_scales.scalar_type() == at::ScalarType::Half)),
      "TurboMind EXL3 fused-MLP tensor dtypes disagree");
  TORCH_CHECK(
      x.is_contiguous() && gate_trellis.is_contiguous() &&
          up_trellis.is_contiguous() && down_trellis.is_contiguous() &&
          gate_state.is_contiguous() && up_state.is_contiguous() &&
          down_state.is_contiguous() && gate_suh.is_contiguous() &&
          up_suh.is_contiguous() && down_suh.is_contiguous() &&
          gate_svh.is_contiguous() && up_svh.is_contiguous() &&
          down_svh.is_contiguous() && gate_metadata.is_contiguous() &&
          gate_offsets.is_contiguous() && gate_locks.is_contiguous() &&
          down_locks.is_contiguous() &&
          (!int8_down ||
           (down_packed_lane.is_contiguous() &&
            down_tile_scales.is_contiguous())),
      "TurboMind EXL3 fused-MLP tensors must be contiguous");
  TORCH_CHECK(
      m > 0 && x.dim() == 2 && x.size(1) == gate_k &&
          (int8_gate ||
           (gate_state.dim() == 4 &&
            gate_state.sizes() == up_state.sizes() &&
            sm70_exl3_tm_state_shape_valid(gate_state, gate_bits))) &&
          (int8_down ||
           (down_state.dim() == 4 &&
            sm70_exl3_tm_state_shape_valid(down_state, down_bits))) &&
          gate_trellis.dim() == 3 && up_trellis.dim() == 3 &&
          down_trellis.dim() == 3 &&
          gate_trellis.size(0) * 16 == gate_k &&
          up_trellis.size(0) * 16 == gate_k &&
          down_trellis.size(0) * 16 == down_k &&
          gate_trellis.size(1) * 16 == intermediate_n &&
          up_trellis.size(1) * 16 == intermediate_n &&
          down_trellis.size(1) * 16 == down_n && down_k == intermediate_n &&
          gate_suh.numel() == gate_k && up_suh.numel() == gate_k &&
          down_suh.numel() == down_k && gate_svh.numel() == intermediate_n &&
          up_svh.numel() == intermediate_n && down_svh.numel() == down_n &&
          gate_metadata.numel() >=
              static_cast<int64_t>(4 * sizeof(tm::StridedPtr) +
                                   (int8_gate ? 2 * sizeof(void*) : 0)) &&
          gate_offsets.numel() >= 6 && down_locks.numel() >= down_n / 128 &&
          gate_k % kHadamard == 0 && intermediate_n % kHadamard == 0 &&
          down_n % kHadamard == 0,
      "TurboMind EXL3 fused-MLP tensor shapes disagree");
  TORCH_CHECK(
      !int8_down ||
          (down_packed_lane.dim() == 4 && down_tile_scales.dim() == 2 &&
           down_packed_lane.size(0) * 16 == down_k &&
           down_packed_lane.size(1) * 32 == down_n &&
           down_packed_lane.size(2) == kWarpSize &&
           down_packed_lane.size(3) == 16 &&
           down_tile_scales.size(0) == down_packed_lane.size(0) &&
           down_tile_scales.size(1) == down_packed_lane.size(1)),
      "TurboMind EXL3 fused-MLP INT8 down metadata disagree");
  TORCH_CHECK(gate_trellis.size(2) == gate_bits * 16 &&
                  up_trellis.size(2) == gate_bits * 16 &&
                  down_trellis.size(2) == down_bits * 16,
              "TurboMind EXL3 fused-MLP bit widths disagree");
  TORCH_CHECK(gate_bits == 5 && down_bits == 6 && gate_k == 5120 &&
                  intermediate_n == 4352 && down_k == 4352 &&
                  down_n == 5120 &&
                  sm70_exl3_tm_use_fused_output_hadamard(
                      down_bits, down_k, down_n, m),
              "TurboMind EXL3 fused-MLP is qualified only for the TP4 "
              "Qwen3.8 K5 5120x4352 -> K6 4352x5120 decode shape");

  auto gate_x_had =
      torch::empty({16, gate_k}, x.options().dtype(at::ScalarType::Half));
  auto projected = torch::empty(
      {16, intermediate_n}, x.options().dtype(at::ScalarType::Half));
  auto gate_partials = torch::empty(
      {16, intermediate_n}, x.options().dtype(at::ScalarType::Float));
  auto down_x_had =
      torch::empty({m, down_k}, x.options().dtype(at::ScalarType::Half));
  auto down_partials =
      torch::empty({8, down_n}, x.options().dtype(at::ScalarType::Float));
  auto out =
      torch::empty({m, down_n}, x.options().dtype(at::ScalarType::Half));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  dim3 const gate_input_grid(static_cast<unsigned>(gate_k / kHadamard),
                             static_cast<unsigned>(m), 2);
  sm70_exl3_input_hadamard_pair_kernel<<<gate_input_grid, kWarpSize, 0,
                                         stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate_suh.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up_suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(gate_x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(gate_k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto const gate_policy =
      int8_gate
          ? sm70_exl3_tm_int8_policy(gate_bits, gate_k, intermediate_n, m)
          : sm70_exl3_tm_state_policy(gate_bits, gate_k, intermediate_n, m);
  if (int8_gate) {
    TORCH_CHECK(gate_bits == 5,
                "TurboMind EXL3 fused-MLP INT8 gate/up requires K5");
    launch_tm_sm70_exl3_int8_gate_up_core_out<5>(
        projected, gate_x_had, gate_metadata, gate_offsets, gate_partials,
        gate_locks, static_cast<int>(m), static_cast<int>(gate_k),
        static_cast<int>(intermediate_n), gate_policy.first,
        gate_policy.second);
  } else {
    bool const gate_interleave = sm70_exl3_tm_use_interleaved_state_decode(
        gate_bits, gate_k, intermediate_n);
    switch (gate_bits) {
      case 4:
        launch_tm_sm70_exl3_state_gate_up_core_out<4>(
            projected, gate_x_had, gate_metadata, gate_offsets, gate_partials,
            gate_locks, static_cast<int>(m), static_cast<int>(gate_k),
            static_cast<int>(intermediate_n), gate_policy.first,
            gate_policy.second);
        break;
      case 5:
        if (gate_interleave) {
          launch_tm_sm70_exl3_state_gate_up_core_out<5, true>(
              projected, gate_x_had, gate_metadata, gate_offsets,
              gate_partials, gate_locks, static_cast<int>(m),
              static_cast<int>(gate_k), static_cast<int>(intermediate_n),
              gate_policy.first, gate_policy.second);
        } else {
          launch_tm_sm70_exl3_state_gate_up_core_out<5>(
              projected, gate_x_had, gate_metadata, gate_offsets,
              gate_partials, gate_locks, static_cast<int>(m),
              static_cast<int>(gate_k), static_cast<int>(intermediate_n),
              gate_policy.first, gate_policy.second);
        }
        break;
      case 6:
        if (gate_interleave) {
          launch_tm_sm70_exl3_state_gate_up_core_out<6, true>(
              projected, gate_x_had, gate_metadata, gate_offsets,
              gate_partials, gate_locks, static_cast<int>(m),
              static_cast<int>(gate_k), static_cast<int>(intermediate_n),
              gate_policy.first, gate_policy.second);
        } else {
          launch_tm_sm70_exl3_state_gate_up_core_out<6>(
              projected, gate_x_had, gate_metadata, gate_offsets,
              gate_partials, gate_locks, static_cast<int>(m),
              static_cast<int>(gate_k), static_cast<int>(intermediate_n),
              gate_policy.first, gate_policy.second);
        }
        break;
      default:
        TORCH_CHECK(false,
                    "TurboMind EXL3 fused-MLP gate/up supports K4/K5/K6");
    }
  }

  dim3 const boundary_grid(
      static_cast<unsigned>(intermediate_n / kHadamard),
      static_cast<unsigned>(m));
  half const* projected_ptr =
      reinterpret_cast<const half*>(projected.data_ptr<at::Half>());
  sm70_exl3_silu_mul_input_hadamard_kernel<<<boundary_grid, kWarpSize, 0,
                                             stream>>>(
      projected_ptr, projected_ptr + 8 * intermediate_n,
      reinterpret_cast<const half*>(down_suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(down_x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(intermediate_n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto const down_policy = int8_down
                               ? sm70_exl3_tm_int8_policy(
                                     down_bits, down_k, down_n, m)
                               : sm70_exl3_tm_state_policy(
                                     down_bits, down_k, down_n, m);
  if (int8_down) {
    TORCH_CHECK(down_bits == 6,
                "TurboMind EXL3 fused-MLP INT8 down requires K6");
    launch_tm_sm70_exl3_int8_hadamard_core_out<6>(
        out, down_x_had, down_packed_lane, down_tile_scales, down_svh,
        down_partials, down_locks, down_policy.first, down_policy.second);
    return out;
  }
  bool const down_interleave =
      sm70_exl3_tm_use_interleaved_state_decode(down_bits, down_k, down_n);
  switch (down_bits) {
    case 4:
      launch_tm_sm70_exl3_state_hadamard_core_out<4>(
          out, down_x_had, down_state, down_svh, down_partials, down_locks,
          down_policy.first, down_policy.second);
      break;
    case 5:
      if (down_interleave) {
        launch_tm_sm70_exl3_state_hadamard_core_out<5, true>(
            out, down_x_had, down_state, down_svh, down_partials, down_locks,
            down_policy.first, down_policy.second);
      } else {
        launch_tm_sm70_exl3_state_hadamard_core_out<5>(
            out, down_x_had, down_state, down_svh, down_partials, down_locks,
            down_policy.first, down_policy.second);
      }
      break;
    case 6:
      if (down_interleave) {
        launch_tm_sm70_exl3_state_hadamard_core_out<6, true>(
            out, down_x_had, down_state, down_svh, down_partials, down_locks,
            down_policy.first, down_policy.second);
      } else {
        launch_tm_sm70_exl3_state_hadamard_core_out<6>(
            out, down_x_had, down_state, down_svh, down_partials, down_locks,
            down_policy.first, down_policy.second);
      }
      break;
    default:
      TORCH_CHECK(false, "TurboMind EXL3 fused-MLP down supports K4/K5/K6");
  }
  return out;
}

torch::Tensor exl3_sm70_tm_dispatch_gemm(
    torch::Tensor const& x, torch::Tensor const& trellis,
    torch::Tensor const& state, torch::Tensor const& suh,
    torch::Tensor const& svh, int64_t bits, bool mcg, bool mul1) {
  int64_t const m = x.size(0);
  int64_t const k = trellis.size(0) * 16;
  int64_t const n = trellis.size(1) * 16;
  TORCH_CHECK(m > 0, "TurboMind EXL3 dispatch requires M > 0");
  TORCH_CHECK(trellis.dim() == 3 && state.dim() == 4 &&
                  trellis.size(2) == bits * 16 &&
                  state.size(0) * 16 == k && state.size(1) * 32 == n &&
                  sm70_exl3_tm_state_shape_valid(state, bits),
              "TurboMind EXL3 state/trellis dispatch metadata disagree");
  if (m > 8) {
    return exl3_sm70_gemm(x, trellis, suh, svh, mcg, mul1);
  }
  auto const [splits, swizzle] =
      sm70_exl3_tm_state_policy(bits, k, n, m);
  return exl3_sm70_tm_state_gemm(x, state, suh, svh, bits, splits,
                                  swizzle);
}

torch::Tensor exl3_sm70_tm_dispatch_gemm_persistent_locks(
    torch::Tensor const& x, torch::Tensor const& trellis,
    torch::Tensor const& state, torch::Tensor const& suh,
    torch::Tensor const& svh, torch::Tensor const& locks, int64_t bits,
    bool mcg, bool mul1) {
  int64_t const m = x.size(0);
  int64_t const k = trellis.size(0) * 16;
  int64_t const n = trellis.size(1) * 16;
  TORCH_CHECK(m > 0, "TurboMind EXL3 dispatch requires M > 0");
  TORCH_CHECK(trellis.dim() == 3 && state.dim() == 4 &&
                  trellis.size(2) == bits * 16 &&
                  state.size(0) * 16 == k && state.size(1) * 32 == n &&
                  sm70_exl3_tm_state_shape_valid(state, bits) &&
                  locks.is_cuda() &&
                  locks.scalar_type() == at::ScalarType::Int &&
                  locks.is_contiguous() && locks.numel() >= n / 128,
              "TurboMind EXL3 persistent-lock dispatch metadata disagree");
  if (m > 8) {
    return exl3_sm70_gemm(x, trellis, suh, svh, mcg, mul1);
  }
  auto const [splits, swizzle] =
      sm70_exl3_tm_state_policy(bits, k, n, m);
  return exl3_sm70_tm_state_gemm_persistent_locks(
      x, state, suh, svh, locks, bits, splits, swizzle);
}

torch::Tensor exl3_sm70_tm_raw_dispatch_gemm_persistent_locks(
    torch::Tensor const& x, torch::Tensor const& trellis,
    torch::Tensor const& suh, torch::Tensor const& svh,
    torch::Tensor const& locks, int64_t bits, bool mcg, bool mul1) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);
  int64_t const k = trellis.size(0) * 16;
  int64_t const n = trellis.size(1) * 16;
  TORCH_CHECK(m > 0, "TurboMind EXL3 raw dispatch requires M > 0");
  TORCH_CHECK(
      x.is_cuda() && trellis.is_cuda() && suh.is_cuda() && svh.is_cuda() &&
          locks.is_cuda() && x.scalar_type() == at::ScalarType::Half &&
          trellis.scalar_type() == at::ScalarType::Short &&
          suh.scalar_type() == at::ScalarType::Half &&
          svh.scalar_type() == at::ScalarType::Half &&
          locks.scalar_type() == at::ScalarType::Int && x.dim() == 2 &&
          trellis.dim() == 3 && x.is_contiguous() &&
          trellis.is_contiguous() && suh.is_contiguous() &&
          svh.is_contiguous() && locks.is_contiguous() &&
          x.size(1) == k && trellis.size(2) == bits * 16 &&
          trellis.stride(2) == 1 &&
          trellis.stride(1) == trellis.size(2) &&
          trellis.stride(0) == trellis.size(1) * trellis.size(2) &&
          suh.numel() == k && svh.numel() == n,
      "TurboMind EXL3 raw dispatch metadata disagree");

  // The copied TurboMind raw-B mainloop reconstructs the MCG codebook.  Keep
  // every other codebook and large symbolic-prefill shape on the established
  // bit-faithful implementation.
  if (m > 8 || !mcg || mul1) {
    return exl3_sm70_gemm(x, trellis, suh, svh, mcg, mul1);
  }
  TORCH_CHECK(
      bits >= 4 && bits <= 6 && k % 128 == 0 && n % 128 == 0 &&
          locks.numel() >= ((m + 7) / 8) * (n / 128),
      "TurboMind EXL3 raw dispatch workspace or dimensions are invalid");

  auto const [splits, swizzle] = sm70_exl3_tm_raw_policy(bits, k, n, m);
  auto out = torch::empty({m, n}, x.options().dtype(at::ScalarType::Half));
  auto x_had = torch::empty_like(x);
  auto partials =
      torch::empty({8, n}, x.options().dtype(at::ScalarType::Float));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  dim3 const block(kWarpSize);
  dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                        static_cast<unsigned>(m));
  sm70_exl3_input_hadamard_kernel<<<input_grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  bool const fused_output =
      sm70_exl3_tm_use_fused_output_hadamard(bits, k, n, m);
  if (fused_output) {
    switch (bits) {
      case 4:
        launch_tm_sm70_exl3_raw_hadamard_core_out<4>(
            out, x_had, trellis, svh, partials, locks, splits, swizzle);
        return out;
      case 5:
        launch_tm_sm70_exl3_raw_hadamard_core_out<5>(
            out, x_had, trellis, svh, partials, locks, splits, swizzle);
        return out;
      case 6:
        launch_tm_sm70_exl3_raw_hadamard_core_out<6>(
            out, x_had, trellis, svh, partials, locks, splits, swizzle);
        return out;
      default:
        TORCH_CHECK(false,
                    "TurboMind EXL3 raw dispatch supports K4/K5/K6");
    }
  }

  auto accum =
      torch::empty({m, n}, x.options().dtype(at::ScalarType::Float));
  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_core_out<4, 0, true>(
          accum, x_had, trellis, partials, locks, splits, swizzle);
      break;
    case 5:
      launch_tm_sm70_exl3_core_out<5, 0, true>(
          accum, x_had, trellis, partials, locks, splits, swizzle);
      break;
    case 6:
      launch_tm_sm70_exl3_core_out<6, 0, true>(
          accum, x_had, trellis, partials, locks, splits, swizzle);
      break;
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 raw dispatch supports K4/K5/K6");
  }
  dim3 const output_grid(static_cast<unsigned>(n / kHadamard),
                         static_cast<unsigned>(m));
  sm70_exl3_output_hadamard_kernel<<<output_grid, block, 0, stream>>>(
      accum.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(svh.data_ptr<at::Half>()),
      static_cast<int>(m), static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::pair<int, int> sm70_exl3_tm_int8_policy(int64_t bits, int64_t k,
                                              int64_t n, int64_t m) {
  // Real-weight v44 sweep.  M=2/3 share verifier-like occupancy and use the
  // qualified M=4 policy; other decode widths retain a conservative split.
  bool const verifier_width = m >= 2 && m <= 4;
  if (bits == 5 && k == 5120 && n == 512) {
    return verifier_width ? std::pair{9, 1} : std::pair{7, 3};
  }
  if (bits == 5 && k == 5120 && n == 1536) {
    return verifier_width ? std::pair{6, 2} : std::pair{6, 1};
  }
  if (bits == 5 && k == 1536 && n == 5120) {
    return verifier_width ? std::pair{5, 1} : std::pair{5, 0};
  }
  if (bits == 5 && k == 5120 && n == 3072) {
    return verifier_width ? std::pair{6, 1} : std::pair{6, 1};
  }
  if (bits == 5 && k == 5120 && n == 256) {
    return verifier_width ? std::pair{7, 2} : std::pair{7, 0};
  }
  if (bits == 5 && k == 5120 && n == 4352) {
    return {7, 0};
  }
  if (bits == 6 && k == 4352 && n == 5120) {
    return verifier_width ? std::pair{6, 1} : std::pair{6, 2};
  }
  if (bits == 6 && k == 5120 && n == 62080) {
    return verifier_width ? std::pair{4, 0} : std::pair{4, 3};
  }
  return {std::min<int64_t>(8, std::max<int64_t>(1, k / 128)), 0};
}

torch::Tensor exl3_sm70_tm_int8_pair_gemm(
    torch::Tensor const& x, torch::Tensor const& trellis0,
    torch::Tensor const& trellis1, torch::Tensor const& packed0,
    torch::Tensor const& scales0, torch::Tensor const& packed1,
    torch::Tensor const& scales1, torch::Tensor const& suh0,
    torch::Tensor const& suh1, torch::Tensor const& svh0,
    torch::Tensor const& svh1, torch::Tensor const& metadata,
    torch::Tensor const& offsets, torch::Tensor const& locks, int64_t bits) {
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t const m = x.size(0);
  int64_t const k = packed0.size(0) * 16;
  int64_t const n = packed0.size(1) * 32;
  constexpr int64_t kMetadataBytes =
      4 * sizeof(tm::StridedPtr) + 2 * sizeof(void*);

  TORCH_CHECK(
      x.is_cuda() && trellis0.is_cuda() && trellis1.is_cuda() &&
          packed0.is_cuda() && scales0.is_cuda() && packed1.is_cuda() &&
          scales1.is_cuda() && suh0.is_cuda() && suh1.is_cuda() &&
          svh0.is_cuda() && svh1.is_cuda() && metadata.is_cuda() &&
          offsets.is_cuda() && locks.is_cuda(),
      "TurboMind EXL3 INT8 paired projection tensors must be CUDA tensors");
  TORCH_CHECK(
      x.scalar_type() == at::ScalarType::Half &&
          trellis0.scalar_type() == at::ScalarType::Short &&
          trellis1.scalar_type() == at::ScalarType::Short &&
          packed0.scalar_type() == at::ScalarType::Char &&
          packed1.scalar_type() == at::ScalarType::Char &&
          scales0.scalar_type() == at::ScalarType::Half &&
          scales1.scalar_type() == at::ScalarType::Half &&
          suh0.scalar_type() == at::ScalarType::Half &&
          suh1.scalar_type() == at::ScalarType::Half &&
          svh0.scalar_type() == at::ScalarType::Half &&
          svh1.scalar_type() == at::ScalarType::Half &&
          metadata.scalar_type() == at::ScalarType::Byte &&
          offsets.scalar_type() == at::ScalarType::Int &&
          locks.scalar_type() == at::ScalarType::Int,
      "TurboMind EXL3 INT8 paired projection tensor dtypes disagree");
  TORCH_CHECK(
      x.is_contiguous() && trellis0.is_contiguous() &&
          trellis1.is_contiguous() && packed0.is_contiguous() &&
          packed1.is_contiguous() && scales0.is_contiguous() &&
          scales1.is_contiguous() && suh0.is_contiguous() &&
          suh1.is_contiguous() && svh0.is_contiguous() &&
          svh1.is_contiguous() && metadata.is_contiguous() &&
          offsets.is_contiguous() && locks.is_contiguous(),
      "TurboMind EXL3 INT8 paired projection tensors must be contiguous");
  TORCH_CHECK(
      m > 0 && x.dim() == 2 && x.size(1) == k && trellis0.dim() == 3 &&
          trellis0.sizes() == trellis1.sizes() && packed0.dim() == 4 &&
          packed0.sizes() == packed1.sizes() && scales0.dim() == 2 &&
          scales0.sizes() == scales1.sizes() &&
          scales0.size(0) == packed0.size(0) &&
          scales0.size(1) == packed0.size(1) &&
          packed0.size(2) == kWarpSize && packed0.size(3) == 16 &&
          trellis0.size(0) * 16 == k && trellis0.size(1) * 16 == n &&
          trellis0.size(2) == bits * 16 && suh0.numel() == k &&
          suh1.numel() == k && svh0.numel() == n && svh1.numel() == n &&
          metadata.numel() >= kMetadataBytes && offsets.numel() >= 9 &&
          locks.numel() >= 2 * (n / kHadamard) && k % kHadamard == 0 &&
          n % kHadamard == 0,
      "TurboMind EXL3 INT8 paired projection tensor shapes disagree");

  if (m > 8) {
    auto out0 = exl3_sm70_gemm(x, trellis0, suh0, svh0, true, false);
    auto out1 = exl3_sm70_gemm(x, trellis1, suh1, svh1, true, false);
    return at::cat({out0, out1}, 1);
  }

  struct PairWs {
    torch::Tensor out, x_had, partials;
    int64_t cached_m = 0, cached_n = 0, cached_k = 0;
    int cached_dev = -1;
  };
  thread_local PairWs pair_slots[8];
  thread_local int pair_slot_count = 0;
  PairWs* pair_ws = nullptr;
  int const pair_dev = x.get_device();
  for (int i = 0; i < pair_slot_count; ++i) {
    if (pair_slots[i].cached_m == m && pair_slots[i].cached_n == n &&
        pair_slots[i].cached_k == k && pair_slots[i].cached_dev == pair_dev &&
        pair_slots[i].x_had.defined()) {
      pair_ws = &pair_slots[i];
      break;
    }
  }
  if (pair_ws == nullptr) {
    pair_ws = &pair_slots[pair_slot_count < 8 ? pair_slot_count++ : 0];
    pair_ws->cached_m = m;
    pair_ws->cached_n = n;
    pair_ws->cached_k = k;
    pair_ws->cached_dev = pair_dev;
    pair_ws->x_had =
        torch::empty({16, k}, x.options().dtype(at::ScalarType::Half));
    pair_ws->partials =
        torch::empty({16, n}, x.options().dtype(at::ScalarType::Float));
  }
  auto out =
      torch::empty({m, 2 * n}, x.options().dtype(at::ScalarType::Half));
  auto& x_had = pair_ws->x_had;
  auto& partials = pair_ws->partials;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  bool const shared_input_transform = suh0.data_ptr() == suh1.data_ptr();
  if (shared_input_transform) {
    dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                          static_cast<unsigned>(m));
    sm70_exl3_input_hadamard_kernel<<<input_grid, kWarpSize, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(suh0.data_ptr<at::Half>()),
        reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
        static_cast<int>(m), static_cast<int>(k));
  } else {
    dim3 const input_grid(static_cast<unsigned>(k / kHadamard),
                          static_cast<unsigned>(m), 2);
    sm70_exl3_input_hadamard_pair_kernel<<<input_grid, kWarpSize, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(suh0.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(suh1.data_ptr<at::Half>()),
        reinterpret_cast<half*>(x_had.data_ptr<at::Half>()),
        static_cast<int>(m), static_cast<int>(k));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto const [splits, swizzle] = sm70_exl3_tm_int8_policy(bits, k, n, m);
  switch (bits) {
    case 4:
      launch_tm_sm70_exl3_int8_pair_core_out<4>(
          out, x_had, metadata, offsets, partials, locks,
          static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
          splits, swizzle);
      break;
    case 5:
      launch_tm_sm70_exl3_int8_pair_core_out<5>(
          out, x_had, metadata, offsets, partials, locks,
          static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
          splits, swizzle);
      break;
    case 6:
      launch_tm_sm70_exl3_int8_pair_core_out<6>(
          out, x_had, metadata, offsets, partials, locks,
          static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
          splits, swizzle);
      break;
    default:
      TORCH_CHECK(false,
                  "TurboMind EXL3 INT8 paired projection supports K4/K5/K6");
  }
  return out;
}

torch::Tensor exl3_sm70_tm_int8_dispatch_gemm_persistent_locks(
    torch::Tensor const& x, torch::Tensor const& trellis,
    torch::Tensor const& packed_lane, torch::Tensor const& tile_scales,
    torch::Tensor const& suh, torch::Tensor const& svh,
    torch::Tensor const& locks, int64_t bits, bool mcg, bool mul1) {
  int64_t const m = x.size(0);
  int64_t const k = trellis.size(0) * 16;
  int64_t const n = trellis.size(1) * 16;
  TORCH_CHECK(m > 0, "TurboMind EXL3 int8 dispatch requires M > 0");
  TORCH_CHECK(
      trellis.is_cuda() && packed_lane.is_cuda() && tile_scales.is_cuda() &&
          locks.is_cuda() && trellis.dim() == 3 && packed_lane.dim() == 4 &&
          tile_scales.dim() == 2 && trellis.size(2) == bits * 16 &&
          packed_lane.size(0) * 16 == k &&
          packed_lane.size(1) * 32 == n &&
          packed_lane.size(2) == kWarpSize && packed_lane.size(3) == 16 &&
          tile_scales.size(0) == packed_lane.size(0) &&
          tile_scales.size(1) == packed_lane.size(1) &&
          locks.scalar_type() == at::ScalarType::Int &&
          locks.is_contiguous() && locks.numel() >= n / 128,
      "TurboMind EXL3 int8 persistent dispatch metadata disagree");
  if (m > 8) {
    return exl3_sm70_gemm(x, trellis, suh, svh, mcg, mul1);
  }

  auto const [splits, swizzle] =
      sm70_exl3_tm_int8_policy(bits, k, n, m);
  struct DispatchWs {
    torch::Tensor out, x_had, partials, accum;
    int64_t cached_m = 0, cached_n = 0, cached_k = 0;
    int cached_dev = -1;
  };
  thread_local DispatchWs dispatch_slots[12];
  thread_local int dispatch_slot_count = 0;
  DispatchWs* dispatch_ws = nullptr;
  int const dispatch_dev = x.get_device();
  for (int i = 0; i < dispatch_slot_count; ++i) {
    if (dispatch_slots[i].cached_m == m && dispatch_slots[i].cached_n == n &&
        dispatch_slots[i].cached_k == k &&
        dispatch_slots[i].cached_dev == dispatch_dev &&
        dispatch_slots[i].x_had.defined()) {
      dispatch_ws = &dispatch_slots[i];
      break;
    }
  }
  if (dispatch_ws == nullptr) {
    dispatch_ws =
        &dispatch_slots[dispatch_slot_count < 12 ? dispatch_slot_count++ : 0];
    dispatch_ws->cached_m = m;
    dispatch_ws->cached_n = n;
    dispatch_ws->cached_k = k;
    dispatch_ws->cached_dev = dispatch_dev;
    dispatch_ws->x_had = torch::empty_like(x);
    dispatch_ws->partials =
        torch::empty({8, n}, x.options().dtype(at::ScalarType::Float));
    dispatch_ws->accum =
        torch::empty({m, n}, x.options().dtype(at::ScalarType::Float));
  }
  auto out =
      torch::empty({m, n}, x.options().dtype(at::ScalarType::Half));
  auto& x_had = dispatch_ws->x_had;
  auto& partials = dispatch_ws->partials;
  auto& accum = dispatch_ws->accum;
  if (sm70_exl3_tm_use_fused_output_hadamard(bits, k, n, m)) {
    exl3_sm70_tm_int8_gemm_hadamard_out(
        out, x, packed_lane, tile_scales, suh, svh, x_had, partials, locks,
        bits, splits, swizzle);
    return out;
  }
  exl3_sm70_tm_int8_gemm_out(
      out, x, packed_lane, tile_scales, suh, svh, x_had, accum, partials,
      locks, bits, splits, swizzle);
  return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def("exl3_sm70_tm_core(Tensor x_had, Tensor trellis) -> Tensor");
  m.def("exl3_sm70_tm_core_out(Tensor(a!) out, Tensor x_had, Tensor trellis, "
        "Tensor(b!) partials, Tensor(c!) locks, int splits, int swizzle) -> ()");
  m.def("exl3_sm70_tm_state_repack(Tensor trellis) -> Tensor");
  // Research-only tile-scaled INT8 path.  Keep this surface independent of
  // the production state dispatcher until its latency and model-level error
  // have both been qualified on the real Qwen3.8 projection set.
  m.def("exl3_sm70_tm_int8_repack(Tensor trellis) -> (Tensor, Tensor)");
  m.def("exl3_sm70_tm_int6_repack(Tensor trellis) -> (Tensor, Tensor)");
  m.def("exl3_sm70_tm_int10_repack(Tensor trellis) -> (Tensor, Tensor)");
  m.def("exl3_sm70_tm_e4m3_repack(Tensor trellis) -> (Tensor, Tensor)");
  m.def("exl3_sm70_tm_int8_gemm_out(Tensor(a!) out, Tensor x, "
        "Tensor packed_lane, Tensor tile_scales, Tensor suh, Tensor svh, "
        "Tensor(b!) x_had, Tensor(c!) accum, Tensor(d!) partials, "
        "Tensor(e!) locks, int bits, int splits, int swizzle) -> ()");
  m.def("exl3_sm70_tm_int8_gemm_hadamard_out(Tensor(a!) out, Tensor x, "
        "Tensor packed_lane, Tensor tile_scales, Tensor suh, Tensor svh, "
        "Tensor(b!) x_had, Tensor(c!) partials, Tensor(d!) locks, int bits, "
        "int splits, int swizzle) -> ()");
  m.def("exl3_sm70_tm_e4m3_gemm_hadamard_out(Tensor(a!) out, Tensor x, "
        "Tensor packed_lane, Tensor tile_scales, Tensor suh, Tensor svh, "
        "Tensor(b!) x_had, Tensor(c!) partials, Tensor(d!) locks, int bits, "
        "int splits, int swizzle) -> ()");
  m.def("exl3_sm70_tm_int6_gemm_hadamard_out(Tensor(a!) out, Tensor x, "
        "Tensor packed_words, Tensor group_scales, Tensor suh, Tensor svh, "
        "Tensor(b!) x_had, Tensor(c!) partials, Tensor(d!) locks, int bits, "
        "int splits, int swizzle) -> ()");
  m.def("exl3_sm70_tm_int10_gemm_hadamard_out(Tensor(a!) out, Tensor x, "
        "Tensor packed_words, Tensor group_scales, Tensor suh, Tensor svh, "
        "Tensor(b!) x_had, Tensor(c!) partials, Tensor(d!) locks, int bits, "
        "int splits, int swizzle) -> ()");
  m.def("exl3_sm70_tm_int8_gemm(Tensor x, Tensor packed_lane, "
        "Tensor tile_scales, Tensor suh, Tensor svh, int bits, int splits, "
        "int swizzle) -> Tensor");
  m.def("exl3_sm70_tm_gate_up_metadata(Tensor gate_state, Tensor up_state, "
        "Tensor gate_svh, Tensor up_svh) -> Tensor");
  m.def("exl3_sm70_tm_raw_pair_metadata(Tensor trellis0, Tensor trellis1, "
        "Tensor svh0, Tensor svh1) -> Tensor");
  m.def("exl3_sm70_tm_int8_pair_metadata(Tensor packed0, Tensor scales0, "
        "Tensor packed1, Tensor scales1, Tensor svh0, Tensor svh1) -> "
        "Tensor");
  m.def("exl3_sm70_tm_state_core_out(Tensor(a!) out, Tensor x_had, "
        "Tensor state, Tensor(b!) partials, Tensor(c!) locks, int bits, "
        "int splits, int swizzle) -> ()");
  m.def("exl3_sm70_tm_state_core_out_n256(Tensor(a!) out, Tensor x_had, "
        "Tensor state, Tensor(b!) partials, Tensor(c!) locks, int bits, "
        "int splits, int swizzle) -> ()");
  m.def("exl3_sm70_tm_state_core_out_k32(Tensor(a!) out, Tensor x_had, "
        "Tensor state, Tensor(b!) partials, Tensor(c!) locks, int bits, "
        "int splits, int swizzle) -> ()");
  m.def("exl3_sm70_tm_state_gemm_out(Tensor(a!) out, Tensor x, Tensor state, "
        "Tensor suh, Tensor svh, Tensor(b!) x_had, Tensor(c!) accum, "
        "Tensor(d!) partials, Tensor(e!) locks, int bits, int splits, "
        "int swizzle) -> ()");
  m.def("exl3_sm70_tm_state_gemm_hadamard_out(Tensor(a!) out, Tensor x, "
        "Tensor state, Tensor suh, Tensor svh, Tensor(b!) x_had, "
        "Tensor(c!) partials, Tensor(d!) locks, int bits, int splits, "
        "int swizzle) -> ()");
  m.def("exl3_sm70_tm_state_gemm_hadamard_tile_reduce_out("
        "Tensor(a!) reduced, Tensor(b!) staging, Tensor x, Tensor state, "
        "Tensor suh, Tensor svh, Tensor(c!) x_had, Tensor(d!) partials, "
        "Tensor(e!) locks, int bits, int splits, int swizzle, int fa_ptr, "
        "int reducer_blocks) -> ()");
  m.def("exl3_sm70_tm_state_gemm(Tensor x, Tensor state, Tensor suh, "
        "Tensor svh, int bits, int splits, int swizzle) -> Tensor");
  m.def("exl3_sm70_tm_dispatch_gemm(Tensor x, Tensor trellis, Tensor state, "
        "Tensor suh, Tensor svh, int bits, bool mcg, bool mul1) -> Tensor");
  m.def("exl3_sm70_tm_dispatch_gemm_persistent_locks(Tensor x, Tensor trellis, "
        "Tensor state, Tensor suh, Tensor svh, Tensor(a!) locks, int bits, "
        "bool mcg, bool mul1) -> Tensor");
  m.def("exl3_sm70_tm_raw_dispatch_gemm_persistent_locks("
        "Tensor x, Tensor trellis, Tensor suh, Tensor svh, Tensor(a!) locks, "
        "int bits, bool mcg, bool mul1) -> Tensor");
  m.def("exl3_sm70_tm_int8_dispatch_gemm_persistent_locks("
        "Tensor x, Tensor trellis, Tensor packed_lane, Tensor tile_scales, "
        "Tensor suh, Tensor svh, Tensor(a!) locks, int bits, bool mcg, "
        "bool mul1) -> Tensor");
  m.def("exl3_sm70_tm_state_pair_gemm(Tensor x, Tensor trellis0, "
        "Tensor trellis1, Tensor state0, Tensor state1, Tensor suh0, "
        "Tensor suh1, Tensor svh0, Tensor svh1, Tensor metadata, "
        "Tensor offsets, Tensor(a!) locks, int bits) -> Tensor");
  m.def("exl3_sm70_tm_raw_pair_gemm(Tensor x, Tensor trellis0, "
        "Tensor trellis1, Tensor suh0, Tensor suh1, Tensor svh0, "
        "Tensor svh1, Tensor metadata, Tensor offsets, Tensor(a!) locks, "
        "int bits, int splits=-1, int swizzle=-1, bool fused_output=True) -> "
        "Tensor");
  m.def("exl3_sm70_tm_int8_pair_gemm(Tensor x, Tensor trellis0, "
        "Tensor trellis1, Tensor packed0, Tensor scales0, Tensor packed1, "
        "Tensor scales1, Tensor suh0, Tensor suh1, Tensor svh0, Tensor svh1, "
        "Tensor metadata, Tensor offsets, Tensor(a!) locks, int bits) -> "
        "Tensor");
  m.def("exl3_sm70_tm_state_gate_up_silu_mul(Tensor x, Tensor gate_trellis, "
        "Tensor up_trellis, Tensor gate_state, Tensor up_state, "
        "Tensor gate_suh, Tensor up_suh, Tensor gate_svh, Tensor up_svh, "
        "Tensor metadata, Tensor offsets, Tensor(a!) locks, int bits, "
        "int splits=-1, int swizzle=-1) -> "
        "Tensor");
  m.def("exl3_sm70_tm_raw_gate_up_silu_mul(Tensor x, Tensor gate_trellis, "
        "Tensor up_trellis, Tensor gate_suh, Tensor up_suh, Tensor gate_svh, "
        "Tensor up_svh, Tensor metadata, Tensor offsets, Tensor(a!) locks, "
        "int bits, int splits=-1, int swizzle=-1, bool fused_output=True) -> "
        "Tensor");
  m.def("exl3_sm70_tm_state_mlp(Tensor x, Tensor gate_trellis, "
        "Tensor up_trellis, Tensor down_trellis, Tensor gate_state, "
        "Tensor up_state, Tensor down_state, Tensor gate_suh, Tensor up_suh, "
        "Tensor down_suh, Tensor gate_svh, Tensor up_svh, Tensor down_svh, "
        "Tensor down_packed_lane, Tensor down_tile_scales, "
        "Tensor gate_metadata, Tensor gate_offsets, Tensor(a!) gate_locks, "
        "Tensor(b!) down_locks, int gate_bits, int down_bits, "
        "bool int8_gate, bool int8_down) -> Tensor");
  m.def("exl3_sm70_silu_mul_input_hadamard(Tensor gate, Tensor up, "
        "Tensor suh) -> Tensor");
  m.def("exl3_sm70_input_hadamard(Tensor x, Tensor suh) -> Tensor");
  m.def("exl3_sm70_silu_mul_input_hadamard_baseline(Tensor gate, Tensor up, "
        "Tensor suh) -> Tensor");
  m.def("exl3_sm70_tm_reconstruct(Tensor trellis) -> Tensor");
#ifndef EXL3_SM70_GATEUP_ONLY
  m.def("exl3_sm70_gemm(Tensor x, Tensor trellis, Tensor suh, Tensor svh, "
        "bool mcg, bool mul1) -> Tensor");
  m.def("exl3_sm70_gemm_n128(Tensor x, Tensor trellis, Tensor suh, Tensor svh, "
        "bool mcg, bool mul1) -> Tensor");
#endif
  m.def("exl3_sm70_gate_up_silu_mul(Tensor x, Tensor gate_trellis, "
        "Tensor up_trellis, Tensor gate_suh, Tensor up_suh, Tensor gate_svh, "
        "Tensor up_svh, bool gate_mcg, bool up_mcg, bool gate_mul1, "
        "bool up_mul1) -> Tensor");
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("exl3_sm70_tm_core", &exl3_sm70_tm_core);
  m.impl("exl3_sm70_tm_core_out", &exl3_sm70_tm_core_out);
  m.impl("exl3_sm70_tm_state_repack", &exl3_sm70_tm_state_repack);
  m.impl("exl3_sm70_tm_int8_repack", &exl3_sm70_tm_int8_repack);
  m.impl("exl3_sm70_tm_int6_repack", &exl3_sm70_tm_int6_repack);
  m.impl("exl3_sm70_tm_int10_repack", &exl3_sm70_tm_int10_repack);
  m.impl("exl3_sm70_tm_e4m3_repack", &exl3_sm70_tm_e4m3_repack);
  m.impl("exl3_sm70_tm_int8_gemm_out", &exl3_sm70_tm_int8_gemm_out);
  m.impl("exl3_sm70_tm_int8_gemm_hadamard_out",
         &exl3_sm70_tm_int8_gemm_hadamard_out);
  m.impl("exl3_sm70_tm_e4m3_gemm_hadamard_out",
         &exl3_sm70_tm_e4m3_gemm_hadamard_out);
  m.impl("exl3_sm70_tm_int6_gemm_hadamard_out",
         &exl3_sm70_tm_int6_gemm_hadamard_out);
  m.impl("exl3_sm70_tm_int10_gemm_hadamard_out",
         &exl3_sm70_tm_int10_gemm_hadamard_out);
  m.impl("exl3_sm70_tm_int8_gemm", &exl3_sm70_tm_int8_gemm);
  m.impl("exl3_sm70_tm_gate_up_metadata",
         &exl3_sm70_tm_gate_up_metadata);
  m.impl("exl3_sm70_tm_raw_pair_metadata",
         &exl3_sm70_tm_raw_pair_metadata);
  m.impl("exl3_sm70_tm_int8_pair_metadata",
         &exl3_sm70_tm_int8_pair_metadata);
  m.impl("exl3_sm70_tm_state_core_out", &exl3_sm70_tm_state_core_out);
  m.impl("exl3_sm70_tm_state_core_out_n256",
         &exl3_sm70_tm_state_core_out_n256);
  m.impl("exl3_sm70_tm_state_core_out_k32",
         &exl3_sm70_tm_state_core_out_k32);
  m.impl("exl3_sm70_tm_state_gemm_out", &exl3_sm70_tm_state_gemm_out);
  m.impl("exl3_sm70_tm_state_gemm_hadamard_out",
         &exl3_sm70_tm_state_gemm_hadamard_out);
  m.impl("exl3_sm70_tm_state_gemm_hadamard_tile_reduce_out",
         &exl3_sm70_tm_state_gemm_hadamard_tile_reduce_out);
  m.impl("exl3_sm70_tm_state_gemm", &exl3_sm70_tm_state_gemm);
  m.impl("exl3_sm70_tm_dispatch_gemm", &exl3_sm70_tm_dispatch_gemm);
  m.impl("exl3_sm70_tm_dispatch_gemm_persistent_locks",
         &exl3_sm70_tm_dispatch_gemm_persistent_locks);
  m.impl("exl3_sm70_tm_raw_dispatch_gemm_persistent_locks",
         &exl3_sm70_tm_raw_dispatch_gemm_persistent_locks);
  m.impl("exl3_sm70_tm_int8_dispatch_gemm_persistent_locks",
         &exl3_sm70_tm_int8_dispatch_gemm_persistent_locks);
  m.impl("exl3_sm70_tm_state_pair_gemm",
         &exl3_sm70_tm_state_pair_gemm);
  m.impl("exl3_sm70_tm_raw_pair_gemm",
         &exl3_sm70_tm_raw_pair_gemm);
  m.impl("exl3_sm70_tm_int8_pair_gemm",
         &exl3_sm70_tm_int8_pair_gemm);
  m.impl("exl3_sm70_tm_state_gate_up_silu_mul",
         &exl3_sm70_tm_state_gate_up_silu_mul);
  m.impl("exl3_sm70_tm_raw_gate_up_silu_mul",
         &exl3_sm70_tm_raw_gate_up_silu_mul);
  m.impl("exl3_sm70_tm_state_mlp", &exl3_sm70_tm_state_mlp);
  m.impl("exl3_sm70_silu_mul_input_hadamard",
         &exl3_sm70_silu_mul_input_hadamard);
  m.impl("exl3_sm70_input_hadamard", &exl3_sm70_input_hadamard);
  m.impl("exl3_sm70_silu_mul_input_hadamard_baseline",
         &exl3_sm70_silu_mul_input_hadamard_baseline);
  m.impl("exl3_sm70_tm_reconstruct", &exl3_sm70_tm_reconstruct);
#ifndef EXL3_SM70_GATEUP_ONLY
  m.impl("exl3_sm70_gemm", &exl3_sm70_gemm);
  m.impl("exl3_sm70_gemm_n128", &exl3_sm70_gemm_n128);
#endif
  m.impl("exl3_sm70_gate_up_silu_mul", &exl3_sm70_gate_up_silu_mul);
}
