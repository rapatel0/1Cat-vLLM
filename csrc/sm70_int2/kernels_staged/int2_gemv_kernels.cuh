// Bandwidth-optimized INT2 small-M GEMV (M <= MAXM). The tile/tensor-core kernels
// are memory-parallelism-starved at small M (~125 GB/s vs cuBLAS ~600); this is a
// GEMV-shaped kernel tuned purely for DRAM bandwidth — the regime that gates MoE
// per-expert GEMMs and single-stream/spec-decode (small M).
//
// Strategy:
//   - One WARP owns one output column n; the 32 lanes split the K-reduction, so a
//     warp's weight reads are 32 consecutive uint32 of row n => one coalesced
//     128-byte transaction. Each warp processes COLS_PER_WARP columns, giving the
//     compiler that many independent in-flight loads (memory-level parallelism).
//   - Activations A[0..M-1, k-chunk] are staged in shared memory once per block and
//     reused across all COLS_PER_BLOCK columns => weights are the only DRAM traffic.
//   - Native signed two's-comp row-major qs (no repack/shuffle): dequant inline.
//
// Layout (matches data_gen k_quantize_int2): qs[n*(K/4) + k/4], 4 signed 2-bit
// vals/byte; scales[n*(K/QK_INT2) + k/QK_INT2], one fp16 per 32 vals.

#pragma once

#include "tc_grid.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace tc_grid::kernels::int2_gemv {

template <int COLS_PER_BLOCK, int WARPS, int KC, int MAXM>
__launch_bounds__(WARPS * 32)
__global__ void mm_int2_gemv(const uint8_t * __restrict__ W_qs,
                             const __half  * __restrict__ W_scales,
                             const float   * __restrict__ A,
                             float         * __restrict__ C,
                             int M, int N, int K) {
    constexpr int COLS_PER_WARP = COLS_PER_BLOCK / WARPS;
    // Pad the m-stride to ODD so the per-lane sA stride (16*PAD) is never a
    // multiple of 32 banks (else all 32 lanes collide on one bank).
    constexpr int PAD = (MAXM & 1) ? MAXM : (MAXM + 1);
    static_assert(KC == 512, "KC=512 -> one uint32 (16 vals) per lane per chunk");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tid = (int) threadIdx.x, threads = WARPS * 32;
    const int col0 = blockIdx.x * COLS_PER_BLOCK + warp * COLS_PER_WARP;
    const int Kpack = K / 4, bpr = K / QK_INT2;

    extern __shared__ float sA[];   // [KC * PAD], k-major, PAD-padded m-stride

    float acc[COLS_PER_WARP][MAXM];
    #pragma unroll
    for (int c = 0; c < COLS_PER_WARP; ++c)
        #pragma unroll
        for (int m = 0; m < MAXM; ++m) acc[c][m] = 0.0f;

    for (int kc = 0; kc < K; kc += KC) {
        // Stage A[0..M-1, kc..kc+KC) into smem, k-MAJOR (sA[kk*MAXM + m]) so the M
        // reads per k land in consecutive banks (m-major would M-way conflict).
        for (int i = tid; i < M * KC; i += threads) {
            const int m = i / KC, kk = i % KC;
            sA[kk * PAD + m] = A[(size_t) m * K + kc + kk];
        }
        __syncthreads();

        const int kk0 = lane * 16;              // this lane's 16 K-values in the chunk
        #pragma unroll
        for (int c = 0; c < COLS_PER_WARP; ++c) {
            const int n = col0 + c;
            if (n >= N) continue;
            const uint32_t w = __ldg(reinterpret_cast<const uint32_t *>(
                &W_qs[(size_t) n * Kpack + (kc + kk0) / 4]));
            const float scale = __half2float(__ldg(W_scales + (size_t) n * bpr + (kc + kk0) / QK_INT2));
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int f = (w >> (2 * j)) & 0x3;
                const int q = (f ^ 0x2) - 0x2;          // sign-extend 2-bit two's-comp
                const float wv = (float) q * scale;
                #pragma unroll
                for (int m = 0; m < MAXM; ++m)
                    if (m < M) acc[c][m] += wv * sA[(kk0 + j) * PAD + m];
            }
        }
        __syncthreads();
    }

    // Warp-reduce each column's M partials across the 32 lanes; lane 0 writes C.
    #pragma unroll
    for (int c = 0; c < COLS_PER_WARP; ++c) {
        const int n = col0 + c;
        #pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m >= M) continue;
            float v = acc[c][m];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            if (lane == 0 && n < N) C[(size_t) m * N + n] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// mm_int2_gemv_m1 — M=1 specialization with NO shared memory. Each lane loads
// its 16-value activation stripe into registers (coalesced) and reuses it across
// the warp's COLS_PER_WARP columns. Eliminates the staging + 2 syncs/chunk.
// ---------------------------------------------------------------------------
template <int COLS_PER_BLOCK, int WARPS, bool ACC_F32 = true>
__launch_bounds__(WARPS * 32)
__global__ void mm_int2_gemv_m1(const uint8_t * __restrict__ W_qs,
                                const __half  * __restrict__ scales,
                                const float   * __restrict__ A,
                                float         * __restrict__ C, int N, int K) {
    constexpr int COLS_PER_WARP = COLS_PER_BLOCK / WARPS;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col0 = blockIdx.x * COLS_PER_BLOCK + warp * COLS_PER_WARP;
    const int Kpack = K / 4, bpr = K / QK_INT2;

    float acc[COLS_PER_WARP]; half acch[COLS_PER_WARP];
    #pragma unroll
    for (int c = 0; c < COLS_PER_WARP; ++c) { acc[c] = 0.0f; acch[c] = __float2half(0.0f); }

    for (int kc = 0; kc < K; kc += 512) {
        const int kk0 = lane * 16;
        float ar[16];                       // this lane's 16 activations (coalesced)
        #pragma unroll
        for (int u = 0; u < 4; ++u)
            *reinterpret_cast<float4 *>(&ar[u * 4]) =
                __ldg(reinterpret_cast<const float4 *>(&A[kc + kk0 + u * 4]));
        #pragma unroll
        for (int c = 0; c < COLS_PER_WARP; ++c) {
            const int n = col0 + c;
            if (n >= N) continue;
            const uint32_t w = __ldg(reinterpret_cast<const uint32_t *>(
                &W_qs[(size_t) n * Kpack + (kc + kk0) / 4]));
            const float scale = __half2float(__ldg(scales + (size_t) n * bpr + (kc + kk0) / QK_INT2));
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int f = (w >> (2 * j)) & 0x3;
                const int q = (f ^ 0x2) - 0x2;
                if (ACC_F32) acc[c] += (float) q * scale * ar[j];
                else acch[c] = __hfma(__float2half((float) q * scale), __float2half(ar[j]), acch[c]);
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < COLS_PER_WARP; ++c) {
        const int n = col0 + c;
        float v = ACC_F32 ? acc[c] : __half2float(acch[c]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0 && n < N) C[n] = v;
    }
}

// ---------------------------------------------------------------------------
// n-major repack for the M>1 GEMV: regroup the row-major uint32s (each = 16 vals
// of one column) into [N/32][K/16][32] order so a warp's 32 columns at one
// k-group are 32 contiguous uint32 (one coalesced 128B load, no shuffle).
//   src element [n][kg]  ->  W_t[(n/32)*Kg*32 + kg*32 + (n%32)]
// Pure transpose of the uint32 array; signed 2-bit payload untouched.
// ---------------------------------------------------------------------------
__global__ void repack_nmajor(const uint32_t * __restrict__ qs, uint32_t * __restrict__ W_t,
                              int N, int Kg) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Kg) return;
    const int n = idx / Kg, kg = idx % Kg;
    W_t[(size_t)(n / 32) * Kg * 32 + (size_t) kg * 32 + (n % 32)] = qs[(size_t) n * Kg + kg];
}

// ---------------------------------------------------------------------------
// mm_int2_gemv_n — n-split GEMV for M in [2,8]. Lane == one output column; the
// 32 lanes of a warp read 32 contiguous uint32 (coalesced). Activations are
// staged in smem and BROADCAST to all lanes (same address => no bank conflict).
// Each lane owns its column's M accumulators => no warp-reduce. f32 accumulate.
// ---------------------------------------------------------------------------
template <int WARPS, int KK, int MAXM>
__launch_bounds__(WARPS * 32)
__global__ void mm_int2_gemv_n(const uint32_t * __restrict__ W_t,
                               const __half  * __restrict__ scales,
                               const float   * __restrict__ A,
                               float         * __restrict__ C,
                               int M, int N, int K) {
    constexpr int KKC = KK * 16;           // K values per chunk
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tid = (int) threadIdx.x, threads = WARPS * 32;
    const int nblk = blockIdx.x * WARPS + warp;
    const int col  = nblk * 32 + lane;
    const int Kg = K / 16, bpr = K / QK_INT2;
    const size_t wbase = (size_t) nblk * Kg * 32;

    extern __shared__ float sAn[];         // [M * KKC]: sAn[m*KKC + kk], broadcast to lanes
    float acc[MAXM];
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) acc[m] = 0.0f;

    for (int kc = 0; kc < K; kc += KKC) {
        for (int i = tid; i < M * KKC; i += threads) {
            const int m = i / KKC, kk = i % KKC;
            sAn[m * KKC + kk] = A[(size_t) m * K + kc + kk];
        }
        __syncthreads();
        #pragma unroll
        for (int kgl = 0; kgl < KK; ++kgl) {
            const int kg = kc / 16 + kgl;
            const uint32_t w = W_t[wbase + (size_t) kg * 32 + lane];      // coalesced
            const float scale = (col < N)
                ? __half2float(__ldg(scales + (size_t) col * bpr + (kg * 16) / QK_INT2)) : 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int f = (w >> (2 * j)) & 0x3;
                const int q = (f ^ 0x2) - 0x2;
                const float wv = (float) q * scale;
                const int kk = kgl * 16 + j;
                #pragma unroll
                for (int m = 0; m < MAXM; ++m)
                    if (m < M) acc[m] += wv * sAn[m * KKC + kk];      // broadcast read
            }
        }
        __syncthreads();
    }
    if (col < N) {
        #pragma unroll
        for (int m = 0; m < MAXM; ++m)
            if (m < M) C[(size_t) m * N + col] = acc[m];
    }
}

}  // namespace tc_grid::kernels::int2_gemv
