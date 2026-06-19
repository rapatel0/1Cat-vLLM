// INT2 rf: register-resident 2-bit GEMM. Port of int8_v13::mm_int8_lut_v13_rf_v2
// (kernels/v13_kernels.cuh:361-603). Keeps the weight unpack in REGISTERS and
// feeds raw mma.sync from register fragments — no dequantized fp16 staged in
// shared (the int2_v3 path stages fp16 via wmma; this closes that gap).
//
// Storage: packed 2-bit, K/4 bytes per row (16 vals = 1 uint32 per BK=16 k-tile),
// but UNSIGNED + zero-point=1 and exllama bit-shuffled (see repack below). The
// register unpack `dequant_2bit_16` (in-place LOP3 + folded scale) recovers the
// 16 values in NATURAL K-order. Both pieces are validated standalone in
// tests/int2_rf_unpack_test.cu (PASS, 0 mismatches / 65536, max_abs 4.8e-4).
//
// The base int2 quantizer (data_gen.cu k_quantize_int2) stores SIGNED two's-comp
// 2-bit. dequant_2bit_16 needs unsigned codes, so the launcher repacks ONCE
// (signed->unsigned + shuffle) via k_shuffle_int2 before timing. The reference
// (d_ref) recovers identical values, so the rf kernel stays consistent with it.

#pragma once

#include "tc_grid.h"
#include "mma_sm70.cuh"

#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace tc_grid::kernels::int2_rf {

using ::tc_grid::mma_sm70::mma_m8n8k4_row_col_acc_f16;

// ---------------------------------------------------------------------------
// Register unpack (exllamav2-derived). Input qa = one shuffled uint32 = 16 vals.
// Recovers q=(u-1) for each field in NATURAL K-order, then * scale.
// dq[j] = half2(v_{2j}, v_{2j+1}). Constants are for zero-point=1.
// immLut 0xea = (0xf0&0xcc)|0xaa == (a&b)|c (Marlin/AWQ/CUTLASS canonical).
// Exactness: worst embed 1024+3*64=1216 < 2048 -> exact in fp16 10-bit mantissa.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void dequant_2bit_16(uint32_t qa, half2 (&dq)[8], half scale) {
    const uint32_t c0 = 0x64006400;
    const half2 y4  = __half2half2(__float2half(1.0f / 4.0f));
    const half2 y16 = __half2half2(__float2half(1.0f / 16.0f));
    const half2 y64 = __half2half2(__float2half(1.0f / 64.0f));
    const half2 z1  = __half2half2(__float2half(-1025.0f));   // -(1024+zero)
    const half2 z4  = __half2half2(__float2half(-257.0f));    // -(256+zero)
    const half2 z16 = __half2half2(__float2half(-65.0f));     // -(64+zero)
    const half2 z64 = __half2half2(__float2half(-17.0f));     // -(16+zero)
    auto MK = [&](uint32_t m) {
        uint32_t r;
        asm("lop3.b32 %0,%1,%2,%3,0xea;" : "=r"(r) : "r"(qa), "r"(m), "r"(c0)); // (qa&m)|c0
        return *reinterpret_cast<half2 *>(&r);
    };
    dq[0] = __hadd2(MK(0x00030003), z1);
    dq[1] = __hfma2(MK(0x000c000c), y4,  z4);
    dq[2] = __hfma2(MK(0x00300030), y16, z16);
    dq[3] = __hfma2(MK(0x00c000c0), y64, z64);
    qa >>= 8;
    dq[4] = __hadd2(MK(0x00030003), z1);
    dq[5] = __hfma2(MK(0x000c000c), y4,  z4);
    dq[6] = __hfma2(MK(0x00300030), y16, z16);
    dq[7] = __hfma2(MK(0x00c000c0), y64, z64);
    const half2 s2 = __half2half2(scale);
    #pragma unroll
    for (int j = 0; j < 8; ++j) dq[j] = __hmul2(dq[j], s2);
}

// ---------------------------------------------------------------------------
// Per-k-step dequant: recovers just the 4 values for inner k-step KS (K=4*KS..
// 4*KS+3) of the 16-value group, i.e. dq[2*KS] and dq[2*KS+1] from
// dequant_2bit_16, *scale. KS is a compile-time constant so the branch folds.
// Used by mm_int2_rf_v2's pipelined mainloop (turbomind MainloopSm70 Transform).
//   KS even -> qa fields {0x0003 +z1, 0x000c *y4+z4}
//   KS odd  -> qa fields {0x0030 *y16+z16, 0x00c0 *y64+z64}
//   KS>=2   -> operate on (qa>>8)
// out[0]=(v_{4KS},v_{4KS+1}), out[1]=(v_{4KS+2},v_{4KS+3}).
// ---------------------------------------------------------------------------
template <int KS>
__device__ __forceinline__ void dequant_2bit_kstep(uint32_t qa, half scale, half2 (&out)[2]) {
    const uint32_t c0  = 0x64006400;
    const uint32_t src = (KS < 2) ? qa : (qa >> 8);
    auto MK = [&](uint32_t m) {
        uint32_t r;
        asm("lop3.b32 %0,%1,%2,%3,0xea;" : "=r"(r) : "r"(src), "r"(m), "r"(c0));
        return *reinterpret_cast<half2 *>(&r);
    };
    const half2 s2 = __half2half2(scale);
    if (KS % 2 == 0) {
        const half2 z1 = __half2half2(__float2half(-1025.0f));
        const half2 y4 = __half2half2(__float2half(1.0f / 4.0f));
        const half2 z4 = __half2half2(__float2half(-257.0f));
        out[0] = __hmul2(__hadd2(MK(0x00030003), z1), s2);
        out[1] = __hmul2(__hfma2(MK(0x000c000c), y4, z4), s2);
    } else {
        const half2 y16 = __half2half2(__float2half(1.0f / 16.0f));
        const half2 z16 = __half2half2(__float2half(-65.0f));
        const half2 y64 = __half2half2(__float2half(1.0f / 64.0f));
        const half2 z64 = __half2half2(__float2half(-17.0f));
        out[0] = __hmul2(__hfma2(MK(0x00300030), y16, z16), s2);
        out[1] = __hmul2(__hfma2(MK(0x00c000c0), y64, z64), s2);
    }
}

// ---------------------------------------------------------------------------
// Repack one natural-packed group (16 signed two's-comp 2-bit fields, value k at
// bits[2k]) into unsigned(+1) + exllama interleave so dequant_2bit_16 emits
// natural K-order. Returns the uint32 to store for rf.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t repack_2bit_group(uint32_t nat_signed) {
    // 1) signed -> unsigned per field: u = q + 1, q = sign_extend2(field)
    uint32_t u = 0;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        int f = (nat_signed >> (2 * k)) & 0x3;
        int q = (f ^ 0x2) - 0x2;                 // sign-extend 2-bit two's-comp
        uint32_t uu = (uint32_t)(q + 1) & 0x3;   // {-1,0,1} -> {0,1,2}
        u |= (uu << (2 * k));
    }
    // 2) exllama shuffle_2bit_16 on the UNSIGNED codes
    uint32_t qa = u, qb = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t qa0 = qa & 0x03, qa1 = (qa & 0x0c) >> 2;
        qa >>= 4;
        qb |= (qa1 << (i * 2 + 16));
        qb |= (qa0 << (i * 2));
    }
    return qb;
}

// Repack kernel: qs region viewed as uint32[N*K/16]; scales copied unchanged by
// the launcher. n_groups = N * (K/16). Run ONCE before the timing loop.
__global__ void k_shuffle_int2(const uint32_t * in, uint32_t * out, int n_groups) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_groups) out[i] = repack_2bit_group(in[i]);
}

// ---------------------------------------------------------------------------
// mm_int2_rf — port of v13_rf_v2. BK=16, ATOM_M=8, ATOM_N=32, ATOM_K=8.
// W_qs is the REPACKED (unsigned+shuffled) packed buffer: K/4 bytes per row.
// ---------------------------------------------------------------------------
template <int BM_, int BN_, int BK_, int WARPS_, int ATOMS_M_, int ATOMS_N_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_int2_rf(
        const uint8_t * __restrict__ W_qs,       // CHANGE: uint8 packed-shuffled
        const __half  * __restrict__ W_scales,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int ATOMS_M = ATOMS_M_;
    constexpr int ATOMS_N = ATOMS_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int ATOM_M = 8, ATOM_N = 32, ATOM_K = 8;
    (void) ATOM_K;
    constexpr int BK_PACK = BK / 4;              // 4 bytes / n-row (16 vals)
    constexpr int BK_PACK_PAD = BK_PACK + 4;     // CHANGE: was BK_INT8_PAD=BK+16
    constexpr int BN_PAD = BN + 8;
    static_assert(BK == 16, "int2_rf: BK=16 fused-K path");
    static_assert(BM == ATOMS_M * ATOM_M, "");
    static_assert(N_PER_WARP == ATOMS_N * ATOM_N, "");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    constexpr int kA_chunks = (BM * BK) / 4;     // float4 A loads (unchanged)
    constexpr int kB_chunks = BN;                // CHANGE: 1 uint32 per n-row
    constexpr int A_PER_THR = (kA_chunks + threads - 1) / threads;
    constexpr int B_PER_THR = (kB_chunks + threads - 1) / threads;

    // smem: sA fp16 (double), sB uint8 packed (double), sS fp16 (double).
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    uint8_t * sB = reinterpret_cast<uint8_t *>(sA + 2 * BM * BK);
    __half * sS = reinterpret_cast<__half *>(sB + 2 * BN * BK_PACK_PAD);
    auto sA_buf = [&](int b) -> __half *  { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> uint8_t * { return sB + (size_t) b * BN * BK_PACK_PAD; };
    auto sS_buf = [&](int b) -> __half *  { return sS + (size_t) b * BN; };

    const int aL_m = (lane / 16) * 4 + (lane % 4);
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;

    half c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = __float2half(0.0f);

    const int k_tiles = K / BK;
    const int Kpack = K / 4;                      // packed bytes per row in DRAM
    const int blocks_per_row = K / QK_INT2;

    float4   A_rmem[A_PER_THR];
    uint32_t B_rmem[B_PER_THR];
    half     S_rmem[B_PER_THR];
    int      A_idx0[A_PER_THR];
    int      B_idx0[B_PER_THR];

    auto load_gmem_to_rmem = [&](int kt) {
        const int k0 = kt * BK;
        // A: float4 loads, VERBATIM v13.
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            float4 v = make_float4(0, 0, 0, 0);
            int idx0 = -1;
            if (c_ < kA_chunks) {
                idx0 = c_ * 4;
                const int mm = idx0 / BK, kk = idx0 % BK;
                const int gm = tile_m + mm, gk = k0 + kk;
                if (gm < M && gk + 3 < K) {
                    v = *(const float4 *) &A[(size_t) gm * K + gk];
                }
            }
            A_rmem[p] = v;
            A_idx0[p] = idx0;
        }
        // B: one uint32 (16 packed 2-bit vals) per n-row.
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            uint32_t vq = 0;
            half s_h = __float2half(0.0f);
            int idx0 = -1;
            if (c_ < kB_chunks) {
                const int nn = c_;                // one n-row per chunk
                const int gn = tile_n + nn, gk = k0;
                idx0 = nn;
                if (gn < N) {
                    vq  = __ldg(reinterpret_cast<const uint32_t *>(&W_qs[(size_t) gn * Kpack + gk / 4]));
                    s_h = __ldg(W_scales + (size_t) gn * blocks_per_row + gk / QK_INT2);
                }
            }
            B_rmem[p] = vq;
            S_rmem[p] = s_h;
            B_idx0[p] = idx0;
        }
    };

    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b  = sA_buf(buf_idx);
        uint8_t * sB_b = sB_buf(buf_idx);
        __half * sS_b  = sS_buf(buf_idx);
        // A: VERBATIM v13.
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int idx0 = A_idx0[p];
            if (idx0 < 0) continue;
            const int mm = idx0 / BK, kk = idx0 % BK;
            const float4 v = A_rmem[p];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        // B: one uint32 per n-row at padded stride.
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int nn = B_idx0[p];
            if (nn < 0) continue;
            *reinterpret_cast<uint32_t *>(&sB_b[nn * BK_PACK_PAD]) = B_rmem[p];
            sS_b[nn] = S_rmem[p];
        }
    };

    auto mainloop = [&](int buf) {
        __half * sA_c  = sA_buf(buf);
        uint8_t * sB_c = sB_buf(buf);
        __half * sS_c  = sS_buf(buf);

        // Per-atom scale (one per warp's atom-N, constant across mainloop).
        half scales[ATOMS_N];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            scales[an] = (n < BN) ? sS_c[n] : __float2half(0.0f);
        }

        // A frags: full BK=16 per lane = 16 halves = uint4 + uint4. VERBATIM v13.
        half a_frags[ATOMS_M][16];
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            const int m = am * ATOM_M + aL_m;
            *reinterpret_cast<uint4 *>(&a_frags[am][0]) =
                *reinterpret_cast<const uint4 *>(&sA_c[m * BK + 0]);
            *reinterpret_cast<uint4 *>(&a_frags[am][8]) =
                *reinterpret_cast<const uint4 *>(&sA_c[m * BK + 8]);
        }

        // B frags: single uint32 (16 packed 2-bit) LDS per lane per atom,
        // register-unpacked into 16 halves covering ALL K's (natural order).
        half b_frags[ATOMS_N][16];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            uint32_t qpk = *reinterpret_cast<const uint32_t *>(&sB_c[n * BK_PACK_PAD]);
            half2 dq[8];
            dequant_2bit_16(qpk, dq, scales[an]);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                *reinterpret_cast<half2 *>(&b_frags[an][2 * j]) = dq[j];
        }

        // 4 back-to-back m8n8k4 atoms per (am, an): K=0..3, 4..7, 8..11, 12..15.
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 0], &b_frags[an][ 0]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 4], &b_frags[an][ 4]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 8], &b_frags[an][ 8]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][12], &b_frags[an][12]);
            }
        }
    };

    load_gmem_to_rmem(0);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(1);
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        store_rmem_to_smem(next_buf);
        if (kt + 2 < k_tiles) load_gmem_to_rmem(kt + 2);
        __syncthreads();
        buf = next_buf;
    }
    mainloop(buf);

    __syncthreads();
    __half * sC = sA;
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_warp + cL_n_f16;
            *reinterpret_cast<uint4 *>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4 *>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;
        const int row    = h2_idx / (BN / 2);
        const int col_h2 = h2_idx % (BN / 2);
        const int col    = col_h2 * 2;
        const half2 v = *reinterpret_cast<const half2 *>(&sC[row * BN_PAD + col]);
        const float2 f = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (gm < M && gn + 1 < N) {
            C[(size_t) gm * N + gn + 0] = f.x;
            C[(size_t) gm * N + gn + 1] = f.y;
        } else if (gm < M && gn < N) {
            C[(size_t) gm * N + gn] = f.x;
        }
    }
}

// ===========================================================================
// mm_int2_rf_sk — mm_int2_rf + SplitK (turbomind V100 ships SplitK on EVERY
// tile) + optional evict-first/streaming cache hint on the weight load.
//   KS      : K-direction CTA count (grid.z). KS==1 == plain mm_int2_rf.
//   STREAM_B: use __ldcs (ld.global.cs, cache-streaming) for the packed weight
//             load instead of __ldg — analog of turbomind's Stream (kEvictFirst)
//             cache policy used for B at small M.
// Splits the K-tile range across blockIdx.z; partials atomicAdd into C (launcher
// pre-zeroes C when KS>1). Decode (M=16) is occupancy-starved — 32 of 80 SMs —
// so the K-split fills the GPU.
// ===========================================================================
template <int BM_, int BN_, int BK_, int WARPS_, int ATOMS_M_, int ATOMS_N_, int KS_, bool STREAM_B_,
          bool ALIGNED_ = false, int MINBLK_ = 2>
__launch_bounds__(WARPS_ * 32, MINBLK_)
__global__ void mm_int2_rf_sk(
        const uint8_t * __restrict__ W_qs,
        const __half  * __restrict__ W_scales,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int ATOMS_M = ATOMS_M_;
    constexpr int ATOMS_N = ATOMS_N_;
    constexpr int KS = KS_;
    constexpr bool STREAM_B = STREAM_B_;
    constexpr bool ALIGNED = ALIGNED_;    // N%BN==0 && K%BK==0 -> drop n/k bounds checks
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int ATOM_M = 8, ATOM_N = 32, ATOM_K = 8;
    (void) ATOM_K;
    constexpr int BK_PACK = BK / 4;
    constexpr int BK_PACK_PAD = BK_PACK + 4;
    constexpr int BN_PAD = BN + 8;
    static_assert(BK == 16, "int2_rf_sk: BK=16 fused-K path");
    static_assert(BM == ATOMS_M * ATOM_M, "");
    static_assert(N_PER_WARP == ATOMS_N * ATOM_N, "");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = BN;
    constexpr int A_PER_THR = (kA_chunks + threads - 1) / threads;
    constexpr int B_PER_THR = (kB_chunks + threads - 1) / threads;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    uint8_t * sB = reinterpret_cast<uint8_t *>(sA + 2 * BM * BK);
    __half * sS = reinterpret_cast<__half *>(sB + 2 * BN * BK_PACK_PAD);
    auto sA_buf = [&](int b) -> __half *  { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> uint8_t * { return sB + (size_t) b * BN * BK_PACK_PAD; };
    auto sS_buf = [&](int b) -> __half *  { return sS + (size_t) b * BN; };

    const int aL_m = (lane / 16) * 4 + (lane % 4);
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;

    half c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = __float2half(0.0f);

    // SplitK: balanced k-tile range for this blockIdx.z (mirrors int4 v3s).
    const int k_tiles_total = K / BK;
    const int k_split = blockIdx.z;
    const int kt_base = (k_tiles_total / KS) * k_split + min(k_split, k_tiles_total % KS);
    const int kt_lim  = kt_base + (k_tiles_total / KS) + ((k_split < (k_tiles_total % KS)) ? 1 : 0);
    const int k_tiles = kt_lim - kt_base;
    if (k_tiles <= 0) return;

    const int Kpack = K / 4;
    const int blocks_per_row = K / QK_INT2;

    float4   A_rmem[A_PER_THR];
    uint32_t B_rmem[B_PER_THR];
    half     S_rmem[B_PER_THR];
    int      A_idx0[A_PER_THR];
    int      B_idx0[B_PER_THR];

    auto load_gmem_to_rmem = [&](int kt) {
        const int k0 = kt * BK;
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            float4 v = make_float4(0, 0, 0, 0);
            int idx0 = -1;
            if (c_ < kA_chunks) {
                idx0 = c_ * 4;
                const int mm = idx0 / BK, kk = idx0 % BK;
                const int gm = tile_m + mm, gk = k0 + kk;
                if (gm < M && (ALIGNED || gk + 3 < K)) v = *(const float4 *) &A[(size_t) gm * K + gk];
            }
            A_rmem[p] = v;
            A_idx0[p] = idx0;
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            uint32_t vq = 0;
            half s_h = __float2half(0.0f);
            int idx0 = -1;
            if (c_ < kB_chunks) {
                const int nn = c_;
                const int gn = tile_n + nn, gk = k0;
                idx0 = nn;
                if (ALIGNED || gn < N) {
                    const uint32_t * bptr = reinterpret_cast<const uint32_t *>(&W_qs[(size_t) gn * Kpack + gk / 4]);
                    vq  = STREAM_B ? __ldcs(bptr) : __ldg(bptr);
                    s_h = __ldg(W_scales + (size_t) gn * blocks_per_row + gk / QK_INT2);
                }
            }
            B_rmem[p] = vq;
            S_rmem[p] = s_h;
            B_idx0[p] = idx0;
        }
    };

    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b  = sA_buf(buf_idx);
        uint8_t * sB_b = sB_buf(buf_idx);
        __half * sS_b  = sS_buf(buf_idx);
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int idx0 = A_idx0[p];
            if (idx0 < 0) continue;
            const int mm = idx0 / BK, kk = idx0 % BK;
            const float4 v = A_rmem[p];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int nn = B_idx0[p];
            if (nn < 0) continue;
            *reinterpret_cast<uint32_t *>(&sB_b[nn * BK_PACK_PAD]) = B_rmem[p];
            sS_b[nn] = S_rmem[p];
        }
    };

    auto mainloop = [&](int buf) {
        __half * sA_c  = sA_buf(buf);
        uint8_t * sB_c = sB_buf(buf);
        __half * sS_c  = sS_buf(buf);
        half scales[ATOMS_N];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            scales[an] = (n < BN) ? sS_c[n] : __float2half(0.0f);
        }
        half a_frags[ATOMS_M][16];
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            const int m = am * ATOM_M + aL_m;
            *reinterpret_cast<uint4 *>(&a_frags[am][0]) = *reinterpret_cast<const uint4 *>(&sA_c[m * BK + 0]);
            *reinterpret_cast<uint4 *>(&a_frags[am][8]) = *reinterpret_cast<const uint4 *>(&sA_c[m * BK + 8]);
        }
        half b_frags[ATOMS_N][16];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            uint32_t qpk = *reinterpret_cast<const uint32_t *>(&sB_c[n * BK_PACK_PAD]);
            half2 dq[8];
            dequant_2bit_16(qpk, dq, scales[an]);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                *reinterpret_cast<half2 *>(&b_frags[an][2 * j]) = dq[j];
        }
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am)
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 0], &b_frags[an][ 0]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 4], &b_frags[an][ 4]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 8], &b_frags[an][ 8]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][12], &b_frags[an][12]);
            }
    };

    load_gmem_to_rmem(kt_base);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(kt_base + 1);
    __syncthreads();

    int buf = 0;
    for (int l = 0; l < k_tiles - 1; ++l) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        store_rmem_to_smem(next_buf);
        if (l + 2 < k_tiles) load_gmem_to_rmem(kt_base + l + 2);
        __syncthreads();
        buf = next_buf;
    }
    mainloop(buf);

    __syncthreads();
    __half * sC = sA;
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_warp + cL_n_f16;
            *reinterpret_cast<uint4 *>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4 *>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;
        const int row    = h2_idx / (BN / 2);
        const int col_h2 = h2_idx % (BN / 2);
        const int col    = col_h2 * 2;
        const half2 v = *reinterpret_cast<const half2 *>(&sC[row * BN_PAD + col]);
        const float2 f = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (gm < M && gn + 1 < N) {
            if (KS > 1) { atomicAdd(&C[(size_t) gm * N + gn + 0], f.x); atomicAdd(&C[(size_t) gm * N + gn + 1], f.y); }
            else        { C[(size_t) gm * N + gn + 0] = f.x;           C[(size_t) gm * N + gn + 1] = f.y; }
        } else if (gm < M && gn < N) {
            if (KS > 1) atomicAdd(&C[(size_t) gm * N + gn], f.x);
            else        C[(size_t) gm * N + gn] = f.x;
        }
    }
}

// ===========================================================================
// mm_int2_rf_bk — BK-generalized rf kernel (BK in {16,32}) + SplitK. turbomind's
// V100 decode tiles use deeper CTA_K (32/64) to amortize tiny-M. K_GROUPS=BK/16
// packed uint32 per n-row per k-tile; K_ATOMS=BK/4 m8n8k4 atoms. One scale per
// QK_INT2=32 block (BK=32 -> 1 scale/tile; BK=16 -> shared across 2 tiles).
// ===========================================================================
template <int BM_, int BN_, int BK_, int WARPS_, int ATOMS_M_, int ATOMS_N_, int KS_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_int2_rf_bk(
        const uint8_t * __restrict__ W_qs,
        const __half  * __restrict__ W_scales,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int ATOMS_M = ATOMS_M_;
    constexpr int ATOMS_N = ATOMS_N_;
    constexpr int KS = KS_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int ATOM_M = 8, ATOM_N = 32;
    constexpr int K_GROUPS = BK / 16;            // packed uint32 groups per n-row
    constexpr int K_ATOMS  = BK / 4;             // m8n8k4 atoms per k-tile
    constexpr int A_UINT4  = BK / 8;             // uint4 loads per A frag (8 halves each)
    constexpr int BK_PACK = BK / 4;
    constexpr int BK_PACK_PAD = BK_PACK + 4;
    constexpr int BN_PAD = BN + 8;
    static_assert(BK == 16 || BK == 32, "int2_rf_bk: BK in {16,32}");
    static_assert(BM == ATOMS_M * ATOM_M, "");
    static_assert(N_PER_WARP == ATOMS_N * ATOM_N, "");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = BN;
    constexpr int A_PER_THR = (kA_chunks + threads - 1) / threads;
    constexpr int B_PER_THR = (kB_chunks + threads - 1) / threads;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    uint8_t * sB = reinterpret_cast<uint8_t *>(sA + 2 * BM * BK);
    __half * sS = reinterpret_cast<__half *>(sB + 2 * BN * BK_PACK_PAD);
    auto sA_buf = [&](int b) -> __half *  { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> uint8_t * { return sB + (size_t) b * BN * BK_PACK_PAD; };
    auto sS_buf = [&](int b) -> __half *  { return sS + (size_t) b * BN; };

    const int aL_m = (lane / 16) * 4 + (lane % 4);
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;

    half c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = __float2half(0.0f);

    const int k_tiles_total = K / BK;
    const int k_split = blockIdx.z;
    const int kt_base = (k_tiles_total / KS) * k_split + min(k_split, k_tiles_total % KS);
    const int kt_lim  = kt_base + (k_tiles_total / KS) + ((k_split < (k_tiles_total % KS)) ? 1 : 0);
    const int k_tiles = kt_lim - kt_base;
    if (k_tiles <= 0) return;

    const int Kpack = K / 4;
    const int blocks_per_row = K / QK_INT2;

    float4   A_rmem[A_PER_THR];
    uint32_t B_rmem[B_PER_THR][K_GROUPS];
    half     S_rmem[B_PER_THR];
    int      A_idx0[A_PER_THR];
    int      B_idx0[B_PER_THR];

    auto load_gmem_to_rmem = [&](int kt) {
        const int k0 = kt * BK;
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            float4 v = make_float4(0, 0, 0, 0);
            int idx0 = -1;
            if (c_ < kA_chunks) {
                idx0 = c_ * 4;
                const int mm = idx0 / BK, kk = idx0 % BK;
                const int gm = tile_m + mm, gk = k0 + kk;
                if (gm < M && gk + 3 < K) v = *(const float4 *) &A[(size_t) gm * K + gk];
            }
            A_rmem[p] = v;
            A_idx0[p] = idx0;
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            half s_h = __float2half(0.0f);
            int idx0 = -1;
            #pragma unroll
            for (int g = 0; g < K_GROUPS; ++g) B_rmem[p][g] = 0;
            if (c_ < kB_chunks) {
                const int nn = c_;
                const int gn = tile_n + nn, gk = k0;
                idx0 = nn;
                if (gn < N) {
                    const uint32_t * bptr = reinterpret_cast<const uint32_t *>(&W_qs[(size_t) gn * Kpack + gk / 4]);
                    #pragma unroll
                    for (int g = 0; g < K_GROUPS; ++g) B_rmem[p][g] = __ldg(bptr + g);
                    s_h = __ldg(W_scales + (size_t) gn * blocks_per_row + gk / QK_INT2);
                }
            }
            S_rmem[p] = s_h;
            B_idx0[p] = idx0;
        }
    };

    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b  = sA_buf(buf_idx);
        uint8_t * sB_b = sB_buf(buf_idx);
        __half * sS_b  = sS_buf(buf_idx);
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int idx0 = A_idx0[p];
            if (idx0 < 0) continue;
            const int mm = idx0 / BK, kk = idx0 % BK;
            const float4 v = A_rmem[p];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int nn = B_idx0[p];
            if (nn < 0) continue;
            #pragma unroll
            for (int g = 0; g < K_GROUPS; ++g)
                *reinterpret_cast<uint32_t *>(&sB_b[nn * BK_PACK_PAD + 4 * g]) = B_rmem[p][g];
            sS_b[nn] = S_rmem[p];
        }
    };

    auto mainloop = [&](int buf) {
        __half * sA_c  = sA_buf(buf);
        uint8_t * sB_c = sB_buf(buf);
        __half * sS_c  = sS_buf(buf);
        half scales[ATOMS_N];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            scales[an] = (n < BN) ? sS_c[n] : __float2half(0.0f);
        }
        half a_frags[ATOMS_M][BK];
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            const int m = am * ATOM_M + aL_m;
            #pragma unroll
            for (int u = 0; u < A_UINT4; ++u)
                *reinterpret_cast<uint4 *>(&a_frags[am][8 * u]) =
                    *reinterpret_cast<const uint4 *>(&sA_c[m * BK + 8 * u]);
        }
        half b_frags[ATOMS_N][BK];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            #pragma unroll
            for (int g = 0; g < K_GROUPS; ++g) {
                uint32_t qpk = *reinterpret_cast<const uint32_t *>(&sB_c[n * BK_PACK_PAD + 4 * g]);
                half2 dq[8];
                dequant_2bit_16(qpk, dq, scales[an]);
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    *reinterpret_cast<half2 *>(&b_frags[an][16 * g + 2 * j]) = dq[j];
            }
        }
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am)
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an)
                #pragma unroll
                for (int ka = 0; ka < K_ATOMS; ++ka)
                    mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][4 * ka], &b_frags[an][4 * ka]);
    };

    load_gmem_to_rmem(kt_base);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(kt_base + 1);
    __syncthreads();

    int buf = 0;
    for (int l = 0; l < k_tiles - 1; ++l) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        store_rmem_to_smem(next_buf);
        if (l + 2 < k_tiles) load_gmem_to_rmem(kt_base + l + 2);
        __syncthreads();
        buf = next_buf;
    }
    mainloop(buf);

    __syncthreads();
    __half * sC = sA;
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_warp + cL_n_f16;
            *reinterpret_cast<uint4 *>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4 *>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;
        const int row    = h2_idx / (BN / 2);
        const int col_h2 = h2_idx % (BN / 2);
        const int col    = col_h2 * 2;
        const half2 v = *reinterpret_cast<const half2 *>(&sC[row * BN_PAD + col]);
        const float2 f = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (gm < M && gn + 1 < N) {
            if (KS > 1) { atomicAdd(&C[(size_t) gm * N + gn + 0], f.x); atomicAdd(&C[(size_t) gm * N + gn + 1], f.y); }
            else        { C[(size_t) gm * N + gn + 0] = f.x;           C[(size_t) gm * N + gn + 1] = f.y; }
        } else if (gm < M && gn < N) {
            if (KS > 1) atomicAdd(&C[(size_t) gm * N + gn], f.x);
            else        C[(size_t) gm * N + gn] = f.x;
        }
    }
}

// ===========================================================================
// mm_int2_rf_ww — 2x2 warp partition (turbomind Blocked<2,2>). Instead of each
// warp owning full-BM x (BN/WARPS) cols (1xW), the 4 warps form a 2x2 grid and
// each owns BM/2 rows x BN/2 cols — more A/B fragment reuse per warp. WARPS=4.
// ATOMS_M/N are PER-WARP atom counts: BM=16*ATOMS_M, BN=64*ATOMS_N. BK=16.
// Only the mainloop frag offsets + C epilogue differ from mm_int2_rf; the
// cooperative gmem/smem staging is identical (CTA-wide, warp-agnostic).
// ===========================================================================
template <int BM_, int BN_, int BK_, int WARPS_, int ATOMS_M_, int ATOMS_N_, bool STREAM_B_ = false, int KS_ = 1>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_int2_rf_ww(
        const uint8_t * __restrict__ W_qs,
        const __half  * __restrict__ W_scales,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int ATOMS_M = ATOMS_M_;   // per-warp (rows = BM/2)
    constexpr int ATOMS_N = ATOMS_N_;   // per-warp (cols = BN/2)
    constexpr bool STREAM_B = STREAM_B_;
    constexpr int KS = KS_;
    constexpr int ATOM_M = 8, ATOM_N = 32;
    constexpr int K_GROUPS = BK / 16;            // packed uint32 groups per n-row
    constexpr int K_ATOMS  = BK / 4;
    constexpr int A_UINT4  = BK / 8;
    constexpr int BK_PACK = BK / 4;
    constexpr int BK_PACK_PAD = BK_PACK + 4;
    constexpr int BN_PAD = BN + 8;
    static_assert(BK == 16 || BK == 32, "int2_rf_ww: BK in {16,32}");
    static_assert(WARPS == 4, "int2_rf_ww: 2x2 warp grid needs WARPS=4");
    static_assert(BM == 2 * ATOMS_M * ATOM_M, "BM = 16*ATOMS_M");
    static_assert(BN == 2 * ATOMS_N * ATOM_N, "BN = 64*ATOMS_N");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int warp_m = warp & 1;          // 2x2 grid: warp_m in {0,1}
    const int warp_n = warp >> 1;         //           warp_n in {0,1}
    const int m_off_warp = warp_m * (BM / 2);
    const int n_off_warp = warp_n * (BN / 2);

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = BN;
    constexpr int A_PER_THR = (kA_chunks + threads - 1) / threads;
    constexpr int B_PER_THR = (kB_chunks + threads - 1) / threads;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    uint8_t * sB = reinterpret_cast<uint8_t *>(sA + 2 * BM * BK);
    __half * sS = reinterpret_cast<__half *>(sB + 2 * BN * BK_PACK_PAD);
    auto sA_buf = [&](int b) -> __half *  { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> uint8_t * { return sB + (size_t) b * BN * BK_PACK_PAD; };
    auto sS_buf = [&](int b) -> __half *  { return sS + (size_t) b * BN; };

    const int aL_m = (lane / 16) * 4 + (lane % 4);
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;

    half c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = __float2half(0.0f);

    // SplitK k-tile range for this blockIdx.z (KS==1 => whole range).
    const int k_tiles_total = K / BK;
    const int k_split = blockIdx.z;
    const int kt_base = (k_tiles_total / KS) * k_split + min(k_split, k_tiles_total % KS);
    const int kt_lim  = kt_base + (k_tiles_total / KS) + ((k_split < (k_tiles_total % KS)) ? 1 : 0);
    const int k_tiles = kt_lim - kt_base;
    if (k_tiles <= 0) return;

    const int Kpack = K / 4;
    const int blocks_per_row = K / QK_INT2;

    float4   A_rmem[A_PER_THR];
    uint32_t B_rmem[B_PER_THR][K_GROUPS];
    half     S_rmem[B_PER_THR];
    int      A_idx0[A_PER_THR];
    int      B_idx0[B_PER_THR];

    auto load_gmem_to_rmem = [&](int kt) {
        const int k0 = kt * BK;
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            float4 v = make_float4(0, 0, 0, 0);
            int idx0 = -1;
            if (c_ < kA_chunks) {
                idx0 = c_ * 4;
                const int mm = idx0 / BK, kk = idx0 % BK;
                const int gm = tile_m + mm, gk = k0 + kk;
                if (gm < M && gk + 3 < K) v = *(const float4 *) &A[(size_t) gm * K + gk];
            }
            A_rmem[p] = v;
            A_idx0[p] = idx0;
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            half s_h = __float2half(0.0f);
            int idx0 = -1;
            #pragma unroll
            for (int g = 0; g < K_GROUPS; ++g) B_rmem[p][g] = 0;
            if (c_ < kB_chunks) {
                const int nn = c_;
                const int gn = tile_n + nn, gk = k0;
                idx0 = nn;
                if (gn < N) {
                    const uint32_t * bptr = reinterpret_cast<const uint32_t *>(&W_qs[(size_t) gn * Kpack + gk / 4]);
                    #pragma unroll
                    for (int g = 0; g < K_GROUPS; ++g) B_rmem[p][g] = STREAM_B ? __ldcs(bptr + g) : __ldg(bptr + g);
                    s_h = __ldg(W_scales + (size_t) gn * blocks_per_row + gk / QK_INT2);
                }
            }
            S_rmem[p] = s_h;
            B_idx0[p] = idx0;
        }
    };

    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b  = sA_buf(buf_idx);
        uint8_t * sB_b = sB_buf(buf_idx);
        __half * sS_b  = sS_buf(buf_idx);
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int idx0 = A_idx0[p];
            if (idx0 < 0) continue;
            const int mm = idx0 / BK, kk = idx0 % BK;
            const float4 v = A_rmem[p];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int nn = B_idx0[p];
            if (nn < 0) continue;
            #pragma unroll
            for (int g = 0; g < K_GROUPS; ++g)
                *reinterpret_cast<uint32_t *>(&sB_b[nn * BK_PACK_PAD + 4 * g]) = B_rmem[p][g];
            sS_b[nn] = S_rmem[p];
        }
    };

    auto mainloop = [&](int buf) {
        __half * sA_c  = sA_buf(buf);
        uint8_t * sB_c = sB_buf(buf);
        __half * sS_c  = sS_buf(buf);
        half scales[ATOMS_N];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            scales[an] = (n < BN) ? sS_c[n] : __float2half(0.0f);
        }
        half a_frags[ATOMS_M][BK];
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            const int m = am * ATOM_M + m_off_warp + aL_m;
            #pragma unroll
            for (int u = 0; u < A_UINT4; ++u)
                *reinterpret_cast<uint4 *>(&a_frags[am][8 * u]) =
                    *reinterpret_cast<const uint4 *>(&sA_c[m * BK + 8 * u]);
        }
        half b_frags[ATOMS_N][BK];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            #pragma unroll
            for (int g = 0; g < K_GROUPS; ++g) {
                uint32_t qpk = *reinterpret_cast<const uint32_t *>(&sB_c[n * BK_PACK_PAD + 4 * g]);
                half2 dq[8];
                dequant_2bit_16(qpk, dq, scales[an]);
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    *reinterpret_cast<half2 *>(&b_frags[an][16 * g + 2 * j]) = dq[j];
            }
        }
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am)
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an)
                #pragma unroll
                for (int ka = 0; ka < K_ATOMS; ++ka)
                    mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][4 * ka], &b_frags[an][4 * ka]);
    };

    load_gmem_to_rmem(kt_base);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(kt_base + 1);
    __syncthreads();

    int buf = 0;
    for (int l = 0; l < k_tiles - 1; ++l) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        store_rmem_to_smem(next_buf);
        if (l + 2 < k_tiles) load_gmem_to_rmem(kt_base + l + 2);
        __syncthreads();
        buf = next_buf;
    }
    mainloop(buf);

    __syncthreads();
    __half * sC = sA;
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + m_off_warp + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_warp + cL_n_f16;
            *reinterpret_cast<uint4 *>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4 *>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;
        const int row    = h2_idx / (BN / 2);
        const int col_h2 = h2_idx % (BN / 2);
        const int col    = col_h2 * 2;
        const half2 v = *reinterpret_cast<const half2 *>(&sC[row * BN_PAD + col]);
        const float2 f = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (gm < M && gn + 1 < N) {
            if (KS > 1) { atomicAdd(&C[(size_t) gm * N + gn + 0], f.x); atomicAdd(&C[(size_t) gm * N + gn + 1], f.y); }
            else        { C[(size_t) gm * N + gn + 0] = f.x;           C[(size_t) gm * N + gn + 1] = f.y; }
        } else if (gm < M && gn < N) {
            if (KS > 1) atomicAdd(&C[(size_t) gm * N + gn], f.x);
            else        C[(size_t) gm * N + gn] = f.x;
        }
    }
}

// ===========================================================================
// mm_int2_rf_v2 — turbomind MainloopSm70-style inner-K software pipeline.
// Splits BK=16 into K_ITERS=4 inner k-steps (one m8n8k4 atom each), register-
// double-buffers the A/B mma fragments across steps, and moves the 2-bit dequant
// into a per-step Transform issued BEFORE the mma so the LOP3/HFMA2 ALU and the
// next A LDS hide under the current step's HMMA (the int8 PRMT path already
// hides cheaply; the heavier 2-bit dequant benefits more). Everything else
// (gmem/smem staging, epilogue) is identical to mm_int2_rf.
// ===========================================================================
template <int BM_, int BN_, int BK_, int WARPS_, int ATOMS_M_, int ATOMS_N_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_int2_rf_v2(
        const uint8_t * __restrict__ W_qs,
        const __half  * __restrict__ W_scales,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int ATOMS_M = ATOMS_M_;
    constexpr int ATOMS_N = ATOMS_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int ATOM_M = 8, ATOM_N = 32, ATOM_K = 8;
    constexpr int K_ITERS = BK / 4;              // 4 inner k-steps (m8n8k4, K=4 each)
    (void) ATOM_K;
    constexpr int BK_PACK = BK / 4;
    constexpr int BK_PACK_PAD = BK_PACK + 4;
    constexpr int BN_PAD = BN + 8;
    static_assert(BK == 16, "int2_rf_v2: BK=16 fused-K path");
    static_assert(BM == ATOMS_M * ATOM_M, "");
    static_assert(N_PER_WARP == ATOMS_N * ATOM_N, "");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = BN;
    constexpr int A_PER_THR = (kA_chunks + threads - 1) / threads;
    constexpr int B_PER_THR = (kB_chunks + threads - 1) / threads;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    uint8_t * sB = reinterpret_cast<uint8_t *>(sA + 2 * BM * BK);
    __half * sS = reinterpret_cast<__half *>(sB + 2 * BN * BK_PACK_PAD);
    auto sA_buf = [&](int b) -> __half *  { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> uint8_t * { return sB + (size_t) b * BN * BK_PACK_PAD; };
    auto sS_buf = [&](int b) -> __half *  { return sS + (size_t) b * BN; };

    const int aL_m = (lane / 16) * 4 + (lane % 4);
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;

    half c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = __float2half(0.0f);

    const int k_tiles = K / BK;
    const int Kpack = K / 4;
    const int blocks_per_row = K / QK_INT2;

    float4   A_rmem[A_PER_THR];
    uint32_t B_rmem[B_PER_THR];
    half     S_rmem[B_PER_THR];
    int      A_idx0[A_PER_THR];
    int      B_idx0[B_PER_THR];

    auto load_gmem_to_rmem = [&](int kt) {
        const int k0 = kt * BK;
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            float4 v = make_float4(0, 0, 0, 0);
            int idx0 = -1;
            if (c_ < kA_chunks) {
                idx0 = c_ * 4;
                const int mm = idx0 / BK, kk = idx0 % BK;
                const int gm = tile_m + mm, gk = k0 + kk;
                if (gm < M && gk + 3 < K) v = *(const float4 *) &A[(size_t) gm * K + gk];
            }
            A_rmem[p] = v;
            A_idx0[p] = idx0;
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            uint32_t vq = 0;
            half s_h = __float2half(0.0f);
            int idx0 = -1;
            if (c_ < kB_chunks) {
                const int nn = c_;
                const int gn = tile_n + nn, gk = k0;
                idx0 = nn;
                if (gn < N) {
                    vq  = __ldg(reinterpret_cast<const uint32_t *>(&W_qs[(size_t) gn * Kpack + gk / 4]));
                    s_h = __ldg(W_scales + (size_t) gn * blocks_per_row + gk / QK_INT2);
                }
            }
            B_rmem[p] = vq;
            S_rmem[p] = s_h;
            B_idx0[p] = idx0;
        }
    };

    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b  = sA_buf(buf_idx);
        uint8_t * sB_b = sB_buf(buf_idx);
        __half * sS_b  = sS_buf(buf_idx);
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int idx0 = A_idx0[p];
            if (idx0 < 0) continue;
            const int mm = idx0 / BK, kk = idx0 % BK;
            const float4 v = A_rmem[p];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int nn = B_idx0[p];
            if (nn < 0) continue;
            *reinterpret_cast<uint32_t *>(&sB_b[nn * BK_PACK_PAD]) = B_rmem[p];
            sS_b[nn] = S_rmem[p];
        }
    };

    // Pipelined mainloop: register double-buffer (2 slots) over K_ITERS=4 steps.
    auto mainloop = [&](int buf) {
        __half * sA_c  = sA_buf(buf);
        uint8_t * sB_c = sB_buf(buf);
        __half * sS_c  = sS_buf(buf);

        half     scales[ATOMS_N];
        uint32_t qpk[ATOMS_N];        // packed 2-bit, ONE LDS per atom, reused all steps
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            scales[an] = (n < BN) ? sS_c[n] : __float2half(0.0f);
            qpk[an]    = *reinterpret_cast<const uint32_t *>(&sB_c[n * BK_PACK_PAD]);
        }

        // A loaded in bulk once (2 LDS.128 per atom, like v13 — no benefit to
        // splitting fp16 A). Only the heavier B dequant is pipelined across steps.
        half a_frags[ATOMS_M][16];
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            const int m = am * ATOM_M + aL_m;
            *reinterpret_cast<uint4 *>(&a_frags[am][0]) =
                *reinterpret_cast<const uint4 *>(&sA_c[m * BK + 0]);
            *reinterpret_cast<uint4 *>(&a_frags[am][8]) =
                *reinterpret_cast<const uint4 *>(&sA_c[m * BK + 8]);
        }

        half b_frag[ATOMS_N][2][4];   // double-buffered B mma fragment [slot][4 halves]

        // prepare(KS) -> slot: per-step 2-bit dequant only (A already resident).
        auto prepare = [&](auto KS_ic, int slot) {
            constexpr int KS = decltype(KS_ic)::value;
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                half2 o[2];
                dequant_2bit_kstep<KS>(qpk[an], scales[an], o);
                *reinterpret_cast<half2 *>(&b_frag[an][slot][0]) = o[0];
                *reinterpret_cast<half2 *>(&b_frag[an][slot][2]) = o[1];
            }
        };

        auto mma_step = [&](int ks, int slot) {
            #pragma unroll
            for (int am = 0; am < ATOMS_M; ++am) {
                #pragma unroll
                for (int an = 0; an < ATOMS_N; ++an) {
                    const int nn = (am & 1) ? (ATOMS_N - 1 - an) : an;  // boustrophedon
                    mma_m8n8k4_row_col_acc_f16(c_frag[am][nn], &a_frags[am][4 * ks], &b_frag[nn][slot][0]);
                }
            }
        };

        // Software pipeline (K_ITERS==4): issue dequant(next) before mma(cur) so
        // the next step's LOP3/HFMA2 ALU overlaps the current HMMA (separate slots).
        static_assert(K_ITERS == 4, "rf_v2 hand-unrolled for K_ITERS=4");
        prepare(std::integral_constant<int, 0>{}, 0);
        prepare(std::integral_constant<int, 1>{}, 1); mma_step(0, 0);
        prepare(std::integral_constant<int, 2>{}, 0); mma_step(1, 1);
        prepare(std::integral_constant<int, 3>{}, 1); mma_step(2, 0);
        mma_step(3, 1);
    };

    load_gmem_to_rmem(0);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(1);
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        store_rmem_to_smem(next_buf);
        if (kt + 2 < k_tiles) load_gmem_to_rmem(kt + 2);
        __syncthreads();
        buf = next_buf;
    }
    mainloop(buf);

    __syncthreads();
    __half * sC = sA;
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_warp + cL_n_f16;
            *reinterpret_cast<uint4 *>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4 *>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;
        const int row    = h2_idx / (BN / 2);
        const int col_h2 = h2_idx % (BN / 2);
        const int col    = col_h2 * 2;
        const half2 v = *reinterpret_cast<const half2 *>(&sC[row * BN_PAD + col]);
        const float2 f = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (gm < M && gn + 1 < N) {
            C[(size_t) gm * N + gn + 0] = f.x;
            C[(size_t) gm * N + gn + 1] = f.y;
        } else if (gm < M && gn < N) {
            C[(size_t) gm * N + gn] = f.x;
        }
    }
}

}  // namespace tc_grid::kernels::int2_rf
