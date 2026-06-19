// int2 SM70 GEMV torch op — bandwidth-optimized 2-bit weight GEMV for V100,
// for the small-M regime (decode / MoE experts) where tensor cores are wasteful.
//
// Generalized affine dequant: w = q*scale + bias  (q unsigned 2-bit 0..3, per
// `group_size` scale+bias). Covers symmetric and asymmetric (GPTQ-style:
// bias = -zero*scale). f32 accumulate (bit-accurate + overflow-safe; on V100
// CUDA cores f32 is *faster* than f16 — see docs/INT2_SM70_INTEGRATION.md §2c).
//
// Weight layout (qweight): row-major per output channel n, K values packed
//   4 per byte; value at K index k is bits [2*(k%4) : +1] of byte qweight[n][k/4].
// scales/bias: half, [N, K/group_size].
//
// v1: M=1 (mm_int2_gemv_m1, register-A). M=2..8 (n-split) is a follow-up.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int CPB = 64;   // output columns per block
constexpr int WG  = 8;    // warps per block
constexpr int CPW = CPB / WG;

__launch_bounds__(WG * 32)
__global__ void gemv_m1_kernel(const uint8_t * __restrict__ qweight,  // [N, K/4]
                               const __half  * __restrict__ scales,   // [N, K/group]
                               const __half  * __restrict__ bias,     // [N, K/group]
                               const __half  * __restrict__ A,        // [K]  (M=1)
                               __half        * __restrict__ C,        // [N]
                               int N, int K, int group_size) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col0 = blockIdx.x * CPB + warp * CPW;
    const int Kpack = K / 4;
    const int ngroups = K / group_size;

    float acc[CPW];
    #pragma unroll
    for (int c = 0; c < CPW; ++c) acc[c] = 0.0f;

    for (int kc = 0; kc < K; kc += 512) {
        const int kk0 = lane * 16;
        // this lane's 16 activations (coalesced across the warp), half -> float
        float ar[16];
        const uint4 a0 = *reinterpret_cast<const uint4 *>(&A[kc + kk0]);     // 8 half
        const uint4 a1 = *reinterpret_cast<const uint4 *>(&A[kc + kk0 + 8]); // 8 half
        const __half * h0 = reinterpret_cast<const __half *>(&a0);
        const __half * h1 = reinterpret_cast<const __half *>(&a1);
        #pragma unroll
        for (int j = 0; j < 8; ++j) { ar[j] = __half2float(h0[j]); ar[8 + j] = __half2float(h1[j]); }

        #pragma unroll
        for (int c = 0; c < CPW; ++c) {
            const int n = col0 + c;
            if (n >= N) continue;
            const uint32_t w = __ldg(reinterpret_cast<const uint32_t *>(
                &qweight[(size_t) n * Kpack + (kc + kk0) / 4]));
            const int g = (kc + kk0) / group_size;          // 16 vals are within one group
            const float scale = __half2float(__ldg(scales + (size_t) n * ngroups + g));
            const float bs    = __half2float(__ldg(bias   + (size_t) n * ngroups + g));
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int q = (w >> (2 * j)) & 0x3;          // unsigned 2-bit
                acc[c] += ((float) q * scale + bs) * ar[j];
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < CPW; ++c) {
        const int n = col0 + c;
        float v = acc[c];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0 && n < N) C[n] = __float2half(v);
    }
}

}  // namespace

torch::Tensor int2_gemv_m1(torch::Tensor A, torch::Tensor qweight,
                           torch::Tensor scales, torch::Tensor bias,
                           int64_t group_size) {
    TORCH_CHECK(A.is_cuda() && qweight.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be fp16");
    TORCH_CHECK(qweight.scalar_type() == torch::kUInt8, "qweight must be uint8");
    TORCH_CHECK(A.size(0) == 1, "v1 supports M==1");
    const int K = A.size(1);
    const int N = qweight.size(0);
    TORCH_CHECK(qweight.size(1) == K / 4, "qweight must be [N, K/4]");
    TORCH_CHECK(K % 512 == 0, "K must be a multiple of 512");
    TORCH_CHECK(group_size % 16 == 0 && K % group_size == 0, "group_size%16==0, K%group==0");
    TORCH_CHECK(A.is_contiguous() && qweight.is_contiguous(), "inputs contiguous");

    auto out = torch::empty({1, N}, A.options());
    const dim3 grid((N + CPB - 1) / CPB), block(WG * 32);
    auto stream = c10::cuda::getCurrentCUDAStream();
    gemv_m1_kernel<<<grid, block, 0, stream>>>(
        qweight.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(bias.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(out.data_ptr<at::Half>()),
        N, K, (int) group_size);
    return out;
}

// ---------------------------------------------------------------------------
// n-major repack + n-split GEMV for M in [2,8]. Lane == one output column; the
// 32 lanes load 32 contiguous uint32 (coalesced); activations broadcast from
// smem; each lane owns its M accumulators (no warp-reduce). Affine dequant.
// ---------------------------------------------------------------------------
namespace {

__global__ void repack_nmajor_kernel(const uint32_t * __restrict__ qs,
                                     uint32_t * __restrict__ W_t, int N, int Kg) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Kg) return;
    const int n = idx / Kg, kg = idx % Kg;
    W_t[(size_t)(n / 32) * Kg * 32 + (size_t) kg * 32 + (n % 32)] = qs[(size_t) n * Kg + kg];
}

template <int WG, int KK, int MAXM>
__launch_bounds__(WG * 32)
__global__ void gemv_n_kernel(const uint32_t * __restrict__ W_t,
                              const __half * __restrict__ scales,
                              const __half * __restrict__ bias,
                              const __half * __restrict__ A,   // [M, K]
                              __half * __restrict__ C,         // [M, N]
                              int M, int N, int K, int group) {
    constexpr int KKC = KK * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tid = (int) threadIdx.x, threads = WG * 32;
    const int nblk = blockIdx.x * WG + warp;
    const int col = nblk * 32 + lane;
    const int Kg = K / 16, ng = K / group;
    const size_t wbase = (size_t) nblk * Kg * 32;
    extern __shared__ float sAn[];           // [M * KKC]: sAn[m*KKC + kk]
    float acc[MAXM];
    #pragma unroll
    for (int m = 0; m < MAXM; ++m) acc[m] = 0.0f;
    for (int kc = 0; kc < K; kc += KKC) {
        for (int i = tid; i < M * KKC; i += threads) {
            const int m = i / KKC, kk = i % KKC;
            sAn[m * KKC + kk] = __half2float(A[(size_t) m * K + kc + kk]);
        }
        __syncthreads();
        #pragma unroll
        for (int kgl = 0; kgl < KK; ++kgl) {
            const int kg = kc / 16 + kgl;
            const uint32_t w = W_t[wbase + (size_t) kg * 32 + lane];
            const int g = (kg * 16) / group;
            const float scale = (col < N) ? __half2float(scales[(size_t) col * ng + g]) : 0.f;
            const float bs    = (col < N) ? __half2float(bias[(size_t) col * ng + g]) : 0.f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int q = (w >> (2 * j)) & 0x3;
                const float wv = (float) q * scale + bs;
                const int kk = kgl * 16 + j;
                #pragma unroll
                for (int m = 0; m < MAXM; ++m)
                    if (m < M) acc[m] += wv * sAn[m * KKC + kk];
            }
        }
        __syncthreads();
    }
    if (col < N) {
        #pragma unroll
        for (int m = 0; m < MAXM; ++m)
            if (m < M) C[(size_t) m * N + col] = __float2half(acc[m]);
    }
}

}  // namespace

torch::Tensor int2_repack_nmajor(torch::Tensor qweight, int64_t K) {
    TORCH_CHECK(qweight.is_cuda() && qweight.scalar_type() == torch::kUInt8, "qweight uint8 CUDA");
    const int N = qweight.size(0), Kg = K / 16;
    auto wt = torch::empty({N * Kg}, qweight.options().dtype(torch::kInt32));
    const int total = N * Kg;
    auto stream = c10::cuda::getCurrentCUDAStream();
    repack_nmajor_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t *>(qweight.data_ptr<uint8_t>()),
        reinterpret_cast<uint32_t *>(wt.data_ptr<int32_t>()), N, Kg);
    return wt;
}

torch::Tensor int2_gemv_n(torch::Tensor A, torch::Tensor wt, torch::Tensor scales,
                          torch::Tensor bias, int64_t group_size) {
    constexpr int WG = 4, KK = 32, KKC = KK * 16;
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A fp16");
    const int M = A.size(0), K = A.size(1);
    const int N = scales.size(0);
    TORCH_CHECK(M >= 2 && M <= 8, "gemv_n: M in [2,8]");
    TORCH_CHECK(N % (32 * WG) == 0 && K % KKC == 0 && group_size % 16 == 0, "shape");
    auto out = torch::empty({M, N}, A.options());
    const dim3 grid(N / (32 * WG)), block(WG * 32);
    const size_t smem = (size_t) M * KKC * sizeof(float);
    auto s = c10::cuda::getCurrentCUDAStream();
    const uint32_t * w = reinterpret_cast<const uint32_t *>(wt.data_ptr<int32_t>());
    const __half * sc = reinterpret_cast<const __half *>(scales.data_ptr<at::Half>());
    const __half * bi = reinterpret_cast<const __half *>(bias.data_ptr<at::Half>());
    const __half * a  = reinterpret_cast<const __half *>(A.contiguous().data_ptr<at::Half>());
    __half * c = reinterpret_cast<__half *>(out.data_ptr<at::Half>());
    if (M <= 4) gemv_n_kernel<WG, KK, 4><<<grid, block, smem, s>>>(w, sc, bi, a, c, M, N, K, group_size);
    else        gemv_n_kernel<WG, KK, 8><<<grid, block, smem, s>>>(w, sc, bi, a, c, M, N, K, group_size);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int2_gemv_m1", &int2_gemv_m1, "int2 SM70 GEMV M=1 (affine grouped dequant)");
    m.def("int2_gemv_n", &int2_gemv_n, "int2 SM70 GEMV M=2..8 (n-split, affine)");
    m.def("int2_repack_nmajor", &int2_repack_nmajor, "row-major qweight -> n-major uint32");
}
