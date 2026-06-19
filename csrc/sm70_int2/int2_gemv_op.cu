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
    auto stream = at::cuda::getCurrentCUDAStream();
    gemv_m1_kernel<<<grid, block, 0, stream>>>(
        qweight.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(bias.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(out.data_ptr<at::Half>()),
        N, K, (int) group_size);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int2_gemv_m1", &int2_gemv_m1, "int2 SM70 GEMV M=1 (affine grouped dequant)");
}
