# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

Q2_TYPE = 42
QK2_0 = 128
Q2_0_BLOCK_BYTES = 34

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_device_capability(70),
    reason="Prism Q2_0 Tensor Core tests require an exact SM70 CUDA device",
)


def _pack_q2_0(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack logical Q2 codes and FP16 scales into Prism type-42 rows."""
    rows, k = codes.shape
    assert k % QK2_0 == 0
    blocks = k // QK2_0
    block_codes = codes.to(torch.uint8).reshape(rows, blocks, QK2_0)
    packed = (
        block_codes[..., 0::4]
        | (block_codes[..., 1::4] << 2)
        | (block_codes[..., 2::4] << 4)
        | (block_codes[..., 3::4] << 6)
    )
    raw = torch.empty((rows, blocks, Q2_0_BLOCK_BYTES), dtype=torch.uint8)
    scale_bytes = scales.to(torch.half).contiguous().view(torch.uint8)
    raw[..., :2] = scale_bytes.reshape(rows, blocks, 2)
    raw[..., 2:] = packed
    return raw.reshape(rows, blocks * Q2_0_BLOCK_BYTES).cuda()


def _dequantize_q2_0(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows, k = codes.shape
    blocks = k // QK2_0
    values = codes.to(torch.float32).reshape(rows, blocks, QK2_0) - 1.0
    return (values * scales.float().unsqueeze(-1)).reshape(rows, k).half()


@torch.inference_mode()
def test_q2_0_mma_unpack_all_bytes_and_scale_edges():
    # Eight blocks contain every possible packed byte exactly once.
    packed = torch.arange(256, dtype=torch.int64).reshape(1, 8, 32)
    codes = torch.empty((1, 8, QK2_0), dtype=torch.int64)
    for index in range(4):
        codes[..., index::4] = (packed >> (2 * index)) & 0x3
    codes = codes.reshape(1, -1)
    scales = torch.tensor(
        [[0.0, -0.0, 2**-24, -(2**-14), 0.5, -1.0, 32752.0, -32752.0]],
        dtype=torch.half,
    )
    qweight = _pack_q2_0(codes, scales)

    # Identity activations expose every independently decoded weight through
    # the wide-prefill MMA route.
    x = torch.eye(codes.shape[1], dtype=torch.half, device="cuda")
    output = ops.ggml_mul_mat_q2_0_sm70(qweight, x, qweight.shape[0])
    reference = _dequantize_q2_0(codes, scales).cuda().T

    torch.testing.assert_close(output, reference, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [4, 8, 64, 65])
@torch.inference_mode()
def test_q2_0_mma_error_no_worse_than_dp4a(num_tokens: int):
    generator = torch.Generator().manual_seed(20260716 + num_tokens)
    rows = 257
    k = 1024
    blocks = k // QK2_0
    codes = torch.randint(0, 4, (rows, k), generator=generator)
    scales = (
        torch.rand((rows, blocks), generator=generator, dtype=torch.float32) * 0.125
        - 0.0625
    ).half()
    qweight = _pack_q2_0(codes, scales)
    weight = _dequantize_q2_0(codes, scales).cuda()
    x = (torch.randn((num_tokens, k), generator=generator) * 0.5).half().cuda()
    reference = x.float() @ weight.float().T

    candidate = ops.ggml_mul_mat_q2_0_sm70(qweight, x, rows).float()
    dp4a = ops.ggml_mul_mat_vec_a8(qweight, x, Q2_TYPE, rows).float()
    candidate_error = (candidate - reference).abs()
    dp4a_error = (dp4a - reference).abs()

    assert candidate_error.max() <= dp4a_error.max() + 1e-3
    assert candidate_error.square().mean() <= dp4a_error.square().mean() + 1e-6


@torch.inference_mode()
def test_q2_0_batch_one_dp4a_reference():
    generator = torch.Generator().manual_seed(42)
    rows = 129
    k = 512
    blocks = k // QK2_0
    codes = torch.randint(0, 4, (rows, k), generator=generator)
    scales = torch.full((rows, blocks), 0.03125, dtype=torch.half)
    qweight = _pack_q2_0(codes, scales)
    weight = _dequantize_q2_0(codes, scales).cuda()
    x = torch.randn((1, k), generator=generator).half().cuda()

    output = ops.ggml_mul_mat_vec_a8(qweight, x, Q2_TYPE, rows)
    reference = x.float() @ weight.float().T

    torch.testing.assert_close(output.float(), reference, atol=0.25, rtol=0.05)
