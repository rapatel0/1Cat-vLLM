# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.dspark import (
    _apply_rope_gptj_last,
    _dspark_query_block_size,
    _rmsnorm_no_weight,
)
from vllm.models.deepseek_v4.nvidia.dspark_triton import (
    dspark_context_kv_store,
    dspark_markov_probs_sample,
    dspark_qkv_postprocess,
    dspark_triton_attention,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="requires CUDA SM70",
)


def test_dspark_runtime_block_size_uses_speculative_token_count() -> None:
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=7)
    )

    assert _dspark_query_block_size(vllm_config) == 7


def _cos_sin_cache(max_position: int, rope_dim: int) -> torch.Tensor:
    angles = torch.randn(
        max_position,
        rope_dim // 2,
        device="cuda",
        dtype=torch.float32,
    )
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.float16)


def test_dspark_fp16_qkv_postprocess_matches_reference() -> None:
    torch.manual_seed(17)
    tokens, heads, head_dim, rope_dim = 7, 8, 128, 64
    eps = 1e-6
    positions = torch.tensor(
        [0, 1, 7, 19, 31, 63, 95], device="cuda", dtype=torch.int64
    )
    cos_sin_cache = _cos_sin_cache(128, rope_dim)
    q = torch.randn(
        tokens,
        heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    kv = torch.randn(tokens, head_dim, device="cuda", dtype=torch.float16)

    q_out, kv_out = dspark_qkv_postprocess(q, kv, positions, cos_sin_cache, eps)
    q_ref = _apply_rope_gptj_last(
        _rmsnorm_no_weight(q, eps).to(q.dtype), positions, cos_sin_cache
    )
    kv_ref = _apply_rope_gptj_last(kv, positions, cos_sin_cache)

    torch.testing.assert_close(q_out, q_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(kv_out, kv_ref, rtol=2e-3, atol=2e-3)


def test_dspark_fp16_context_kv_store_matches_reference() -> None:
    torch.manual_seed(23)
    tokens, head_dim, rope_dim, window = 7, 128, 64, 128
    eps = 1e-6
    positions = torch.tensor(
        [0, 1, 7, 19, 31, 63, 95], device="cuda", dtype=torch.int64
    )
    query_start_loc = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    cos_sin_cache = _cos_sin_cache(window, rope_dim)
    kv = torch.randn(tokens, head_dim, device="cuda", dtype=torch.float16)
    weight = torch.randn(head_dim, device="cuda", dtype=torch.float16)
    cache = torch.zeros(1, window, head_dim, device="cuda", dtype=torch.float16)

    dspark_context_kv_store(
        kv,
        cache,
        positions,
        query_start_loc,
        1,
        None,
        weight,
        cos_sin_cache,
        eps,
    )
    kv_float = kv.float()
    normalized = (
        kv_float
        * torch.rsqrt(kv_float.square().mean(-1, keepdim=True) + eps)
        * weight.float()
    ).to(kv.dtype)
    expected = _apply_rope_gptj_last(normalized, positions, cos_sin_cache)

    torch.testing.assert_close(
        cache[0, positions % window], expected, rtol=2e-3, atol=2e-3
    )


def test_dspark_fp16_attention_matches_reference() -> None:
    torch.manual_seed(29)
    # Exercise the checkpoint's real attention shape.  The 512-wide head is
    # what constrains the Triton tile on V100's 96 KiB shared-memory limit.
    batch, block, heads, head_dim, window = 1, 5, 32, 512, 128
    q = torch.randn(
        batch,
        block,
        heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    main_kv = torch.randn(batch, window, head_dim, device="cuda", dtype=torch.float16)
    draft_kv = torch.randn(batch, block, head_dim, device="cuda", dtype=torch.float16)
    main_positions = torch.tensor([63], device="cuda", dtype=torch.int64)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    scale = head_dim**-0.5

    actual = dspark_triton_attention(q, main_kv, draft_kv, main_positions, sink, scale)
    all_kv = torch.cat((main_kv, draft_kv), dim=1)
    valid_main = torch.arange(window, device="cuda")[None, :] <= main_positions[:, None]
    valid = torch.cat(
        (
            valid_main,
            torch.ones(batch, block, device="cuda", dtype=torch.bool),
        ),
        dim=1,
    )
    scores = torch.einsum("bqhd,bkd->bqhk", q.float(), all_kv.float()) * scale
    scores.masked_fill_(~valid[:, None, None, :], -torch.inf)
    sink_scores = sink[None, None, :, None].expand(batch, block, heads, 1)
    probs = torch.softmax(torch.cat((scores, sink_scores), dim=-1), dim=-1)[..., :-1]
    expected = torch.einsum("bqhk,bkd->bqhd", probs, all_kv.float()).to(q.dtype)

    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=8e-3)


@pytest.mark.parametrize("invalid_value", [-torch.inf, torch.nan])
def test_dspark_markov_sampler_handles_nonfinite_row(invalid_value: float) -> None:
    batch_size, vocab_size, block_v = 1, 4096, 1024
    num_blocks = vocab_size // block_v
    logits = torch.full(
        (batch_size, vocab_size),
        invalid_value,
        device="cuda",
        dtype=torch.float16,
    )
    inv_temp = torch.ones(batch_size, device="cuda", dtype=torch.float32)
    is_greedy = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
    tokens = torch.empty(batch_size, device="cuda", dtype=torch.int64)
    probs = torch.empty_like(logits, dtype=torch.float32)
    scratch = {
        "block_max": torch.empty(
            batch_size, num_blocks, device="cuda", dtype=torch.float32
        ),
        "block_sumexp": torch.empty(
            batch_size, num_blocks, device="cuda", dtype=torch.float32
        ),
        "block_gval": torch.empty(
            batch_size, num_blocks, device="cuda", dtype=torch.float32
        ),
        "block_maxid": torch.empty(
            batch_size, num_blocks, device="cuda", dtype=torch.int32
        ),
        "block_gid": torch.empty(
            batch_size, num_blocks, device="cuda", dtype=torch.int32
        ),
        "row_max": torch.empty(batch_size, device="cuda", dtype=torch.float32),
        "row_invz": torch.empty(batch_size, device="cuda", dtype=torch.float32),
    }

    dspark_markov_probs_sample(
        logits,
        inv_temp,
        is_greedy,
        tokens,
        probs,
        scratch,
        seed=17,
        block_v=block_v,
    )
    torch.accelerator.synchronize()

    assert tokens.item() == 0
    assert torch.count_nonzero(probs).item() == 0
    assert torch.isfinite(probs).all()
