"""Check causal alignment in the Flash-V100 dense LSE prefill.

The dense forward emits the softmax LSE that the learned-sink correction
consumes, so its causal masking and its ``[query_heads, tokens]`` LSE layout are
both under test. Right-aligned rows for a shorter query segment (decode and
chunk continuations) are served by the paged decode kernel and by
``_prefill_prefix_with_sinks``; the dense forward only has to fail closed for
that shape instead of quietly mis-aligning it.
"""

import pytest
import torch

from vllm.v1.attention.backends.flash_v100.dense_prefill import (
    flash_v100_dense_prefill_lse,
)

HEADS_Q, HEADS_KV, DIM = 8, 2, 128
SCALE = DIM**-0.5


def _kv_head_of(query_head: int) -> int:
    return int(query_head * HEADS_KV // HEADS_Q)


def _inputs(q_len: int, k_len: int, device: str = "cuda"):
    generator = torch.Generator(device=device).manual_seed(0)

    def stacked(rows: int, heads: int) -> torch.Tensor:
        raw = torch.randn(
            rows, heads, DIM, generator=generator, device=device, dtype=torch.float16
        )
        return raw

    q = stacked(q_len, HEADS_Q)
    k = stacked(k_len, HEADS_KV)
    v = stacked(k_len, HEADS_KV)
    cu_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, k_len], device=device, dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def _scores(q: torch.Tensor, k: torch.Tensor, row: int, length: int) -> torch.Tensor:
    """Causal scores for one query row, bottom-right aligned: row ``i`` sees
    keys ``0 .. i + (k_len - q_len)``."""
    qi = q[row].float()
    out_scores = torch.empty(length, HEADS_Q, device=q.device)
    for head in range(HEADS_Q):
        kv = k[:length, _kv_head_of(head)].float()
        out_scores[:, head] = (kv * qi[head]).sum(dim=-1) * SCALE
    return out_scores


def _reference(q, k, v, row: int, length: int) -> torch.Tensor:
    weights = torch.softmax(_scores(q, k, row, length), dim=0)
    values = v[:length].float()
    expanded = values.repeat_interleave(HEADS_Q // HEADS_KV, dim=1)
    return (weights.unsqueeze(-1) * expanded).sum(dim=0)


def _run(q, k, v, cu_q, cu_k):
    out = torch.empty_like(q)
    lse = torch.empty((HEADS_Q, q.shape[0]), device=q.device, dtype=torch.float32)
    flash_v100_dense_prefill_lse(
        query=q,
        key=k,
        value=v,
        output=out,
        softmax_lse=lse,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        num_actual_tokens=q.shape[0],
        softmax_scale=SCALE,
        causal=True,
        window_size=(-1, -1),
    )
    return out, lse


@pytest.mark.parametrize("length", [1, 3, 17])
def test_causal_equal_lengths_match_reference(length: int):
    """Square causal prefill matches a torch reference for output and LSE."""
    q, k, v, cu_q, cu_k = _inputs(length, length)
    out, lse = _run(q, k, v, cu_q, cu_k)
    for row in range(length):
        torch.testing.assert_close(
            out[row].float(), _reference(q, k, v, row, row + 1), atol=2e-3, rtol=1e-3
        )
        recomputed = torch.logsumexp(_scores(q, k, row, row + 1), dim=0)
        torch.testing.assert_close(lse[:, row], recomputed, atol=5e-3, rtol=1e-3)


def test_unequal_causal_fails_closed():
    """A shorter causal query segment is refused rather than left-aligned."""
    q, k, v, cu_q, cu_k = _inputs(3, 7)
    with pytest.raises(RuntimeError, match="requires q_len == k_len"):
        _run(q, k, v, cu_q, cu_k)


def test_unequal_non_causal_is_allowed():
    """Without causal masking a shorter query segment is a plain cross-attention
    row set, so the dense forward accepts it."""
    q, k, v, cu_q, cu_k = _inputs(3, 7)
    out = torch.empty_like(q)
    lse = torch.empty((HEADS_Q, 3), device=q.device, dtype=torch.float32)
    flash_v100_dense_prefill_lse(
        query=q,
        key=k,
        value=v,
        output=out,
        softmax_lse=lse,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        num_actual_tokens=3,
        softmax_scale=SCALE,
        causal=False,
        window_size=(-1, -1),
    )
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(lse).all()
