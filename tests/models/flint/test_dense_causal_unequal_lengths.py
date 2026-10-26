"""Check causal attention with M != N in the Flash-V100 dense LSE forward.

Decodes and chunked prefills need query rows right-aligned against a longer key
segment. This compares that behaviour against a torch reference.
"""

import torch

from vllm.v1.attention.backends.flash_v100.dense_prefill import (
    flash_v100_dense_prefill_lse,
)

torch.manual_seed(0)
dev = "cuda"
heads_q, heads_kv, dim = 8, 2, 128
q_len, k_len = 3, 7

q = torch.randn(q_len, heads_q, dim, device=dev, dtype=torch.float16)
k = torch.randn(k_len, heads_kv, dim, device=dev, dtype=torch.float16)
v = torch.randn(k_len, heads_kv, dim, device=dev, dtype=torch.float16)
out = torch.empty_like(q)
lse = torch.empty((heads_q, q_len), device=dev, dtype=torch.float32)
cu_q = torch.tensor([0, q_len], device=dev, dtype=torch.int32)
cu_k = torch.tensor([0, k_len], device=dev, dtype=torch.int32)
scale = dim**-0.5

flash_v100_dense_prefill_lse(
    query=q,
    key=k,
    value=v,
    output=out,
    softmax_lse=lse,
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
    num_actual_tokens=q_len,
    softmax_scale=scale,
    causal=True,
    window_size=(-1, -1),
)


def kv_head_of(query_head: int) -> int:
    return int(query_head * heads_kv // heads_q)


def visible_keys(row: int) -> int:
    # Bottom-right causal alignment: query row i sees keys 0 .. i + (N - M).
    return row + (k_len - q_len) + 1


def scores_for(row: int) -> torch.Tensor:
    length = visible_keys(row)
    qi = q[row].float()  # [heads_q, dim]
    out_scores = torch.empty(length, heads_q, device=dev)
    for head in range(heads_q):
        kv = k[:length, kv_head_of(head)].float()  # [length, dim]
        out_scores[:, head] = (kv * qi[head]).sum(dim=-1) * scale
    return out_scores


def reference(row: int) -> torch.Tensor:
    length = visible_keys(row)
    weights = torch.softmax(scores_for(row), dim=0)  # [length, heads_q]
    values = v[:length].float()  # [length, heads_kv, dim]
    expanded = values.repeat_interleave(heads_q // heads_kv, dim=1)
    return (weights.unsqueeze(-1) * expanded).sum(dim=0)


worst = 0.0
worst_lse = 0.0
for row in range(q_len):
    diff = (out[row].float() - reference(row)).abs().max().item()
    worst = max(worst, diff)
    recomputed = torch.logsumexp(scores_for(row), dim=0)
    worst_lse = max(worst_lse, (lse[:, row] - recomputed).abs().max().item())

print(f"max_output_abs_diff={worst:.5f} max_lse_abs_diff={worst_lse:.5f}")
assert worst < 0.02, f"causal M!=N output mismatch: {worst}"
assert worst_lse < 0.05, f"causal M!=N LSE mismatch: {worst_lse}"
print("OK bottom-right causal alignment for M != N")
