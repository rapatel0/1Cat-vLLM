# SPDX-License-Identifier: Apache-2.0
"""Apply Flint-style learned attention sinks to a FlashAttention result.

A learned sink is an extra softmax logit with no value vector. If ``lse`` is
the log-sum-exp of the key scores, the corrected output is

    out * exp(lse - logaddexp(lse, sink))

``lse`` uses the FlashAttention layout ``[query_heads, tokens]``. ``sinks`` is
``[query_heads]``.
"""

from __future__ import annotations

import torch


def apply_learned_attention_sinks(
    output: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    if output.ndim != 3:
        raise ValueError(f"sink correction expects [tokens, heads, dim], got {tuple(output.shape)}")
    tokens, heads, _ = output.shape
    if tuple(lse.shape) != (heads, tokens):
        raise ValueError(
            f"LSE shape {tuple(lse.shape)} does not match heads={heads} tokens={tokens}"
        )
    if tuple(sinks.shape) != (heads,):
        raise ValueError(f"sinks shape {tuple(sinks.shape)} does not match heads={heads}")
    sink = sinks.to(device=output.device, dtype=torch.float32).view(1, heads, 1)
    lse_f = lse.to(dtype=torch.float32).transpose(0, 1).unsqueeze(-1)
    corrected = output.float() * torch.exp(lse_f - torch.logaddexp(lse_f, sink))
    output.copy_(corrected.to(dtype=output.dtype))
    return output
