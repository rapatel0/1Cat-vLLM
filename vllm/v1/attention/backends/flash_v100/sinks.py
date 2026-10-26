# SPDX-License-Identifier: Apache-2.0
"""Apply Flint-style learned attention sinks to a FlashAttention result.

A learned sink is an extra softmax logit with no value vector. If ``lse`` is
the log-sum-exp of the key scores, the corrected output is

    out * exp(lse - logaddexp(lse, sink))

which is exactly ``out * sigmoid(lse - sink)``. The sigmoid form needs one
exponential instead of three, and it lets the multiply stay in the output
dtype. ``lse`` uses the FlashAttention layout ``[query_heads, tokens]``.
``sinks`` is ``[query_heads]``.
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton


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
    scale = torch.sigmoid(lse_f - sink)
    output.mul_(scale.to(dtype=output.dtype))
    return output


@triton.jit
def _paged_decode_sink_kernel(
    out_ptr,
    max_ptr,
    sum_ptr,
    seq_ptr,
    sink_ptr,
    out_stride_b,
    out_stride_h,
    max_stride_b,
    max_stride_h,
    sum_stride_b,
    sum_stride_h,
    partition_size,
    launch_num_partitions,
    D: tl.constexpr,
):
    """Scale a paged-decode row by the learned-sink softmax factor.

    ``max_ptr`` / ``sum_ptr`` are the Flash-V100 partition stats
    ``[batch, heads, partitions]``. Partitions merge in one online pass, the
    same recurrence the reduce kernel uses, so the merged LSE matches:
    ``global_max + log(sum(exp_sums * exp(part_max - global_max)))`` over
    ``min(launch_num_partitions, ceil(seq_len / partition_size))`` partitions.
    """
    batch = tl.program_id(0)
    head = tl.program_id(1)
    seq_len = tl.load(seq_ptr + batch)
    if seq_len <= 0:
        return
    num_parts = (seq_len + partition_size - 1) // partition_size
    if num_parts > launch_num_partitions:
        num_parts = launch_num_partitions
    if num_parts <= 0:
        return

    max_base = max_ptr + batch * max_stride_b + head * max_stride_h
    sum_base = sum_ptr + batch * sum_stride_b + head * sum_stride_h
    merged_max = -1.0e20
    merged_sum = 0.0
    part = 0
    while part < num_parts:
        part_max = tl.load(max_base + part)
        part_sum = tl.load(sum_base + part)
        new_max = tl.maximum(merged_max, part_max)
        merged_sum = merged_sum * tl.exp(merged_max - new_max) + part_sum * tl.exp(
            part_max - new_max
        )
        merged_max = new_max
        part += 1

    if merged_sum <= 0:
        return

    lse = merged_max + tl.log(merged_sum)
    sink = tl.load(sink_ptr + head)
    scale = tl.sigmoid(lse - sink)

    offs = tl.arange(0, D)
    out_ptrs = out_ptr + batch * out_stride_b + head * out_stride_h + offs
    vals = tl.load(out_ptrs).to(tl.float32) * scale
    tl.store(out_ptrs, vals)


def apply_paged_decode_sinks(
    output: torch.Tensor,
    max_logits: torch.Tensor,
    exp_sums: torch.Tensor,
    seq_lens: torch.Tensor,
    sinks: torch.Tensor,
    *,
    partition_size: int,
    launch_num_partitions: int,
) -> torch.Tensor:
    """Apply learned sinks to paged Flash-V100 decode output in place.

    ``output`` is ``[batch, heads, dim]``. Partition stats are the workspace
    the decode kernel just wrote. This launch allocates nothing, so it can
    run inside a CUDA graph.
    """
    if output.numel() == 0 or os.environ.get("FLINT_SKIP_DECODE_SINKS") == "1":
        return output
    if output.ndim != 3:
        raise ValueError(f"paged decode output must be [B, H, D], got {tuple(output.shape)}")
    batch, heads, _dim = output.shape
    if tuple(sinks.shape) != (heads,):
        raise ValueError(f"sinks shape {tuple(sinks.shape)} does not match heads={heads}")
    if launch_num_partitions <= 0 or partition_size <= 0:
        raise ValueError("paged decode sink correction needs a positive partition plan")
    if sinks.dtype != torch.float32 or not sinks.is_contiguous():
        raise RuntimeError("learned sinks must already be contiguous fp32")
    if (
        os.environ.get("FLINT_SINK_DEBUG") == "1"
        and not torch.cuda.is_current_stream_capturing()
    ):
        longest = int(seq_lens.max().item())
        parts = min(
            launch_num_partitions, (longest + partition_size - 1) // partition_size
        )
        part_max = max_logits[:, :, :parts]
        part_sum = exp_sums[:, :, :parts]
        merged_max = part_max.amax(dim=2)
        merged_sum = (
            part_sum * torch.exp(part_max - merged_max.unsqueeze(2))
        ).sum(dim=2)
        lse = merged_max + torch.log(merged_sum.clamp_min(1.0e-20))
        scale = torch.sigmoid(lse - sinks.view(1, -1))
        print(
            f"[sink-debug] seq={longest} parts={parts} "
            f"lse_mean={lse.mean().item():.3f} sink_mean={sinks.mean().item():.3f} "
            f"scale_min={scale.min().item():.3f} "
            f"scale_mean={scale.mean().item():.3f} "
            f"stats={tuple(max_logits.shape)} "
            f"max0={float(part_max[0, 0, 0]):.3f} "
            f"sum0={float(part_sum[0, 0, 0]):.3f} "
            f"seqvals={seq_lens[:4].to(torch.float32).tolist()}",
            flush=True,
        )
    _paged_decode_sink_kernel[(batch, heads)](  # type: ignore[index]
        output,
        max_logits,
        exp_sums,
        seq_lens,
        sinks,
        output.stride(0),
        output.stride(1),
        max_logits.stride(0),
        max_logits.stride(1),
        exp_sums.stride(0),
        exp_sums.stride(1),
        int(partition_size),
        int(launch_num_partitions),
        D=output.shape[-1],
    )
    return output
