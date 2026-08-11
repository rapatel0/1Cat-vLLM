# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Functional Inkling relative-attention fallback for SM70 GPUs.

Inkling's attention adds a learned, per-distance relative bias to the attention
scores. Upstream implements it two ways, neither of which exists on Volta:

* Hopper: standard FA4 with a CuTe-DSL ``score_mod`` gather.
* Blackwell: tml-fa4's sheared relative-bias layout.

Both need the CUTLASS Python DSL and SM90+ hardware. This module provides a
Triton paged varlen attention kernel that folds the bias in directly, following
the same shape as the DeepSeek-V4 SM70 fallbacks in
``vllm/models/deepseek_v4/nvidia/sm70.py``: correctness first, on kernels that
actually compile for ``sm_70``.

Bias semantics, transcribed from ``ops/fa4_rel_attention.py::_get_score_mod``:

    seqlen_local_offset = seqlen_k - seqlen_q
    rel_dist  = (q_idx + seqlen_local_offset) - kv_idx
    rel_idx   = clamp(rel_dist, 0, rel_extent - 1)
    rel_bias  = rel_logits[offset_q + q_idx, h_idx, rel_idx]
                    if rel_dist == rel_idx else 0.0
    scores   += rel_bias

The ``rel_dist == rel_idx`` guard makes the bias *finite support*: it applies
only for distances in ``[0, rel_extent)`` and is exactly zero beyond, rather
than saturating at the edge entry. Getting that wrong silently biases every
long-range key by the ``rel_extent - 1`` value, so the guard is reproduced
verbatim below.

Volta notes:
  * ``tl.dot`` lowers to ``mma.sync.m8n8k4`` with FP16 inputs and FP32
    accumulate. Scores, the running max and the denominator are kept in FP32
    throughout -- Volta has no BF16 arithmetic, and Inkling's log scaling can
    push pre-softmax scores past the FP16 range.
  * The bias gather is a per-element ``tl.load`` with a computed index. It is
    not free, but it avoids materializing a [BLOCK_M, BLOCK_N] bias tensor per
    KV step in HBM.
"""

from __future__ import annotations

from functools import cache

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@cache
def use_sm70_rel_attention() -> bool:
    """True when the FA4 relative-attention path is unavailable.

    Both upstream implementations need SM90+: standard FA4's CuTe-DSL
    ``score_mod`` (Hopper) and tml-fa4's sheared bias layout (Blackwell). On
    anything older -- Volta included -- Inkling has to use the Triton kernel
    in this module.
    """
    capability = current_platform.get_device_capability()
    return capability is None or capability.major < 9


@triton.jit
def _inkling_rel_attn_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    out_ptr,
    rel_logits_ptr,
    block_table_ptr,
    cache_seqlens_ptr,
    cu_seqlens_q_ptr,
    softmax_scale,
    # strides
    stride_q_t,
    stride_q_h,
    stride_kc_blk,
    stride_kc_pos,
    stride_kc_h,
    stride_vc_blk,
    stride_vc_pos,
    stride_vc_h,
    stride_o_t,
    stride_o_h,
    stride_rel_t,
    stride_rel_h,
    stride_bt_b,
    # sizes
    num_kv_heads,
    max_blocks_per_seq,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,  # -1 => unbounded (full causal)
    Q_PER_KV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m_tile = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q_ptr + batch)
    q_end = tl.load(cu_seqlens_q_ptr + batch + 1)
    seqlen_q = q_end - q_start
    seqlen_k = tl.load(cache_seqlens_ptr + batch)

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    m_valid = offs_m < seqlen_q
    if m_tile * BLOCK_M >= seqlen_q:
        return

    offs_d = tl.arange(0, HEAD_DIM)

    # Absolute row index into the packed varlen q/out/rel_logits tensors.
    q_rows = q_start + offs_m

    q = tl.load(
        q_ptr + q_rows[:, None] * stride_q_t + head * stride_q_h + offs_d[None, :],
        mask=m_valid[:, None],
        other=0.0,
    )

    # Position of each query within the full KV sequence. Prefill packs the
    # new tokens at the tail of the cache, so the first new token sits at
    # seqlen_k - seqlen_q.
    q_pos = offs_m + (seqlen_k - seqlen_q)

    kv_head = head // Q_PER_KV

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Only keys at distance <= WINDOW_LEFT can contribute on local layers, so
    # start the scan at the first in-window key instead of at zero.
    if WINDOW_LEFT >= 0:
        lo = tl.maximum(0, (m_tile * BLOCK_M + (seqlen_k - seqlen_q)) - WINDOW_LEFT)
        lo = (lo // BLOCK_N) * BLOCK_N
    else:
        lo = 0
    # Causal: no key beyond the last query position in this tile.
    hi = tl.minimum(seqlen_k, (m_tile + 1) * BLOCK_M + (seqlen_k - seqlen_q))

    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < seqlen_k

        # Paged gather: map each KV position to (page, slot) via block_table.
        page_idx = offs_n // BLOCK_SIZE
        slot = offs_n % BLOCK_SIZE
        page = tl.load(
            block_table_ptr + batch * stride_bt_b + page_idx,
            mask=(page_idx < max_blocks_per_seq) & n_valid,
            other=0,
        )

        k = tl.load(
            k_cache_ptr
            + page[:, None] * stride_kc_blk
            + slot[:, None] * stride_kc_pos
            + kv_head * stride_kc_h
            + offs_d[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )

        # FP16 inputs, FP32 accumulate (Volta mma.sync).
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * softmax_scale

        rel_dist = q_pos[:, None] - offs_n[None, :]

        # Finite-support relative bias, guarded exactly as the FA4 score_mod:
        # in range -> gather, out of range -> 0.0 (never edge-clamped).
        in_rel = (rel_dist >= 0) & (rel_dist < REL_EXTENT)
        rel_bias = tl.load(
            rel_logits_ptr
            + q_rows[:, None] * stride_rel_t
            + head * stride_rel_h
            + tl.where(in_rel, rel_dist, 0),
            mask=in_rel & m_valid[:, None] & n_valid[None, :],
            other=0.0,
        )
        qk += rel_bias.to(tl.float32)

        causal = rel_dist >= 0
        if WINDOW_LEFT >= 0:
            causal = causal & (rel_dist <= WINDOW_LEFT)
        keep = causal & n_valid[None, :] & m_valid[:, None]
        qk = tl.where(keep, qk, float("-inf"))

        # Online softmax.
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # A fully masked row keeps m_i == -inf; force the correction factor to
        # zero there instead of producing NaN from (-inf) - (-inf).
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(tl.where(m_i == float("-inf"), -float("inf"), m_i - m_safe))
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        p = tl.exp(qk - m_safe[:, None])
        p = tl.where(keep, p, 0.0)

        v = tl.load(
            v_cache_ptr
            + page[:, None] * stride_vc_blk
            + slot[:, None] * stride_vc_pos
            + kv_head * stride_vc_h
            + offs_d[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Rows with no visible key (possible only on degenerate windows) stay zero.
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    tl.store(
        out_ptr + q_rows[:, None] * stride_o_t + head * stride_o_h + offs_d[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=m_valid[:, None],
    )


def inkling_sm70_rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    num_splits: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """SM70 stand-in for ``inkling_fa4_rel_attention`` with an identical
    signature, so ``InklingAttention._attention`` can swap implementations
    without branching on shapes.

    ``num_splits`` is accepted and ignored: split-KV exists to fill an SM90+
    scheduler, and this kernel does not implement the split/combine pass.
    Reduction is a single online-softmax scan per (q_tile, head, request).
    """
    assert causal, "Inkling attention is causal on every layer"
    assert q.dim() == 3, f"expected (tokens, heads, dim) q, got {tuple(q.shape)}"

    num_tokens, num_heads, head_dim = q.shape
    # InklingAttention._split_kv_cache transposes and splits into
    # (num_blocks, block_size, num_kv_heads, head_dim). The kernel indexes
    # those axes positionally, so pin the assumption: if the cache layout ever
    # changes, the gather would silently read the wrong slots rather than
    # fail, and the model would just get quietly worse.
    assert key_cache.dim() == 4 and key_cache.shape[-1] == head_dim, (
        "expected (num_blocks, block_size, num_kv_heads, head_dim) kv cache, "
        f"got {tuple(key_cache.shape)} for head_dim {head_dim}"
    )
    assert value_cache.shape == key_cache.shape
    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    batch = cache_seqlens.shape[0]

    assert num_heads % num_kv_heads == 0, (
        f"{num_heads} query heads is not a multiple of {num_kv_heads} KV heads"
    )
    q_per_kv = num_heads // num_kv_heads

    if out is None:
        out = torch.empty_like(q)

    q = q.contiguous()
    rel_logits = rel_logits.contiguous()

    window_left = window_size[0] if window_size[0] >= 0 else -1

    # Volta has 96 KB of shared memory per SM; 64x64 tiles at head_dim 128 keep
    # q/k/v tiles resident with room for the FP32 accumulator.
    #
    # BLOCK_M is sized to the actual query length. A decode step has
    # seqlen_q == 1, and a fixed BLOCK_M of 64 made the kernel compute a 64x64
    # tl.dot and gather a 64x64 relative-bias tile for a single valid row --
    # about 64x wasted work. A decode profile put this kernel at 33.8% of GPU
    # time, 326us/call, by far the largest single consumer, which is what that
    # waste looks like. Correctness never depended on the tile size: rows are
    # masked by m_valid and the store is masked identically, so shrinking
    # BLOCK_M only removes work that was being discarded.
    #
    # 16 is the floor: tl.dot requires M >= 16 on this Triton/arch pair.
    block_m = 64 if max_seqlen_q > 32 else 16
    block_n = 64

    grid = (triton.cdiv(max_seqlen_q, block_m), num_heads, batch)

    _inkling_rel_attn_kernel[grid](
        q,
        key_cache,
        value_cache,
        out,
        rel_logits,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        out.stride(0),
        out.stride(1),
        rel_logits.stride(0),
        rel_logits.stride(1),
        block_table.stride(0),
        num_kv_heads,
        block_table.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        REL_EXTENT=rel_extent,
        WINDOW_LEFT=window_left,
        Q_PER_KV=q_per_kv,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    return out
