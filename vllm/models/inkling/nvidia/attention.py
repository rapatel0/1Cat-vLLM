# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import cast

import os

import torch

from vllm.logger import init_logger
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from ..configs import InklingModelConfig
from .layernorm import InklingRMSNorm
from .ops.fa4_rel_attention import (
    bucket_max_seqlen_q,
    inkling_fa4_num_splits,
    inkling_fa4_rel_attention,
)
from .ops.fa4_warmup import InklingFA4WarmupConfig, register_fa4_warmup
from .ops.qkvr_prep import fused_qkvr_prep
from .sconv_swa_attn import _K, _V, InklingConvState, InklingSconvMetadata
from .short_conv import InklingShortConv
from .sm70 import inkling_sm70_rel_attention, use_sm70_rel_attention


logger = init_logger(__name__)

_NAN_DUMPED = False

# wo_ud's output is a residual contribution and so carries the model's
# unnormalised range; measured at 7.2e3 (9x under FP16) on a chat-template
# prompt, in the same family as the MoE stores that did overflow. wo_ud is
# linear in its input, so a power-of-two scale on the input scales the output
# exactly and the unscale is done here in FP32 -- callers see true magnitude.
#
# Stays a constant, unlike the sconv wire formats in model.py, which take their
# scale from the data. The tensor that overflows here is wo_ud's *output*, and
# the only tensor in hand before the GEMM runs is its input; an input maximum
# does not bound an output maximum without knowing the weight. Measuring the
# input harder does not help. The sound version is a load-time bound --
# |out| <= |in| * max_row_sum(|W|) -- which is guaranteed rather than observed,
# but worst-case, so it may well spend more headroom than the 9x this constant
# was measured to need. See vllm/model_executor/layers/fp16_range.py.
_ATTN_OUTPUT_SCALE = 1.0 / 64.0


def _snapshot(attn, md):
    """Clone q-adjacent state that the attention call is not supposed to touch."""
    key_cache, value_cache = attn._split_kv_cache()
    pages = torch.unique(md.block_table.flatten()).to(torch.long)
    return {
        "pages": pages.cpu(),
        "key_pages": key_cache[pages].clone(),
        "value_pages": value_cache[pages].clone(),
    }


def _absmax(t: "torch.Tensor") -> float:
    """abs().max() that tolerates the empty tensors of the profiling pass.

    The startup profiling run executes shapes with zero tokens. torch's max()
    raises "Expected reduction dim to be specified for input.numel() == 0" on
    those, which killed the engine before the NaN bisect could report anything.
    """
    return float(t.abs().max().item()) if t.numel() else 0.0


def _dump_failing_attention(attn, q, rel_logits, out, before, q_before) -> None:
    """Save a non-finite SM70 attention call, once per process.

    Captures the inputs both *before* and *after* the kernel runs. The probe
    logged q as finite immediately before the call while the post-call dump
    showed NaN, and the surviving row's absmax (23.562) was larger than the
    probe's all-row absmax (23.31) -- impossible for one tensor -- so the two
    reads disagree and only a paired before/after snapshot settles whether the
    kernel corrupts state it does not own.
    """
    global _NAN_DUMPED
    try:
        md = cast(
            FlashAttentionMetadata,
            get_forward_context().attn_metadata[attn.prefix],
        )
        after = _snapshot(attn, md)
        payload = {
            "prefix": attn.prefix,
            "is_local": attn.is_local,
            "window_size": attn.window_size,
            "rel_extent": attn.rel_extent,
            "scaling": attn.scaling,
            "num_actual_tokens": int(md.num_actual_tokens),
            "max_query_len": int(md.max_query_len),
            "max_seqlen_q_bucketed": int(bucket_max_seqlen_q(md.max_query_len)),
            "q_before": q_before.cpu(),
            "q_after": q.detach().cpu(),
            "rel_logits": rel_logits.detach().cpu(),
            "out": out.detach().cpu(),
            "block_table": md.block_table.detach().cpu(),
            "seq_lens": md.seq_lens.detach().cpu(),
            "query_start_loc": md.query_start_loc.detach().cpu(),
            "pages": before["pages"],
            "key_pages_before": before["key_pages"].cpu(),
            "value_pages_before": before["value_pages"].cpu(),
            "key_pages_after": after["key_pages"].cpu(),
            "value_pages_after": after["value_pages"].cpu(),
            # Aliasing check: if these overlap, "corruption" is just a write.
            "ptr_q": q.data_ptr(),
            "ptr_out": out.data_ptr(),
            "ptr_rel": rel_logits.data_ptr(),
            "nbytes_q": q.numel() * q.element_size(),
            "nbytes_out": out.numel() * out.element_size(),
        }
        path = f"/workspace/inkling-nan-{get_tensor_model_parallel_rank()}.pt"
        torch.save(payload, path)
        logger.info("INKLING_ATTN dumped failing invocation to %s", path)
    except Exception:  # noqa: BLE001
        logger.exception("INKLING_ATTN dump failed")
    finally:
        _NAN_DUMPED = True


def compute_log_scaling_tau(
    positions: torch.Tensor, n_floor: int, alpha: float
) -> torch.Tensor:
    effective_n = (positions + 1).to(torch.float32)
    return 1.0 + alpha * torch.log(torch.clamp(effective_n / float(n_floor), min=1.0))


class RelLogitsProj(nn.Module):
    """Project the per-head relative branch ``r`` to per-distance logits."""

    def __init__(self, d_rel: int, rel_extent: int) -> None:
        super().__init__()
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.proj = nn.Parameter(torch.empty(d_rel, rel_extent), requires_grad=False)

    def forward(self, r_out: torch.Tensor) -> torch.Tensor:
        # r_out: (T, num_heads, d_rel) -> (T, num_heads, rel_extent)
        return torch.einsum("thd,de->the", r_out, self.proj)


class InklingAttention(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        config: InklingModelConfig,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rel_extent: int,
        local_extent: int,
        is_local: bool,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        conv_owner: InklingConvState,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.d_rel = config.d_rel
        self.log_scaling_n_floor = config.log_scaling_n_floor
        self.log_scaling_alpha = config.log_scaling_alpha
        # q/k are per-head RMS-normed (unit norm), so Inkling scales by 1/head_dim.
        self.scaling = 1.0 / head_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.num_total_heads = num_heads
        self.num_total_kv_heads = num_kv_heads
        assert self.num_total_heads % tp_size == 0
        self.num_heads = self.num_total_heads // tp_size
        if self.num_total_kv_heads >= tp_size:
            assert self.num_total_kv_heads % tp_size == 0
        else:
            assert tp_size % self.num_total_kv_heads == 0
        self.num_kv_heads = max(1, self.num_total_kv_heads // tp_size)
        # When tp_size > num_kv_heads the K/V projections are padded up to
        # tp_size heads so each rank gets at least one (GQA replication).
        kv_total_for_sizing = max(self.num_total_kv_heads, tp_size)

        # Upstream fuses q/k/v/r into one MergedColumnParallelLinear. That
        # cannot represent the served checkpoint: it quantizes wq_du/wk_dv/
        # wv_dv to INT4 on 40 of 42 layers but keeps wr_du in BF16 on *all*
        # 42 (it is in the quantization ignore list), and one linear carries a
        # single quant method for all its shards.
        #
        # So split off the relative branch, as toncao/vllm's
        # inkling-w4a16-mixed-precision does. q/k/v stay fused -- they always
        # share a precision -- and wr_du becomes its own unquantized
        # projection. forward() re-concatenates them into the [q, k, v, r]
        # layout fused_qkvr_prep expects; per-rank that ordering is preserved
        # because each shard is sharded along the same output axis.
        self.qkv = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[
                head_dim * self.num_total_heads,
                head_dim * kv_total_for_sizing,
                head_dim * kv_total_for_sizing,
            ],
            bias=config.q_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )
        # quant_config is deliberately omitted rather than relying on the
        # checkpoint's ignore list matching: the ignore entries are spelled in
        # checkpoint naming (model.llm.layers.N.attn.wr_du) while this layer's
        # runtime prefix is model.layers.N.attn.wr_du. The checkpoint carries
        # 42 plain wr_du weights and zero quantized ones, so unquantized is
        # not a guess.
        self.wr_du = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=self.d_rel * self.num_total_heads,
            bias=config.q_bias,
            quant_config=None,
            prefix=f"{prefix}.wr_du",
        )
        self.wo_ud = RowParallelLinear(
            input_size=head_dim * self.num_total_heads,
            output_size=config.hidden_size,
            bias=config.o_bias,
            quant_config=quant_config,
            # reduce_results=False: the partial output is all-reduced below
            # (one-shot custom AR) so the attention-output sconv can run on the
            # full hidden width fused with the residual add + rmsnorm.
            reduce_results=False,
            prefix=f"{prefix}.wo_ud",
        )
        self.rel_extent = local_extent if is_local else rel_extent
        self.local_extent = local_extent if is_local else None
        self.rel_logits_proj = RelLogitsProj(self.d_rel, self.rel_extent)
        self.q_norm = InklingRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = InklingRMSNorm(head_dim, eps=config.rms_norm_eps)

        # Short convolution on the K/V streams (per-head-width, TP sharded),
        # applied after the qkvr projection and before q/k norm.
        kv_conv_dim = self.num_kv_heads * head_dim
        self.conv_owner = conv_owner
        self.k_sconv = InklingShortConv(
            kv_conv_dim, config.sconv_kernel_size, owner=conv_owner, stream_idx=_K
        )
        self.v_sconv = InklingShortConv(
            kv_conv_dim, config.sconv_kernel_size, owner=conv_owner, stream_idx=_V
        )

        # FA4 left/right window; right=0 keeps it causal. local_extent-1 mirrors
        # the source (sliding_window_size - 1).
        self.window_size: tuple[int, int] = (
            (local_extent - 1, 0) if is_local else (-1, -1)
        )
        # Static per-layer-type KV length bound for the split heuristic: local
        # layers never see more than the sliding window.
        vllm_config = get_current_vllm_config()
        self._max_kv_len = (
            local_extent if is_local else vllm_config.model_config.max_model_len
        )

        # ---- KV-cache wiring (reuse FlashAttentionBackend for metadata) ----
        cache_config = vllm_config.cache_config
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        self.register_buffer("k_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer("v_scale", torch.ones((), dtype=torch.float32))

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache

        register_fa4_warmup(
            InklingFA4WarmupConfig(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                rel_extent=self.rel_extent,
                window_size=self.window_size,
                is_local=self.is_local,
                max_kv_len=self._max_kv_len,
                dtype=vllm_config.model_config.dtype,
                kv_dtype=self.kv_cache_torch_dtype,
                block_size=vllm_config.cache_config.block_size,
                max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
                max_num_batched_tokens=(
                    vllm_config.scheduler_config.max_num_batched_tokens
                ),
            )
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return FlashAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = vllm_config.cache_config.block_size
        if self.is_local:
            assert self.local_extent is not None
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                dtype=self.kv_cache_torch_dtype,
                sliding_window=self.local_extent,
            )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
        )

    def _split_kv_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the K and V views as (num_blocks, block_size, heads, dim).

        The two FlashAttention backends disagree on how the bound cache holds
        K and V, and upstream's split only handles its own:

          upstream  (num_blocks, num_kv_heads, block_size, 2 * head_size)
                    K/V packed into the content dim
          this fork (num_blocks, 2, block_size, num_kv_heads, head_size)
                    K/V on a separate leading axis

        Upstream's `transpose(1, 2).split(head_dim, dim=-1)` on the fork's
        5-D layout leaves a trailing dim of exactly head_dim, so `split`
        returns a single chunk and the unpack raises
        "not enough values to unpack (expected 2, got 1)".

        Both layouts are handled here by rank, since the intermediate that
        `sm70.inkling_sm70_rel_attention` and `fused_qkvr_prep` consume is the
        same either way.
        """
        cache = self.kv_cache
        if cache.dim() == 5:
            # (num_blocks, 2, block_size, num_kv_heads, head_size)
            key_cache, value_cache = cache.unbind(1)
        else:
            # (num_blocks, num_kv_heads, block_size, 2 * head_size)
            key_cache, value_cache = cache.transpose(1, 2).split(
                self.head_dim, dim=-1
            )
        return (
            canonicalize_singleton_dim_strides(key_cache),
            canonicalize_singleton_dim_strides(value_cache),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        log_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        # Rebuild the fused [q, k, v, r] activation the prep kernel consumes.
        # Both projections shard along the output axis, so each rank's local
        # concatenation is the same ordering the fused linear would produce.
        qkv, _ = self.qkv(hidden_states)
        r, _ = self.wr_du(hidden_states)
        if os.environ.get("INKLING_DEBUG_NAN") == "1":
            for _tag, _t in (("in", hidden_states), ("qkv", qkv), ("r", r)):
                logger.info(
                    "INKLING_QKV %s %s finite=%s absmax=%.4g",
                    self.prefix,
                    _tag,
                    bool(torch.isfinite(_t).all().item()),
                    _absmax(_t),
                )
        qkvr = torch.cat((qkv, r), dim=-1)

        attn_metadata = get_forward_context().attn_metadata
        attn_output = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=qkvr.dtype,
            device=qkvr.device,
        )
        if not isinstance(attn_metadata, dict):
            attn_output.zero_()
        else:
            conv_meta = attn_metadata[self.conv_owner.prefix]
            md = attn_metadata[self.prefix]
            assert isinstance(conv_meta, InklingSconvMetadata)
            fa_md = cast(FlashAttentionMetadata, md)
            assert self.kv_cache.numel() > 0
            assert self.conv_owner.kv_cache.numel() > 0
            # One launch: K/V sconv (conv-cache insert + conv + residual),
            # Q/K per-head rmsnorm, and the attention KV-cache write. K/V are
            # consumed via the KV cache; only normed q is materialized.
            key_cache, value_cache = self._split_kv_cache()
            off_k, _ = self.conv_owner.stream_ranges[_K]
            off_v, _ = self.conv_owner.stream_ranges[_V]
            q, rel_logits = fused_qkvr_prep(
                qkvr,
                self.k_sconv.weight.squeeze(1),
                self.v_sconv.weight.squeeze(1),
                self.q_norm.weight,
                self.k_norm.weight,
                self.rel_logits_proj.proj,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.d_rel,
                self.conv_owner.kv_cache,
                key_cache,
                value_cache,
                positions,
                conv_meta.block_table,
                conv_meta.seq_idx,
                conv_meta.slot_mapping,
                conv_meta.query_start,
                fa_md.slot_mapping,
                off_k,
                off_v,
                self.conv_owner.block_size,
                log_scaling if not self.is_local else None,
            )
            q = q.view(num_tokens, self.num_heads, self.head_dim)
            _adbg = os.environ.get("INKLING_DEBUG_NAN") == "1"
            if _adbg:
                for _tag, _t in (
                    ("qkvr", qkvr),
                    ("q_normed", q),
                    ("rel_logits", rel_logits),
                ):
                    logger.info(
                        "INKLING_ATTN %s local=%s finite=%s absmax=%.4g",
                        _tag,
                        self.is_local,
                        bool(torch.isfinite(_t).all().item()),
                        _absmax(_t),
                    )
            if _adbg:
                _kvw = self._split_kv_cache()
                _slots = fa_md.slot_mapping[fa_md.slot_mapping >= 0].to(torch.long)
                _bs = _kvw[0].shape[1]
                _kw = _kvw[0][_slots // _bs, _slots % _bs]
                _vw = _kvw[1][_slots // _bs, _slots % _bs]
                logger.info(
                    "INKLING_KVW %s k_finite=%s k_absmax=%.4g "
                    "v_finite=%s v_absmax=%.4g",
                    self.prefix,
                    bool(torch.isfinite(_kw).all().item()),
                    _absmax(_kw),
                    bool(torch.isfinite(_vw).all().item()),
                    _absmax(_vw),
                )
                _before = _snapshot(self, fa_md)
                _q_before = q.detach().clone()
            self._attention(q, rel_logits, attn_output)
            if _adbg:
                _fin = bool(torch.isfinite(attn_output).all().item())
                logger.info(
                    "INKLING_ATTN kernel_out local=%s finite=%s absmax=%.4g",
                    self.is_local,
                    _fin,
                    _absmax(attn_output),
                )
                if not _fin and not _NAN_DUMPED:
                    # Capture the exact failing invocation so it can be
                    # replayed offline in seconds instead of via a 15-minute
                    # model reload. Only the referenced KV pages are saved --
                    # the whole cache is far too large.
                    _dump_failing_attention(
                        self, q, rel_logits, attn_output, _before, _q_before
                    )

        flat = attn_output.view(num_tokens, -1).mul(_ATTN_OUTPUT_SCALE)
        output, _ = self.wo_ud(flat)
        return output.float().mul_(1.0 / _ATTN_OUTPUT_SCALE)

    @eager_break_during_capture
    def _attention(
        self,
        q: torch.Tensor,
        rel_logits: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict)
        md = cast(FlashAttentionMetadata, attn_metadata[self.prefix])

        nt = md.num_actual_tokens
        key_cache, value_cache = self._split_kv_cache()
        max_seqlen_q = bucket_max_seqlen_q(md.max_query_len)

        if use_sm70_rel_attention():
            # Volta: no FA4/CuTe. Same signature, single-pass Triton scan.
            inkling_sm70_rel_attention(
                q[:nt],
                key_cache,
                value_cache,
                block_table=md.block_table,
                cache_seqlens=md.seq_lens,
                cu_seqlens_q=md.query_start_loc,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=self.scaling,
                causal=True,
                window_size=self.window_size,
                rel_extent=self.rel_extent,
                rel_logits=rel_logits[:nt],
                out=output[:nt],
            )
            return

        num_splits = inkling_fa4_num_splits(
            is_local=self.is_local,
            batch_size=md.seq_lens.shape[0],
            max_query_len=max_seqlen_q,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            max_kv_len=self._max_kv_len,
        )
        inkling_fa4_rel_attention(
            q[:nt],
            key_cache,
            value_cache,
            block_table=md.block_table,
            cache_seqlens=md.seq_lens,
            cu_seqlens_q=md.query_start_loc,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=self.scaling,
            causal=True,
            window_size=self.window_size,
            rel_extent=self.rel_extent,
            rel_logits=rel_logits[:nt],
            num_splits=num_splits,
            out=output[:nt],
        )
