# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling mixture-of-experts on vLLM's FusedMoE abstraction.

Overfit to the served checkpoint: sigmoid gate (+ selection bias) top-k over
the routed experts, log-sigmoid renormalization over the k routed + S shared
"sink" logits, scaled by route_scale * global_scale. The routed top-k goes
through vLLM's FusedMoE (which handles TP/EP); the sink experts run in
:class:`InklingSinkExperts` -- replicated across EP ranks (every token
activates every sink) and always bf16 (the checkpoint excludes every
``shared_experts`` from quantization).

NVFP4 routed experts reuse vLLM's fused-MoE methods; excluded (bf16) layers
fall back to the unquantized method. Checkpoint fused stacked tensors are
translated to the standard per-expert loads in
:meth:`InklingMoE.load_expert_weight`.
"""

from __future__ import annotations

import os

import math
import re
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream

from ..configs import InklingModelConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts,
    )
    from vllm.model_executor.layers.quantization import QuantizationConfig

# ---------------------------------------------------------------------------
# Gate / expert selection
# ---------------------------------------------------------------------------

_INKLING_LL_BF16_MAX_TOKENS = 64
_NVFP4_INPUT_SCALE_DENOMINATOR = torch.finfo(torch.float8_e4m3fn).max * 6.0


def _linear_with_fp32_out(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    leading = list(x.shape[:-1])
    flat = x.flatten(0, -2)
    if (
        flat.shape[0] <= _INKLING_LL_BF16_MAX_TOKENS
        and flat.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and flat.is_cuda
        and flat.is_contiguous()
        and weight.is_contiguous()
        and flat.shape[1] % 8 == 0
        and current_platform.has_device_capability(90)
        and ll_bf16.is_available()
    ):
        out = ll_bf16.ll_bf16_gemm(flat, weight)
    else:
        out = torch.mm(flat, weight.T, out_dtype=torch.float32)
    return out.view(*leading, weight.shape[0])


@triton.jit(do_not_specialize=["T", "route_scale"])
def _inkling_gate_select_kernel(
    logits_ptr,  # [T, G] fp32 gate logits (stride_logits_0 may include pad)
    bias_ptr,  # [R] fp32 selection bias (or 0 ptr if HAS_BIAS=False)
    global_scale_ptr,  # [1] fp32 (or unused if HAS_GSCALE=False)
    ids_ptr,  # [T, K + S] int32 out: selected expert ids
    weights_ptr,  # [T, K + S] fp32 out: renormalized weights
    route_scale,
    T,
    G: tl.constexpr,  # total gate experts (routed + shared)
    stride_logits_0,
    R: tl.constexpr,  # routed experts
    K: tl.constexpr,  # top-k routed
    S: tl.constexpr,  # shared (sink) experts
    HAS_BIAS: tl.constexpr,
    HAS_GSCALE: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    offs = tl.arange(0, BLOCK_G)
    mask_r = offs < R
    logits = tl.load(
        logits_ptr + pid * stride_logits_0 + offs,
        mask=offs < G,
        other=float("-inf"),
    ).to(tl.float32)

    # Selection scores: sigmoid(routed logits) (+ bias), non-routed lanes -inf.
    sel = tl.where(mask_r, tl.sigmoid(logits), float("-inf"))
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs, mask=mask_r, other=0.0).to(tl.float32)
        sel = tl.where(mask_r, sel + bias, float("-inf"))

    scale = route_scale
    if HAS_GSCALE:
        scale = scale * tl.load(global_scale_ptr).to(tl.float32)

    # Iterative top-K (K is small); argmax tie-breaks to the lowest index
    # (stable ordering).
    A: tl.constexpr = K + S
    offs_a = tl.arange(0, A)
    top_ids = tl.zeros([A], dtype=tl.int32)
    active = tl.zeros([A], dtype=tl.float32)
    for kk in tl.static_range(K):
        idx = tl.argmax(sel, axis=0).to(tl.int32)
        raw = tl.max(tl.where(offs == idx, logits, float("-inf")), axis=0)
        top_ids = tl.where(offs_a == kk, idx, top_ids)
        active = tl.where(offs_a == kk, raw, active)
        sel = tl.where(offs == idx, float("-inf"), sel)
    if S > 0:
        # Shared sink logits sit at the tail of the gate output; their expert
        # ids continue after the routed range (R + j).
        for jj in tl.static_range(S):
            raw = tl.max(tl.where(offs == R + jj, logits, float("-inf")), axis=0)
            top_ids = tl.where(offs_a == K + jj, tl.full([], R + jj, tl.int32), top_ids)
            active = tl.where(offs_a == K + jj, raw, active)

    # Log-sigmoid renormalization over the K + S active logits.
    abs_l = tl.abs(active)
    min_l = tl.minimum(active, 0.0)
    log_probs = min_l - tldevice.log1p(tldevice.exp(-abs_l))
    max_lp = tl.max(log_probs, axis=0)
    exp_shifted = tldevice.exp(log_probs - max_lp)
    sum_exp = tl.sum(exp_shifted, axis=0)
    weights = exp_shifted / sum_exp * scale

    tl.store(ids_ptr + pid * A + offs_a, top_ids)
    tl.store(weights_ptr + pid * A + offs_a, weights)


def inkling_gate_select(
    logits: torch.Tensor,  # [T, >=G] fp32 (rows may carry GEMM padding)
    n_gate_experts: int,
    n_routed_experts: int,
    topk: int,
    n_shared_experts: int,
    bias: torch.Tensor | None,
    route_scale: float,
    global_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid + bias + top-k + log-sigmoid renorm; returns (weights, ids)."""
    assert logits.dtype == torch.float32
    tokens = logits.shape[0]
    active = topk + n_shared_experts
    topk_ids = torch.empty((tokens, active), dtype=torch.int32, device=logits.device)
    topk_weights = torch.empty(
        (tokens, active), dtype=torch.float32, device=logits.device
    )
    if tokens == 0:
        return topk_weights, topk_ids
    _inkling_gate_select_kernel[(tokens,)](
        logits,
        bias if bias is not None else logits,
        global_scale if global_scale is not None else logits,
        topk_ids,
        topk_weights,
        route_scale,
        tokens,
        n_gate_experts,
        logits.stride(0),
        n_routed_experts,
        topk,
        n_shared_experts,
        HAS_BIAS=bias is not None,
        HAS_GSCALE=global_scale is not None,
        BLOCK_G=triton.next_power_of_2(n_gate_experts),
    )
    return topk_weights, topk_ids


class InklingGate(nn.Module):
    """Sigmoid gate with selection bias, log-sigmoid renorm after top-k, and
    global scale (the served checkpoint's only configuration)."""

    def __init__(
        self,
        d_model: int,
        n_routed_experts: int,
        n_shared_experts: int,
        experts_per_token: int,
        route_scale: float,
        *,
        use_global_scale: bool = False,
        use_gate_bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_total_experts = n_routed_experts + n_shared_experts
        self.topk = experts_per_token
        self.route_scale = route_scale

        padded_experts = self.n_total_experts + (-self.n_total_experts) % 8
        self.weight = Parameter(
            torch.empty(padded_experts, d_model), requires_grad=False
        )
        set_weight_attrs(self.weight, {"weight_loader": self._load_weight})
        if use_global_scale:
            self.global_scale = Parameter(
                torch.empty(1, dtype=torch.float32), requires_grad=False
            )
        else:
            self.global_scale = None
        if use_gate_bias:
            self.bias = Parameter(
                torch.empty(n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.bias = None

    @staticmethod
    def _load_weight(param: Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.zero_()
        param.data[: loaded_weight.shape[0]].copy_(loaded_weight)

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        """fp32 gate logits [T, n_total_experts + pad] (pad columns are junk)."""
        return _linear_with_fp32_out(x, self.weight)

    def select_experts(
        self, gating_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full selection: (weights, ids) of [T, K + S]. The first K entries
        are the routed top-k; the S trailing entries are the sink gammas."""
        return inkling_gate_select(
            gating_output,
            self.n_total_experts,
            self.n_routed_experts,
            self.topk,
            self.n_shared_experts,
            self.bias,
            self.route_scale,
            self.global_scale,
        )


# ---------------------------------------------------------------------------
# MoE layer
# ---------------------------------------------------------------------------


def _inkling_moe_ep_size() -> int:
    """EP size the FusedMoE layer will run with (mirrors
    FusedMoEParallelConfig.make: experts shard over tp * dp * pcp when
    expert parallelism is enabled)."""
    parallel_config = get_current_vllm_config().parallel_config
    if not parallel_config.enable_expert_parallel:
        return 1
    world = (
        get_tensor_model_parallel_world_size()
        * get_dp_group().world_size
        * get_pcp_group().world_size
    )
    return world if world > 1 else 1


# The sink-expert output is a residual contribution, so it carries the model's
# unnormalised dynamic range: measured at 5.1e4 over 42 layers, only 1.3x under
# FP16's 65504. The value is produced by a plain ``h @ w2.T`` whose FP16 store
# is the narrowing step, so a deeper prompt would land +inf there and no
# after-the-fact cast could recover it.
#
# ``w2`` is linear in ``h``, so scaling ``h`` by a power of two scales the
# output exactly (exponent shift only, no mantissa loss) and the FP16 store
# then has 64x headroom. The unscale happens here, in FP32, so callers receive
# a true-magnitude tensor and there is no cross-module scaling contract to keep
# in sync -- the same reasoning as the conv-state cache scale in model.py,
# which does need such a contract because the cache itself holds scaled values.
# Must match VLLM_SM70_MOE_W13_UP_DIV, which the shared SM70 AWQ MoE path folds
# into the UP half of the W13 dequant scales at load time. Read from the same
# env so the two cannot drift apart -- if they do, outputs are silently scaled
# wrong rather than failing loudly.
_W13_UP_DIV = float(os.getenv("VLLM_SM70_MOE_W13_UP_DIV", "1") or "1")

# Both stay constants rather than being derived from the data the way the sconv
# wire formats in model.py now are, but for two different reasons.
#
# _SINK_OUTPUT_SCALE guards the W2 GEMM's FP16 output store while the only
# tensor available before the GEMM is its input, and an input maximum does not
# bound an output maximum without the weight -- the "operator output" case in
# vllm/model_executor/layers/fp16_range.py, where the sound fix is a load-time
# max_row_sum(|W|) bound rather than a measurement.
#
# _ROUTED_OUTPUT_SCALE is not guarding a store at all. It is folded into the
# router weights, which the fused op applies *after* the expert GEMMs, so it
# never covered the per-expert FP16 stages (see forward_partials below) -- which
# is why widening it did nothing for the tool-prompt NaN. _W13_UP_DIV is what
# actually protects those. Making this one dynamic would buy nothing.
_SINK_OUTPUT_SCALE = 1.0 / 64.0
_ROUTED_OUTPUT_SCALE = 1.0 / 64.0


class InklingSinkExperts(nn.Module):
    """Shared "sink" experts with per-token gammas, in bf16.

    Replicated across EP ranks (every token activates every sink, so
    EP-sharding them would hotspot the owning rank) and TP-sharded on the
    intermediate dim so the output remains a TP-partial sum like the routed
    output. The sinks are always bf16 (the checkpoint excludes every
    ``shared_experts`` from quantization): the experts concatenate into two
    plain dense GEMMs with the fused sink epilogue between them.
    """

    def __init__(
        self, n_experts: int, d_model: int, d_mlp: int, *, prefix: str = ""
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        intermediate_pp = d_mlp // tp_size
        self.w13_weight = Parameter(
            torch.empty(n_experts, 2 * intermediate_pp, d_model),
            requires_grad=False,
        )
        self.w2_weight = Parameter(
            torch.empty(d_model, n_experts * intermediate_pp),
            requires_grad=False,
        )
        self._unit: torch.Tensor | None = None

    def load_weight(self, key: str, weight: torch.Tensor) -> list[str]:
        """Load one checkpoint sink tensor (stacked over the S experts)."""
        if key == "w13_weight":
            if weight.shape != self.w13_weight.shape:
                shard = self.w13_weight.shape[1]
                weight = weight.narrow(1, self.tp_rank * shard, shard)
            self.w13_weight.data.copy_(weight)
            return [key]

        assert key == "w2_weight"
        shard = self.w2_weight.shape[1] // self.n_experts
        shard_start = 0 if weight.shape[2] == shard else self.tp_rank * shard
        for expert_idx, expert_weight in enumerate(weight):
            local_weight = expert_weight.narrow(1, shard_start, shard)
            start = expert_idx * shard
            self.w2_weight.data[:, start : start + shard].copy_(local_weight)
        return [key]

    def forward(self, x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        """``sum_e gammas[:, e] * MLP_e(x)`` (TP-partial along d_mlp)."""
        from .ops import sink_silu_mul_epilogue

        # One GEMM over the experts' stacked w13 (a view), fused epilogue,
        # then one GEMM whose K-reduction over the K-concatenated w2 performs
        # the expert sum.
        if self._unit is None or self._unit.device != x.device:
            self._unit = torch.ones(
                self.n_experts, dtype=torch.float32, device=x.device
            )
        raw = x @ self.w13_weight.view(-1, x.shape[-1]).T  # (T, S*2F)
        h = sink_silu_mul_epilogue(
            raw, self._unit, gammas, self._unit, self.n_experts, x.dtype
        )
        # Scale into the FP16 store, unscale out of it in FP32.
        h = h.mul_(_SINK_OUTPUT_SCALE)
        out = h @ self.w2_weight.T  # (T, D)
        return out.float().mul_(1.0 / _SINK_OUTPUT_SCALE)


class InklingSinkExpertsLinear(nn.Module):
    """LoRA-capable implementation of the Inkling sink experts."""

    def __init__(
        self,
        n_experts: int,
        d_model: int,
        d_mlp: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        from vllm.model_executor.layers.linear import (
            MergedColumnParallelLinear,
            RowParallelLinear,
        )

        self.n_experts = n_experts
        self.d_mlp = d_mlp
        total = n_experts * d_mlp
        self.w13 = MergedColumnParallelLinear(
            input_size=d_model,
            output_sizes=[total, total],
            bias=False,
            prefix=f"{prefix}.w13",
        )
        self.w2 = RowParallelLinear(
            input_size=total,
            output_size=d_model,
            bias=False,
            reduce_results=False,
            prefix=f"{prefix}.w2",
        )
        self._w2_input_pp = self.w2.input_size_per_partition
        self._col_expert: torch.Tensor | None = None

    def _gamma_expand(self, gammas: torch.Tensor) -> torch.Tensor:
        if self._col_expert is None or self._col_expert.device != gammas.device:
            local = self._w2_input_pp
            start = get_tensor_model_parallel_rank() * local
            cols = torch.arange(start, start + local, device=gammas.device)
            self._col_expert = (cols // self.d_mlp).long()
        return gammas[:, self._col_expert]

    def load_weight(self, key: str, weight: torch.Tensor) -> list[str]:
        if key == "w13_weight":
            d_model = weight.shape[-1]
            gate = weight[:, 0::2, :].reshape(-1, d_model).contiguous()
            up = weight[:, 1::2, :].reshape(-1, d_model).contiguous()
            self.w13.weight_loader(self.w13.weight, gate, 0)
            self.w13.weight_loader(self.w13.weight, up, 1)
            return ["w13.weight"]
        w = weight.permute(1, 0, 2).reshape(weight.shape[1], -1).contiguous()
        self.w2.weight_loader(self.w2.weight, w)
        return ["w2.weight"]

    def forward(self, x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w13(x)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = torch.nn.functional.silu(gate) * up
        hidden_states = (hidden_states * self._gamma_expand(gammas)).to(x.dtype)
        hidden_states = hidden_states.mul_(_SINK_OUTPUT_SCALE)
        output, _ = self.w2(hidden_states)
        return output.float().mul_(1.0 / _SINK_OUTPUT_SCALE)


# `experts.<expert_id>.<gate|up|down>_proj.<suffix>` -- the per-expert module
# layout llm-compressor emits, as opposed to the reference release's tensors
# stacked over the expert dimension.
_PER_EXPERT_RE = re.compile(
    r"^experts\.(?P<eid>\d+)\.(?P<proj>gate|up|down)_proj\.(?P<suffix>.+)$"
)

# vLLM's fused expert slab is [w1(gate); w3(up)] with w2 the down projection.
_CKPT_PROJ_TO_SHARD = {"gate": "w1", "up": "w3", "down": "w2"}


class InklingMoE(nn.Module):
    def __init__(
        self,
        config: InklingModelConfig,
        *,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        # Overfit to the served checkpoint: sigmoid gate renormalized after
        # top-k, shared sink experts, interleaved gate/up checkpoint rows.
        assert config.gate_activation == "sigmoid" and config.norm_after_topk
        assert config.n_shared_experts > 0 and config.shared_expert_sink
        assert config.inference_moe_w13_interleaved
        n_routed = config.n_routed_experts
        n_shared = config.n_shared_experts
        self.n_routed_experts = n_routed
        self.gate = InklingGate(
            d_model=config.hidden_size,
            n_routed_experts=n_routed,
            n_shared_experts=n_shared,
            experts_per_token=config.num_experts_per_tok,
            route_scale=config.route_scale,
            use_global_scale=config.use_global_scale,
            use_gate_bias=config.use_gate_bias,
        )

        # TRTLLM MoE kernels assume equal, contiguous per-rank expert slabs
        # (local_expert_offset = ep_rank * local_num_experts), so pad the
        # expert count to a multiple of the EP size. A no-op for the usual
        # power-of-two EP sizes (n_routed is a power of two).
        num_experts = n_routed + (-n_routed) % _inkling_moe_ep_size()

        self.experts = FusedMoEFactory(
            num_experts=num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            renormalize=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            custom_routing_function=self._select_routed,
            router_logits_dtype=torch.float32,
            activation="silu",
        )
        # The decoder layer reduce-scatters the MoE delta into the sconv
        # stream itself (RS -> shard sconv -> AG); the runner must return the
        # per-rank partial sum instead of all-reducing.
        self.experts.moe_config.skip_final_all_reduce = True

        self._routed_sel: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

        sink_experts_cls = (
            InklingSinkExpertsLinear
            if get_current_vllm_config().lora_config is not None
            else InklingSinkExperts
        )
        self.sink_experts = sink_experts_cls(
            n_experts=n_shared,
            d_model=config.hidden_size,
            d_mlp=config.intermediate_size,
            prefix=f"{prefix}.shared_experts",
        )

        # Sink chain overlaps the routed MoE call on the aux stream for
        # decode-sized batches (same pattern as the runner's SharedExperts
        # multi-stream overlap). The routed GEMM runs on the default stream and
        # the sink chain on the aux stream, joined via these two events by
        # ``maybe_execute_in_parallel``.
        self._sink_stream: torch.cuda.Stream | None = aux_stream()
        self._sink_events = (torch.cuda.Event(), torch.cuda.Event())

    def _select_routed(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FusedMoE ``custom_routing_function``: the routed top-k slice of the
        full (routed + sink) selection.

        forward() stashes its selection (keyed by logits identity) so the
        gate select runs once per layer; the fallback covers paths where the
        runner re-derives the logits (e.g. naive DP dispatch).
        """
        del hidden_states, renormalize
        assert topk == self.gate.topk
        cached = self._routed_sel
        self._routed_sel = None
        if cached is not None and cached[0] is gating_output:
            return cached[1], cached[2]
        weights, ids = self.gate.select_experts(gating_output)
        # Scaled here, unscaled in forward_partials -- see _ROUTED_OUTPUT_SCALE.
        return (
            weights[:, :topk].contiguous().mul_(_ROUTED_OUTPUT_SCALE),
            ids[:, :topk].contiguous(),
        )

    def forward_partials(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.gate.compute_logits(x)
        num_tokens = x.shape[0]
        # One gate select per layer: the routed slice is stashed for the
        # routing function inside the FusedMoE op; the sink gammas are the
        # trailing columns.
        k = self.gate.topk
        weights, ids = self.gate.select_experts(router_logits)
        self._routed_sel = (
            router_logits,
            weights[:, :k].contiguous().mul_(_ROUTED_OUTPUT_SCALE),
            ids[:, :k].contiguous(),
        )
        gammas = weights[:, k:]

        out, sink_out = maybe_execute_in_parallel(
            lambda: self.experts(hidden_states=x, router_logits=router_logits),
            lambda: self.sink_experts(x, gammas),
            self._sink_events[0],
            self._sink_events[1],
            self._sink_stream
            if num_tokens <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            else None,
        )
        self._routed_sel = None

        # Both partials leave here at true magnitude in FP32. The routed store
        # overflowed FP16 on a chat-template prompt (rank 3, +inf at 1.5x
        # headroom) and the sink store sat at 1.3x, so neither is safe to keep
        # in FP16 -- these are residual contributions and carry the model's
        # unnormalised range, which on a BF16-native model FP16 does not have.
        # Two scales come off here, not one. _ROUTED_OUTPUT_SCALE was folded
        # into the router weights, which the fused op applies AFTER the expert
        # GEMMs -- so it never protected the FP16 per-expert stage stores
        # (the up projection, the SwiGLU product, the W2 output). That is why
        # widening it did nothing for the tool-prompt NaN.
        #
        # _W13_UP_DIV is folded into the UP half of the W13 dequant scales, so
        # every one of those stores is smaller by that factor while silu(gate)
        # is untouched. Undone here, in FP32, where the true magnitude fits.
        out = out.float().mul_(_W13_UP_DIV / _ROUTED_OUTPUT_SCALE)
        return out, sink_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, sink_out = self.forward_partials(x)
        # sink_out is FP32 (see _SINK_OUTPUT_SCALE); an in-place add of FP32
        # into FP16 raises, and this sum is a residual contribution, so widen
        # rather than narrow. Inkling itself takes forward_partials.
        return sink_out.add_(out)

    # -- weight loading ----------------------------------------------------

    def _local_expert_slots(self) -> dict[int, int]:
        """Global expert id -> local slot for this rank's expert partition."""
        manager = self.experts.routed_experts.expert_map_manager
        if manager.expert_map is None:
            return {g: g for g in range(manager.global_num_experts)}
        emap = manager.expert_map.tolist()
        return {g: slot for g, slot in enumerate(emap) if slot >= 0}

    def load_expert_weight(self, name: str, weight: torch.Tensor) -> list[str]:
        """Load one checkpoint expert tensor.

        ``name`` is relative to the mlp module: ``experts.<t>`` (routed
        stack) or ``shared_experts.shared_<t>`` (sink experts). Returns the
        loaded param names (relative to this module).
        """
        if name.startswith("shared_experts."):
            key = name.split(".", 1)[1].replace("shared_", "", 1)
            return [
                f"sink_experts.{p}" for p in self.sink_experts.load_weight(key, weight)
            ]

        # Two routed-expert checkpoint layouts exist in the wild. The code
        # below this branch expects the reference release's *stacked* tensors
        # (`experts.w13_weight_packed` indexed by expert id). llm-compressor
        # builds -- including the served cyankiwi W4A16 one -- instead emit one
        # module per expert (`experts.<id>.gate_proj.weight_packed`). Route
        # those through vLLM's standard FusedMoE expert loader.
        per_expert = _PER_EXPERT_RE.match(name)
        if per_expert is not None:
            return self._load_per_expert_weight(
                int(per_expert.group("eid")),
                per_expert.group("proj"),
                per_expert.group("suffix"),
                weight,
            )

        experts: RoutedExperts = self.experts.routed_experts
        key = name.split(".", 1)[1]

        # original_shape is unused by the vLLM serving layout.
        if key.endswith(".original_shape"):
            return []
        if key.endswith(".input_amax"):
            projection = "w13" if key.startswith("w13") else "w2"
            amax = float(weight.max())
            assert math.isfinite(amax) and amax > 0, (
                f"bad {projection} input_amax: {amax}"
            )
            input_scale = getattr(experts, f"{projection}_input_scale")
            input_scale.data.fill_(amax / _NVFP4_INPUT_SCALE_DENOMINATOR)
            return [f"experts.routed_experts.{projection}_input_scale"]

        param = getattr(experts, key)
        slots = self._local_expert_slots()
        gids = sorted(slots)
        lids = [slots[g] for g in gids]
        tp_rank = experts.moe_config.moe_parallel_config.tp_rank

        if key.endswith(("_scale_2", "_global_scale")):
            # Per-expert scalars, vectorized over the local experts. The
            # fused w13 params carry one slot per gate/up half. A single
            # checkpoint value is shared by both halves.
            vals = weight[gids].float().reshape(len(gids), -1)
            target_width = math.prod(param.shape[1:])
            if vals.shape[1] == 1:
                vals = vals.expand(-1, target_width)
            elif vals.shape[1] != target_width:
                raise ValueError(
                    f"cannot load {tuple(weight.shape)} into {tuple(param.shape)}"
                )
            param.data[lids] = vals.reshape(len(gids), *param.shape[1:]).to(
                param.device
            )
        elif key == "w2_weight_scale" and weight.shape[-1] == 1:
            # Per-output-channel scales are replicated across TP ranks.
            param.data[lids] = weight[gids].to(device=param.device, dtype=param.dtype)
        elif key.startswith("w13"):
            # Checkpoint w13 rows are interleaved [g0, u0, g1, u1, ...]; the
            # fused param layout is [w1(gate); w3(up)]. The TP-local rows form
            # one contiguous slab of the interleaved tensor, so upload just
            # that slab (a single bounded synchronous H2D; pre-uploading whole
            # untrimmed tensors pins the mmap pages of the entire checkpoint
            # and OOMs the host) and de-interleave on device.
            half = param.shape[1] // 2
            for gid, lid in slots.items():
                slab = weight[gid].narrow(0, tp_rank * 2 * half, 2 * half)
                slab = slab.to(param.device)
                param.data[lid, :half].copy_(slab[0::2])
                param.data[lid, half:].copy_(slab[1::2])
        else:
            # w2: shard the packed intermediate (last) dim.
            shard = param.shape[2]
            for gid, lid in slots.items():
                param.data[lid].copy_(weight[gid].narrow(1, tp_rank * shard, shard))
        return [f"experts.routed_experts.{key}"]

    def _load_per_expert_weight(
        self, expert_id: int, proj: str, suffix: str, weight: torch.Tensor
    ) -> list[str]:
        """Load a single `experts.<id>.<proj>_proj.<suffix>` checkpoint tensor.

        Delegates to `FusedMoE.weight_loader`, which is the path every other
        per-expert compressed-tensors MoE in this fork uses (Laguna included).
        It already knows how to place a gate/up half into the fused w13 slab
        and how to shard each projection for the current TP rank, including
        the packed/scale/zero-point companions -- none of which is worth
        re-deriving here.

        Returns the loaded param name so the caller can mark it seen, or an
        empty list when the expert does not live on this rank (the loader
        reports that via `return_success`).
        """
        routed: RoutedExperts = self.experts.routed_experts
        shard_id = _CKPT_PROJ_TO_SHARD[proj]
        stem = "w2_" if shard_id == "w2" else "w13_"
        attr = f"{stem}{suffix}"

        param = getattr(routed, attr, None)
        if param is None:
            # Some companions are genuinely optional: weight_shape and the
            # actorder indices are only materialized by certain backends.
            #
            # Zero points are not optional on an asymmetric checkpoint, and
            # skipping them silently is how 30720 of them were quietly dropped
            # while the load still "succeeded" -- the failure only surfaced
            # later, in the prepare step. Fail loudly instead.
            weight_quant = getattr(
                getattr(routed, "quant_method", None), "weight_quant", None
            )
            asymmetric = weight_quant is not None and not weight_quant.symmetric
            if suffix == "weight_zero_point" and asymmetric:
                raise RuntimeError(
                    f"asymmetric checkpoint supplies {attr} but the quant "
                    f"method did not allocate it; the MoE would silently "
                    f"dequantize with uninitialized zero points"
                )
            return []

        success = routed.weight_loader(
            param,
            weight,
            f"experts.{attr}",
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=True,
        )
        return [f"experts.routed_experts.{attr}"] if success else []

    def finalize_load(self) -> list[str]:
        """Post-load fixups for zeroed padding experts."""
        experts = self.experts.routed_experts
        out: list[str] = []
        # Zero the EP-alignment padding experts (if any) so their
        # (never-routed) slots hold defined values.
        slots = self._local_expert_slots()
        for gid in range(self.n_routed_experts, experts.global_num_experts):
            lid = slots.get(gid)
            if lid is None:
                continue
            for pname in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_weight_scale_2",
                "w2_weight_scale_2",
            ):
                p = getattr(experts, pname, None)
                if p is not None:
                    p.data[lid].zero_()
        return out
