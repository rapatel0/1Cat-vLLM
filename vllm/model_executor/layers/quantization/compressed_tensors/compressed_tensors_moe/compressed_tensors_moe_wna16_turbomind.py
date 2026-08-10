# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 TurboMind route for groupwise compressed-tensors W4A16 MoE.

The Laguna checkpoint stores packed uint4 values in compressed-tensors layout.
The SM70 TurboMind grouped-MoE runner accepts the same numerical format once
each expert has been converted by ``uint4_sm70_prepare``.  This keeps the
checkpoint loader and routing behaviour shared with the established Marlin
implementation while using the TurboMind execution path after loading.

Both symmetric and asymmetric checkpoints are supported. TurboMind
dequantizes as ``(q - zero) * scale`` over raw uint4 values, so a symmetric
checkpoint is just the special case ``zero == 8``; asymmetric checkpoints
(Inkling-Small) carry real per-group zero points packed alongside the weights.
"""

import os

import torch
from compressed_tensors.quantization import QuantizationArgs
from torch.nn import Parameter

from vllm import _sm70_ops as sm70_ops
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    AWQSM70MoEMethod,
    _batched_gemm_enabled_for_layer,
    _get_layer_id,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_marlin import (  # noqa: E501
    CompressedTensorsWNA16MarlinMoEMethod,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


def _active_group_experiment_enabled() -> bool:
    """Whether to use the experimental compact active-expert GEMM route.

    This route must be enabled deliberately because it changes the MoE dispatch
    ordering. Keep the quality-safe per-expert path as the default.
    """
    enabled = (
        os.getenv("VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND_ACTIVE_GROUPS", "0") == "1"
    )
    if enabled:
        missing_ops = [
            name
            for name in (
                "awq_moe_build_active_group_indices_out",
                "awq_moe_gemm_sm70_active_groups_out",
            )
            if not hasattr(torch.ops._C, name)
        ]
        if missing_ops:
            raise RuntimeError(
                "SM70 active-group experiment requested but the loaded extension "
                f"does not provide: {', '.join(missing_ops)}"
            )
    return enabled


class CompressedTensorsWNA16TurboMindMoEMethod(CompressedTensorsWNA16MarlinMoEMethod):
    """TurboMind MoE executor for groupwise compressed W4 weights.

    Weight construction intentionally uses the Marlin compressed-tensors
    loader layout.  It is a faithful transpose of the checkpoint layout, and
    keeps compressed-tensors zero points in their own convention rather than
    reinterpreting them as AWQ zero-points (which carry a +1 bias).
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ) -> None:
        if (
            weight_quant.num_bits != 4
            or weight_quant.group_size not in sm70_tm.COMPRESSED_UINT4_GROUP_SIZES
            or weight_quant.actorder
            or moe.has_bias
        ):
            raise ValueError(
                "SM70 TurboMind compressed-tensors MoE requires groupwise W4 "
                "weights without actorder or bias."
            )
        super().__init__(weight_quant, input_quant, moe, layer_name)
        # The inherited weight loader has a Marlin and a FlashInfer layout.
        # TurboMind consumes the former before the Marlin repack step.
        self.kernel_backend = "Marlin"
        self.use_batched_gemm = envs.VLLM_SM70_AWQ_MOE_BATCHED_GEMM

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create the standard compressed-tensors transposed loader layout.

        This is deliberately independent of the Marlin class-name dispatch in
        ``FusedMoE``. Unlike the Marlin loader it does not need the full
        unsharded intermediate size because this route rejects actorder.
        """
        extra_weight_attrs.update(
            {"is_transposed": True, "quant_method": self.strategy}
        )
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        w13_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.packed_factor,
                w13_num_shards * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.packed_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.group_size
        num_groups_w2 = intermediate_size_per_partition // self.group_size
        w13_scale = Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                w13_num_shards * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)
        w2_scale = Parameter(
            torch.ones(
                num_experts,
                num_groups_w2,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": False})

        # Asymmetric checkpoints carry a packed zero point per group. This
        # route used to be symmetric-only, so these were never created and the
        # prepare step died on getattr(layer, "w13_weight_zero_point").
        #
        # Same transposed layout as the scales, with the output axis packed
        # four values to an int32 word: (num_groups, N / packed_factor)
        # unpacks to the (num_groups, N) scale grid.
        if not self.weight_quant.symmetric:
            w13_zp = Parameter(
                torch.empty(
                    num_experts,
                    num_groups_w13,
                    w13_num_shards
                    * intermediate_size_per_partition
                    // self.packed_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_zero_point", w13_zp)
            set_weight_attrs(w13_zp, extra_weight_attrs)

            w2_zp = Parameter(
                torch.empty(
                    num_experts,
                    num_groups_w2,
                    hidden_size // self.packed_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_zero_point", w2_zp)
            set_weight_attrs(w2_zp, extra_weight_attrs)
            set_weight_attrs(w2_zp, {"load_full_w2": False})

        for name, size in (
            ("w2_weight_shape", intermediate_size_per_partition),
            ("w13_weight_shape", hidden_size),
            ("w13_weight_g_idx", hidden_size),
            ("w2_weight_g_idx", intermediate_size_per_partition),
            ("w13_g_idx_sort_indices", hidden_size),
            ("w2_g_idx_sort_indices", intermediate_size_per_partition),
        ):
            shape = (
                (num_experts, 2)
                if name.endswith("weight_shape")
                else (
                    num_experts,
                    size,
                )
            )
            value = Parameter(
                torch.empty(*shape, dtype=torch.int32), requires_grad=False
            )
            layer.register_parameter(name, value)
            set_weight_attrs(value, extra_weight_attrs)
        layer.a13_scale = None
        layer.a2_scale = None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if not hasattr(torch.ops._C, "uint4_sm70_prepare"):
            raise RuntimeError(
                "SM70 TurboMind compressed-tensors MoE requires a build with "
                "CUDA arch 7.0 and the uint4_sm70_prepare extension."
            )

        hidden_logical_size = int(layer.w13_weight_packed.shape[1]) * self.packed_factor
        intermediate_logical_size = (
            int(layer.w2_weight_packed.shape[1]) * self.packed_factor
        )
        w13_logical_out = int(layer.w13_weight_packed.shape[2])
        if w13_logical_out != 2 * intermediate_logical_size:
            raise ValueError(
                "SM70 TurboMind compressed-tensors MoE expects a gated W13 "
                "projection with twice the W2 input size."
            )
        if (
            hidden_logical_size % self.group_size
            or intermediate_logical_size % self.group_size
        ):
            raise ValueError(
                "SM70 TurboMind compressed-tensors MoE requires complete "
                "groupwise W13 and W2 partitions."
            )

        num_experts = int(layer.w13_weight_packed.shape[0])
        batched_gemm = _batched_gemm_enabled_for_layer(layer, self.use_batched_gemm)
        build_legacy_w13 = (
            batched_gemm
            and envs.VLLM_SM70_AWQ_MOE_LEGACY_SINGLE_TOKEN_COMPACT
            and hasattr(torch.ops._C, "awq_moe_single_token_sm70_out")
        )
        # Match the native AWQ route: one interleaved W13 representation serves
        # both the grouped path and the monolithic single-token compact path.
        # The latter avoids the general 256-expert routing machinery at M=1.
        w13_interleaved = build_legacy_w13
        w13_tm_weights: list[torch.Tensor] = []
        w13_tm_scales: list[torch.Tensor] = []
        w13_meta: list[torch.Tensor] = []
        w2_tm_weights: list[torch.Tensor] = []
        w2_tm_scales: list[torch.Tensor] = []
        w2_meta: list[torch.Tensor] = []

        # TurboMind dequantizes as (q - zero) * scale over raw uint4 values, so
        # a symmetric int4 checkpoint has zero == 8 (the midpoint of [0, 15]).
        # Asymmetric checkpoints carry real per-group zero points, packed the
        # same way as the weights.
        symmetric = bool(self.weight_quant.symmetric)

        def _zeros_for(expert_id: int, packed_zp_name: str, scales: torch.Tensor):
            if symmetric:
                return sm70_tm.symmetric_int4_zeros_like(scales)
            zeros = sm70_tm.unpack_compressed_moe_zeros(
                getattr(layer, packed_zp_name)[expert_id]
            )
            assert zeros.shape == scales.shape, (
                f"{packed_zp_name} unpacked to {tuple(zeros.shape)}, expected "
                f"{tuple(scales.shape)} to match the scale grid"
            )
            return zeros.contiguous()

        for expert_id in range(num_experts):
            w13_scales = layer.w13_weight_scale[expert_id].to(torch.float16)
            r13 = sm70_ops.uint4_sm70_prepare(
                sm70_tm.unpack_gptq_weight(layer.w13_weight_packed[expert_id]),
                w13_scales.contiguous(),
                _zeros_for(expert_id, "w13_weight_zero_point", w13_scales),
                self.group_size,
                w13_interleaved,
            )
            w13_tm_weights.append(r13[0])
            w13_tm_scales.append(r13[1])
            w13_meta.append(r13[2])

            w2_scales = layer.w2_weight_scale[expert_id].to(torch.float16)
            r2 = sm70_ops.uint4_sm70_prepare(
                sm70_tm.unpack_gptq_weight(layer.w2_weight_packed[expert_id]),
                w2_scales.contiguous(),
                _zeros_for(expert_id, "w2_weight_zero_point", w2_scales),
                self.group_size,
                False,
            )
            w2_tm_weights.append(r2[0])
            w2_tm_scales.append(r2[1])
            w2_meta.append(r2[2])

        layer.w13_tm_weight = Parameter(
            torch.stack(w13_tm_weights), requires_grad=False
        )
        layer.w13_tm_scales = Parameter(torch.stack(w13_tm_scales), requires_grad=False)
        layer.w2_tm_weight = Parameter(torch.stack(w2_tm_weights), requires_grad=False)
        layer.w2_tm_scales = Parameter(torch.stack(w2_tm_scales), requires_grad=False)

        w13_k_ld, w13_q_ld = int(w13_meta[0][0]), int(w13_meta[0][1])
        w2_k_ld, w2_q_ld = int(w2_meta[0][0]), int(w2_meta[0][1])
        w13_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            layer.w13_tm_weight,
            layer.w13_tm_scales,
            w13_k_ld,
            w13_q_ld,
            num_experts,
        )
        w2_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            layer.w2_tm_weight,
            layer.w2_tm_scales,
            w2_k_ld,
            w2_q_ld,
            num_experts,
        )
        layer.w13_strided_ptrs_w = Parameter(w13_ptrs[0], requires_grad=False)
        layer.w13_strided_ptrs_s = Parameter(w13_ptrs[1], requires_grad=False)
        layer.w2_strided_ptrs_w = Parameter(w2_ptrs[0], requires_grad=False)
        layer.w2_strided_ptrs_s = Parameter(w2_ptrs[1], requires_grad=False)
        ptr_row_bytes = int(layer.w13_strided_ptrs_w.numel() // num_experts)
        layer.sm70_ptr_row_bytes = ptr_row_bytes
        layer.w13_strided_ptrs_w_rows = layer.w13_strided_ptrs_w.view(
            num_experts, ptr_row_bytes
        )
        layer.w13_strided_ptrs_s_rows = layer.w13_strided_ptrs_s.view(
            num_experts, ptr_row_bytes
        )
        if build_legacy_w13:
            layer.w13_legacy_strided_ptrs_w_rows = layer.w13_strided_ptrs_w_rows
            layer.w13_legacy_strided_ptrs_s_rows = layer.w13_strided_ptrs_s_rows
        layer.w2_strided_ptrs_w_rows = layer.w2_strided_ptrs_w.view(
            num_experts, ptr_row_bytes
        )
        layer.w2_strided_ptrs_s_rows = layer.w2_strided_ptrs_s.view(
            num_experts, ptr_row_bytes
        )

        layer.sm70_hidden_logical_size = hidden_logical_size
        layer.sm70_hidden_aligned_size = hidden_logical_size
        layer.sm70_intermediate_logical_size = intermediate_logical_size
        layer.sm70_intermediate_aligned_size = intermediate_logical_size
        layer.sm70_num_experts = num_experts
        layer.sm70_w13_k_dim = int(layer.w13_tm_weight.shape[1])
        layer.sm70_w13_n_dim = int(layer.w13_tm_weight.shape[2]) * self.packed_factor
        layer.sm70_w2_k_dim = int(layer.w2_tm_weight.shape[1])
        layer.sm70_w2_n_dim = int(layer.w2_tm_weight.shape[2]) * self.packed_factor
        layer.sm70_w13_k_ld = w13_k_ld
        layer.sm70_w13_q_ld = w13_q_ld
        layer.sm70_w2_k_ld = w2_k_ld
        layer.sm70_w2_q_ld = w2_q_ld
        layer.sm70_intermediate_size = layer.sm70_w2_k_dim
        layer.sm70_awq_moe_batched_gemm = batched_gemm
        # Default to the quality-safe per-expert route. The compact route has
        # identical packed-U4 register dequantization and Volta FP16 HMMA, but
        # changes the dispatch ordering and therefore needs an explicit
        # model-level quality gate before it can become a default.
        active_groups = _active_group_experiment_enabled()
        layer.sm70_awq_moe_batched_w13_per_expert_dispatch = not active_groups
        layer.sm70_awq_moe_batched_w2_per_expert_dispatch = not active_groups
        layer.sm70_awq_moe_batched_active_groups = active_groups
        layer.sm70_awq_moe_layer_id = _get_layer_id(layer)
        layer.sm70_awq_moe_w13_interleaved = w13_interleaved
        layer.sm70_awq_moe_legacy_single_token_compact = build_legacy_w13

        AWQSM70MoEMethod._allocate_buffers(self, layer)
        for name in (
            "w13_weight_packed",
            "w13_weight_scale",
            "w2_weight_packed",
            "w2_weight_scale",
            "w13_weight_shape",
            "w2_weight_shape",
            "w13_weight_g_idx",
            "w2_weight_g_idx",
            "w13_g_idx_sort_indices",
            "w2_g_idx_sort_indices",
        ):
            delattr(layer, name)
        # Zero points are baked into tm_weight/tm_scales by uint4_sm70_prepare
        # and never read at runtime, so drop them too. Leaving them resident
        # costs ~0.5 GiB per GPU across 40 MoE layers, straight out of the KV
        # cache budget.
        for name in ("w13_weight_zero_point", "w2_weight_zero_point"):
            if hasattr(layer, name):
                delattr(layer, name)
        logger.info_once(
            "SM70 TurboMind compressed-tensors W4 MoE %s path enabled "
            "(%d experts, group_size=%d).",
            "batched" if batched_gemm else "per-expert dense",
            num_experts,
            self.group_size,
        )
        if active_groups:
            logger.warning_once(
                "SM70 TurboMind compressed-tensors active-group route enabled; "
                "this is an output-quality experiment."
            )

    def get_fused_moe_quant_config(self, layer: RoutedExperts):
        return AWQSM70MoEMethod.get_fused_moe_quant_config(self, layer)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        return AWQSM70MoEMethod.apply(
            self,
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return AWQSM70MoEMethod.apply_monolithic(
            self, layer, x, router_logits, input_ids
        )

    _allocate_buffers = AWQSM70MoEMethod._allocate_buffers
    _get_buffers = AWQSM70MoEMethod._get_buffers
    _apply_legacy_single_token_compact = (
        AWQSM70MoEMethod._apply_legacy_single_token_compact
    )
