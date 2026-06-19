# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from compressed_tensors.quantization import ActivationOrdering

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.marlin import (
    MarlinLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
    marlin_repeat_scales_on_all_ranks,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_TYPES_MAP = {4: scalar_types.uint4b8, 8: scalar_types.uint8b128}
WNA16_ZP_SUPPORTED_TYPES_MAP = {4: scalar_types.uint4, 8: scalar_types.uint8}
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())

# Group sizes the TurboMind SM70 (V100) AWQ GEMM kernel accepts.
_SM70_AWQ_GROUP_SIZES = (32, 64, 128)


def _is_sm70_capable() -> bool:
    """True iff the current CUDA device is SM70 (V100) and the TurboMind AWQ
    custom op is built in. The dense compressed-tensors W4A16 path has no native
    SM70 kernel (Marlin is SM75+), so on V100 we convert the checkpoint to the
    TurboMind AWQ format at load time -- the same trick the CT MoE path uses."""
    if not torch.cuda.is_available():
        return False
    try:
        if torch.cuda.get_device_capability() != (7, 0):
            return False
    except Exception:
        return False
    return hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "awq_sm70_prepare")


def _ct_dense_to_awq_qweight(ct_packed: torch.Tensor) -> torch.Tensor:
    """Convert a dense compressed-tensors qweight to AWQ packing.

    CT stores a dense Linear weight as ``[N, K/8]`` (logical ``[N, K]`` =
    [out, in]), packing 8 sequential 4-bit values along the input dim. AWQ wants
    ``[K, N/8]`` (logical ``[K, N]`` = [in, out]), packing 8 interleaved values
    along the output dim -- i.e. the *transpose* of the CT layout. (The CT MoE
    helper packs the same axis it unpacks and so needs no transpose; the dense
    case does.) The interleave order mirrors the MoE path: AWQ unpacking gathers
    with ``[0,4,1,5,2,6,3,7]``, so packing uses the inverse ``[0,2,4,6,1,3,5,7]``.
    """
    N, K_div_8 = ct_packed.shape
    K = K_div_8 * 8
    # Unpack CT along the input dim: int32 [n, m] holds K-values 8m..8m+7.
    unpacked = torch.zeros(N, K, dtype=torch.uint8, device=ct_packed.device)
    tmp = ct_packed.clone()
    for i in range(8):
        unpacked[:, i::8] = (tmp & 0xF).to(torch.uint8)
        tmp = tmp >> 4
    # Transpose to logical [K, N], then repack 8 interleaved N-values per int32.
    unpacked = unpacked.t().contiguous()  # [K, N]
    awq_pack_order = [0, 2, 4, 6, 1, 3, 5, 7]
    grouped = unpacked.view(K, -1, 8)  # [K, N/8, 8]
    result = grouped[:, :, awq_pack_order[7]].to(torch.int32)
    for i in range(6, -1, -1):
        result = (result << 4) | grouped[:, :, awq_pack_order[i]].to(torch.int32)
    return result  # [K, N/8]


class CompressedTensorsWNA16(CompressedTensorsScheme):
    _kernel_backends_being_used: set[str] = set()

    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: int | None = None,
        symmetric: bool | None = True,
        actorder: ActivationOrdering | None = None,
        layer_name: str | None = None,
    ):
        self.pack_factor = 32 // num_bits
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP
        self.layer_name = layer_name

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError(
                "Marlin kernels require group quantization or "
                "channelwise quantization, but found no group "
                "size and strategy is not channelwise."
            )

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {WNA16_SUPPORTED_TYPES_MAP.keys()}"
            )

        self.quant_type = (
            WNA16_ZP_SUPPORTED_TYPES_MAP[num_bits]
            if not self.symmetric
            else WNA16_SUPPORTED_TYPES_MAP[num_bits]
        )

        # V100 path: dense 4-bit symmetric group-quant has no native SM70 kernel,
        # so convert to TurboMind AWQ at load time (see _is_sm70_capable). Gated to
        # the formats the AWQ GEMM accepts; everything else keeps the Marlin path.
        self.num_bits = num_bits
        self.use_sm70_awq = (
            num_bits == 4
            and bool(self.symmetric)
            and not self.has_g_idx
            and self.group_size in _SM70_AWQ_GROUP_SIZES
            and _is_sm70_capable()
        )

    def get_min_capability(self) -> int:
        # SM70 (V100) is supported only via the TurboMind AWQ conversion above;
        # the native Marlin kernel still requires Turing (SM75) and up.
        if getattr(self, "use_sm70_awq", False):
            return 70
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)

        # On SM70 there is no native mixed-precision kernel (Marlin is SM75+); the
        # weights are loaded in CT format and converted to TurboMind AWQ in
        # process_weights_after_loading. Skip kernel selection in that case.
        if self.use_sm70_awq:
            self.kernel = None
            if "CT->TurboMindAWQ(SM70)" not in self._kernel_backends_being_used:
                logger.info(
                    "Using CT->TurboMind AWQ (SM70) for CompressedTensorsWNA16"
                )
                self._kernel_backends_being_used.add("CT->TurboMindAWQ(SM70)")
        else:
            mp_linear_kernel_config = MPLinearLayerConfig(
                full_weight_shape=(input_size, output_size),
                partition_weight_shape=(
                    input_size_per_partition,
                    output_size_per_partition,
                ),
                weight_type=self.quant_type,
                act_type=params_dtype,
                group_size=self.group_size,
                zero_points=not self.symmetric,
                has_g_idx=self.has_g_idx,
            )

            kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

            if kernel_type.__name__ not in self._kernel_backends_being_used:
                logger.info(
                    "Using %s for CompressedTensorsWNA16", kernel_type.__name__
                )
                self._kernel_backends_being_used.add(kernel_type.__name__)

            if kernel_type is MarlinLinearKernel:
                input_dtype = get_marlin_input_dtype(self.layer_name)
                if input_dtype is not None:
                    mp_linear_kernel_config.act_type = input_dtype

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = input_size != input_size_per_partition
        partition_scales = not marlin_repeat_scales_on_all_ranks(
            self.has_g_idx, self.group_size, row_parallel
        )

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self.pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
        )

        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
        }

        zeros_args = {
            "weight_loader": weight_loader,
            "data": torch.zeros(
                output_size_per_partition // self.pack_factor,
                scales_and_zp_size,
                dtype=torch.int32,
            ),
        }

        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0, **weight_scale_args)

            if not self.symmetric:
                qzeros = PackedColumnParameter(
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args
            )
            if not self.symmetric:
                qzeros = PackedvLLMParameter(
                    input_dim=1,
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        if not self.symmetric:
            layer.register_parameter("weight_zero_point", qzeros)

        # group index (for activation reordering)
        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(
                data=torch.empty(
                    input_size_per_partition,
                    dtype=torch.int32,
                ),
                input_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_g_idx", weight_g_idx)

        if not self.use_sm70_awq:
            self.kernel = kernel_type(
                mp_linear_kernel_config,
                w_q_param_name="weight_packed",
                w_s_param_name="weight_scale",
                w_zp_param_name="weight_zero_point",
                w_gidx_param_name="weight_g_idx",
            )

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.use_sm70_awq:
            self._process_weights_sm70_awq(layer)
        else:
            self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        if self.use_sm70_awq:
            return self._apply_sm70_awq(layer, x, bias)
        return self.kernel.apply_weights(layer, x, bias)

    def _process_weights_sm70_awq(self, layer: torch.nn.Module) -> None:
        """Convert the loaded dense CT W4A16 weights to TurboMind AWQ (SM70).

        Mirrors the CT MoE SM70 path: repack qweight CT [N, K/8] -> AWQ [K, N/8],
        transpose scales [N, K/gs] -> [K/gs, N], synthesise symmetric AWQ qzeros
        (zero_point = 8 -> 0x88888888), then run the TurboMind prepare op. Frees
        the CT tensors afterwards to keep V100 residency low.
        """
        from vllm import _custom_ops as ops

        gs = self.group_size
        qweight_awq = _ct_dense_to_awq_qweight(layer.weight_packed.data)  # [K, N/8]
        scales_awq = (
            layer.weight_scale.data.t().contiguous().to(torch.float16)
        )  # [K/gs, N]
        k_gs, n = scales_awq.shape
        # 0x88888888 overflows signed int32; build via the unsigned view.
        zp = torch.tensor([0x88888888], dtype=torch.uint32).view(torch.int32).item()
        qzeros = torch.full(
            (k_gs, n // self.pack_factor),
            zp,
            dtype=torch.int32,
            device=qweight_awq.device,
        )

        tm_weight, tm_scales, meta = ops.awq_sm70_prepare(
            qweight_awq, scales_awq, qzeros, gs
        )
        layer._awq_sm70_weight = tm_weight
        layer._awq_sm70_scales = tm_scales
        layer._awq_sm70_k_ld = int(meta[0])
        layer._awq_sm70_q_ld = int(meta[1])
        layer._awq_sm70_group_size = gs

        dev = tm_weight.device
        layer.weight_packed = torch.nn.Parameter(
            torch.empty(0, dtype=torch.int32, device=dev), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.empty(0, dtype=torch.float16, device=dev), requires_grad=False
        )

    def _apply_sm70_awq(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        from vllm import _custom_ops as ops

        reshaped_x = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (layer._awq_sm70_weight.shape[-1] * 8,)
        out = ops.awq_gemm_sm70(
            reshaped_x,
            layer._awq_sm70_weight,
            layer._awq_sm70_scales,
            layer._awq_sm70_group_size,
            layer._awq_sm70_k_ld,
            layer._awq_sm70_q_ld,
        )
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
