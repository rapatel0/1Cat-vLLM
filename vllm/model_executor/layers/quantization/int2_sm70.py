# SPDX-License-Identifier: Apache-2.0
"""int2 SM70 (V100) weight-only 2-bit quantization.

Affine grouped format: w = q*scale + bias, q unsigned 2-bit (0..3), per
`group_size` scale+bias (covers symmetric and GPTQ-style asymmetric). Targets
the small-M decode / MoE-expert regime via custom SM70 GEMV kernels.

apply() currently uses a torch-fallback dequant (validates the integration math
on hardware without nvcc); the compiled int2 GEMV op (csrc/sm70_int2) replaces
it once built. See docs/INT2_SM70_INTEGRATION.md.

Weight tensors (per linear partition, output O x input I):
  qweight : uint8  [O, I//4]            (4 unsigned 2-bit vals/byte, val k at bits 2*(k%4))
  scales  : fp16   [O, I//group_size]
  qbias   : fp16   [O, I//group_size]   (the affine bias = min of the group)
"""
from typing import Any, Optional

import torch
from torch.nn import Module, Parameter

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


def dequantize_affine_2bit(qweight: torch.Tensor, scales: torch.Tensor,
                           qbias: torch.Tensor, group_size: int) -> torch.Tensor:
    """qweight[O,I//4] uint8 + scales/qbias[O,I//g] -> W [O,I] (params_dtype)."""
    O, Kp = qweight.shape
    K = Kp * 4
    # unpack 4 vals/byte -> [O, K] in 0..3
    q = torch.empty(O, K, dtype=torch.int32, device=qweight.device)
    qw = qweight.to(torch.int32)
    for t in range(4):
        q[:, t::4] = (qw >> (t * 2)) & 0x3
    ng = K // group_size
    s = scales.to(torch.float32).view(O, ng, 1)
    b = qbias.to(torch.float32).view(O, ng, 1)
    W = (q.float().view(O, ng, group_size) * s + b).view(O, K)
    return W.to(scales.dtype)


def quantize_affine_2bit(W: torch.Tensor, group_size: int):
    """W [O,I] -> (qweight uint8 [O,I//4], scales fp16, qbias fp16, Wdq)."""
    O, K = W.shape
    ng = K // group_size
    Wg = W.view(O, ng, group_size).float()
    mn = Wg.amin(dim=2, keepdim=True)
    mx = Wg.amax(dim=2, keepdim=True)
    scale = (mx - mn) / 3.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round((Wg - mn) / scale).clamp_(0, 3).to(torch.int32)
    Wdq = (q.float() * scale + mn).view(O, K).to(W.dtype)
    qf = q.view(O, K)
    qweight = torch.zeros(O, K // 4, dtype=torch.uint8, device=W.device)
    for t in range(4):
        qweight |= (qf[:, t::4].to(torch.uint8) << (t * 2))
    scales = scale.squeeze(-1).to(torch.float16).contiguous()
    qbias = mn.squeeze(-1).to(torch.float16).contiguous()
    return qweight.contiguous(), scales, qbias, Wdq


@register_quantization_config("int2_sm70")
class Int2Sm70Config(QuantizationConfig):
    def __init__(self, group_size: int = 128):
        super().__init__()
        self.group_size = group_size

    def get_name(self):
        return "int2_sm70"

    def get_supported_act_dtypes(self):
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Int2Sm70Config":
        return cls(group_size=config.get("group_size", 128))

    def get_quant_method(self, layer: Module, prefix: str):
        # mixed precision: quantize Linear layers; leave others (incl. MoE for
        # now) unquantized. MoE expert method is a follow-up (uses the same op).
        if isinstance(layer, LinearBase):
            return Int2Sm70LinearMethod(self)
        return None


class Int2Sm70LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Int2Sm70Config):
        self.quant_config = quant_config

    def create_weights(self, layer: Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        g = self.quant_config.group_size
        I = input_size_per_partition
        O = sum(output_partition_sizes)
        if I % g != 0 or I % 4 != 0:
            raise ValueError(f"int2_sm70: input {I} must be divisible by group {g} and 4")
        dev = "cuda"
        qweight = Parameter(torch.empty(O, I // 4, dtype=torch.uint8, device=dev),
                            requires_grad=False)
        scales = Parameter(torch.empty(O, I // g, dtype=torch.float16, device=dev),
                           requires_grad=False)
        qbias = Parameter(torch.empty(O, I // g, dtype=torch.float16, device=dev),
                          requires_grad=False)
        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qbias", qbias)
        for p in (qweight, scales, qbias):
            extra_weight_attrs and torch.nn.init.zeros_(p)
        layer.input_size_per_partition = I
        layer.output_size_per_partition = O

    def process_weights_after_loading(self, layer: Module) -> None:
        # cache the dequantized weight for the torch-fallback path (P1b replaces
        # this with the compiled GEMV op on the packed tensors directly).
        W = dequantize_affine_2bit(layer.qweight, layer.scales, layer.qbias,
                                   self.quant_config.group_size)
        layer.register_parameter("_w_dq",
                                 Parameter(W, requires_grad=False))

    def apply(self, layer: Module, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # torch-fallback: y = x @ dequant(W).T  (+ bias)
        out = torch.nn.functional.linear(x, layer._w_dq, bias)
        return out
