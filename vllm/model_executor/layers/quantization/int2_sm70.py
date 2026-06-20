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
import os
from typing import Any, Optional

import torch
from torch.nn import Module, Parameter

_OP = None


def _get_op():
    """Lazily load the compiled int2 GEMV op (cached build). Returns None if
    unavailable so apply() falls back to the torch dequant path."""
    global _OP
    if _OP is None:
        try:
            from torch.utils.cpp_extension import load
            src = os.environ.get(
                "INT2_SM70_OP_SRC", "/workspace/csrc/sm70_int2/int2_gemv_op.cu")
            _OP = load(name="int2_sm70_gemv", sources=[src],
                       extra_cuda_cflags=["-O3", "-arch=sm_70"], verbose=False)
            import sys
            print("[int2_sm70] compiled GEMV op LOADED", file=sys.stderr, flush=True)
        except Exception as e:
            import sys
            print(f"[int2_sm70] op load failed ({e}); using fallback",
                  file=sys.stderr, flush=True)
            _OP = False
    return _OP or None

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
        # mixed precision: 2-bit the MoE experts (memory bulk) + Linear layers.
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        if isinstance(layer, FusedMoE):
            return _make_moe_method(layer.moe_config, self)
        if isinstance(layer, LinearBase):
            return Int2Sm70LinearMethod(self)
        return None


def fake_quantize_2bit(W: torch.Tensor, group_size: int) -> torch.Tensor:
    """dequant(quantize(W)) over the last dim — puts weights through the exact
    2-bit affine format. For dev/validation (random dummy weights, gibberish OK)
    until the production quantized-checkpoint loader lands."""
    shp = W.shape
    K = shp[-1]
    g = group_size if (K % group_size == 0) else K
    _, _, _, Wdq = quantize_affine_2bit(W.reshape(-1, K), g)
    return Wdq.reshape(shp).to(W.dtype)


def _unpack_2bit_experts(qweight: torch.Tensor, scales: torch.Tensor,
                         qbias: torch.Tensor, group: int) -> torch.Tensor:
    """[E,O,K//4] uint8 + [E,O,K//g] -> [E,O,K] fp16 (affine dequant, batched).
    Used to dequant a *slice* of experts on the fly (never all at once)."""
    E, O, Kp = qweight.shape
    K = Kp * 4
    q = torch.empty(E, O, K, dtype=torch.int32, device=qweight.device)
    qw = qweight.to(torch.int32)
    for t in range(4):
        q[:, :, t::4] = (qw >> (t * 2)) & 0x3
    ng = K // group
    s = scales.to(torch.float32).view(E, O, ng, 1)
    b = qbias.to(torch.float32).view(E, O, ng, 1)
    return (q.float().view(E, O, ng, group) * s + b).view(E, O, K).to(torch.float16)


def _make_moe_method(moe_config, quant_config: "Int2Sm70Config"):
    """Pick the MoE 2-bit path. INT2_PACKED_MOE=1 -> real packed-2-bit expert
    storage (fits 256GB; experts dequantized per-token on the fly). Default ->
    fake-quant (fp16 storage; for dummy-weight smokes / numeric validation)."""
    if os.environ.get("INT2_PACKED_MOE", "0") == "1":
        return _make_packed_moe_method(moe_config, quant_config)

    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    class Int2Sm70MoEMethod(UnquantizedFusedMoEMethod):
        def process_weights_after_loading(self, layer):
            g = quant_config.group_size
            layer.w13_weight.data = fake_quantize_2bit(layer.w13_weight.data, g)
            layer.w2_weight.data = fake_quantize_2bit(layer.w2_weight.data, g)
            super().process_weights_after_loading(layer)

    return Int2Sm70MoEMethod(moe_config)


def _make_packed_moe_method(moe_config, quant_config: "Int2Sm70Config"):
    """Real packed-2-bit FusedMoE: experts stored as 2-bit (≈8x smaller, fits
    GLM-5.2 on 8xV100). apply() dequantizes only the experts a token routes to
    (small transient), then runs the FFN — validated bit-exact in
    tools/test_int2_moe.py. The grouped int2 kernel (no dequant) is the perf
    follow-up; this is the fits-and-runs path."""
    import torch.nn.functional as F
    from torch.nn import Parameter
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.utils import set_weight_attrs

    g = quant_config.group_size

    class Int2Sm70PackedMoEMethod(FusedMoEMethodBase):
        def __init__(self, moe):
            super().__init__(moe)

        def create_weights(self, layer, num_experts, hidden_size,
                           intermediate_size_per_partition, params_dtype,
                           **extra):
            E, H, I = num_experts, hidden_size, intermediate_size_per_partition
            w13_o = 2 * I if self.moe.is_act_and_mul else I
            dev = "cuda"

            def pk(o, k):
                return Parameter(torch.zeros(E, o, k // 4, dtype=torch.uint8,
                                             device=dev), requires_grad=False)

            def sc(o, k):
                return Parameter(torch.zeros(E, o, k // g, dtype=torch.float16,
                                             device=dev), requires_grad=False)

            params = {
                "w13_qweight": pk(w13_o, H), "w13_scales": sc(w13_o, H),
                "w13_qbias": sc(w13_o, H),
                "w2_qweight": pk(H, I), "w2_scales": sc(H, I), "w2_qbias": sc(H, I),
            }
            for name, p in params.items():
                layer.register_parameter(name, p)
                set_weight_attrs(p, extra)
            layer.w13_o, layer.w2_o = w13_o, H

        def process_weights_after_loading(self, layer):
            pass

        def get_fused_moe_quant_config(self, layer):
            # apply() is a custom dequant-loop, not the fused kernel; the
            # standard quant config is unused but required by the ABC.
            from vllm.model_executor.layers.fused_moe.config import (
                FUSED_MOE_UNQUANTIZED_CONFIG,
            )
            return FUSED_MOE_UNQUANTIZED_CONFIG

        def _expert_gemm(self, op, A, qweight, scales, qbias):
            """A [M,K] fp16 @ dequant(qweight [N,K//4]).T -> [M,N], via the
            compiled int2 GEMV kernel (dequant inside the kernel — no fp16
            materialization). M-adaptive: gemv_m1 (M=1 decode), gemv_n (M=2-8);
            dequant+matmul fallback (M>8 / unsupported shape / no op)."""
            M, K = A.shape
            if op is not None and K % 512 == 0:
                Ac = A.contiguous()
                qw = qweight.contiguous()
                try:
                    if M == 1:
                        return op.int2_gemv_m1(Ac, qw, scales, qbias, g)
                    if 2 <= M <= 8:
                        wt = op.int2_repack_nmajor(qw, K)
                        return op.int2_gemv_n(Ac, wt, scales, qbias, g)
                except Exception:
                    pass
            W = _unpack_2bit_experts(qweight.unsqueeze(0), scales.unsqueeze(0),
                                     qbias.unsqueeze(0), g)[0]   # [N,K] fp16
            return A @ W.t()

        def apply(self, layer, x, topk_weights, topk_ids,
                  shared_experts=None, shared_experts_input=None, **kwargs):
            E = layer.w13_qweight.shape[0]
            emap = getattr(layer, "expert_map", None)
            # topk_ids are GLOBAL expert ids; map to local (-1 == not on rank).
            local_ids = emap[topk_ids] if emap is not None else topk_ids
            out = torch.zeros_like(x)
            op = _get_op()
            row_on_input = getattr(layer, "apply_router_weight_on_input", False)
            for le in range(E):                      # local experts only
                sel = (local_ids == le)
                rows, slots = sel.nonzero(as_tuple=True)
                if rows.numel() == 0:
                    continue
                xe = x[rows].contiguous()
                wt = topk_weights[rows, slots].unsqueeze(-1).to(xe.dtype)
                if row_on_input:
                    xe = xe * wt
                gu = self._expert_gemm(op, xe, layer.w13_qweight[le],
                                       layer.w13_scales[le], layer.w13_qbias[le])
                gate, up = gu.chunk(2, dim=-1)
                h = (F.silu(gate.float()) * up.float()).to(xe.dtype).contiguous()
                ye = self._expert_gemm(op, h, layer.w2_qweight[le],
                                       layer.w2_scales[le], layer.w2_qbias[le])
                if not row_on_input:
                    ye = ye * wt
                out.index_add_(0, rows, ye.to(out.dtype))
            if shared_experts is not None:
                out = out + shared_experts(shared_experts_input)
            return out

    return Int2Sm70PackedMoEMethod(moe_config)


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
        # fallback dequantized weight (M>8 / unsupported shapes)
        W = dequantize_affine_2bit(layer.qweight, layer.scales, layer.qbias,
                                   self.quant_config.group_size)
        layer.register_parameter("_w_dq", Parameter(W, requires_grad=False))
        # n-major repack for the M=2..8 op (once, on load), if shapes fit
        O, Kp = layer.qweight.shape
        K = Kp * 4
        layer._wt = None
        if O % 128 == 0 and K % 512 == 0:
            op = _get_op()
            if op is not None:
                try:
                    layer._wt = op.int2_repack_nmajor(layer.qweight.contiguous(), K)
                except Exception:
                    layer._wt = None

    def apply(self, layer: Module, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # compiled int2 GEMV for M==1 decode (validated rel ~6e-4); torch-fallback
        # otherwise (prefill / M>1; the n-split op is a follow-up).
        K = x.shape[-1]
        g = self.quant_config.group_size
        use_op = os.environ.get("INT2_SM70_USE_OP", "1") == "1"
        if use_op and x.dim() == 2 and K % 512 == 0:
            M = x.shape[0]
            op = _get_op()
            if op is not None:
                try:
                    if M == 1:
                        out = op.int2_gemv_m1(x.contiguous(), layer.qweight,
                                              layer.scales, layer.qbias, g)
                        return out if bias is None else out + bias
                    if 2 <= M <= 8 and getattr(layer, "_wt", None) is not None:
                        out = op.int2_gemv_n(x.contiguous(), layer._wt,
                                             layer.scales, layer.qbias, g)
                        return out if bias is None else out + bias
                except Exception:
                    pass
        return torch.nn.functional.linear(x, layer._w_dq, bias)
