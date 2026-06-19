#!/usr/bin/env python3
"""Numerical validation of the int2_sm70 quant method (torch-fallback path).

Validates, on hardware, without nvcc:
  1. pack/unpack roundtrip: dequantize(quantize(W)) == the Wdq quantize returned
  2. quant error ||W - Wdq|| / ||W||  (informative)
  3. LinearMethod.apply() == x @ dequant(W).T  (the integration math)

  /opt/venv/bin/python tests/int2_sm70/test_quant_method.py
"""
import torch
from vllm.model_executor.layers.quantization.int2_sm70 import (  # noqa: E402
    Int2Sm70Config, Int2Sm70LinearMethod,
    quantize_affine_2bit, dequantize_affine_2bit,
)


def main():
    torch.manual_seed(0)
    print("[int2_sm70 quant method] numerical validation")
    bad = 0
    for (O, I, g) in [(256, 2048, 128), (512, 4096, 128), (128, 8192, 64)]:
        W = (torch.randn(O, I, device="cuda") * 0.05).half()
        qw, sc, qb, Wdq = quantize_affine_2bit(W, g)

        # 1. roundtrip: the packed tensors dequantize back to exactly Wdq
        Wre = dequantize_affine_2bit(qw, sc, qb, g)
        rt = (Wre.float() - Wdq.float()).abs().max().item()

        # 2. quantization error (informative — 2-bit is lossy by design)
        qerr = ((W.float() - Wdq.float()).norm() / W.float().norm()).item()

        # 3. LinearMethod.apply() vs x @ dequant(W).T
        layer = torch.nn.Module()
        layer.qweight = torch.nn.Parameter(qw, requires_grad=False)
        layer.scales = torch.nn.Parameter(sc, requires_grad=False)
        layer.qbias = torch.nn.Parameter(qb, requires_grad=False)
        method = Int2Sm70LinearMethod(Int2Sm70Config(g))
        method.process_weights_after_loading(layer)
        x = torch.randn(3, I, device="cuda").half()
        out = method.apply(layer, x)
        ref = torch.nn.functional.linear(x, Wdq)
        apply_rel = ((out.float() - ref.float()).norm()
                     / ref.float().norm().clamp_min(1e-6)).item()

        ok = rt < 1e-3 and apply_rel < 1e-3
        bad |= (0 if ok else 1)
        print(f"  O={O:5d} I={I:6d} g={g:4d}  roundtrip_maxabs={rt:.2e}  "
              f"apply_rel={apply_rel:.2e}  quant_err={qerr:.3f}  {'ok' if ok else 'FAIL'}")
    print("[QUANT METHOD OK]" if not bad else "[QUANT METHOD FAIL]")
    raise SystemExit(bad)


if __name__ == "__main__":
    main()
