#!/usr/bin/env python3
"""Write a mixed NVFP4 + serialized MTP FP8 overlay checkpoint.

Sources:
  official  — copy Qwen3.8-Flash-Next-FP8 MTP expert tensors (fused stack)
  runtime-amax — quantize this snapshot's BF16 MTP experts with amax/448

The overlay keeps NVFP4 main experts and PLE. MTP attn/shared/HC stay BF16.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
BLOCK = 128


def block_fp8_quantize(weight: torch.Tensor):
    num_experts, n, k = weight.shape
    ns = (n + BLOCK - 1) // BLOCK
    ks = (k + BLOCK - 1) // BLOCK
    pad_n = ns * BLOCK - n
    pad_k = ks * BLOCK - k
    weight_f = weight.to(torch.float32)
    if pad_n or pad_k:
        weight_f = torch.nn.functional.pad(weight_f, (0, pad_k, 0, pad_n))
    blocks = weight_f.view(num_experts, ns, BLOCK, ks, BLOCK)
    amax = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-12)
    scale_f = amax / E4M3_MAX
    quant = blocks / scale_f.unsqueeze(2).unsqueeze(4)
    quant = quant.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    quant = quant.reshape(num_experts, ns * BLOCK, ks * BLOCK)[:, :n, :k]
    return quant.contiguous(), scale_f.to(torch.bfloat16).contiguous()
MTP_GATE_UP = "mtp.layers.0.mlp.experts.gate_up_proj"
MTP_DOWN = "mtp.layers.0.mlp.experts.down_proj"
MTP_W13_SCALE = "mtp.layers.0.mlp.experts.w13_weight_scale_inv"
MTP_W2_SCALE = "mtp.layers.0.mlp.experts.w2_weight_scale_inv"
MIXED_MODULES = {
    "mtp.layers.*.mlp.experts": "fp8_block128",
    "model.layers.*.mlp.experts": "fp8_block128",
}


def _sha256_tensor(t: torch.Tensor) -> str:
    u8 = t.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(u8.tobytes()).hexdigest()


def _load_index(root: Path) -> dict[str, str]:
    return json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]


def _open(root: Path, wm: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(str(root / wm[key]), framework="pt", device="cpu") as st:
        return st.get_tensor(key)


def _dequant(fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = fp8.shape
    s = scale.float().repeat_interleave(BLOCK, 0)[:n].repeat_interleave(BLOCK, 1)[:, :k]
    return fp8.float() * s


def _stack_official(fp8_root: Path, fp8_wm: dict[str, str], num_experts: int):
    keys = []
    for eid in range(num_experts):
        base = f"mtp.layers.0.mlp.experts.{eid}"
        keys.extend(
            [
                f"{base}.gate_proj.weight",
                f"{base}.gate_proj.weight_scale_inv",
                f"{base}.up_proj.weight",
                f"{base}.up_proj.weight_scale_inv",
                f"{base}.down_proj.weight",
                f"{base}.down_proj.weight_scale_inv",
            ]
        )
    tensors: dict[str, torch.Tensor] = {}
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(fp8_wm[key], []).append(key)
    for fname, file_keys in by_file.items():
        with safe_open(str(fp8_root / fname), framework="pt", device="cpu") as st:
            for key in file_keys:
                tensors[key] = st.get_tensor(key)
        print("loaded", fname, "n", len(file_keys), flush=True)
    w13 = []
    w13_s = []
    w2 = []
    w2_s = []
    for eid in range(num_experts):
        base = f"mtp.layers.0.mlp.experts.{eid}"
        gate = tensors[f"{base}.gate_proj.weight"]
        gate_s = tensors[f"{base}.gate_proj.weight_scale_inv"]
        up = tensors[f"{base}.up_proj.weight"]
        up_s = tensors[f"{base}.up_proj.weight_scale_inv"]
        down = tensors[f"{base}.down_proj.weight"]
        down_s = tensors[f"{base}.down_proj.weight_scale_inv"]
        w13.append(torch.cat([gate, up], dim=0))
        w13_s.append(torch.cat([gate_s, up_s], dim=0))
        w2.append(down)
        w2_s.append(down_s)
        if eid == 0:
            print("official e0 gate", tuple(gate.shape), gate.dtype, tuple(gate_s.shape))
    return (
        torch.stack(w13),
        torch.stack(w13_s).contiguous(),
        torch.stack(w2),
        torch.stack(w2_s).contiguous(),
    )


def _runtime_amax(nv_root: Path, nv_wm: dict[str, str]):
    up = _open(nv_root, nv_wm, MTP_GATE_UP)
    down = _open(nv_root, nv_wm, MTP_DOWN)
    w13, w13_s = block_fp8_quantize(up)
    w2, w2_s = block_fp8_quantize(down)
    return w13, w13_s, w2, w2_s


def convert(nvfp4: Path, fp8: Path, out: Path, source: str) -> dict:
    nv_wm = _load_index(nvfp4)
    fp8_wm = _load_index(fp8)
    nv_up = _open(nvfp4, nv_wm, MTP_GATE_UP)
    nv_down = _open(nvfp4, nv_wm, MTP_DOWN)
    num_experts = int(nv_up.shape[0])
    if source == "official":
        w13, w13_s, w2, w2_s = _stack_official(fp8, fp8_wm, num_experts)
        gate0 = _open(fp8, fp8_wm, "mtp.layers.0.mlp.experts.0.gate_proj.weight")
        gate0_s = _open(fp8, fp8_wm, "mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv")
        recon = _dequant(gate0, gate0_s)
        ref = nv_up[0, :640].float()
        rel = float((recon - ref).norm() / ref.norm())
        if rel >= 0.05:
            raise SystemExit(f"official MTP vs NVFP4 BF16 rel L2 {rel} >= 0.05")
        identity_rel = rel
    elif source == "runtime-amax":
        w13, w13_s, w2, w2_s = _runtime_amax(nvfp4, nv_wm)
        identity_rel = None
    else:
        raise SystemExit(f"unknown source {source}")

    if w13.shape != nv_up.shape:
        raise SystemExit(f"fused shape mismatch {tuple(w13.shape)} vs {tuple(nv_up.shape)}")

    out.mkdir(parents=True, exist_ok=True)
    for item in nvfp4.iterdir():
        dest = out / item.name
        if dest.exists() or dest.is_symlink():
            continue
        if item.name in {
            "model.safetensors.index.json",
            "hf_quant_config.json",
            "config.json",
            "conversion_manifest.json",
        }:
            continue
        os.symlink(item.resolve(), dest)

    shard_name = "model-mtp-fp8.safetensors"
    save_file(
        {
            MTP_GATE_UP: w13,
            MTP_W13_SCALE: w13_s,
            MTP_DOWN: w2,
            MTP_W2_SCALE: w2_s,
        },
        str(out / shard_name),
    )

    index = json.loads((nvfp4 / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    for key in (MTP_GATE_UP, MTP_DOWN):
        wm[key] = shard_name
    wm[MTP_W13_SCALE] = shard_name
    wm[MTP_W2_SCALE] = shard_name
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    hf_quant = json.loads((nvfp4 / "hf_quant_config.json").read_text())
    hf_quant["quantization"]["mixed_modules"] = MIXED_MODULES
    hf_quant["quantization"]["fp8_serialized"] = True
    hf_quant["quantization"]["fp8_scale_convention"] = "amax_div_448"
    hf_quant["quantization"]["fp8_block_size"] = [128, 128]
    (out / "hf_quant_config.json").write_text(json.dumps(hf_quant, indent=2) + "\n")

    cfg = json.loads((nvfp4 / "config.json").read_text())
    q = cfg.setdefault("quantization_config", {})
    q["mixed_modules"] = MIXED_MODULES
    q["fp8_serialized"] = True
    q["fp8_scale_convention"] = "amax_div_448"
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    manifest = {
        "source_nvfp4": str(nvfp4),
        "source_fp8": str(fp8),
        "conversion_source": source,
        "scale_convention": "amax_div_448",
        "block_shape": [128, 128],
        "included": ["mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}"],
        "excluded": ["mtp attn", "shared expert", "HC", "router", "norms"],
        "output_dtypes": {
            MTP_GATE_UP: str(w13.dtype),
            MTP_W13_SCALE: str(w13_s.dtype),
            MTP_DOWN: str(w2.dtype),
            MTP_W2_SCALE: str(w2_s.dtype),
        },
        "shapes": {
            MTP_GATE_UP: list(w13.shape),
            MTP_W13_SCALE: list(w13_s.shape),
            MTP_DOWN: list(w2.shape),
            MTP_W2_SCALE: list(w2_s.shape),
        },
        "hashes": {
            "nvfp4_gate_up": _sha256_tensor(nv_up[:1]),
            "nvfp4_down": _sha256_tensor(nv_down[:1]),
            "out_w13": _sha256_tensor(w13[:1]),
        },
        "official_vs_bf16_rel_l2_expert0_gate": identity_rel,
        "e4m3_max": E4M3_MAX,
    }
    (out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("WROTE", out)
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nvfp4", type=Path, required=True)
    p.add_argument("--fp8", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", choices=["official", "runtime-amax"], default="official")
    args = p.parse_args()
    convert(args.nvfp4, args.fp8, args.out, args.source)


if __name__ == "__main__":
    main()
