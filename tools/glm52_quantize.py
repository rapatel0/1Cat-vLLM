#!/usr/bin/env python3
"""GLM-5.2 FP8 -> 2-bit (information-preserving) converter / quantizer.

Source: zai-org/GLM-5.2-FP8 — block-FP8 (e4m3, 128x128 blocks; each weight has a
per-block fp32 `weight_scale_inv`). Target: mixed precision for 8xV100/int2_sm70:
  * MoE experts (gate/up/down)  -> 2-bit affine, group 128, MSE-optimal clip
  * everything else (MLA attn, router/gate, shared expert, embed, lm_head, norms)
    -> fp16   (quality-sensitive; kept high-bit per design doc §3d)

The 2-bit format matches vllm int2_sm70 (affine `w = q*s + b`, q in {0,1,2,3},
per `group` along the input/K dim): qweight uint8 [O, I//4], scales/qbias fp16
[O, I//group].

"Preserve the most information": instead of naive per-group min/max (outlier
sensitive — one large value stretches the range and wastes levels), we grid-
search the clip range that *minimizes per-group reconstruction MSE*, so the 4
levels land where the weight mass actually is.

Usage:
  # validate the quantizer on real downloaded weights (no full model needed):
  python tools/glm52_quantize.py --selftest --src /models/GLM-5.2-FP8
  # full conversion (after the download completes):
  python tools/glm52_quantize.py --src /models/GLM-5.2-FP8 --dst /models/GLM-5.2-int2
"""
import argparse
import os
import torch


# --------------------------------------------------------------------------- #
# block-FP8 dequant
# --------------------------------------------------------------------------- #
def dequant_block_fp8(w_fp8: torch.Tensor, scale_inv: torch.Tensor,
                      block: int = 128) -> torch.Tensor:
    """w_fp8 [O,I] e4m3 + scale_inv [ceil(O/blk), ceil(I/blk)] f32 -> bf16 [O,I]."""
    O, I = w_fp8.shape
    w = w_fp8.to(torch.float32)
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:O, :I]
    return (w * s).to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# information-preserving 2-bit affine quantizer (group along input dim)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def quantize_2bit_mse(W: torch.Tensor, group: int = 128,
                      grid: int = 8, device: str = "cuda"):
    """W [O,I] -> (qweight uint8 [O,I//4], scales f16 [O,I//g], qbias f16,
    Wdq, rel_err). 4 levels (q in 0..3). MSE-optimal *asymmetric* clip per
    group: independently search shrink factors on the lower gap (mean->min) and
    upper gap (mean->max), minimizing per-group reconstruction MSE. Handles
    one-sided outliers that a symmetric shrink can't."""
    O, I = W.shape
    assert I % group == 0, f"I={I} not divisible by group={group}"
    ng = I // group
    Wf = W.to(device=device, dtype=torch.float32).reshape(O, ng, group)

    wmin = Wf.amin(dim=2, keepdim=True)           # [O,ng,1]
    wmax = Wf.amax(dim=2, keepdim=True)
    wmean = Wf.mean(dim=2, keepdim=True)

    best_mse = torch.full((O, ng, 1), float("inf"), device=device)
    best_s = torch.ones((O, ng, 1), device=device)
    best_b = wmin.clone()
    # 2-D grid: lower-gap and upper-gap shrink factors searched independently.
    facs = [1.0 - 0.55 * i / (grid - 1) for i in range(grid)]   # 1.0 .. 0.45
    for flo in facs:
        lo = wmean + (wmin - wmean) * flo
        for fhi in facs:
            hi = wmean + (wmax - wmean) * fhi
            s = (hi - lo) / 3.0
            s = torch.where(s <= 0, torch.ones_like(s), s)
            q = torch.round((Wf - lo) / s).clamp_(0, 3)
            recon = q * s + lo
            mse = ((recon - Wf) ** 2).mean(dim=2, keepdim=True)
            upd = mse < best_mse
            best_mse = torch.where(upd, mse, best_mse)
            best_s = torch.where(upd, s, best_s)
            best_b = torch.where(upd, lo, best_b)

    q = torch.round((Wf - best_b) / best_s).clamp_(0, 3).to(torch.int32)  # [O,ng,group]
    Wdq = (q.to(torch.float32) * best_s + best_b).reshape(O, I)
    qf = q.reshape(O, I)
    qweight = torch.zeros(O, I // 4, dtype=torch.uint8, device=device)
    for t in range(4):
        qweight |= (qf[:, t::4].to(torch.uint8) << (t * 2))
    scales = best_s.squeeze(-1).to(torch.float16)
    qbias = best_b.squeeze(-1).to(torch.float16)
    rel = ((Wdq - Wf.reshape(O, I)).norm() / Wf.reshape(O, I).norm().clamp_min(1e-9)).item()
    return (qweight.cpu(), scales.cpu(), qbias.cpu(),
            Wdq.to(torch.bfloat16).cpu(), rel)


@torch.no_grad()
def quantize_2bit_minmax(W: torch.Tensor, group: int = 128, device: str = "cuda"):
    """Naive per-group min/max affine (baseline, to quantify the MSE win)."""
    O, I = W.shape
    ng = I // group
    Wf = W.to(device=device, dtype=torch.float32).reshape(O, ng, group)
    mn = Wf.amin(dim=2, keepdim=True)
    mx = Wf.amax(dim=2, keepdim=True)
    s = (mx - mn) / 3.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = torch.round((Wf - mn) / s).clamp_(0, 3)
    Wdq = (q * s + mn).reshape(O, I)
    rel = ((Wdq - Wf.reshape(O, I)).norm() / Wf.reshape(O, I).norm().clamp_min(1e-9)).item()
    return rel


# --------------------------------------------------------------------------- #
# self-test on real downloaded shards
# --------------------------------------------------------------------------- #
def selftest(src: str):
    import glob
    from safetensors import safe_open
    shards = sorted(glob.glob(os.path.join(src, "model-*-of-*.safetensors")))
    assert shards, f"no shards in {src}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[selftest] device={dev}  shards available={len(shards)}")
    tested = 0
    for sh in shards:
        with safe_open(sh, "pt") as f:
            keys = list(f.keys())
            # find an expert proj with its scale, in this shard
            for k in keys:
                if k.endswith("_proj.weight") and ".experts." in k and \
                        (k + "_scale_inv") in keys:
                    w = f.get_tensor(k)
                    s = f.get_tensor(k + "_scale_inv")
                    Wbf = dequant_block_fp8(w, s)
                    name = k.split("model.")[-1]
                    rel_mm = quantize_2bit_minmax(Wbf, 128, dev)
                    _, _, _, _, rel_g128 = quantize_2bit_mse(Wbf, 128, device=dev)
                    _, _, _, _, rel_g64 = quantize_2bit_mse(Wbf, 64, device=dev)
                    print(f"  {name:46s} {tuple(Wbf.shape)}  2bit rel: "
                          f"minmax={rel_mm:.4f}  MSEopt/g128={rel_g128:.4f}  "
                          f"MSEopt/g64={rel_g64:.4f}")
                    tested += 1
                    break
        if tested >= 6:
            break
    print(f"[selftest] done ({tested} expert tensors) — MSE-opt preserves more info")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/models/GLM-5.2-FP8")
    ap.add_argument("--dst", default="/models/GLM-5.2-int2")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.src)
        return
    raise SystemExit("full conversion path: implemented in next step (needs full "
                     "download); use --selftest now to validate the quantizer.")


if __name__ == "__main__":
    main()
