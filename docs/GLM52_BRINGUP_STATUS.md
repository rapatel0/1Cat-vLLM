# GLM-5.2 on 8×V100 — bring-up status (snapshot 2026-06-20)

Anti-regression snapshot. The full 2-bit GLM-5.2 inference path is **built and
validated on real weights**; only the full-checkpoint TP=8 coherence run remains
(gated on the 745 GB download). Do not regress the items marked ✅.

## Goal
`dev, test, validate until the real GLM-5.2 runs across all 8 V100 GPUs.`
GLM-5.2 = `GlmMoeDsaForCausalLM` (`glm_moe_dsa`): DeepSeek-V3.2-style **MLA + DSA +
MoE**, 743B / 256 experts / 78 layers / hidden 6144. FP8 ckpt 745 GB, BF16 1.51 TB.
2-bit experts are *mandatory* to fit 256 GB VRAM (4-bit = 372 GB > budget).

## What works (✅ validated)
| Item | How validated | Commit |
|---|---|---|
| Real `GlmMoeDsaForCausalLM` arch on V100 | `[DSA SMOKE OK]` TP=2 & **TP=8** | b6d3ec5d3 |
| Dense-MLA sm70 fallback (DSA indexer is DeepGEMM/sm90-only) | `_dsa_runtime_enabled()` → dense MLA; FLASH_ATTN_V100 | b6d3ec5d3 |
| `glm_moe_dsa` → DeepseekV3Config registration (transformers 4.57) | real config.json loads | b6d3ec5d3 |
| Info-preserving FP8→2-bit quantizer (MSE-opt asymmetric, g128) | real experts 0.50→**0.333** rel | 409388a07 |
| Packed-2-bit MoE compute (dequant-in-kernel, per expert) bit-exact | `tools/test_int2_moe.py` rel 0.00, 7.1× smaller | 2cf904096 |
| Packed-2-bit MoE in vLLM (storage + int2 GEMV kernel) | `[DSA SMOKE OK]` + `INT2_PACKED_MOE=1`, TP=2 & **TP=8** | 6bddce20c |
| Offline FP8→packed-2bit converter | real layers 0-4 written | 7bbe58e7d |
| Mixed precision (experts 2-bit, attn/embed fp16, §3d) | `get_quant_method` gated on INT2_PACKED_MOE | 7bbe58e7d |
| Sharding-aware loader (EP dim0 + TP gate_up dim1 / down dim2) | — | 7bbe58e7d |
| **REAL weights load + generate, full pipeline** | `[REAL-LOAD OK]` 5 layers, real-vocab tokens, TP=2 | 7bbe58e7d |

## Remaining (⏳)
1. Full 745 GB download (in progress; `kubectl -n llm logs job/hydrate-glm52-fp8`).
2. Convert all 78 layers: `tools/glm52_quantize.py --src /models/GLM-5.2-FP8 --dst /models/GLM-5.2-int2`.
3. Load `/models/GLM-5.2-int2` at TP=8 (EP=8), verify coherent text.

## Key files (worktree `~/repos/1Cat-vLLM-int2`, branch `int2-v100-gemv`)
- `vllm/model_executor/models/deepseek_v2.py` — `_dsa_runtime_enabled()` + dense-MLA gate + indexer-weight skip + `GlmMoeDsaForCausalLM`.
- `vllm/transformers_utils/config.py` — `glm_moe_dsa="DeepseekV3Config"`.
- `vllm/model_executor/layers/quantization/int2_sm70.py` — `Int2Sm70Config`,
  `Int2Sm70PackedMoEMethod` (packed storage + sharding loader + M-adaptive
  kernel apply), mixed-precision `get_quant_method`.
- `csrc/sm70_int2/int2_gemv_op.cu` — compiled int2 GEMV ops (gemv_m1 / gemv_n / repack).
- `tools/glm52_quantize.py` — block-FP8 dequant + MSE-opt 2-bit quantizer + `convert()`.
- `tools/test_int2_moe.py` — standalone 2-bit MoE compute validation.
- `tests/int2_sm70/{smoke_dsa,load_real}.py`, overlays `dsa_overlay.sh` + `dev_overlay.sh`.
- Design: `docs/INT2_SM70_INTEGRATION.md` (§0 = GLM-5.2 reality + status).

## Reproduce (dev pod `int2-dev`, all 8 V100 free)
Env (every op build/run):
```
export CUDA_HOME=/workspace/cuda128 PATH=/workspace/cuda128/bin:$PATH \
  TORCH_CUDA_ARCH_LIST=7.0 TORCH_EXTENSIONS_DIR=/workspace/torch_ext \
  LD_LIBRARY_PATH=/workspace/cuda128/nvvm/lib64:$LD_LIBRARY_PATH \
  INT2_SM70_OP_SRC=/workspace/csrc/sm70_int2/int2_gemv_op.cu
```
Apply overlays onto the image vLLM (no rebuild):
```
kubectl -n llm exec int2-dev -- bash /workspace/tests/int2_sm70/dsa_overlay.sh
kubectl -n llm exec int2-dev -- bash /workspace/tests/int2_sm70/dev_overlay.sh
```
Real-arch + packed-2bit MoE smoke (dummy weights, gibberish OK):
```
CONFIG_JSON=/workspace/tmp/glm52/config.json CUDA_VISIBLE_DEVICES=0..7 TP=8 \
  INT2_QUANT=int2_sm70 INT2_PACKED_MOE=1 LAYERS=8 EXPERTS=32 \
  python /workspace/tests/int2_sm70/smoke_dsa.py
```
Load a converted real checkpoint:
```
MODEL_DIR=/models/GLM-5.2-int2 TP=8 INT2_QUANT=int2_sm70 INT2_PACKED_MOE=1 \
  python /workspace/tests/int2_sm70/load_real.py
```

## Infra notes
- Dev pod `int2-dev`: hostPath `/localpool/dev/int2-vllm` → `/workspace`; cu12.8
  toolkit staged at `/workspace/cuda128`; image `localhost:32000/onecat-vllm:v1-v100-110`.
- Converter pod `glm52-convert`: same image + `/models` (PVC `/srv/models`) +
  `/workspace`. torch 2.9.1.
- Download job `hydrate-glm52-fp8`: `rotorquant:v0-v100`, dnsConfig 1.1.1.1/8.8.8.8
  for internet, writes to PVC `llm-models-local` (pool has 2.1 TB free).
- The worktree vLLM is **1.2.0** (has GlmMoeDsa + DSA); the pod image is **1.1.0**
  but already carries the DSA infra (MLAModules is_sparse, indexer backend), so
  the overlays patch the pod's files in place rather than swapping the tree.
- Mixed precision: `INT2_PACKED_MOE=1` → experts 2-bit + attn/embed/lm_head fp16.
  Without it → int2 on linears too (dummy-weight numeric smokes only).

## Update (2026-06-20): long-context via MLA latent KV

The dense-MLA fallback was silently using FLASH_ATTN_V100 (materializes full
multi-head K/V, ~600 KB/token) because the pod's is_deepseek_mla list lacked
glm_moe_dsa -> use_mla=False. Fixed (dsa_overlay.sh patches the convertor +
kv_b_proj kept fp16) -> TRITON_MLA (latent KV, ~84 KB/token, ~7x smaller).

Measured (real GLM-5.2, 8xV100, MLA latent fp16 KV):
| config | KV for 131K | max context | stable? |
|---|---|---|---|
| FLASH_ATTN_V100 (materialized) | 78 GiB | ~3.6K | yes |
| TRITON_MLA latent, TP4xPP2 | 11 GiB | ~14K | knife-edge (PP1 tight) |
| **TRITON_MLA latent, TP8** | 11 GiB | **~10.7K, [REAL-LOAD OK]** | **yes** |

Stable production config: TP8 + MLA latent, GMU 0.97 -> 10,752 tok KV, generates.
Path to 120K (follow-on, not one-flag): fp8 latent KV (worktree 1.2.0 TritonMLA
supports it; pod 1.1.0 rejects -> overlay), g256 expert scales (+1.4GB/GPU ->
~40K), or higher PP (more KV sharding, needs the tight-stage margin fixed).
