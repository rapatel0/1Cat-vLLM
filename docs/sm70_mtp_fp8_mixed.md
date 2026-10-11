# Mixed NVFP4/FP8 MTP

## Scale convention (Phase 1)

Official Flash Next FP8 MTP experts:

- E4M3 weights, 128×128 blocks
- `weight_scale_inv` is bfloat16
- dequant = `fp8.float() * scale`
- scale = `amax / 448`
- MTP attn, shared expert, HC, router stay BF16

See `docs/sm70_mtp_fp8_scale_forensics.md`.

Runtime `amax / 448` matches official scales (median ratio 1.0). Do not call it ModelOpt MSE.

## Conversion (Phase 2–3)

Official-parity overlay (copy Flash Next FP8 experts, fused):

```
python3 tools/convert_nvfp4_mtp_fp8.py \
  --nvfp4 /models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 \
  --fp8 /workspace/iron-002/hf-cache/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13 \
  --out /workspace/iron-002/ckpts/Qwen3.8-Flash-Next-ABLITERATED-NVFP4-MTP-FP8 \
  --source official
```

Runtime-amax overlay:

```
python3 tools/convert_nvfp4_mtp_fp8.py ... --source runtime-amax \
  --out /workspace/iron-002/ckpts/Qwen3.8-Flash-Next-ABLITERATED-NVFP4-MTP-FP8-AMAX
```

The draft loader's first matching glob is `model-bf16-*.safetensors`. The converter writes `model-bf16-mtp-fp8.safetensors`.

Written overlay (official source):

- `mtp.layers.0.mlp.experts.gate_up_proj` float8 `[512,1280,2560]`
- `w13_weight_scale_inv` bf16 `[512,10,20]`
- `down_proj` float8 `[512,2560,640]`
- `w2_weight_scale_inv` bf16 `[512,20,5]`
- `hf_quant_config.json` `mixed_modules` + `fp8_serialized=true`
- `official_vs_bf16_rel_l2_expert0_gate` = 0.026585

## Dispatch (Phase 4)

`mixed_modules` selects the expert path:

- `fp8_block128` + `fp8_serialized` → `Sm70SerializedBlockFp8MoEMethod`
- `fp8_block128_runtime` / architecture fallback + `VLLM_SM70_MTP_BLOCK_FP8=1` → `Sm70OnlineBlockFp8MoEMethod`
- main `language_model` experts → NVFP4
- missing MTP format with mixed metadata → fail closed
- `VLLM_SM70_MTP_ARCH_FALLBACK=1` (default) warns on old NVFP4 checkpoints

## Three-way bench (Phase 5)

Shared settings: TP4, INT8 `int8_block32` KV, MTP4, `max_model_len=32768`, `max_num_seqs=1`, `gpu_memory_utilization=0.90`, temperature 0, 64 completion tokens, four prompts (coding, factual, reasoning, tool-use). One warmup.

| config | coding tok/s | factual | reasoning | tool_use | GPU0 mem | MTP AL | pos accept | draft accept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 MTP | 59.07 | 17.19 | 49.72 | 13.55 | 30522 MiB | n/a (metrics not flushed) | n/a | n/a |
| runtime-amax FP8 | 84.81 | 51.84 | 76.99 | 34.89 | 30322 MiB | 4.30 | 1.000, 0.848, 0.758, 0.697 | 82.6% |
| serialized official FP8 | 24.17 | 21.70 | 23.90 | 21.23 | 30522 MiB | 1.01 | 0.013, 0, 0, 0 | 0.3% |

Temperature-0 coding/factual/tool-use prefixes matched across BF16 and runtime-amax. Serialized reasoning wording diverged (`t + b` vs `b + t`); verification still ran on the target model.

Serialized official fused tensors load, but TP4 scale sharding makes drafts nearly useless. Do not treat that overlay as ModelOpt-parity serving until scale-shard loading is fixed.

## Default

**Use runtime-amax FP8 MTP.** It improved end-to-end tok/s, slightly reduced loaded memory, kept high MTP acceptance, and used the TurboMind block-FP8 kernel (no BF16 expert GEMM).

Do not default to serialized official-parity until acceptance recovers.

## Launch (selected config)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
FLASH_ATTN_V100=1 \
VLLM_PLE_CPU_OFFLOAD=1 \
VLLM_SM70_NVFP4_TURBOMIND=1 \
VLLM_SM70_MTP_BLOCK_FP8=1 \
VLLM_SM70_MTP_ARCH_FALLBACK=1 \
VLLM_KV_CACHE_LAYOUT=NHD \
vllm serve /models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 \
  --served-model-name qwen38 \
  --trust-remote-code \
  --language-model-only \
  --dtype float16 \
  --kv-cache-dtype int8_block32 \
  --kv-offloading-size 32 \
  --kv-offloading-backend native \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --attention-backend FLASH_ATTN_V100 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --host 0.0.0.0 --port 8100
```

## 256K integration (runtime-amax)

`max_model_len=262144`, MTP4, INT8 KV, 1 sequence, `gpu_memory_utilization=0.90`:

- GPU KV cache: **637,233 tokens** (5.16 GiB)
- loaded GPU0: 30322 MiB
- short prompts: 83.4 / 46.6 / 76.1 / 34.9 tok/s
- SpecDecoding: AL **3.83**, positions 0.894, 0.766, 0.638, 0.532, draft accept **70.7%**
- 4820-token prompt probe: 13 completion tokens in 6.78 s, HTTP 200

4 concurrent 256K slots do not fit: 4 × 262144 > 637233. About **2** 256K slots fit. 4-slot 256K remains blocked without throttling or a second island.

Other leftovers:

- serialized official MTP acceptance collapse on TP4 fused scales
- SIGKILL of the API server leaks `VLLM::Worker_TP*` GPU memory; kill those PIDs
