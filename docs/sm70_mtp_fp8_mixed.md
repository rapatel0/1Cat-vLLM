# Mixed NVFP4/FP8 MTP

## Conversion

Official-parity (copy Flash Next FP8 MTP experts, fused):

```
python3 tools/convert_nvfp4_mtp_fp8.py \
  --nvfp4 /models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 \
  --fp8 /workspace/iron-002/hf-cache/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13 \
  --out /models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4-MTP-FP8 \
  --source official
```

Runtime-amax comparison overlay:

```
python3 tools/convert_nvfp4_mtp_fp8.py \
  --nvfp4 /models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 \
  --fp8 /workspace/iron-002/hf-cache/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13 \
  --out /models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4-MTP-FP8-AMAX \
  --source runtime-amax
```

## Launch (serialized ModelOpt-parity FP8 MTP)

```
CUDA_VISIBLE_DEVICES=4,5,6,7 \
VLLM_SLEEP_CALIBRATE=0 \
vllm serve /models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4-MTP-FP8 \
  --served-model-name qwen38 \
  --tensor-parallel-size 4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --kv-cache-dtype int8_block32 \
  --max-model-len 262144 \
  --port 8083
```

Architecture fallback (original NVFP4 checkpoint, runtime-amax):

```
VLLM_SM70_MTP_BLOCK_FP8=1 VLLM_SM70_MTP_ARCH_FALLBACK=1
```

BF16 MTP baseline:

```
VLLM_SM70_MTP_BLOCK_FP8=0 VLLM_SM70_MTP_ARCH_FALLBACK=1
```

Fail closed without mixed metadata:

```
VLLM_SM70_MTP_ARCH_FALLBACK=0
```

## Default

Do not set FP8 MTP as production default until the three-way bench table in this file has measured throughput, memory, and acceptance. Serialized official-parity is the candidate default if it wins those gates.
