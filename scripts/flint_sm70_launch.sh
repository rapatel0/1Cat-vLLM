#!/usr/bin/env bash
# First Flint service on Island B. FP16 KV. INT8 block KV is deferred.
# Requires the four Island B GPU UUIDs and an explicit lease acknowledgement.
set -euo pipefail

if [[ "${SWARM_CANARY_GPU_LEASE_ACK:-0}" != 1 ]]; then
  echo "An explicit operator GPU lease is required." >&2
  exit 64
fi

export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES// /}"
expected='GPU-c1aa8bd9-f642-328d-96f8-79d7c38ce61e,GPU-c275f176-56b9-eb8d-14d7-a3fd15a0ec03,GPU-d5302639-be81-61f0-5e1a-b8f421bb100b,GPU-dd6f7287-63a3-17c4-64c2-1eb597391f4b'
mapfile -t gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d ' ' | sort)
actual=$(IFS=,; echo "${gpu_uuids[*]}")
if [[ "$actual" != "$expected" ]]; then
  echo "Flint requires exactly Island B." >&2
  exit 64
fi

MODEL=${FLINT_MODEL_DIR:-/models/flint-nvfp4-gguf}
exec taskset -c 20-39,60-79 \
  python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --served-model-name flint \
  --trust-remote-code \
  --dtype float16 \
  --kv-cache-dtype auto \
  --tensor-parallel-size "${FLINT_TP_SIZE:-4}" \
  --max-model-len 262144 \
  --max-num-seqs "${FLINT_MAX_NUM_SEQS:-8}" \
  --max-num-batched-tokens "${FLINT_MAX_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization "${FLINT_GPU_MEMORY_UTILIZATION:-0.90}" \
  --attention-backend FLASH_ATTN_V100 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --host 0.0.0.0 \
  --port 8002
