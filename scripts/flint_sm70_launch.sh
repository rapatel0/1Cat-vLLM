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
# A layout may claim part of the island, so every visible device must belong to
# Island B and the count must match the requested tensor-parallel width.
tp_size="${FLINT_TP_SIZE:-4}"
# The NVIDIA runtime rewrites NVIDIA_VISIBLE_DEVICES to "void" once it has
# claimed the devices, so the live device list is the only reliable count.
mapfile -t gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d ' \r' | sort)
if [[ ${#gpu_uuids[@]} -ne "$tp_size" ]]; then
  echo "Flint expects $tp_size GPUs, saw ${#gpu_uuids[@]}." >&2
  exit 64
fi
for uuid in "${gpu_uuids[@]}"; do
  case ",$expected," in
    *",$uuid,"*) ;;
    *)
      printf 'GPU %q is outside Island B.\n' "$uuid" >&2
      exit 64
      ;;
  esac
done

MODEL=${FLINT_MODEL_DIR:-/models/flint-nvfp4-gguf}
# FP8 KV halves the cache and doubles full-context concurrency on V100.
KV_CACHE_DTYPE=${FLINT_KV_CACHE_DTYPE:-fp8_e5m2}
MAX_NUM_SEQS=${FLINT_MAX_NUM_SEQS:-16}

args=(
  --model "$MODEL"
  --tokenizer "$MODEL"
  --served-model-name flint
  --trust-remote-code
  --dtype float16
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --tensor-parallel-size "${FLINT_TP_SIZE:-4}"
  --max-model-len "${FLINT_MAX_MODEL_LEN:-262144}"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "${FLINT_MAX_BATCHED_TOKENS:-8192}"
  --gpu-memory-utilization "${FLINT_GPU_MEMORY_UTILIZATION:-0.90}"
  --attention-backend FLASH_ATTN_V100
  --enable-prefix-caching
  --enable-auto-tool-choice
  --tool-call-parser glm47
  --host 0.0.0.0
  --port "${FLINT_PORT:-8002}"
)

# Graph capture sizes must cover the batch ceiling, or larger batches decode eagerly.
if [[ -n "${FLINT_COMPILATION_CONFIG:-}" ]]; then
  args+=(--compilation-config "$FLINT_COMPILATION_CONFIG")
fi

# Flint keeps 256 experts with top-6 routing, so sharding experts across ranks
# instead of splitting each expert's hidden width can move MoE-heavy decode.
if [[ "${FLINT_EXPERT_PARALLEL:-0}" == 1 ]]; then
  args+=(--enable-expert-parallel)
fi

exec taskset -c "${FLINT_CPUSET:-20-39,60-79}" \
  python3 -m vllm.entrypoints.openai.api_server "${args[@]}"
