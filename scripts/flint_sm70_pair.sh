#!/bin/bash
# Two independent TP2 Flint instances across Island B: GPUs 4-5 and GPUs 6-7.
# Each instance keeps its own KV cache and CUDA graphs, so a coding-agent fleet
# gets two schedulers instead of one. Ports 18002 and 18003.
set -euo pipefail

IMAGE=localhost:32000/onecat-vllm:flint-sm70-6d421910
OVERLAY=/localpool/flint-sm70/src/overlay
UUID_A='GPU-c1aa8bd9-f642-328d-96f8-79d7c38ce61e,GPU-c275f176-56b9-eb8d-14d7-a3fd15a0ec03'
UUID_B='GPU-d5302639-be81-61f0-5e1a-b8f421bb100b,GPU-dd6f7287-63a3-17c4-64c2-1eb597391f4b'
CPUSET_A='20-29,60-69'
CPUSET_B='30-39,70-79'
SEQS="${1:-16}"
shift || true
EXTRA=("$@")

mounts=(
  --mount type=bind,source=/localpool/flint-sm70/models/flint-nvfp4-gguf,target=/models/flint-nvfp4-gguf,readonly
  --mount type=bind,source=/tmp/flint_launch.sh,target=/opt/flint_sm70_launch.sh,readonly
  --mount type=bind,source=$OVERLAY/vllm/models/flint/ops_moe.py,target=/usr/local/lib/python3.12/dist-packages/vllm/models/flint/ops/moe.py,readonly
  --mount type=bind,source=$OVERLAY/vllm/models/flint/model.py,target=/usr/local/lib/python3.12/dist-packages/vllm/models/flint/model.py,readonly
  --mount type=bind,source=$OVERLAY/vllm/v1/attention/backends/flash_v100/impl.py,target=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_v100/impl.py,readonly
  --mount type=bind,source=$OVERLAY/vllm/v1/attention/backends/flash_v100/decode.py,target=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_v100/decode.py,readonly
  --mount type=bind,source=$OVERLAY/vllm/v1/attention/backends/flash_v100/prefill.py,target=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_v100/prefill.py,readonly
  --mount type=bind,source=$OVERLAY/vllm/v1/attention/backends/flash_v100/prefill_candidates.py,target=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_v100/prefill_candidates.py,readonly
  --mount type=bind,source=$OVERLAY/vllm/v1/attention/backends/flash_v100/sinks.py,target=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_v100/sinks.py,readonly
  --mount type=bind,source=$OVERLAY/vllm/v1/attention/backends/flash_v100/dense_prefill.py,target=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_v100/dense_prefill.py,readonly
  --mount type=bind,source=$OVERLAY/flash_attn_v100/flash_attn_interface.py,target=/usr/local/lib/python3.12/dist-packages/flash_attn_v100/flash_attn_interface.py,readonly
)

launch() {
  local name="$1" uuids="$2" host_port="$3" cpuset="$4"
  sudo -n docker rm -f "$name" >/dev/null 2>&1
  sudo -n docker run -d --name "$name" \
    --runtime nvidia --shm-size=16g --entrypoint bash \
    -p "$host_port:8002" \
    -e SWARM_CANARY_GPU_LEASE_ACK=1 \
    -e NVIDIA_VISIBLE_DEVICES="$uuids" \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e FLASH_ATTN_V100=1 -e VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
    -e VLLM_SM70_FLASH_ATTN_V100=1 -e VLLM_SM70_NVFP4_TURBOMIND=1 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e HF_HUB_OFFLINE=1 \
    -e VLLM_KV_CACHE_LAYOUT=NHD -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e TOKENIZERS_PARALLELISM=false -e FLINT_TP_SIZE=2 \
    -e FLINT_KV_CACHE_DTYPE=fp8_e5m2 -e FLINT_MAX_NUM_SEQS="$SEQS" \
    -e FLINT_MAX_BATCHED_TOKENS=${BATCHED:-8192} \
    -e FLINT_GPU_MEMORY_UTILIZATION=${MEMUTIL:-0.88} \
    -e FLINT_PORT=8002 -e FLINT_CPUSET="$cpuset" \
    -e "FLINT_COMPILATION_CONFIG={\"cudagraph_capture_sizes\":[1,2,3,4,6,8,12,16]}" \
    "${EXTRA[@]}" "${mounts[@]}" \
    "$IMAGE" /opt/flint_sm70_launch.sh >/dev/null
}

# The single-instance container holds port 18002 and all four GPUs.
sudo -n docker rm -f flint-sm70 >/dev/null 2>&1

launch flint-a "$UUID_A" 18002 "$CPUSET_A"
launch flint-b "$UUID_B" 18003 "$CPUSET_B"

for i in $(seq 1 30); do
  sleep 15
  ok=0
  for port in 18002 18003; do
    curl -sf -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null && ok=$((ok + 1))
  done
  if [[ $ok -eq 2 ]]; then
    echo "both UP ~$((i * 15))s"
    exit 0
  fi
done
echo "timeout waiting for both instances" >&2
exit 1
