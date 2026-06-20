#!/bin/bash
# Bring up the GLM-5.2 int2 OpenAI-compatible API server on 8xV100.
# TP4xPP2 (NVLink-island map) + fp8 e5m2 latent KV + small logits batch.
#   kubectl -n llm exec int2-dev -- bash /workspace/tests/int2_sm70/serve.sh
set -e
export CUDA_HOME=/workspace/cuda128 PATH=/workspace/cuda128/bin:$PATH
export TORCH_CUDA_ARCH_LIST="7.0" TORCH_EXTENSIONS_DIR=/workspace/torch_ext
export LD_LIBRARY_PATH=/workspace/cuda128/nvvm/lib64:$LD_LIBRARY_PATH
export INT2_SM70_OP_SRC=/workspace/csrc/sm70_int2/int2_gemv_op.cu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export INT2_QUANT=int2_sm70 INT2_PACKED_MOE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL=/workspace/GLM-5.2-int2
PORT="${PORT:-8000}"

exec /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name glm-5.2-int2 \
  --quantization int2_sm70 \
  --tensor-parallel-size "${TP:-4}" \
  --pipeline-parallel-size "${PP:-2}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e5m2}" \
  --max-model-len "${MAX_MODEL_LEN:-24576}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-512}" \
  --gpu-memory-utilization "${GMU:-0.96}" \
  --enforce-eager \
  --trust-remote-code \
  --host 0.0.0.0 --port "$PORT"
