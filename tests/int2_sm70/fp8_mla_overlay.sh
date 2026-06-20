#!/bin/bash
# Enable fp8 latent KV cache for TRITON_MLA on the dev pod (1.1.0 image), by
# overlaying the worktree (1.2.0) MLA backend + decode kernel which carry the
# fp8 path (k_scale/v_scale dequant in-kernel, supported_kv_cache_dtypes incl
# fp8). The pod's stock TRITON_MLA raises NotImplementedError for fp8 and its
# decode kernel lacks the fp8 dequant. fp8 latent KV ~= 2x context vs fp16.
#
# Run from the repo root (needs the worktree files at vllm/...):
#   bash tests/int2_sm70/fp8_mla_overlay.sh        # copies + patches in pod
# (assumes worktree is mounted at /workspace in int2-dev)
set -e
SP=/opt/venv/lib/python3.12/site-packages/vllm
WT=/workspace/vllm

cp "$WT/v1/attention/backends/mla/triton_mla.py" "$SP/v1/attention/backends/mla/triton_mla.py"
cp "$WT/v1/attention/ops/triton_decode_attention.py" "$SP/v1/attention/ops/triton_decode_attention.py"

# 1.1.0 API drift: is_quantized_kv_cache lives in vllm.v1.attention.backend
# (the 1.2.0 worktree imports it from vllm.utils.torch_utils).
/opt/venv/bin/python - "$SP/v1/attention/backends/mla/triton_mla.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
if "from vllm.utils.torch_utils import is_quantized_kv_cache" in s:
    s = s.replace(
        "from vllm.utils.torch_utils import is_quantized_kv_cache\n", "")
    s = s.replace(
        "from vllm.v1.attention.backend import (\n",
        "from vllm.v1.attention.backend import (\n    is_quantized_kv_cache,\n", 1)
    open(p, "w").write(s)
    print("[fp8-mla-overlay] patched is_quantized_kv_cache import")
else:
    print("[fp8-mla-overlay] import already patched / not present")
# 1.1.0 platform lacks num_compute_units() (returns None -> TypeError). Fall
# back to V100's 80 SMs for the kv-split heuristic.
old = "self._sm_count = current_platform.num_compute_units()"
new = ('self._sm_count = (getattr(current_platform, "num_compute_units", None) '
       'and current_platform.num_compute_units()) or 80')
if old in s:
    s = s.replace(old, new); open(p, "w").write(s)
    print("[fp8-mla-overlay] patched _sm_count fallback")
# 1.1.0 envs lacks VLLM_BATCH_INVARIANT.
old = "if envs.VLLM_BATCH_INVARIANT:"
new = 'if getattr(envs, "VLLM_BATCH_INVARIANT", False):'
if old in s:
    s = s.replace(old, new); open(p, "w").write(s)
    print("[fp8-mla-overlay] patched VLLM_BATCH_INVARIANT")
PY
/opt/venv/bin/python -c "
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend
assert 'fp8_e4m3' in TritonMLABackend.supported_kv_cache_dtypes, 'fp8 not enabled'
print('[fp8-mla-overlay] OK — supported:', TritonMLABackend.supported_kv_cache_dtypes)
"
echo "[fp8-mla-overlay] done — run with KV_CACHE_DTYPE=fp8_e4m3"
