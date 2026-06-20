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
# V100 Triton supports fp8 e5m2 ('fp8e5') but NOT e4m3 ('fp8e4nv' -> Hopper).
# Add fp8_e5m2 to the supported list and USE IT (KV_CACHE_DTYPE=fp8_e5m2).
/opt/venv/bin/python - "$SP/v1/attention/backends/mla/triton_mla.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
if '"fp8_e5m2"' not in s:
    s = s.replace('"fp8_e4m3",\n', '"fp8_e4m3",\n        "fp8_e5m2",\n', 1)
    open(p, "w").write(s)
    print("[fp8-mla-overlay] added fp8_e5m2 to supported list")
PY
/opt/venv/bin/python -c "
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend
assert 'fp8_e5m2' in TritonMLABackend.supported_kv_cache_dtypes, 'fp8_e5m2 not enabled'
print('[fp8-mla-overlay] OK — supported:', TritonMLABackend.supported_kv_cache_dtypes)
"

# --- MLA fp8 path: route e5m2 (not e4m3) through cache view + bf16 query ------
# mla_attention.py hardwires e4m3 via current_platform.fp8_dtype() in 2 spots.
# For e5m2 (the V100-capable fp8): view the cache as e5m2, and keep the query
# in bf16 (skip the fp8 query-quant) so the kernel does bf16-q x fp8e5-KV.
/opt/venv/bin/python - "$SP/model_executor/layers/attention/mla_attention.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read(); n = 0
old1 = "            kv_cache = kv_cache.view(current_platform.fp8_dtype())"
new1 = ('            _fp8dt = (torch.float8_e5m2 if self.kv_cache_dtype == "fp8_e5m2"\n'
        '                      else current_platform.fp8_dtype())\n'
        '            kv_cache = kv_cache.view(_fp8dt)')
if old1 in s and "_fp8dt" not in s:
    s = s.replace(old1, new1, 1); n += 1
old2 = "            if fp8_attention:\n                assert mqa_ql_nope.shape[0] == mqa_q_pe.shape[0]"
new2 = '            if fp8_attention and self.kv_cache_dtype != "fp8_e5m2":\n                assert mqa_ql_nope.shape[0] == mqa_q_pe.shape[0]'
if old2 in s:
    s = s.replace(old2, new2, 1); n += 1
open(p, "w").write(s)
print(f"[fp8-mla-overlay] mla_attention e5m2 patches: {n}/2")
PY

# --- decode kernel: fit V100's 96KB shared memory ----------------------------
# The grouped MLA decode kernel (BLOCK_DMODEL=512 + BLOCK_DPE=64) needs 100KB
# smem at BLOCK=32; V100 has 96KB. Halve the KV iteration tile -> ~55KB.
/opt/venv/bin/python - "$SP/v1/attention/ops/triton_decode_attention.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "    BLOCK = 32\n    if is_hip_:\n        BLOCK = 16\n"
new = "    BLOCK = 16  # V100: 96KB smem (was 32)\n    if is_hip_:\n        BLOCK = 16\n"
if old in s:
    s = s.replace(old, new, 1); open(p, "w").write(s)
    print("[fp8-mla-overlay] decode-kernel BLOCK 32 -> 16 (V100 smem)")
else:
    print("[fp8-mla-overlay] decode BLOCK already patched")
PY
echo "[fp8-mla-overlay] done — run with KV_CACHE_DTYPE=fp8_e5m2 (e4m3 unsupported on sm70)"
