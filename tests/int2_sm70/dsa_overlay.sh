#!/bin/bash
# Overlay GLM-5.2 (glm_moe_dsa) support onto the installed (image) vLLM in the
# dev pod, WITHOUT a rebuild. Idempotent. Mirrors the worktree edits to
# deepseek_v2.py / registry.py / config.py.
#
# What it adds:
#   1. GlmMoeDsaForCausalLM class (alias of DeepseekV2ForCausalLM) + registry
#   2. glm_moe_dsa -> DeepseekV3Config registration (transformers<5.12 shim)
#   3. dense-MLA sm70 fallback: _dsa_runtime_enabled() gates the DSA sparse
#      indexer on DeepGEMM availability; on V100 -> dense MLA, indexer weights
#      skipped on load.
#
#   kubectl -n llm exec int2-dev -- bash /workspace/tests/int2_sm70/dsa_overlay.sh
set -e
SP=/opt/venv/lib/python3.12/site-packages/vllm

/opt/venv/bin/python - "$SP" <<'PY'
import sys, re
sp = sys.argv[1]

# ---- 1/3: deepseek_v2.py -------------------------------------------------
p = f"{sp}/model_executor/models/deepseek_v2.py"
s = open(p).read()
orig = s

if "import os\nimport typing" not in s:
    s = s.replace("import typing", "import os\nimport typing", 1)

imp = "from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer"
if "from vllm.utils.deep_gemm import has_deep_gemm" not in s:
    s = s.replace(imp, imp + "\nfrom vllm.utils.deep_gemm import has_deep_gemm", 1)

helper = '''

def _dsa_runtime_enabled(config) -> bool:
    """Run the DSA *sparse* path only when its kernels exist (DeepGEMM/sm90);
    else dense MLA (sm70). Dense attention is a superset of the indexer's top-k
    selection, so correctness is preserved. VLLM_SM70_DSA_DENSE=1 forces dense."""
    if not hasattr(config, "index_topk"):
        return False
    force = os.environ.get("VLLM_SM70_DSA_DENSE")
    if force is not None:
        return force == "0"
    return has_deep_gemm()
'''
anchor = "logger = init_logger(__name__)"
if "_dsa_runtime_enabled" not in s:
    s = s.replace(anchor, anchor + "\n" + helper, 1)

s = s.replace('self.is_v32 = hasattr(config, "index_topk")',
              'self.is_v32 = _dsa_runtime_enabled(config)')

# load_weights: skip indexer ckpt weights when running the dense fallback
lw_anchor = "        params_dict = dict(self.named_parameters())\n"
fb = ('        _dsa_dense_fallback = hasattr(self.config, "index_topk") and '
      'not getattr(\n            self.model, "is_v32", True\n        )\n')
if "_dsa_dense_fallback" not in s:
    s = s.replace(lw_anchor, lw_anchor + fb, 1)
loop_anchor = ('        for name, loaded_weight in weights:\n'
               '            if "rotary_emb.inv_freq" in name:\n'
               '                continue\n')
skip = ('\n            if _dsa_dense_fallback and ".indexer." in name:\n'
        '                continue\n')
if ".indexer." not in s.split("def load_weights", 1)[-1][:2000] or skip not in s:
    if skip not in s:
        s = s.replace(loop_anchor, loop_anchor + skip, 1)

if "class GlmMoeDsaForCausalLM" not in s:
    s = s.rstrip() + (
        "\n\n\nclass GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):\n"
        "    # GLM-5.x DSA: same MLA+DSA+MoE stack as DeepSeek-V3.2.\n"
        "    pass\n")

if s != orig:
    open(p, "w").write(s)
    print("[dsa-overlay] patched deepseek_v2.py")
else:
    print("[dsa-overlay] deepseek_v2.py already patched")

# ---- 2/3: registry.py ----------------------------------------------------
p = f"{sp}/model_executor/models/registry.py"
s = open(p).read()
line = '    "DeepseekV32ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),\n'
add = '    "GlmMoeDsaForCausalLM": ("deepseek_v2", "GlmMoeDsaForCausalLM"),\n'
if "GlmMoeDsaForCausalLM" not in s:
    s = s.replace(line, line + add, 1)
    open(p, "w").write(s)
    print("[dsa-overlay] patched registry.py")
else:
    print("[dsa-overlay] registry.py already has GlmMoeDsa")

# ---- 3/3: config.py ------------------------------------------------------
p = f"{sp}/transformers_utils/config.py"
s = open(p).read()
line = '    deepseek_v32="DeepseekV3Config",\n'
add = '    glm_moe_dsa="DeepseekV3Config",\n'
if "glm_moe_dsa=" not in s:
    s = s.replace(line, line + add, 1)
    open(p, "w").write(s)
    print("[dsa-overlay] patched config.py")
else:
    print("[dsa-overlay] config.py already has glm_moe_dsa")

# ---- 4/4: is_deepseek_mla — flip use_mla=True so the model uses the MLA-native
#           backend (TRITON_MLA, latent KV) instead of FLASH_ATTN_V100 (which
#           materializes full multi-head KV → ~7x more KV memory). GLM-5.2 has
#           kv_lora_rank=512 so the method's kv_lora_rank check passes.
p = f"{sp}/transformers_utils/model_arch_config_convertor.py"
s = open(p).read()
if '"glm_moe_dsa",' not in s:
    s = s.replace('            "deepseek_v32",\n',
                  '            "deepseek_v32",\n            "glm_moe_dsa",\n', 1)
    open(p, "w").write(s)
    print("[dsa-overlay] patched model_arch_config_convertor.py (use_mla)")
else:
    print("[dsa-overlay] convertor already has glm_moe_dsa")
PY
echo "[dsa-overlay] done"
