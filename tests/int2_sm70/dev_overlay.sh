#!/bin/bash
# Apply the int2_sm70 quant method as an overlay onto the installed (image) vLLM
# in the dev pod, so we iterate without a full rebuild. Idempotent.
#
#   kubectl -n llm exec int2-dev -- bash /workspace/tests/int2_sm70/dev_overlay.sh
set -e
SP=/opt/venv/lib/python3.12/site-packages/vllm
QDIR=$SP/model_executor/layers/quantization

# 1. drop in the quant method
cp /workspace/vllm/model_executor/layers/quantization/int2_sm70.py "$QDIR/"

# 2. register it lazily in the resolver (avoids the import-time circular import:
#    int2_sm70 -> linear -> config). Done by (a) allowing the name and (b)
#    importing the module right before the custom-registry merge so the
#    @register_quantization_config decorator populates the map.
/opt/venv/bin/python - "$QDIR/__init__.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a1 = 'QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))'
if 'int2_sm70 registration' not in s:
    s = s.replace(a1, a1 + '\nif "int2_sm70" not in QUANTIZATION_METHODS:  # int2_sm70 registration\n    QUANTIZATION_METHODS.append("int2_sm70")')
a2 = '    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)'
if 'int2_sm70 lazy-populate' not in s:
    s = s.replace(a2, '    if quantization == "int2_sm70":  # int2_sm70 lazy-populate\n        from . import int2_sm70 as _i2  # noqa: F401\n' + a2)
open(p, 'w').write(s)
print('[overlay] int2_sm70 registered in', p)
PY
echo "[overlay] done"
