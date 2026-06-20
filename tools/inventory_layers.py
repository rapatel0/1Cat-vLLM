#!/usr/bin/env python3
"""Inventory GLM-5.2-FP8 layer types + dtypes -> decide 2-bit / fp8 / fp16."""
import glob
import os
import re
import sys
from collections import defaultdict
from safetensors import safe_open

root = sys.argv[1] if len(sys.argv) > 1 else "/models/GLM-5.2-FP8"
cat = defaultdict(lambda: defaultdict(lambda: [0, 0]))


def categorize(k):
    if ".mlp.experts." in k:
        return "routed_experts (gate/up/down)"
    if ".mlp.shared_experts." in k:
        return "shared_experts"
    if ".mlp.gate." in k:
        return "router gate"
    if re.search(r"\.mlp\.(gate|up|down)_proj", k):
        return "dense_mlp (layers 0-2)"
    if ".self_attn.indexer." in k:
        return "indexer (DROPPED on V100)"
    if ".self_attn." in k and "layernorm" in k:
        return "attn layernorm"
    if ".self_attn." in k:
        return "attention proj (q/kv/o)"
    if "layernorm" in k or k.endswith(".norm.weight"):
        return "layernorm"
    if "embed_tokens" in k:
        return "embed_tokens"
    if "lm_head" in k:
        return "lm_head"
    return "other:" + k.split(".")[-1]


for sh in sorted(glob.glob(os.path.join(root, "model-*-of-*.safetensors"))):
    with safe_open(sh, "pt") as f:
        for k in f.keys():
            if k.endswith("_scale_inv"):
                continue
            t = f.get_slice(k)
            n = 1
            for d in t.get_shape():
                n *= d
            cat[categorize(k)][str(t.get_dtype())][0] += 1
            cat[categorize(k)][str(t.get_dtype())][1] += n

print("%-34s %-9s %8s %10s" % ("category", "dtype", "tensors", "params(B)"))
print("-" * 65)
tot = defaultdict(float)
for c in sorted(cat):
    for dt, (cnt, pn) in sorted(cat[c].items()):
        print("%-34s %-9s %8d %10.3f" % (c, dt, cnt, pn / 1e9))
        tot[dt] += pn
print("-" * 65)
for dt, pn in sorted(tot.items()):
    print("%-34s %-9s %8s %10.3f" % ("TOTAL", dt, "", pn / 1e9))
