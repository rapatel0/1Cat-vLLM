# SPDX-License-Identifier: Apache-2.0
"""CPU check for the FlashAttention learned-sink correction."""

import importlib.util
from pathlib import Path

import torch

_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm/v1/attention/backends/flash_v100/sinks.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("flash_v100_sinks_under_test", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sink_correction_matches_extra_logit():
    sinks_mod = _load()
    # Two keys with scores 0 and 0, value basis e0 and e1, plus a sink of 0.
    # Without the sink, output is 0.5 * e0 + 0.5 * e1. With sink logit 0 and
    # value 0, each key weight becomes 1/3.
    output = torch.tensor([[[0.5, 0.5]]], dtype=torch.float16)
    lse = torch.tensor([[0.693147]], dtype=torch.float32)  # log(2)
    sinks = torch.tensor([0.0], dtype=torch.float32)
    sinks_mod.apply_learned_attention_sinks(output, lse, sinks)
    expected = 2.0 / 3.0 * 0.5
    assert torch.allclose(output.float(), torch.tensor([[[expected, expected]]]), atol=1e-3)
