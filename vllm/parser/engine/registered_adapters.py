# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapters exposing ParserEngine parsers through the split registries.

These are created via :func:`make_adapters` and exposed as module-level
names so the tool/reasoning registries can load them lazily by name.

Trimmed to Inkling. Upstream registers an adapter pair per ParserEngine
parser, but this fork vendors only ``vllm/parser/inkling.py``; importing the
full upstream list would fail on the nine parser modules it does not carry
(deepseek_v4, deepseek_v32, gemma4, glm47_moe, kimi_k2, minimax_m2, mistral,
nemotron_v3, qwen3, seed_oss). Add entries here as those parsers are vendored.
"""

from vllm.parser.engine.adapters import make_adapters
from vllm.parser.inkling import InklingParser

(
    InklingParserReasoningAdapter,
    InklingParserToolAdapter,
) = make_adapters(InklingParser)
