# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling MTP proposer: drafting across multiple KV cache groups.

Inkling's drafter is hybrid. Every MTP depth is a full Inkling transformer
block, so alongside its attention layer it owns an ``InklingConvState``
short-conv cache, registered with a different spec (``block_size`` =
``sconv_kernel_size`` = 4, against the attention block size). The checkpoint
confirms this is not optional -- each depth ships ``k_sconv``, ``v_sconv``,
``attn_sconv`` and ``mlp_sconv`` weights -- so the draft cannot simply drop the
conv the way Qwen3-Next's MTP pins its block to ``layer_type="full_attention"``
and leaves its linear-attention state out.

The draft layers therefore span two KV cache groups, which
``SpecDecodeBaseProposer`` rejects ("All drafting layers should belong to the
same kv cache group") because it builds one ``AttentionMetadata`` for all of
them.

``Step3p5MTPProposer`` already implements exactly the machinery that requires:
per-group ``AttentionGroup``s, per-group block tables and slot mappings, and
per-group metadata building. None of it is Step3.5-specific except the two
hooks overridden below, so it is reused rather than copied. That shared part
would be worth lifting into a common base for the next hybrid drafter.
"""

from __future__ import annotations

import torch.nn as nn

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer


class InklingMTPProposer(Step3p5MTPProposer):
    """Multi-KV-cache-group MTP proposer for Inkling."""

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        # Step3.5 suppresses this because its checkpoint carries a per-MTP-layer
        # head. Inkling's draft shares the target's token embedding table and LM
        # head (see models/inkling/nvidia/mtp.py), so the base behaviour is what
        # is wanted here, not Step3.5's.
        SpecDecodeBaseProposer._maybe_share_lm_head(self, target_language_model)
