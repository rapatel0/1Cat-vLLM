# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 Unified (omni: text + vision + audio) configuration.

`google/gemma-4-12B-it` declares `model_type: gemma4_unified` /
`architectures: [Gemma4UnifiedForConditionalGeneration]` and targets
transformers 5.10. transformers 5.5 (what this fork ships) does NOT register
`gemma4_unified` or its `gemma4_unified_{text,vision,audio}` sub-configs, so
vLLM supplies thin local config classes.

The text sub-config is field-identical to the known `gemma4_text` (same head
dims, layer_types, p-RoPE, k_eq_v, PLE-off), so we subclass that. The vision and
audio embedders are **encoder-free** (no SigLIP ViT / audio tower — raw patches
projected to LM space), so their configs are small.
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

_Gemma4TextConfig = CONFIG_MAPPING["gemma4_text"]


class Gemma4UnifiedTextConfig(_Gemma4TextConfig):
    model_type = "gemma4_unified_text"


class Gemma4UnifiedVisionConfig(PretrainedConfig):
    model_type = "gemma4_unified_vision"

    def __init__(
        self,
        mm_embed_dim=3840,
        mm_posemb_size=1120,
        model_patch_size=48,
        patch_size=16,
        num_soft_tokens=280,
        output_proj_dims=3840,
        pooling_kernel_size=3,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        self.mm_embed_dim = mm_embed_dim
        self.mm_posemb_size = mm_posemb_size
        self.model_patch_size = model_patch_size
        self.patch_size = patch_size
        self.num_soft_tokens = num_soft_tokens
        self.output_proj_dims = output_proj_dims
        self.pooling_kernel_size = pooling_kernel_size
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class Gemma4UnifiedAudioConfig(PretrainedConfig):
    model_type = "gemma4_unified_audio"

    def __init__(
        self,
        audio_embed_dim=640,
        audio_samples_per_token=640,
        hidden_size=640,
        output_proj_dims=640,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        self.audio_embed_dim = audio_embed_dim
        self.audio_samples_per_token = audio_samples_per_token
        self.hidden_size = hidden_size
        self.output_proj_dims = output_proj_dims
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


def _build(cls, cfg):
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cls(**cfg)
    return cfg


class Gemma4UnifiedConfig(PretrainedConfig):
    model_type = "gemma4_unified"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        image_token_id=258880,
        audio_token_id=258881,
        video_token_id=None,
        boi_token_id=255999,
        eoi_token_id=258882,
        boa_token_id=None,
        eoa_token_index=None,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.text_config = _build(Gemma4UnifiedTextConfig, text_config) or (
            Gemma4UnifiedTextConfig()
        )
        self.vision_config = _build(Gemma4UnifiedVisionConfig, vision_config)
        self.audio_config = _build(Gemma4UnifiedAudioConfig, audio_config)
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.video_token_id = video_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.boa_token_id = boa_token_id
        self.eoa_token_index = eoa_token_index
        super().__init__(**kwargs)
        # Set after super().__init__() to avoid transformers' default overwrite.
        self.tie_word_embeddings = tie_word_embeddings

    def get_text_config(self, *args, **kwargs):
        return self.text_config


__all__ = [
    "Gemma4UnifiedConfig",
    "Gemma4UnifiedTextConfig",
    "Gemma4UnifiedVisionConfig",
    "Gemma4UnifiedAudioConfig",
]
