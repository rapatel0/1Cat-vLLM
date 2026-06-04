# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 assistant (MTP speculative-decoding drafter) configuration.

The ``google/gemma-4-*-it-assistant`` checkpoints declare
``model_type: gemma4_assistant`` / ``architectures: [Gemma4AssistantForCausalLM]``.
transformers <= 5.5 does not register this architecture (it is newer than the
gemma4 target model), so vLLM supplies this thin wrapper.

The nested ``text_config`` is the standard ``gemma4_text`` config, which *is*
known to transformers (the target model uses it), so we resolve it via the
transformers CONFIG_MAPPING rather than redefining the gemma4 text fields. The
wrapper only adds the MTP-specific fields the drafter needs:
``backbone_hidden_size`` (sizes the pre/post projections that bridge the draft
hidden dim to the target's last-layer hidden dim) and the optional
centroids-masking knobs (``use_ordered_embeddings`` is False for the 31B
assistant, so that path stays off).
"""

from transformers.configuration_utils import PretrainedConfig


def _gemma4_text_config_class():
    # Resolve lazily so import does not hard-depend on a particular
    # transformers layout; gemma4_text is registered by the target model.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    return CONFIG_MAPPING["gemma4_text"]


class Gemma4AssistantConfig(PretrainedConfig):
    model_type = "gemma4_assistant"

    def __init__(
        self,
        text_config=None,
        backbone_hidden_size=5376,
        use_ordered_embeddings=False,
        num_centroids=2048,
        centroid_intermediate_top_k=32,
        tie_word_embeddings=True,
        **kwargs,
    ):
        text_cls = _gemma4_text_config_class()
        if isinstance(text_config, dict):
            self.text_config = text_cls(**text_config)
        elif text_config is None:
            self.text_config = text_cls()
        else:
            self.text_config = text_config

        self.backbone_hidden_size = backbone_hidden_size
        self.use_ordered_embeddings = use_ordered_embeddings
        self.num_centroids = num_centroids
        self.centroid_intermediate_top_k = centroid_intermediate_top_k

        super().__init__(**kwargs)
        # Set after super().__init__(): transformers' PretrainedConfig.__init__
        # has tie_word_embeddings as an explicit param with a default that would
        # otherwise overwrite the checkpoint's value.
        self.tie_word_embeddings = tie_word_embeddings

    def get_text_config(self, *args, **kwargs):
        return self.text_config


__all__ = ["Gemma4AssistantConfig"]
