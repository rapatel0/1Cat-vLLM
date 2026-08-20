# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch


class _QuantConfig:
    pass


def test_sm70_mtp4_enables_fused_gdn_projection_extract_by_default():
    from vllm.model_executor.models.qwen3_5 import (
        _sm70_gdn_fused_zba_extract_enabled,
    )

    speculative_config = Mock(method="mtp", num_speculative_tokens=4)
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("vllm.model_executor.models.qwen3_5.current_platform") as mock_platform,
    ):
        mock_platform.is_device_capability.return_value = True
        assert _sm70_gdn_fused_zba_extract_enabled(speculative_config)


def test_fused_gdn_projection_extract_default_is_narrow_to_sm70_mtp4():
    from vllm.model_executor.models.qwen3_5 import (
        _sm70_gdn_fused_zba_extract_enabled,
    )

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("vllm.model_executor.models.qwen3_5.current_platform") as mock_platform,
    ):
        mock_platform.is_device_capability.return_value = True
        assert not _sm70_gdn_fused_zba_extract_enabled(None)
        assert not _sm70_gdn_fused_zba_extract_enabled(
            Mock(method="mtp", num_speculative_tokens=3)
        )
        mock_platform.is_device_capability.return_value = False
        assert not _sm70_gdn_fused_zba_extract_enabled(
            Mock(method="mtp", num_speculative_tokens=4)
        )


def test_fused_gdn_projection_extract_explicit_disable_wins():
    from vllm.model_executor.models.qwen3_5 import (
        _sm70_gdn_fused_zba_extract_enabled,
    )

    with patch.dict("os.environ", {"VLLM_SM70_GDN_FUSED_ZBA_EXTRACT": "0"}, clear=True):
        assert not _sm70_gdn_fused_zba_extract_enabled(
            Mock(method="mtp", num_speculative_tokens=4)
        )


def test_qwen3_5_split_gdn_detects_compressed_tensors_ignore():
    from vllm.model_executor.models.qwen3_5 import (
        _uses_split_gdn_input_projections,
    )

    quant_config = _QuantConfig()
    quant_config.ignore = [
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.linear_attn.in_proj_a",
    ]
    quant_config.config = {}

    assert _uses_split_gdn_input_projections(quant_config)


def test_qwen3_5_split_gdn_detects_compressed_tensors_config_ignore():
    from vllm.model_executor.models.qwen3_5 import (
        _uses_split_gdn_input_projections,
    )

    quant_config = _QuantConfig()
    quant_config.config = {
        "ignore": [
            "model.language_model.layers.0.linear_attn.in_proj_b",
            "model.language_model.layers.0.linear_attn.in_proj_a",
        ],
    }

    assert _uses_split_gdn_input_projections(quant_config)


def test_qwen3_5_lm_head_receives_quant_config():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config
    mock_vllm_config.lora_config = None

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5.Qwen3_5Model") as MockModel,
        patch("vllm.model_executor.models.qwen3_5.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()

        Qwen3_5ForCausalLMBase(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_qwen3_5_mtp_lm_head_receives_quant_config():
    from vllm.config import CompilationMode
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64
    mock_hf_config.quantization_config = None

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.model_config.hf_config = None
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.compilation_config.mode = CompilationMode.NONE
    mock_vllm_config.quant_config = mock_quant_config

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MultiTokenPredictor"),
        patch("vllm.model_executor.models.qwen3_5_mtp.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5_mtp.LogitsProcessor"),
        patch.dict("os.environ", {"VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS": "0"}),
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        Qwen3_5MTP(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config
