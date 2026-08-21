# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
import torch


class _QuantConfig:
    pass


@pytest.mark.parametrize("num_speculative_tokens", [3, 4])
def test_sm70_mtp_enables_fused_gdn_projection_extract_by_default(
    num_speculative_tokens: int,
):
    from vllm.model_executor.models.qwen3_5 import (
        _sm70_gdn_fused_zba_extract_enabled,
    )

    speculative_config = Mock(
        method="mtp", num_speculative_tokens=num_speculative_tokens
    )
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("vllm.model_executor.models.qwen3_5.current_platform") as mock_platform,
    ):
        mock_platform.is_device_capability.return_value = True
        assert _sm70_gdn_fused_zba_extract_enabled(speculative_config)


def test_fused_gdn_projection_extract_default_scope():
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
            Mock(method="mtp", num_speculative_tokens=2)
        )
        assert not _sm70_gdn_fused_zba_extract_enabled(
            Mock(method="other", num_speculative_tokens=3)
        )
        mock_platform.is_device_capability.return_value = False
        assert not _sm70_gdn_fused_zba_extract_enabled(
            Mock(method="mtp", num_speculative_tokens=3)
        )


@pytest.mark.parametrize("num_speculative_tokens", [3, 4])
def test_fused_gdn_projection_extract_explicit_disable_wins(
    num_speculative_tokens: int,
):
    from vllm.model_executor.models.qwen3_5 import (
        _sm70_gdn_fused_zba_extract_enabled,
    )

    with patch.dict("os.environ", {"VLLM_SM70_GDN_FUSED_ZBA_EXTRACT": "0"}, clear=True):
        assert not _sm70_gdn_fused_zba_extract_enabled(
            Mock(method="mtp", num_speculative_tokens=num_speculative_tokens)
        )


def test_fused_gdn_projection_extract_explicit_enable_wins():
    from vllm.model_executor.models.qwen3_5 import (
        _sm70_gdn_fused_zba_extract_enabled,
    )

    with patch.dict("os.environ", {"VLLM_SM70_GDN_FUSED_ZBA_EXTRACT": "1"}, clear=True):
        assert _sm70_gdn_fused_zba_extract_enabled(None)


@pytest.mark.parametrize("num_tokens", [4, 128])
@pytest.mark.parametrize("padded_rows", [False, True])
def test_sm70_fused_gdn_projection_extract_cuda_graph_parity(
    num_tokens: int,
    padded_rows: bool,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("The fused extraction kernel requires an SM70 CUDA device")

    from vllm.model_executor.models.qwen3_5 import _sm70_gdn_extract_zba

    qkv_size = 512
    z_size = 192
    ba_size = 12
    mixed_width = qkv_size + z_size
    ba_width = ba_size * 2
    mixed_padding = 16 if padded_rows else 0
    ba_padding = 8 if padded_rows else 0
    device = torch.device("cuda")

    mixed_storage = torch.randn(
        num_tokens,
        mixed_width + mixed_padding,
        dtype=torch.float16,
        device=device,
    )
    ba_storage = torch.randn(
        num_tokens,
        ba_width + ba_padding,
        dtype=torch.float16,
        device=device,
    )
    mixed = mixed_storage[:, :mixed_width]
    ba = ba_storage[:, :ba_width]

    def reference():
        return (
            mixed[:, qkv_size:].contiguous(),
            ba[:, :ba_size].contiguous(),
            ba[:, ba_size:].contiguous(),
        )

    # Warm Triton compilation before graph capture.
    actual = _sm70_gdn_extract_zba(mixed, ba, qkv_size, z_size, ba_size)
    torch.cuda.synchronize()
    for result, expected in zip(actual, reference()):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _sm70_gdn_extract_zba(mixed, ba, qkv_size, z_size, ba_size)

    mixed_storage.normal_()
    ba_storage.normal_()
    graph.replay()
    torch.cuda.synchronize()
    for result, expected in zip(captured, reference()):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


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
