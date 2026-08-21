# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch


def _config(**overrides):
    values = {
        "model_config": SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3_5_text"),
            quantization="fp8",
        ),
        "speculative_config": SimpleNamespace(method="mtp", num_speculative_tokens=3),
        "parallel_config": SimpleNamespace(
            tensor_parallel_size=8, decode_context_parallel_size=2
        ),
        "scheduler_config": SimpleNamespace(max_num_seqs=32),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen35_mtp3_fp8_verifier_tune_scope():
    from vllm.config.vllm import _sm70_qwen35_mtp3_fp8_verifier_tune_m

    config = _config()
    assert _sm70_qwen35_mtp3_fp8_verifier_tune_m(config) == 128

    config.scheduler_config.max_num_seqs = 31
    assert _sm70_qwen35_mtp3_fp8_verifier_tune_m(config) is None
    config.scheduler_config.max_num_seqs = 32

    config.speculative_config.num_speculative_tokens = 4
    assert _sm70_qwen35_mtp3_fp8_verifier_tune_m(config) is None
    config.speculative_config.num_speculative_tokens = 3

    config.parallel_config.tensor_parallel_size = 4
    assert _sm70_qwen35_mtp3_fp8_verifier_tune_m(config) is None


def test_fp8_warmup_adds_only_captured_tune_endpoint():
    from vllm.model_executor.warmup.awq_sm70_warmup import (
        _get_decode_m_values,
        _get_fp8_dense_m_values,
        _should_warmup_fp8_dense_shape,
    )

    compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[1, 2, 4, 8, 12, 16, 128]
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=3),
            compilation_config=compilation_config,
        )
    )
    with patch.dict(
        "os.environ",
        {
            "VLLM_SM70_AWQ_WARMUP_MAX_M": "16",
            "VLLM_SM70_FP8_DENSE_TUNE_MAX_M": "128",
        },
    ):
        assert 128 not in _get_decode_m_values(worker)
        assert _get_fp8_dense_m_values(worker) == [1, 2, 4, 8, 12, 16, 128]

        compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8, 12, 16]
        assert _get_fp8_dense_m_values(worker) == [1, 2, 4, 8, 12, 16]

        assert _should_warmup_fp8_dense_shape(128, 2176, 5120, False)
        assert _should_warmup_fp8_dense_shape(128, 768, 5120, False)
        assert not _should_warmup_fp8_dense_shape(128, 5120, 4352, True)
        assert not _should_warmup_fp8_dense_shape(128, 5120, 2560, False)
        assert _should_warmup_fp8_dense_shape(16, 5120, 2560, False)
