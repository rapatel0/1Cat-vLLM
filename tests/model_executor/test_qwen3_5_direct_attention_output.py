# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class _CopyCounter(TorchDispatchMode):
    def __init__(self) -> None:
        self.count = 0
        self.allocations = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.copy_.default:
            self.count += 1
        if func is torch.ops.aten.empty.memory_format:
            self.allocations += 1
        return func(*args, **(kwargs or {}))


class _Projection:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.calls = 0

    def __call__(self, _input: torch.Tensor):
        self.calls += 1
        return self.output, None


def _full_attention_harness(projection: _Projection):
    qkv = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    return SimpleNamespace(
        qkv_proj=lambda _hidden: (qkv, None),
        layer_idx=1,
        attn_output_gate=False,
        q_size=2,
        kv_size=2,
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
        q_norm=lambda value: value,
        k_norm=lambda value: value,
        rotary_emb=lambda _positions, q, k: (q, k),
        attn=lambda _q, _k, v: v,
        o_proj=projection,
    )


def _gdn_projection_harness(projection: _Projection):
    return SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        _compute_output_projection=lambda _core, _z, _tokens: projection(_core)[0],
    )


def _run_full_attention(layer, output: torch.Tensor | None) -> torch.Tensor:
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

    return Qwen3NextAttention.forward(
        layer,
        positions=torch.arange(4),
        output=output,
        hidden_states=torch.zeros(4, 8),
    )


def _run_gdn_projection(layer, output: torch.Tensor | None) -> torch.Tensor:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    return QwenGatedDeltaNetAttention._output_projection(
        layer,
        core_attn_out=torch.zeros(4, 2),
        z=torch.zeros(4, 2),
        output=output,
        num_tokens=4,
    )


def test_sm70_mtp3_direct_attention_output_scope():
    from vllm.model_executor.models.qwen3_5 import (
        _sm70_mtp3_direct_attention_output_enabled,
    )

    config = Mock()
    config.speculative_config = Mock(method="mtp", num_speculative_tokens=3)
    enabled_env = {
        "VLLM_SM70_MTP3_DIRECT_ATTENTION_OUTPUT": "1",
        "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH": "1",
    }
    with (
        patch.dict("os.environ", enabled_env, clear=True),
        patch("vllm.model_executor.models.qwen3_5.current_platform") as platform,
    ):
        platform.is_device_capability.return_value = True
        assert _sm70_mtp3_direct_attention_output_enabled(config)

        for depth in (1, 2, 4):
            config.speculative_config.num_speculative_tokens = depth
            assert not _sm70_mtp3_direct_attention_output_enabled(config)
        config.speculative_config = None
        assert not _sm70_mtp3_direct_attention_output_enabled(config)


def test_sm70_mtp3_direct_attention_output_fallbacks():
    from vllm.model_executor.models.qwen3_5 import (
        _sm70_mtp3_direct_attention_output_enabled,
    )

    config = Mock()
    config.speculative_config = Mock(method="mtp", num_speculative_tokens=3)
    base_env = {
        "VLLM_SM70_MTP3_DIRECT_ATTENTION_OUTPUT": "1",
        "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH": "1",
    }
    with patch("vllm.model_executor.models.qwen3_5.current_platform") as platform:
        platform.is_device_capability.return_value = True
        with patch.dict("os.environ", {}, clear=True):
            assert not _sm70_mtp3_direct_attention_output_enabled(config)
        with patch.dict(
            "os.environ",
            {"VLLM_SM70_MTP3_DIRECT_ATTENTION_OUTPUT": "1"},
            clear=True,
        ):
            assert not _sm70_mtp3_direct_attention_output_enabled(config)
        for debug_env in (
            {"VLLM_SM70_DUMP_QWEN_LAYER_DIR": "/tmp/qwen"},
            {"VLLM_SM70_DUMP_GDN_PROJ_DIR": "/tmp/gdn"},
            {"VLLM_SM70_DUMP_GDN_GRAPH_BUFFERS": "1"},
        ):
            with patch.dict("os.environ", base_env | debug_env, clear=True):
                assert not _sm70_mtp3_direct_attention_output_enabled(config)
        platform.is_device_capability.return_value = False
        with patch.dict("os.environ", base_env, clear=True):
            assert not _sm70_mtp3_direct_attention_output_enabled(config)


def test_full_attention_direct_output_matches_preallocated_buffer():
    projection_output = torch.randn(4, 8)
    projection = _Projection(projection_output)
    layer = _full_attention_harness(projection)

    direct = _run_full_attention(layer, None)
    preallocated = torch.empty_like(projection_output)
    fallback = _run_full_attention(layer, preallocated)

    assert direct.data_ptr() == projection_output.data_ptr()
    assert fallback.data_ptr() == preallocated.data_ptr()
    assert fallback.data_ptr() != direct.data_ptr()
    torch.testing.assert_close(direct, fallback, rtol=0, atol=0)
    assert projection.calls == 2


def test_gdn_direct_output_matches_preallocated_buffer_and_padding():
    projection_output = torch.randn(4, 8)
    projection = _Projection(projection_output)
    layer = _gdn_projection_harness(projection)
    padded_storage = torch.full((7, 10), -17.0)
    padded_output = padded_storage[:, :8]

    with patch(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
        "_sm70_gdn_graph_buffer_copy"
    ):
        direct = _run_gdn_projection(layer, None)
        returned = _run_gdn_projection(layer, padded_output)

    assert direct.data_ptr() == projection_output.data_ptr()
    assert returned.data_ptr() == projection_output.data_ptr()
    torch.testing.assert_close(padded_output[:4], direct, rtol=0, atol=0)
    assert torch.all(padded_output[4:] == -17.0)
    assert projection.calls == 2


def test_direct_route_removes_64_copies_without_more_projections():
    """Instrument the 48 GDN and 16 full-attention projection boundaries."""
    gdn_projection = _Projection(torch.randn(4, 8))
    full_projection = _Projection(torch.randn(4, 8))
    gdn_layer = _gdn_projection_harness(gdn_projection)
    full_layer = _full_attention_harness(full_projection)

    fallback_copies = _CopyCounter()
    with (
        patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
            "_sm70_gdn_graph_buffer_copy"
        ),
        fallback_copies,
    ):
        for _ in range(48):
            _run_gdn_projection(gdn_layer, torch.empty(4, 8))
        for _ in range(16):
            _run_full_attention(full_layer, torch.empty(4, 8))

    direct_gdn_projection = _Projection(torch.randn(4, 8))
    direct_full_projection = _Projection(torch.randn(4, 8))
    direct_gdn_layer = _gdn_projection_harness(direct_gdn_projection)
    direct_full_layer = _full_attention_harness(direct_full_projection)
    direct_copies = _CopyCounter()
    with (
        patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
            "_sm70_gdn_graph_buffer_copy"
        ),
        direct_copies,
    ):
        for _ in range(48):
            _run_gdn_projection(direct_gdn_layer, None)
        for _ in range(16):
            _run_full_attention(direct_full_layer, None)

    assert fallback_copies.count == 64
    assert direct_copies.count == 0
    assert fallback_copies.allocations == 64
    assert direct_copies.allocations == 0
    assert gdn_projection.calls + full_projection.calls == 64
    assert direct_gdn_projection.calls + direct_full_projection.calls == 64


def test_qwen_gdn_direct_custom_op_returns_projection_allocation():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        qwen_gdn_full_forward_direct_fake,
    )

    hidden_states = torch.randn(4, 8)
    projection = torch.randn(4, 8)
    layer = SimpleNamespace(
        _forward_method=Mock(return_value=projection),
    )
    context = SimpleNamespace(no_compile_layers={"model.layers.0.linear_attn": layer})
    conv_cache = torch.empty(0)
    ssm_cache = torch.empty(0)
    with (
        patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
            "get_forward_context",
            return_value=context,
        ),
        patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
            "_log_runtime_route_once"
        ),
    ):
        output = torch.ops.vllm.qwen_gdn_full_forward_direct(
            hidden_states,
            conv_cache,
            ssm_cache,
            "model.layers.0.linear_attn",
        )

    assert output.data_ptr() == projection.data_ptr()
    layer._forward_method.assert_called_once_with(hidden_states, None)
    fake_output = qwen_gdn_full_forward_direct_fake(
        hidden_states,
        conv_cache,
        ssm_cache,
        "model.layers.0.linear_attn",
    )
    assert fake_output.shape == hidden_states.shape
    assert fake_output.is_contiguous()
    assert fake_output.data_ptr() != hidden_states.data_ptr()

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as fake_mode:
        fake_hidden = fake_mode.from_tensor(hidden_states)
        fake_conv = fake_mode.from_tensor(conv_cache)
        fake_ssm = fake_mode.from_tensor(ssm_cache)
        dispatched_fake = torch.ops.vllm.qwen_gdn_full_forward_direct(
            fake_hidden,
            fake_conv,
            fake_ssm,
            "model.layers.0.linear_attn",
        )
    assert dispatched_fake.shape == hidden_states.shape
    assert dispatched_fake.is_contiguous()


@pytest.mark.parametrize("num_tokens", [4, 128])
def test_direct_projection_changed_input_cuda_graph_replay(num_tokens: int):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("The direct MTP3 graph route requires an SM70 CUDA device")

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    device = torch.device("cuda")
    hidden = torch.randn(num_tokens, 32, dtype=torch.float16, device=device)
    weight = torch.randn(32, 32, dtype=torch.float16, device=device)
    layer = SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        _compute_output_projection=lambda core, _z, _tokens: core @ weight,
    )
    z = torch.empty_like(hidden)

    # Warm the GEMM before capture. The graph retains the returned allocation.
    QwenGatedDeltaNetAttention._output_projection(layer, hidden, z, None, num_tokens)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = QwenGatedDeltaNetAttention._output_projection(
            layer, hidden, z, None, num_tokens
        )
    captured_ptr = captured.data_ptr()

    for replay in range(100):
        hidden.fill_(replay / 128)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, hidden @ weight, rtol=0, atol=0)
        assert captured.data_ptr() == captured_ptr
