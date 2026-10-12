# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU dispatch regressions for SM70 preparation wrappers, not Triton math.

Load the real wrapper modules with inert JIT/custom-op registration. Native
kernel tests cover arithmetic; these tests cover the context-owned cache route
that model compilation and graph capture execute during startup.
"""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

SOURCE_ROOT = (
    Path(os.environ["VLLM_TEST_SOURCE_ROOT"])
    if "VLLM_TEST_SOURCE_ROOT" in os.environ
    else Path(__file__).resolve().parents[3]
)


@pytest.fixture(params=["qwen", "dflash"])
def wrapper(request, monkeypatch):
    kind = request.param
    jit = ModuleType("vllm.triton_utils")
    jit.__dict__.update(triton=SimpleNamespace(jit=lambda function: function),
                        tl=SimpleNamespace(constexpr=object()))
    utils = ModuleType("vllm.utils.torch_utils")
    utils.__dict__.update(LayerNameType=str, direct_register_custom_op=Mock(),
                          _resolve_layer_name=lambda name: name)
    context = ModuleType("vllm.model_executor.layers.attention.attention")
    layout = ModuleType("vllm.v1.attention.backends.flash_v100.kv_layout")

    # Exercise the actual conventional splitter, including its two-page axis
    # preference, without importing the entire GPU backend on a CPU machine.
    path = SOURCE_ROOT / "vllm/v1/attention/backends/flash_v100/kv_layout.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    definition = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "split_paged_kv_cache")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
         namespace)
    splitter = Mock(wraps=namespace["split_paged_kv_cache"])
    layout.__dict__["_split_paged_kv_cache"] = splitter
    for module in (jit, utils, context, layout):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    filename = "sm70_qwen38_qk_rope.py" if kind == "qwen" else "sm70_dflash2_qk_rope.py"
    path = SOURCE_ROOT / "vllm/model_executor/layers/attention" / filename
    spec = importlib.util.spec_from_file_location("sm70_dispatch_test_" + kind, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def compute(*args, **kwargs):
        kwargs["q_out"].fill_(7)
        kwargs["k_out"].fill_(9)
        if kind == "qwen":
            kwargs["gate_out"].fill_(11)

    compute_mock = Mock(side_effect=compute)
    monkeypatch.setattr(module, "qk_norm_rope" if kind == "qwen"
                        else "qk_norm_rope_cache", compute_mock)
    return SimpleNamespace(kind=kind, module=module, context=context,
                           splitter=splitter, compute=compute_mock)


def invoke(wrapper, cache, slots, cache_dtype):
    tokens = 3
    raw_width = 3584 if wrapper.kind == "qwen" else 1536
    raw = torch.arange(tokens * raw_width, dtype=torch.float16).view(tokens, raw_width)
    q = torch.empty(tokens, 1536 if wrapper.kind == "qwen" else 1024)
    k = torch.empty(tokens, 256)
    writer = Mock()
    layer = SimpleNamespace(kv_cache_dtype=cache_dtype,
                            _k_scale=torch.ones(1), _v_scale=torch.ones(1),
                            impl=SimpleNamespace(do_kv_cache_update=writer))
    wrapper.context.get_attention_context = lambda name: (None, layer, cache, slots)
    weights = torch.ones(256 if wrapper.kind == "qwen" else 128)
    positions = torch.arange(tokens)
    rope = torch.zeros(8, 64 if wrapper.kind == "qwen" else 128)
    gate = torch.empty_like(q)
    if wrapper.kind == "qwen":
        wrapper.module._prepare_with_cache(raw, q, k, gate, weights, weights,
                                           positions, rope, 1e-6, "layer")
        assert torch.all(gate == 11)
    else:
        wrapper.module.sm70_dflash2_qk_norm_rope_cache(
            raw, q, k, weights, weights, positions, rope, 1e-6, "layer")
    assert torch.all(q == 7) and torch.all(k == 9)
    wrapper.compute.assert_called_once()
    return SimpleNamespace(raw=raw, q=q, k=k, layer=layer, writer=writer)


def assert_backend_write(wrapper, result, cache, slots):
    result.writer.assert_called_once()
    layer, key, value, storage, mapping = result.writer.call_args.args
    assert layer is result.layer and storage is cache and mapping is slots
    heads, head_size, start = (
        (1, 256, 3328) if wrapper.kind == "qwen" else (2, 128, 1280)
    )
    assert key.shape == value.shape == (3, heads, head_size)
    assert key.data_ptr() == result.k.data_ptr()
    torch.testing.assert_close(value, result.raw[:, start:].view(3, heads, head_size))
    kwargs = wrapper.compute.call_args.kwargs
    assert "key_cache" not in kwargs and "value_cache" not in kwargs


@pytest.mark.parametrize("pages", [2, 3])
@pytest.mark.parametrize("padding", [0, 128])
def test_packed_int8_uses_backend_once_without_splitting(wrapper, pages, padding):
    page_bytes = 2 * 16 * 256 + 2 * 8 * 2 + 4
    backing = torch.zeros(pages, page_bytes + padding, dtype=torch.int8)
    cache = backing[:, :page_bytes]
    slots = torch.tensor([3, -1, 5], dtype=torch.int64)
    result = invoke(wrapper, cache, slots, "int8_block32")
    wrapper.splitter.assert_not_called()
    assert cache.stride(0) == page_bytes + padding
    assert_backend_write(wrapper, result, cache, slots)


def test_storage_guard_precedes_conventional_split(wrapper):
    # Format spelling alone must not admit a packed cache into either writer.
    cache = torch.zeros(3, 100, dtype=torch.int8)
    slots = torch.tensor([3, -1, 5], dtype=torch.int64)
    result = invoke(wrapper, cache, slots, "fp8" if wrapper.kind == "qwen" else "auto")
    wrapper.splitter.assert_not_called()
    assert_backend_write(wrapper, result, cache, slots)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("no_slots", [False, True])
def test_empty_or_no_slot_dispatch(wrapper, empty, no_slots):
    cache = (
        torch.empty(0, dtype=torch.int8)
        if empty else torch.zeros(3, 100, dtype=torch.int8)
    )
    slots = None if no_slots else torch.tensor([3, -1, 5], dtype=torch.int64)
    result = invoke(wrapper, cache, slots, "int8_block32")
    wrapper.splitter.assert_not_called()
    if no_slots:
        result.writer.assert_not_called()
    else:
        assert_backend_write(wrapper, result, cache, slots)


@pytest.mark.parametrize("kv_first", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_conventional_fused_writer_is_preserved(wrapper, kv_first, alias):
    heads, head_size = (1, 256) if wrapper.kind == "qwen" else (2, 128)
    dtype = torch.uint8 if wrapper.kind == "qwen" else torch.float16
    shape = (2, 3, 16, heads, head_size) if kv_first else (3, 2, 16, heads, head_size)
    cache = torch.zeros(shape, dtype=dtype)
    cache_dtype = ("fp8_e4m3" if alias else "fp8") if wrapper.kind == "qwen" else (
        "float16" if alias else "auto")
    slots = torch.tensor([3, -1, 5], dtype=torch.int64)
    result = invoke(wrapper, cache, slots, cache_dtype)
    wrapper.splitter.assert_called_once_with(cache)
    result.writer.assert_not_called()
    kwargs = wrapper.compute.call_args.kwargs
    assert kwargs["slots"] is slots
    assert kwargs["key_cache"].shape == kwargs["value_cache"].shape == (
        3, 16, heads, head_size
    )
