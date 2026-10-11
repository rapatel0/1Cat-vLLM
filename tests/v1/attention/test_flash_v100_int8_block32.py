# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Signed INT8 block-32 KV-cache routes for Flash-V100, with fake native ops."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import decode as _decode
from vllm.v1.attention.backends.flash_v100 import int8_block32 as _int8
from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.flash_v100.backend import FlashAttnV100Backend
from vllm.v1.attention.backends.flash_v100.spec import attention as _spec_attention
from vllm.v1.attention.backends.flash_v100.spec import int8_verify as _iv
from vllm.v1.attention.kv_codecs import (
    FP16,
    INT8_BLOCK32,
    canonical_kv_cache_dtype,
    resolve_kv_codec,
)

HEADS, HEAD_SIZE = 1, 256


def page_bytes(block_size: int) -> int:
    payload = 2 * block_size * HEADS * HEAD_SIZE
    scales = 2 * HEADS * (HEAD_SIZE // 32) * 2
    return payload + scales + 4


def make_cache(num_blocks: int = 2, block_size: int = 1728) -> torch.Tensor:
    return torch.zeros(num_blocks, page_bytes(block_size), dtype=torch.int8)


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args[-1] if args else None


@pytest.fixture
def routes(monkeypatch):
    seen = []
    monkeypatch.setattr(_routing, "record_route", seen.append)
    return seen


def make_ops(**overrides) -> _int8.Int8Block32Ops:
    fields = dict(
        scale=0.0625,
        num_kv_heads=HEADS,
        head_size=HEAD_SIZE,
        decode_paged=Recorder(),
        prefill_paged=Recorder(),
        reshape_and_cache=Recorder(),
        paged_kv_to_fp16=Recorder(),
        xqa=Recorder(),
    )
    fields.update(overrides)
    return _int8.Int8Block32Ops(**fields)


def make_metadata(num_tokens: int = 1, num_requests: int = 1, **overrides):
    qsl = torch.arange(
        0, (num_requests + 1) * num_tokens, num_tokens, dtype=torch.int32
    )
    fields = dict(
        block_table=torch.zeros(num_requests, 4, dtype=torch.int32),
        seq_lens=torch.full((num_requests,), 5000, dtype=torch.int32),
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.clone(),
        num_actual_tokens=num_tokens * num_requests,
        max_query_len=num_tokens,
        max_model_len=1 << 18,
        is_dflash_selector_target=True,
        causal=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_codec_is_registered_and_distinct():
    assert resolve_kv_codec("int8_block32") is INT8_BLOCK32
    assert INT8_BLOCK32 is not FP16
    assert canonical_kv_cache_dtype("int8_block32") == "int8_block32"
    cache = make_cache()
    assert INT8_BLOCK32.stores(cache, cache)
    assert not FP16.stores(cache, cache)


def test_backend_admits_int8_block32_kv_cache():
    assert "int8_block32" in FlashAttnV100Backend.supported_kv_cache_dtypes
    assert FlashAttnV100Backend.supported_dtypes == [torch.float16]


def test_int8_routes_are_declared_for_the_int8_codec_only():
    names = [
        "decode_int8_block32_register",
        "decode_int8_block32_xqa",
        "decode_int8_block32_register_fallback",
        "prefill_smallq_int8_block32_register",
        "prefill_prefix_int8_block32_register",
        "prefill_prefix_int8_block32_bridge_fp16",
        "prefill_prefix_int8_block32_bridge_fp16_multi_request",
        "prefill_smallq_dflash2_int8_block32_grouped_verify",
    ]
    for name in names:
        assert _routing.route_spec(name).codecs == frozenset((INT8_BLOCK32,))


def test_triton_is_never_an_int8_fallback():
    policy = SimpleNamespace(allow_triton_fallback=True, use_flash_v100=True)
    with pytest.raises(RuntimeError, match="unsupported"):
        _spec_attention.unsupported(
            policy, {"kv_cache_dtype": "int8_block32"}, "unsupported", True
        )
    _spec_attention.unsupported(policy, {"kv_cache_dtype": "auto"}, "unsupported", True)


def test_split_views_follow_the_page_layout():
    cache = make_cache(num_blocks=3)
    key, value, key_scales, value_scales, owners = (
        _kv_layout.split_int8_block32_kv_cache(
            cache, num_kv_heads=HEADS, head_size=HEAD_SIZE
        )
    )
    assert key.shape == (3, 1728, HEADS, HEAD_SIZE)
    assert key_scales.shape == (3, HEADS, HEAD_SIZE // 32)
    assert key.stride(0) == cache.stride(0)
    assert owners.shape == (3,)
    with pytest.raises(ValueError, match="int8 storage"):
        _kv_layout.split_int8_block32_kv_cache(
            cache.view(torch.uint8),
            num_kv_heads=HEADS,
            head_size=HEAD_SIZE,
        )
    with pytest.raises(ValueError, match="does not match"):
        _kv_layout.split_int8_block32_kv_cache(
            cache[:, :-5], num_kv_heads=HEADS, head_size=HEAD_SIZE
        )


def test_packed_verify_offsets_are_constant_and_persistent():
    first = _iv.packed_verify_offsets(3, torch.device("cpu"))
    assert first.tolist() == [0, 8, 16, 24]
    assert first.dtype == torch.int32
    assert _iv.packed_verify_offsets(3, torch.device("cpu")) is first


@pytest.mark.parametrize(
    ("offsets", "tokens", "limit", "valid"),
    [
        ([0, 8], 8, 8, True),
        ([0, 16], 16, 16, True),
        ([0, 16], 16, 8, False),
        ([0, 8, 16], 16, 8, True),
        ([0, 8, 8], 8, 8, False),
        ([1, 9], 8, 8, False),
        ([0, 8], 16, 8, False),
    ],
)
def test_query_partition_validation(offsets, tokens, limit, valid):
    qsl = torch.tensor(offsets, dtype=torch.int32)
    assert (
        _iv.query_partition_is_valid(
            qsl,
            num_requests=len(offsets) - 1,
            num_query_tokens=tokens,
            max_query_tokens=limit,
        )
        is valid
    )
    assert not _iv.query_partition_is_valid(
        qsl.to(torch.int64),
        num_requests=len(offsets) - 1,
        num_query_tokens=tokens,
        max_query_tokens=limit,
    )


def test_update_cache_writes_through_split_views():
    ops = make_ops()
    cache = make_cache()
    key = torch.zeros(2, HEADS, HEAD_SIZE, dtype=torch.float16)
    slots = torch.tensor([0, 1], dtype=torch.int64)
    _int8.Int8Block32Attention(ops).update_cache(key, key, cache, slots)
    ((args, _),) = ops.reshape_and_cache.calls
    assert args[0] is key
    assert args[2].dtype == torch.int8 and args[2].shape[1] == 1728
    assert args[4].dtype == torch.float16
    assert args[-1] is slots
    with pytest.raises(RuntimeError, match="writer is unavailable"):
        _int8.Int8Block32Attention(make_ops(reshape_and_cache=None)).update_cache(
            key, key, cache, slots
        )


def test_decode_runs_xqa_with_block_scales(routes):
    ops = make_ops()
    cache = make_cache()
    query = torch.zeros(1, 6, HEAD_SIZE, dtype=torch.float16)
    output = torch.zeros_like(query)
    metadata = make_metadata()
    result = _int8.Int8Block32Attention(ops).decode(query, cache, metadata, output)
    assert result is output
    ((args, kwargs),) = ops.xqa.calls
    assert kwargs["kv_cache_dtype"] == "int8_block32"
    assert kwargs["key_scales"].dtype == torch.float16
    assert kwargs["value_scales"].dtype == torch.float16
    assert kwargs["partition_size_hint"] == 256
    assert kwargs["softmax_scale"] == 0.0625
    assert not ops.decode_paged.calls
    assert routes == ["decode_int8_block32_xqa"]


def test_decode_uses_register_kernel_without_xqa(routes):
    ops = make_ops(xqa=None)
    query = torch.zeros(1, 6, HEAD_SIZE, dtype=torch.float16)
    output = torch.zeros_like(query)
    _int8.Int8Block32Attention(ops).decode(query, make_cache(), make_metadata(), output)
    assert len(ops.decode_paged.calls) == 1
    assert routes == ["decode_int8_block32_register_fallback"]


def test_decode_guards():
    attention = _int8.Int8Block32Attention(make_ops(window_size=lambda causal: (8, 0)))
    query = torch.zeros(1, 6, HEAD_SIZE, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="sliding-window"):
        attention.decode(query, make_cache(), make_metadata(), torch.zeros_like(query))
    attention = _int8.Int8Block32Attention(make_ops())
    two = torch.zeros(2, 6, HEAD_SIZE, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="one query token per request"):
        attention.decode(
            two,
            make_cache(),
            make_metadata(num_tokens=2, num_actual_tokens=2),
            torch.zeros_like(two),
        )
    empty = torch.zeros(0, 6, HEAD_SIZE, dtype=torch.float16)
    output = torch.zeros_like(empty)
    assert (
        attention.decode(
            empty, make_cache(), make_metadata(num_actual_tokens=0), output
        )
        is output
    )


def grouped(**overrides) -> _iv.Int8GroupedVerify:
    fields = dict(
        op=Recorder(),
        scale=0.0625,
        bridge_available=True,
        enabled=True,
        max_query_tokens=16,
        max_requests=1,
        min_model_len=1024,
    )
    fields.update(overrides)
    return _iv.Int8GroupedVerify(**fields)


def verify_inputs(tokens: int = 8):
    cache = make_cache()
    key, value, *_ = _kv_layout.split_int8_block32_kv_cache(
        cache, num_kv_heads=HEADS, head_size=HEAD_SIZE
    )
    query = torch.zeros(tokens, 6, HEAD_SIZE, dtype=torch.float16)
    return query, key, value, make_metadata(num_tokens=tokens)


@pytest.mark.parametrize("tokens", [8, 16])
def test_grouped_verify_admits_single_request_q8_and_q16(tokens):
    attention = _int8.Int8Block32Attention(make_ops(grouped=grouped()))
    query, key, value, metadata = verify_inputs(tokens)
    assert attention.ops.grouped.allowed(
        query, key, value, metadata, num_query_tokens=tokens
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda g, m, q: (grouped(enabled=False), m, q),
        lambda g, m, q: (grouped(op=None), m, q),
        lambda g, m, q: (g, make_metadata(8, is_dflash_selector_target=False), q),
        lambda g, m, q: (g, make_metadata(8, max_model_len=100), q),
        lambda g, m, q: (g, make_metadata(8, causal=False), q),
        lambda g, m, q: (
            g,
            make_metadata(
                8,
                query_start_loc=torch.tensor([0, 8], dtype=torch.int64),
                query_start_loc_cpu=None,
            ),
            q,
        ),
        lambda g, m, q: (g, m, q.to(torch.bfloat16)),
        lambda g, m, q: (grouped(max_query_tokens=8), make_metadata(16), q),
    ],
)
def test_grouped_verify_gate_rejects(change):
    attention = _int8.Int8Block32Attention(make_ops(grouped=grouped()))
    query, key, value, metadata = verify_inputs(8)
    group, metadata, query = change(grouped(), metadata, query)
    attention = _int8.Int8Block32Attention(make_ops(grouped=group))
    tokens = 16 if metadata.max_query_len == 16 else 8
    if tokens == 16:
        query = torch.zeros(16, 6, HEAD_SIZE, dtype=torch.float16)
    assert not attention.ops.grouped.allowed(
        query, key, value, metadata, num_query_tokens=tokens
    )


def test_grouped_verify_requires_the_bridge_op_and_int8_source():
    attention = _int8.Int8Block32Attention(
        make_ops(grouped=grouped(bridge_available=False))
    )
    query, key, value, metadata = verify_inputs(8)
    assert not attention.ops.grouped.allowed(
        query, key, value, metadata, num_query_tokens=8
    )
    attention = _int8.Int8Block32Attention(make_ops(grouped=grouped()))
    assert not attention.ops.grouped.allowed(
        query, key.to(torch.uint8), value.to(torch.uint8), metadata, num_query_tokens=8
    )


def test_multi_request_packing_needs_batched_flag_or_int8_request_limit():
    group = grouped(max_requests=4)
    attention = _int8.Int8Block32Attention(make_ops(grouped=group))
    cache = make_cache()
    key, value, *_ = _kv_layout.split_int8_block32_kv_cache(
        cache, num_kv_heads=HEADS, head_size=HEAD_SIZE
    )
    query = torch.zeros(16, 6, HEAD_SIZE, dtype=torch.float16)
    metadata = make_metadata(num_tokens=8, num_requests=2)
    # The single-request or batched-flag shape must hold first, as deployed.
    assert not attention.ops.grouped.allowed(
        query, key, value, metadata, num_query_tokens=16
    )
    batched = grouped(max_requests=4, batched_enabled=True, request_major_abi=1)
    attention = _int8.Int8Block32Attention(make_ops(grouped=batched))
    assert attention.ops.grouped.allowed(
        query, key, value, metadata, num_query_tokens=16
    )


def test_small_query_prefill_prefers_packed_verifier(routes):
    group = grouped()
    ops = make_ops(grouped=group)
    attention = _int8.Int8Block32Attention(ops)
    query, *_ = verify_inputs(8)
    output = torch.zeros_like(query)
    metadata = make_metadata(num_tokens=8)
    result = attention.small_query_prefill_as_decode(
        query, make_cache(), metadata, output
    )
    assert result is output
    ((args, kwargs),) = group.op.calls
    assert kwargs["kv_cache_dtype"] == "int8_block32"
    assert kwargs["one_pass"] is True
    assert kwargs["query_start_loc"] is None
    assert kwargs["_query_partition_validated"] is True
    assert not ops.prefill_paged.calls and not ops.decode_paged.calls
    assert routes == ["prefill_smallq_dflash2_int8_block32_grouped_verify"]


def test_packed_requests_use_the_constant_graph_safe_partition(routes):
    group = grouped(max_requests=4, batched_enabled=True, request_major_abi=1)
    attention = _int8.Int8Block32Attention(make_ops(grouped=group))
    cache = make_cache()
    key, value, *_ = _kv_layout.split_int8_block32_kv_cache(
        cache, num_kv_heads=HEADS, head_size=HEAD_SIZE
    )
    query = torch.zeros(16, 6, HEAD_SIZE, dtype=torch.float16)
    metadata = make_metadata(num_tokens=8, num_requests=2)
    # Replay metadata padded by repeating the real token count.
    padded = torch.tensor([0, 8, 16, 16, 16], dtype=torch.int32)
    metadata.query_start_loc = padded[:3]
    attention.ops.grouped.call(query, key, value, metadata, out=query)
    ((_, kwargs),) = group.op.calls
    assert kwargs["query_start_loc"].tolist() == [0, 8, 16]
    assert kwargs["query_start_loc"] is _iv.packed_verify_offsets(2, query.device)


def test_small_query_prefill_uses_register_decode_with_persistent_metadata(routes):
    ops = make_ops()
    attention = _int8.Int8Block32Attention(ops)
    query = torch.zeros(4, 6, HEAD_SIZE, dtype=torch.float16)
    output = torch.zeros_like(query)
    metadata = make_metadata(
        num_tokens=4,
        smallq_decode_block_table=torch.zeros(4, 4, dtype=torch.int32),
        smallq_decode_seq_lens=torch.full((4,), 5000, dtype=torch.int32),
    )
    attention.small_query_prefill_as_decode(query, make_cache(), metadata, output)
    ((args, _),) = ops.decode_paged.calls
    assert args[5].shape[0] == 4 and args[6].shape[0] == 4
    assert routes == ["prefill_smallq_int8_block32_register"]


def test_small_query_prefill_falls_back_to_native_prefill(routes):
    ops = make_ops()
    attention = _int8.Int8Block32Attention(ops)
    query = torch.zeros(4, 6, HEAD_SIZE, dtype=torch.float16)
    output = torch.zeros_like(query)
    attention.small_query_prefill_as_decode(
        query, make_cache(), make_metadata(num_tokens=4), output
    )
    ((args, _),) = ops.prefill_paged.calls
    assert args[7].tolist() == [0, 4]
    assert routes == ["prefill_prefix_int8_block32_register"]


def test_non_causal_prefix_prefill_is_rejected():
    attention = _int8.Int8Block32Attention(make_ops())
    query = torch.zeros(4, 6, HEAD_SIZE, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="requires causal attention"):
        attention.small_query_prefill_as_decode(
            query,
            make_cache(),
            make_metadata(num_tokens=4, causal=False),
            torch.zeros_like(query),
        )


def test_decode_candidate_admits_only_the_int8_cache(routes):
    attention = _int8.Int8Block32Attention(make_ops())

    def executor(kv_cache_dtype, int8):
        return SimpleNamespace(
            config=SimpleNamespace(kv_cache_dtype=kv_cache_dtype),
            ops=SimpleNamespace(int8_block32=int8, profile_trace=lambda *a: None),
        )

    assert _decode.DecodeInt8Block32(executor("int8_block32", attention)).admit(None)
    assert not _decode.DecodeInt8Block32(executor("auto", None)).admit(None)
    assert not _decode.DecodeInt8Block32(executor("fp8_e5m2", None)).admit(None)
    assert _decode.DECODE_CANDIDATES.index(
        _decode.DecodeInt8Block32
    ) < _decode.DECODE_CANDIDATES.index(_decode.DecodePagedPrefill)
    with pytest.raises(RuntimeError, match="decoder is unavailable"):
        _decode.DecodeInt8Block32(executor("int8_block32", None)).run(
            SimpleNamespace(layer_name="l"), routes.append
        )


def test_grouped_verify_uses_query_start_loc_when_it_is_already_on_cpu():
    attention = _int8.Int8Block32Attention(make_ops(grouped=grouped()))
    query, key, value, metadata = verify_inputs(8)
    metadata.query_start_loc_cpu = None
    assert attention.ops.grouped.allowed(
        query, key, value, metadata, num_query_tokens=8
    )
