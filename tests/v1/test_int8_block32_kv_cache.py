import pytest
import torch

from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVQuantMode,
    MambaSpec,
    get_kv_quant_mode,
    make_int8_block32_kv_cache_views,
)


def test_int8_block32_mode_and_page_size() -> None:
    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_BLOCK32,
    )

    payload_bytes = 2 * 16 * 2 * 64
    scale_bytes = (
        2 * 2 * (64 // 32) * torch.empty([], dtype=torch.float16).element_size()
    )
    assert get_kv_quant_mode("int8_block32") == KVQuantMode.INT8_BLOCK32
    owner_bytes = torch.empty([], dtype=torch.int32).element_size()
    assert spec.page_size_bytes == payload_bytes + scale_bytes + owner_bytes


def test_int8_block32_cache_views_share_one_allocation() -> None:
    num_blocks, block_size, num_heads, head_size = 3, 16, 2, 64
    scale_bytes = 2 * num_blocks * num_heads * (head_size // 32) * 2
    owner_bytes = num_blocks * torch.empty([], dtype=torch.int32).element_size()
    raw = torch.zeros(
        2 * num_blocks * block_size * num_heads * head_size + scale_bytes + owner_bytes,
        dtype=torch.int8,
    )

    key, value, key_scales, value_scales, page_owners = (
        make_int8_block32_kv_cache_views(
            raw,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_heads,
            head_size=head_size,
        )
    )
    key[1, 2, 0, 3] = -17
    value_scales[2, 1, 1] = 0.5
    page_owners[1] = 7

    assert key.shape == (num_blocks, block_size, num_heads, head_size)
    assert value.shape == key.shape
    assert key_scales.shape == (num_blocks, num_heads, head_size // 32)
    assert value_scales.shape == key_scales.shape

    side_payload_bytes = block_size * num_heads * head_size
    side_scale_bytes = num_heads * (head_size // 32) * 2
    page_bytes = 2 * side_payload_bytes + 2 * side_scale_bytes + 4
    assert key.stride(0) == page_bytes
    assert value.stride(0) == page_bytes
    assert key_scales.stride(0) == page_bytes // 2
    assert value_scales.stride(0) == page_bytes // 2
    assert page_owners.stride(0) == page_bytes // 4
    key_byte = page_bytes + 2 * num_heads * head_size + 3
    assert raw[key_byte] == -17
    assert value_scales[2, 1, 1] == torch.tensor(0.5, dtype=torch.float16)
    owner_byte = page_bytes + 2 * side_payload_bytes + 2 * side_scale_bytes
    owner_from_raw = raw[owner_byte : owner_byte + 4].view(torch.int32)
    assert owner_from_raw.item() == 7


def test_int8_block32_padded_page_views_preserve_page_stride() -> None:
    num_blocks, block_size, num_heads, head_size = 2, 16, 2, 64
    base_spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_heads,
        head_size=head_size,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_BLOCK32,
    )
    padded_page_size = base_spec.page_size_bytes + 128
    padded_spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_heads,
        head_size=head_size,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_BLOCK32,
        page_size_padded=padded_page_size,
    )
    raw = torch.zeros(num_blocks * padded_page_size, dtype=torch.int8)

    key, _, _, _, page_owners = make_int8_block32_kv_cache_views(
        raw,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_heads,
        head_size=head_size,
        page_stride_bytes=padded_page_size,
    )
    key[1, 0, 0, 0] = 11
    page_owners[1] = 19

    assert padded_spec.page_size_bytes == padded_page_size
    assert key.stride(0) == padded_page_size
    assert page_owners.stride(0) == padded_page_size // 4
    assert raw[padded_page_size] == 11


def test_int8_block32_views_preserve_nonzero_storage_offset() -> None:
    num_blocks, block_size, num_heads, head_size = 2, 16, 2, 64
    prefix_bytes = 12
    side_payload_bytes = block_size * num_heads * head_size
    side_scale_elements = num_heads * (head_size // 32)
    page_bytes = 2 * side_payload_bytes + 2 * side_scale_elements * 2 + 4
    page_stride = page_bytes + 128
    backing = torch.full(
        (prefix_bytes + num_blocks * page_stride + 16,), 99, dtype=torch.int8
    )
    raw = backing[prefix_bytes : prefix_bytes + num_blocks * page_stride]

    key, value, key_scales, value_scales, page_owners = (
        make_int8_block32_kv_cache_views(
            raw,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_heads,
            head_size=head_size,
            page_stride_bytes=page_stride,
        )
    )

    assert key.storage_offset() == prefix_bytes
    assert value.storage_offset() == prefix_bytes + side_payload_bytes
    assert key_scales.storage_offset() == (prefix_bytes + 2 * side_payload_bytes) // 2
    assert (
        value_scales.storage_offset()
        == (prefix_bytes + 2 * side_payload_bytes) // 2 + side_scale_elements
    )
    assert (
        page_owners.storage_offset()
        == (prefix_bytes + 2 * side_payload_bytes + 2 * side_scale_elements * 2) // 4
    )

    key[0, 0, 0, 0] = 1
    value[0, 0, 0, 0] = 2
    key_scales[0, 0, 0] = 3
    value_scales[0, 0, 0] = 4
    page_owners[0] = 5
    assert torch.all(backing[:prefix_bytes] == 99)
    assert raw[0] == 1
    assert raw[side_payload_bytes] == 2
    assert key_scales[0, 0, 0] == 3
    assert value_scales[0, 0, 0] == 4
    assert page_owners[0] == 5


def test_int8_block32_hybrid_unification_uses_tail_padding() -> None:
    int8_spec = FullAttentionSpec(
        block_size=1648,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_BLOCK32,
    )
    draft_spec = FullAttentionSpec(
        block_size=1648,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=1648,
        shapes=((int8_spec.page_size_bytes,),),
        dtypes=(torch.uint8,),
        page_size_padded=int8_spec.page_size_bytes,
    )

    unified = unify_kv_cache_spec_page_size(
        {"target.attn": int8_spec, "target.mamba": mamba_spec, "draft": draft_spec}
    )

    common_page = unified["draft"].page_size_bytes
    unified_int8 = unified["target.attn"]
    unified_mamba = unified["target.mamba"]
    assert isinstance(unified_int8, AttentionSpec)
    assert isinstance(unified_mamba, MambaSpec)
    assert {spec.page_size_bytes for spec in unified.values()} == {common_page}
    assert common_page == draft_spec.page_size_bytes
    assert unified["draft"].block_size == draft_spec.block_size
    assert unified_int8.block_size == int8_spec.block_size
    assert unified_int8.page_size_padded == common_page
    assert unified_mamba.block_size == mamba_spec.block_size
    assert unified_mamba.page_size_padded == common_page


def test_int8_block32_rejects_asymmetric_pages() -> None:
    with pytest.raises(ValueError, match="equal key and value head sizes"):
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=64,
            head_size_v=32,
            dtype=torch.int8,
            kv_quant_mode=KVQuantMode.INT8_BLOCK32,
        )


def test_int8_block32_rejects_partial_channel_blocks() -> None:
    raw = torch.zeros(4096, dtype=torch.int8)
    with pytest.raises(ValueError, match="head size divisible by 32"):
        make_int8_block32_kv_cache_views(
            raw,
            num_blocks=1,
            block_size=16,
            num_kv_heads=1,
            head_size=80,
        )
