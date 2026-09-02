import math

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    make_int8_block32_kv_cache_views,
)


def _require_sm70_extension():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("This test requires a CUDA SM70 device")
    return pytest.importorskip("flash_attn_v100.flash_attn_v100_cuda")


def test_int8_block32_publish_requantize_and_decode() -> None:
    extension = _require_sm70_extension()
    torch.manual_seed(7)

    num_blocks = 2
    block_size = 16
    num_kv_heads = 1
    num_query_heads = 2
    head_size = 64
    num_tokens = 20
    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_BLOCK32,
    )
    raw = torch.zeros(
        num_blocks * spec.page_size_bytes,
        dtype=torch.int8,
        device="cuda",
    )
    (
        key_cache,
        value_cache,
        key_scales,
        value_scales,
        page_owners,
    ) = make_int8_block32_kv_cache_views(
        raw,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )

    key = torch.randn(
        num_tokens, num_kv_heads, head_size, dtype=torch.float16, device="cuda"
    )
    value = torch.randn_like(key)
    # Force several scale increases in one publication batch. The kernel must
    # compute the final scale first instead of repeatedly rounding old codes.
    key[:15, 0, :32] = 0.05
    key[0, 0, :32] = 0.1
    key[1, 0, :32] = 0.2
    key[2, 0, :32] = 0.3
    value[:15, 0, 32:] = 0.05
    value[0, 0, 32:] = 0.1
    value[1, 0, 32:] = 0.2
    value[2, 0, 32:] = 0.3
    # A later call expands both scales again and exercises historical-page
    # re-quantization across publication calls.
    key[15, 0, :32] = 12.0
    value[15, 0, 32:] = -10.0

    # Logical page zero maps to physical page one. Logical page one maps to zero.
    slot_mapping = torch.cat(
        (
            torch.arange(16, 32, device="cuda"),
            torch.arange(0, 4, device="cuda"),
        )
    ).to(torch.int64)
    # Interleave writes to the two physical pages and defer the scale-growing
    # token. Publication must elect one owner per page rather than assuming
    # page runs are contiguous in slot_mapping.
    first_indices = torch.tensor(
        [0, 16, 1, 17, 2, 18, 3, 19, *range(4, 15)],
        dtype=torch.long,
        device="cuda",
    )
    extension.int8_block32_reshape_and_cache(
        key[first_indices],
        value[first_indices],
        key_cache,
        value_cache,
        key_scales,
        value_scales,
        page_owners,
        slot_mapping[first_indices],
    )
    key_scale_before = key_scales[1, 0].repeat_interleave(32).float()
    value_scale_before = value_scales[1, 0].repeat_interleave(32).float()
    expected_key_codes = (
        torch.round(key[:15, 0].float() / key_scale_before)
        .clamp(-127, 127)
        .to(torch.int8)
    )
    expected_value_codes = (
        torch.round(value[:15, 0].float() / value_scale_before)
        .clamp(-127, 127)
        .to(torch.int8)
    )
    torch.testing.assert_close(key_cache[1, :15, 0], expected_key_codes)
    torch.testing.assert_close(value_cache[1, :15, 0], expected_value_codes)

    key_before_growth = (
        key_cache[1, :15].float() * key_scales[1].repeat_interleave(32, dim=-1).float()
    )
    value_before_growth = (
        value_cache[1, :15].float()
        * value_scales[1].repeat_interleave(32, dim=-1).float()
    )

    extension.int8_block32_reshape_and_cache(
        key[15:16],
        value[15:16],
        key_cache,
        value_cache,
        key_scales,
        value_scales,
        page_owners,
        slot_mapping[15:16],
    )

    key_after_growth = (
        key_cache[1, :15].float() * key_scales[1].repeat_interleave(32, dim=-1).float()
    )
    value_after_growth = (
        value_cache[1, :15].float()
        * value_scales[1].repeat_interleave(32, dim=-1).float()
    )
    torch.testing.assert_close(
        key_after_growth,
        key_before_growth,
        atol=float(key_scales[1].max()),
        rtol=0.0,
    )
    torch.testing.assert_close(
        value_after_growth,
        value_before_growth,
        atol=float(value_scales[1].max()),
        rtol=0.0,
    )

    assert torch.all(key_scales > 0)
    assert torch.all(value_scales > 0)
    assert key_scales[1, 0, 0] >= torch.tensor(
        12.0 / 127.0, dtype=torch.float16, device="cuda"
    )

    query = torch.randn(
        1, num_query_heads, head_size, dtype=torch.float16, device="cuda"
    )
    block_table = torch.tensor([[1, 0]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    softmax_scale = 1.0 / math.sqrt(head_size)
    extension.int8_block32_decode_paged(
        query,
        key_cache,
        value_cache,
        key_scales,
        value_scales,
        block_table,
        seq_lens,
        output,
        softmax_scale,
    )

    key_scale_values = key_scales.repeat_interleave(32, dim=-1).unsqueeze(1)
    value_scale_values = value_scales.repeat_interleave(32, dim=-1).unsqueeze(1)
    dequant_key = (key_cache.float() * key_scale_values.float()).half().float()
    dequant_value = (value_cache.float() * value_scale_values.float()).half().float()
    token_indices = torch.arange(num_tokens, device="cuda")
    physical_pages = block_table[0, token_indices // 16].to(torch.long)
    page_offsets = token_indices % 16
    key_sequence = dequant_key[physical_pages, page_offsets, 0]
    value_sequence = dequant_value[physical_pages, page_offsets, 0]
    scores = torch.einsum("bhd,td->bht", query.float(), key_sequence)
    probabilities = torch.softmax(scores * softmax_scale, dim=-1)
    reference = torch.einsum("bht,td->bhd", probabilities, value_sequence)

    torch.testing.assert_close(output.float(), reference, atol=2e-3, rtol=2e-3)

    chunk_query = torch.randn(
        3, num_query_heads, head_size, dtype=torch.float16, device="cuda"
    )
    chunk_output = torch.empty_like(chunk_query)
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    extension.int8_block32_prefill_paged(
        chunk_query,
        key_cache,
        value_cache,
        key_scales,
        value_scales,
        block_table,
        seq_lens,
        query_start_loc,
        chunk_output,
        softmax_scale,
    )

    chunk_reference = torch.empty_like(chunk_output, dtype=torch.float32)
    for query_idx in range(3):
        visible_tokens = num_tokens - 3 + query_idx + 1
        chunk_scores = torch.einsum(
            "hd,td->ht",
            chunk_query[query_idx].float(),
            key_sequence[:visible_tokens],
        )
        chunk_probabilities = torch.softmax(chunk_scores * softmax_scale, dim=-1)
        chunk_reference[query_idx] = torch.einsum(
            "ht,td->hd",
            chunk_probabilities,
            value_sequence[:visible_tokens],
        )

    torch.testing.assert_close(
        chunk_output.float(), chunk_reference, atol=2e-3, rtol=2e-3
    )


def test_int8_block32_bridge_preserves_paged_layout_and_graph_replay() -> None:
    extension = _require_sm70_extension()
    torch.manual_seed(19)

    num_blocks = 4
    block_size = 16
    num_kv_heads = 2
    head_size = 64
    page_padding = 128
    prefix_bytes = 12
    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_BLOCK32,
    )
    page_stride = spec.page_size_bytes + page_padding
    backing = torch.zeros(
        prefix_bytes + num_blocks * page_stride + 16,
        dtype=torch.int8,
        device="cuda",
    )
    raw = backing[prefix_bytes : prefix_bytes + num_blocks * page_stride]
    key_cache, value_cache, key_scales, value_scales, _ = (
        make_int8_block32_kv_cache_views(
            raw,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            page_stride_bytes=page_stride,
        )
    )
    key_cache.random_(-127, 128)
    value_cache.random_(-127, 128)
    key_scales.uniform_(0.001, 0.02)
    value_scales.uniform_(0.001, 0.02)

    batch_size = 2
    output_block_size = 8
    output_blocks_per_seq = 4
    key_out = torch.empty(
        batch_size * output_blocks_per_seq,
        output_block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    value_out = torch.empty_like(key_out)
    block_table = torch.tensor(
        [[3, 1], [2, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor([27, 19], dtype=torch.int32, device="cuda")

    def run_bridge() -> None:
        extension.int8_block32_paged_kv_to_fp16(
            key_cache,
            value_cache,
            key_scales,
            value_scales,
            block_table,
            seq_lens,
            key_out,
            value_out,
        )

    def check_output() -> None:
        key_scale_values = key_scales.repeat_interleave(32, dim=-1)
        value_scale_values = value_scales.repeat_interleave(32, dim=-1)
        for batch_idx in range(batch_size):
            seq_len = int(seq_lens[batch_idx].item())
            token_indices = torch.arange(seq_len, device="cuda")
            pages = block_table[batch_idx, token_indices // block_size].long()
            offsets = token_indices % block_size
            expected_key = (
                key_cache[pages, offsets].float() * key_scale_values[pages].float()
            ).half()
            expected_value = (
                value_cache[pages, offsets].float() * value_scale_values[pages].float()
            ).half()
            output_start = batch_idx * output_blocks_per_seq
            actual_key = key_out[
                output_start : output_start + output_blocks_per_seq
            ].flatten(0, 1)[:seq_len]
            actual_value = value_out[
                output_start : output_start + output_blocks_per_seq
            ].flatten(0, 1)[:seq_len]
            torch.testing.assert_close(actual_key, expected_key, atol=0, rtol=0)
            torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0)

    run_bridge()
    torch.cuda.synchronize()
    check_output()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_bridge()
    block_table.copy_(torch.tensor([[0, 2], [1, 3]], dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    check_output()


def test_grouped_verify_reads_int8_block32_scales_directly() -> None:
    _require_sm70_extension()
    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    torch.manual_seed(23)

    query_len = 8
    num_query_heads = 6
    num_kv_heads = 1
    head_size = 256
    block_size = 16
    seq_len = 24
    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_BLOCK32,
    )
    raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.int8, device="cuda")
    key_cache, value_cache, key_scales, value_scales, _ = (
        make_int8_block32_kv_cache_views(
            raw,
            num_blocks=2,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
    )
    key_cache.random_(-127, 128)
    value_cache.random_(-127, 128)
    key_scales.uniform_(0.001, 0.02)
    value_scales.uniform_(0.001, 0.02)
    query = torch.randn(
        query_len,
        num_query_heads,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    block_table = torch.tensor([[1, 0]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    softmax_scale = 1.0 / math.sqrt(head_size)

    flash_attn_v100.flash_attn_grouped_verify_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        softmax_scale=softmax_scale,
        out=output,
        kv_cache_dtype="int8_block32",
        one_pass=True,
    )

    token_indices = torch.arange(seq_len, device="cuda")
    pages = block_table[0, token_indices // block_size].long()
    offsets = token_indices % block_size
    key_scale_values = key_scales.repeat_interleave(32, dim=-1)
    value_scale_values = value_scales.repeat_interleave(32, dim=-1)
    key_sequence = (
        (key_cache[pages, offsets, 0].float() * key_scale_values[pages, 0].float())
        .half()
        .float()
    )
    value_sequence = (
        (value_cache[pages, offsets, 0].float() * value_scale_values[pages, 0].float())
        .half()
        .float()
    )
    reference = torch.empty_like(output, dtype=torch.float32)
    prefix_len = seq_len - query_len
    for query_idx in range(query_len):
        visible_len = prefix_len + query_idx + 1
        scores = torch.einsum(
            "hd,td->ht",
            query[query_idx].float(),
            key_sequence[:visible_len],
        )
        probabilities = torch.softmax(scores * softmax_scale, dim=-1)
        reference[query_idx] = torch.einsum(
            "ht,td->hd",
            probabilities,
            value_sequence[:visible_len],
        )

    torch.testing.assert_close(output.float(), reference, atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_grouped_verify_batches_eight_int8_block32_requests() -> None:
    _require_sm70_extension()
    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    torch.manual_seed(29)

    num_requests = 8
    query_len = 8
    num_query_heads = 6
    num_kv_heads = 1
    head_size = 256
    block_size = 16
    num_blocks = num_requests * 2
    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_BLOCK32,
    )
    raw = torch.zeros(
        num_blocks * spec.page_size_bytes,
        dtype=torch.int8,
        device="cuda",
    )
    key_cache, value_cache, key_scales, value_scales, _ = (
        make_int8_block32_kv_cache_views(
            raw,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
    )
    key_cache.random_(-127, 128)
    value_cache.random_(-127, 128)
    key_scales.uniform_(0.001, 0.02)
    value_scales.uniform_(0.001, 0.02)
    query = torch.randn(
        num_requests * query_len,
        num_query_heads,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    block_table = torch.tensor(
        [[2 * request + 1, 2 * request] for request in range(num_requests)],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.arange(
        24,
        24 + num_requests,
        dtype=torch.int32,
        device="cuda",
    )
    query_start_loc = torch.arange(
        0,
        num_requests * query_len + 1,
        query_len,
        dtype=torch.int32,
        device="cuda",
    )
    output = torch.empty_like(query)
    softmax_scale = 1.0 / math.sqrt(head_size)

    def run_grouped_verify() -> None:
        flash_attn_v100.flash_attn_grouped_verify_paged(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=softmax_scale,
            out=output,
            kv_cache_dtype="int8_block32",
            one_pass=True,
            query_start_loc=query_start_loc,
        )

    def reference() -> torch.Tensor:
        result = torch.empty_like(output, dtype=torch.float32)
        key_scale_values = key_scales.repeat_interleave(32, dim=-1)
        value_scale_values = value_scales.repeat_interleave(32, dim=-1)
        for request in range(num_requests):
            seq_len = int(seq_lens[request].item())
            token_indices = torch.arange(seq_len, device="cuda")
            pages = block_table[request, token_indices // block_size].long()
            offsets = token_indices % block_size
            keys = (
                key_cache[pages, offsets, 0].float()
                * key_scale_values[pages, 0].float()
            ).half()
            values = (
                value_cache[pages, offsets, 0].float()
                * value_scale_values[pages, 0].float()
            ).half()
            prefix_len = seq_len - query_len
            for local_query in range(query_len):
                query_idx = request * query_len + local_query
                visible_len = prefix_len + local_query + 1
                scores = torch.einsum(
                    "hd,td->ht",
                    query[query_idx].float(),
                    keys[:visible_len].float(),
                )
                probabilities = torch.softmax(
                    scores * softmax_scale,
                    dim=-1,
                )
                result[query_idx] = torch.einsum(
                    "ht,td->hd",
                    probabilities,
                    values[:visible_len].float(),
                )
        return result

    assert flash_attn_v100.flash_attn_grouped_verify_max_requests() >= num_requests
    run_grouped_verify()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), reference(), atol=3e-2, rtol=3e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_grouped_verify()
    query.copy_(torch.randn_like(query))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), reference(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("num_queries", [8, 16])
@pytest.mark.parametrize("block_size", [16, 1648])
def test_int8_block32_expanded_metadata_cuda_graph_replay(
    num_queries: int,
    block_size: int,
) -> None:
    extension = _require_sm70_extension()
    torch.manual_seed(11 + num_queries)

    num_blocks = 4
    num_kv_heads = 1
    num_query_heads = 2
    head_size = 64
    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_BLOCK32,
    )
    raw = torch.zeros(
        num_blocks * spec.page_size_bytes,
        dtype=torch.int8,
        device="cuda",
    )
    key_cache, value_cache, key_scales, value_scales, _ = (
        make_int8_block32_kv_cache_views(
            raw,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
    )
    key_cache.random_(-80, 81)
    value_cache.random_(-80, 81)
    key_scales.uniform_(0.002, 0.02)
    value_scales.uniform_(0.002, 0.02)

    query = torch.randn(
        num_queries,
        num_query_heads,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    block_table = torch.empty(
        num_queries,
        2,
        dtype=torch.int32,
        device="cuda",
    )
    for query_idx in range(num_queries):
        if query_idx % 2 == 0:
            block_table[query_idx] = torch.tensor([0, 1], device="cuda")
        else:
            block_table[query_idx] = torch.tensor([2, 3], device="cuda")
    seq_lens = torch.arange(
        block_size + 1,
        block_size + 1 + num_queries,
        dtype=torch.int32,
        device="cuda",
    ).clamp_max(2 * block_size)
    output = torch.empty_like(query)
    softmax_scale = 1.0 / math.sqrt(head_size)

    def reference() -> torch.Tensor:
        key_scale_values = key_scales.repeat_interleave(32, dim=-1).float()
        value_scale_values = value_scales.repeat_interleave(32, dim=-1).float()
        dequant_key = (key_cache.float() * key_scale_values[:, None]).half().float()
        dequant_value = (
            (value_cache.float() * value_scale_values[:, None]).half().float()
        )
        result = torch.empty_like(output, dtype=torch.float32)
        for query_idx in range(num_queries):
            seq_len = int(seq_lens[query_idx].item())
            token_indices = torch.arange(seq_len, device="cuda")
            physical_pages = block_table[
                query_idx,
                token_indices // block_size,
            ].to(torch.long)
            page_offsets = token_indices % block_size
            key_sequence = dequant_key[physical_pages, page_offsets, 0]
            value_sequence = dequant_value[physical_pages, page_offsets, 0]
            scores = torch.einsum(
                "hd,td->ht",
                query[query_idx].float(),
                key_sequence,
            )
            probabilities = torch.softmax(scores * softmax_scale, dim=-1)
            result[query_idx] = torch.einsum(
                "ht,td->hd",
                probabilities,
                value_sequence,
            )
        return result

    def run_decode() -> None:
        extension.int8_block32_decode_paged(
            query,
            key_cache,
            value_cache,
            key_scales,
            value_scales,
            block_table,
            seq_lens,
            output,
            softmax_scale,
        )

    run_decode()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_decode()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), reference(), atol=2e-3, rtol=2e-3)

    seq_lens.copy_(torch.flip(seq_lens, dims=[0]))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), reference(), atol=2e-3, rtol=2e-3)
