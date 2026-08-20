# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

flash_attn_v100 = pytest.importorskip("flash_attn_v100.flash_attn_interface")


def _clear_decode_caches() -> None:
    flash_attn_v100._decode_plan_cache.clear()
    flash_attn_v100._decode_workspace_cache.clear()


def _sm70_device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    for index in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(index) == (7, 0):
            return torch.device(f"cuda:{index}")
    pytest.skip("SM70/V100 CUDA device is required")


def test_short_hd256_decode_default_partition_stays_256(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)

    partition = flash_attn_v100._get_decode_partition_size(
        max_seq_capacity=4096,
        head_dim=256,
        num_q_heads=4,
        num_kv_heads=1,
        max_seq_len_hint=1,
        batch_size_hint=1,
    )

    assert partition == 256


def test_long_hd256_gqa_decode_default_partition_preserves_256(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)

    partition = flash_attn_v100._get_decode_partition_size(
        max_seq_capacity=8192,
        head_dim=256,
        num_q_heads=8,
        num_kv_heads=1,
        max_seq_len_hint=4097,
        batch_size_hint=1,
    )

    assert partition == 256


def test_32k_hd256_gqa_decode_default_partition_uses_1024(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)

    partition = flash_attn_v100._get_decode_partition_size(
        max_seq_capacity=65536,
        head_dim=256,
        num_q_heads=8,
        num_kv_heads=1,
        max_seq_len_hint=32769,
        batch_size_hint=1,
    )

    assert partition == 1024


def test_static_decode_plan_uses_fixed_launch_and_active_runtime(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    _clear_decode_caches()

    q = torch.empty((1, 8, 256), dtype=torch.float16)
    k_cache = torch.empty((512, 16, 1, 256), dtype=torch.float16)
    block_table = torch.zeros((1, 512), dtype=torch.int32)

    plan = flash_attn_v100._get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=4097,
        workspace_seq_capacity_hint=8192,
    )

    assert plan.partition_size == 256
    assert plan.actual_num_partitions == 17
    assert plan.launch_num_partitions == 32
    assert plan.workspace_num_partitions == 32

    tmp_out, max_logits, exp_sums, final_lse, active_num_partitions = (
        flash_attn_v100._get_decode_workspace_for_plan(
            q,
            batch_capacity=1,
            num_heads=8,
            head_dim=256,
            plan=plan,
        )
    )

    assert tmp_out.shape[2] >= plan.launch_num_partitions
    assert max_logits.shape[:3] == tmp_out.shape[:3]
    assert exp_sums.shape[:3] == tmp_out.shape[:3]
    assert final_lse.shape == (1, 8)
    assert final_lse.dtype == torch.float32
    assert active_num_partitions.dtype == torch.int32
    assert active_num_partitions.item() == plan.actual_num_partitions

    reused = flash_attn_v100._get_decode_workspace_for_plan(
        q,
        batch_capacity=1,
        num_heads=8,
        head_dim=256,
        plan=plan,
    )
    assert reused[3] is final_lse
    assert reused[3].data_ptr() == final_lse.data_ptr()


def test_static_decode_cuda_graph_capture_uses_workspace_active(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.setattr(flash_attn_v100, "_cuda_graph_capture_active", lambda: True)
    _clear_decode_caches()

    q = torch.empty((1, 8, 256), dtype=torch.float16)
    k_cache = torch.empty((512, 16, 1, 256), dtype=torch.float16)
    block_table = torch.zeros((1, 512), dtype=torch.int32)

    plan = flash_attn_v100._get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=4097,
        workspace_seq_capacity_hint=8192,
    )

    assert plan.partition_size == 256
    assert plan.actual_num_partitions == 32
    assert plan.launch_num_partitions == 32
    assert plan.workspace_num_partitions == 32


def test_static_decode_cuda_graph_capture_runtime_active_keeps_short_plan(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.setattr(flash_attn_v100, "_cuda_graph_capture_active", lambda: True)
    _clear_decode_caches()

    q = torch.empty((1, 8, 256), dtype=torch.float16)
    k_cache = torch.empty((512, 16, 1, 256), dtype=torch.float16)
    block_table = torch.zeros((1, 512), dtype=torch.int32)
    active_num_partitions = torch.empty((1,), dtype=torch.int32)

    plan = flash_attn_v100._get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=4097,
        workspace_seq_capacity_hint=8192,
        active_num_partitions=active_num_partitions,
    )

    assert plan.partition_size == 256
    assert plan.actual_num_partitions == 17
    assert plan.launch_num_partitions == 32
    assert plan.workspace_num_partitions == 32


@torch.inference_mode()
def test_stale_active_num_partitions_does_not_truncate_decode(
    monkeypatch,
) -> None:
    device = _sm70_device_or_skip()
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.setenv("VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS", "1")
    _clear_decode_caches()

    torch.manual_seed(0)
    seq_len = 513
    block_size = 16
    num_blocks = (seq_len + block_size - 1) // block_size
    q = torch.randn((1, 2, 64), dtype=torch.float16, device=device)
    k_cache = torch.randn(
        (num_blocks, block_size, 1, 64), dtype=torch.float16, device=device
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        1, num_blocks
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    expected_active = torch.tensor([3], dtype=torch.int32, device=device)
    stale_active = torch.tensor([1], dtype=torch.int32, device=device)
    expected = flash_attn_v100.flash_attn_decode_paged(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        max_seq_len_hint=seq_len,
        workspace_seq_capacity_hint=seq_len,
        active_num_partitions=expected_active,
    )
    actual = flash_attn_v100.flash_attn_decode_paged(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        max_seq_len_hint=seq_len,
        workspace_seq_capacity_hint=seq_len,
        active_num_partitions=stale_active,
    )

    torch.cuda.synchronize(device)
    assert torch.equal(actual, expected)


def _workspace_lse_reference(
    max_logits: torch.Tensor,
    exp_sums: torch.Tensor,
    seq_lens: torch.Tensor,
    partition_size: int,
) -> torch.Tensor:
    num_partitions = torch.div(
        seq_lens + partition_size - 1,
        partition_size,
        rounding_mode="floor",
    )
    partition_ids = torch.arange(
        max_logits.shape[-1],
        dtype=num_partitions.dtype,
        device=max_logits.device,
    )
    valid = partition_ids.view(1, 1, -1) < num_partitions.view(-1, 1, 1)
    partition_lse = torch.where(
        (exp_sums > 0) & valid,
        max_logits + torch.log(exp_sums),
        torch.full_like(max_logits, -torch.inf),
    )
    return torch.logsumexp(partition_lse, dim=-1)


def _dense_paged_lse_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    flat_k = k_cache.flatten(0, 1)
    q_per_kv = q.shape[1] // k_cache.shape[2]
    rows = []
    for batch_idx, seq_len_tensor in enumerate(seq_lens.cpu()):
        seq_len = int(seq_len_tensor)
        if seq_len == 0:
            rows.append(
                torch.full(
                    (q.shape[1],),
                    -torch.inf,
                    dtype=torch.float32,
                    device=q.device,
                )
            )
            continue
        keys = flat_k[:seq_len].repeat_interleave(q_per_kv, dim=1)
        scores = torch.einsum("hd,thd->ht", q[batch_idx].float(), keys.float())
        rows.append(torch.logsumexp(scores * (q.shape[-1] ** -0.5), dim=-1))
    return torch.stack(rows)


def _direct_lse_inputs(device: torch.device, *, use_xqa: bool):
    torch.manual_seed(812)
    batch_size = 3
    block_size = 16
    head_dim = 256 if use_xqa else 64
    num_q_heads = 6 if use_xqa else 2
    num_kv_heads = 1
    seq_lens = torch.tensor([0, 257, 513], dtype=torch.int32, device=device)
    max_seq_len = int(seq_lens.max().item())
    num_blocks = (max_seq_len + block_size - 1) // block_size
    q = torch.randn(
        (batch_size, num_q_heads, head_dim),
        dtype=torch.float16,
        device=device,
    )
    k_cache = torch.randn(
        (num_blocks, block_size, num_kv_heads, head_dim),
        dtype=torch.float16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    block_table = (
        torch.arange(num_blocks, dtype=torch.int32, device=device)
        .expand(batch_size, -1)
        .contiguous()
    )
    return q, k_cache, v_cache, block_table, seq_lens, max_seq_len


@pytest.mark.parametrize("use_xqa", [False, True], ids=["scalar", "xqa"])
@torch.inference_mode()
def test_decode_direct_lse_matches_workspace_and_dense_reference(
    monkeypatch, use_xqa: bool
) -> None:
    device = _sm70_device_or_skip()
    if use_xqa and not flash_attn_v100.flash_attn_decode_paged_xqa_available():
        pytest.skip("Flash-V100 extension lacks XQA decode")
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    monkeypatch.setenv("VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS", "1")
    _clear_decode_caches()

    q, k_cache, v_cache, block_table, seq_lens, max_seq_len = _direct_lse_inputs(
        device, use_xqa=use_xqa
    )
    decode = (
        flash_attn_v100.flash_attn_decode_paged_xqa
        if use_xqa
        else flash_attn_v100.flash_attn_decode_paged
    )
    output, direct_lse = decode(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        max_seq_len_hint=max_seq_len,
        workspace_seq_capacity_hint=max_seq_len,
        return_lse=True,
    )

    plan = flash_attn_v100._get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=max_seq_len,
        batch_size_hint=q.shape[0],
        workspace_seq_capacity_hint=max_seq_len,
    )
    _, max_logits, exp_sums, workspace_lse, _ = (
        flash_attn_v100._get_decode_workspace_for_plan(
            q,
            batch_capacity=q.shape[0],
            num_heads=q.shape[1],
            head_dim=q.shape[2],
            plan=plan,
        )
    )
    expected_from_workspace = _workspace_lse_reference(
        max_logits, exp_sums, seq_lens, plan.partition_size
    )
    expected_dense = _dense_paged_lse_reference(q, k_cache, seq_lens)

    torch.cuda.synchronize(device)
    assert output.shape == q.shape
    assert direct_lse.dtype == torch.float32
    assert direct_lse.data_ptr() == workspace_lse.data_ptr()
    assert torch.isneginf(direct_lse[0]).all()
    assert torch.count_nonzero(output[0]) == 0
    torch.testing.assert_close(
        direct_lse, expected_from_workspace, rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(direct_lse, expected_dense, rtol=5e-3, atol=5e-3)


@torch.inference_mode()
def test_decode_without_lse_does_not_write_lse_workspace(monkeypatch) -> None:
    device = _sm70_device_or_skip()
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    _clear_decode_caches()
    q, k_cache, v_cache, block_table, seq_lens, max_seq_len = _direct_lse_inputs(
        device, use_xqa=False
    )
    plan = flash_attn_v100._get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=max_seq_len,
        batch_size_hint=q.shape[0],
        workspace_seq_capacity_hint=max_seq_len,
    )
    workspace = flash_attn_v100._get_decode_workspace_for_plan(
        q,
        batch_capacity=q.shape[0],
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        plan=plan,
    )
    workspace_lse = workspace[3]
    workspace_lse.fill_(123.0)

    result = flash_attn_v100.flash_attn_decode_paged(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        max_seq_len_hint=max_seq_len,
        workspace_seq_capacity_hint=max_seq_len,
        return_lse=False,
    )

    torch.cuda.synchronize(device)
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(workspace_lse, torch.full_like(workspace_lse, 123.0))


@torch.inference_mode()
def test_decode_extension_writes_strided_final_lse(monkeypatch) -> None:
    device = _sm70_device_or_skip()
    monkeypatch.delenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE", raising=False)
    _clear_decode_caches()
    q, k_cache, v_cache, block_table, seq_lens, max_seq_len = _direct_lse_inputs(
        device, use_xqa=False
    )
    plan = flash_attn_v100._get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=max_seq_len,
        batch_size_hint=q.shape[0],
        workspace_seq_capacity_hint=max_seq_len,
    )
    tmp_out, max_logits, exp_sums, _, active_num_partitions = (
        flash_attn_v100._get_decode_workspace_for_plan(
            q,
            batch_capacity=q.shape[0],
            num_heads=q.shape[1],
            head_dim=q.shape[2],
            plan=plan,
        )
    )
    storage = torch.full(
        (q.shape[0], q.shape[1] * 2),
        321.0,
        dtype=torch.float32,
        device=device,
    )
    strided_lse = storage[:, ::2]
    assert not strided_lse.is_contiguous()

    flash_attn_v100.flash_attn_v100_cuda.decode_paged_fwd(
        q,
        k_cache,
        v_cache,
        None,
        block_table,
        seq_lens,
        tmp_out,
        max_logits,
        exp_sums,
        active_num_partitions,
        q.shape[-1] ** -0.5,
        plan.partition_size,
        plan.launch_num_partitions,
        "auto",
        1.0,
        1.0,
        -1,
        -1,
        strided_lse,
    )
    expected = _workspace_lse_reference(
        max_logits, exp_sums, seq_lens, plan.partition_size
    )

    torch.cuda.synchronize(device)
    torch.testing.assert_close(strided_lse, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        storage[:, 1::2], torch.full_like(storage[:, 1::2], 321.0)
    )
