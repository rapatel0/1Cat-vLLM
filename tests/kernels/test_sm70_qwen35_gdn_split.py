# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.qwen3_5 import (
    _sm70_materialize_qwen35_gdn_splits,
)


@pytest.mark.parametrize("num_rows", [1, 8, 32])
def test_qwen35_gdn_split_materialization_is_bitwise_exact(num_rows: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Qwen3.5 GDN split kernel")

    torch.manual_seed(7)
    qkv_size = 2560
    z_size = 1536
    ba_size = 12

    # Slice wider allocations so the source row strides are deliberately
    # larger than their logical widths, matching the projection-view contract.
    qkvz_storage = torch.randn(
        (num_rows, qkv_size + z_size + 7),
        dtype=torch.float16,
        device="cuda",
    )
    ba_storage = torch.randn(
        (num_rows, 2 * ba_size + 5),
        dtype=torch.float16,
        device="cuda",
    )
    mixed_qkvz = qkvz_storage[:, : qkv_size + z_size]
    mixed_ba = ba_storage[:, : 2 * ba_size]

    actual_z, actual_b, actual_a = _sm70_materialize_qwen35_gdn_splits(
        mixed_qkvz,
        mixed_ba,
        qkv_size,
        z_size,
        ba_size,
    )

    expected_z = mixed_qkvz[:, qkv_size : qkv_size + z_size].contiguous()
    expected_b = mixed_ba[:, :ba_size].contiguous()
    expected_a = mixed_ba[:, ba_size : 2 * ba_size].contiguous()
    assert torch.equal(actual_z, expected_z)
    assert torch.equal(actual_b, expected_b)
    assert torch.equal(actual_a, expected_a)


def test_qwen35_gdn_split_graph_replay_reads_current_projection_values():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Qwen3.5 GDN split kernel")

    num_rows, qkv_size, z_size, ba_size = 8, 2560, 1536, 12
    mixed_qkvz = torch.randn(
        (num_rows, qkv_size + z_size), dtype=torch.float16, device="cuda"
    )
    mixed_ba = torch.randn(
        (num_rows, 2 * ba_size), dtype=torch.float16, device="cuda"
    )
    _sm70_materialize_qwen35_gdn_splits(
        mixed_qkvz, mixed_ba, qkv_size, z_size, ba_size
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_z, actual_b, actual_a = _sm70_materialize_qwen35_gdn_splits(
            mixed_qkvz, mixed_ba, qkv_size, z_size, ba_size
        )

    mixed_qkvz.copy_(torch.randn_like(mixed_qkvz))
    mixed_ba.copy_(torch.randn_like(mixed_ba))
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(actual_z, mixed_qkvz[:, qkv_size:].contiguous())
    assert torch.equal(actual_b, mixed_ba[:, :ba_size].contiguous())
    assert torch.equal(actual_a, mixed_ba[:, ba_size:].contiguous())
