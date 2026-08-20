# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DCP A2A communication backend (no GPU required).

Tests cover:
1. DCP A2A config validation (--dcp-comm-backend)
2. KVP group function exists
3. LSE-weighted combination correctness
"""

import math
from types import SimpleNamespace

try:
    import multiprocess as mp
except ImportError:  # pragma: no cover - minimal GPU qualification images.
    import multiprocessing as mp
import pytest
import torch
import torch.distributed as dist

from vllm.config.parallel import ParallelConfig
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)


class _FakeCPGroup:
    def __init__(self, world_size: int, device_group: dist.ProcessGroup):
        self.world_size = world_size
        self.device_group = device_group


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def _packed_a2a_reference(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    world_size: int,
    h_per_rank: int,
    is_lse_base_on_e: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

    B, _H, D = cp_attn_out.shape
    outputs = (
        cp_attn_out.view(B, world_size, h_per_rank, D)
        .permute(1, 0, 2, 3)
        .contiguous()
        .float()
    )
    lses = cp_attn_lse.view(B, world_size, h_per_rank).permute(1, 0, 2).contiguous()
    return _lse_weighted_combine(
        outputs,
        lses,
        return_lse=True,
        is_lse_base_on_e=is_lse_base_on_e,
    )


def _assert_packed_a2a_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=3e-2, atol=3e-2
        )


def _distributed_run(fn, world_size: int, extra_env: dict[str, str]) -> None:
    port = str(get_open_port())
    processes: list[mp.Process] = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
            **extra_env,
        }
        process = mp.Process(target=fn, args=(env,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join(timeout=120)

    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
        assert process.exitcode == 0


class TestDCPCommBackendConfig:
    """Test --dcp-comm-backend config validation."""

    def test_default_is_ag_rs(self):
        """Default comm backend is ag_rs."""
        config = ParallelConfig()
        assert config.dcp_comm_backend == "ag_rs"

    def test_a2a_requires_dcp_greater_than_1(self):
        """A2A backend requires decode_context_parallel_size > 1."""
        with pytest.raises(
            ValueError, match="requires decode_context_parallel_size > 1"
        ):
            ParallelConfig(
                dcp_comm_backend="a2a",
                decode_context_parallel_size=1,
            )

    def test_a2a_with_dcp_valid(self):
        """A2A backend is valid when DCP > 1."""
        config = ParallelConfig(
            dcp_comm_backend="a2a",
            tensor_parallel_size=2,
            decode_context_parallel_size=2,
        )
        assert config.dcp_comm_backend == "a2a"

    def test_invalid_backend_rejected(self):
        """Invalid backend values are rejected."""
        with pytest.raises(ValueError, match="must be one of|Input should be"):
            ParallelConfig(
                dcp_comm_backend="invalid",
            )

    def test_ag_rs_with_dcp_1_valid(self):
        """ag_rs backend is valid with DCP=1 (no DCP)."""
        config = ParallelConfig(
            dcp_comm_backend="ag_rs",
            decode_context_parallel_size=1,
        )
        assert config.dcp_comm_backend == "ag_rs"


class TestLSEWeightedCombine:
    """Test LSE-weighted combination logic (CPU only, no GPU).

    The _lse_weighted_combine function is the reference implementation
    that verifies the Triton kernel's correctness. It computes:

        result[b,h,d] = sum_n(w_n * output_n[b,h,d])

    where w_n = softmax(lse_n) = exp(lse_n) / sum_k(exp(lse_k))
    """

    def test_importable(self):
        """Verify _lse_weighted_combine is importable."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        assert callable(_lse_weighted_combine)

    def test_single_rank(self):
        """Single rank: output unchanged."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        # N=1, B=2, H=4, D=8
        outputs = torch.randn(1, 2, 4, 8)
        lses = torch.randn(1, 2, 4)

        result = _lse_weighted_combine(outputs, lses)

        assert result.shape == (2, 4, 8)
        torch.testing.assert_close(result, outputs.squeeze(0), rtol=1e-5, atol=1e-5)

    def test_equal_lse(self):
        """Equal LSE values: outputs averaged equally."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        _N, B, H, D = 2, 1, 1, 4
        outputs = torch.tensor(
            [
                [[[1.0, 2.0, 3.0, 4.0]]],  # Rank 0
                [[[5.0, 6.0, 7.0, 8.0]]],  # Rank 1
            ]
        )
        lses = torch.tensor(
            [
                [[0.0]],  # Rank 0
                [[0.0]],  # Rank 1
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        expected = (outputs[0] + outputs[1]) / 2
        assert result.shape == (B, H, D)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_dominant_rank(self):
        """Different LSE values: larger LSE gets more weight."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        B, H, D = 1, 1, 2
        outputs = torch.tensor(
            [
                [[[0.0, 0.0]]],  # Rank 0
                [[[1.0, 1.0]]],  # Rank 1
            ]
        )
        lses = torch.tensor(
            [
                [[-100.0]],  # Rank 0: negligible contribution
                [[0.0]],  # Rank 1: dominant
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        assert result.shape == (B, H, D)
        torch.testing.assert_close(result, outputs[1], atol=1e-5, rtol=1e-5)

    def test_mathematically_correct(self):
        """Verify mathematical correctness of LSE combination."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        outputs = torch.tensor(
            [
                [[[2.0, 4.0]]],
                [[[6.0, 8.0]]],
            ]
        )
        lses = torch.tensor(
            [
                [[1.0]],  # exp(1) ≈ 2.718
                [[2.0]],  # exp(2) ≈ 7.389
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        w0 = math.exp(1) / (math.exp(1) + math.exp(2))
        w1 = math.exp(2) / (math.exp(1) + math.exp(2))
        expected = torch.tensor([[[w0 * 2.0 + w1 * 6.0, w0 * 4.0 + w1 * 8.0]]])

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_return_lse(self):
        """return_lse=True returns global LSE (logsumexp of inputs)."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        B, H, D = 1, 1, 2
        outputs = torch.tensor(
            [
                [[[1.0, 2.0]]],
                [[[3.0, 4.0]]],
            ]
        )
        lses = torch.tensor(
            [
                [[1.0]],
                [[2.0]],
            ]
        )

        result, global_lse = _lse_weighted_combine(outputs, lses, return_lse=True)

        expected_global_lse = math.log(math.exp(1) + math.exp(2))

        assert result.shape == (B, H, D)
        assert global_lse.shape == (B, H)
        assert abs(global_lse.item() - expected_global_lse) < 1e-5

    def test_base2_return_lse(self):
        """Base-2 LSE mode returns log2-sum-exp2 global LSE."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        outputs = torch.tensor(
            [
                [[[1.0, 2.0]]],
                [[[3.0, 4.0]]],
            ]
        )
        lses = torch.tensor(
            [
                [[1.0]],
                [[2.0]],
            ]
        )

        result, global_lse = _lse_weighted_combine(
            outputs,
            lses,
            return_lse=True,
            is_lse_base_on_e=False,
        )

        expected_global_lse = math.log2(2**1 + 2**2)
        w0 = 2**1 / (2**1 + 2**2)
        w1 = 2**2 / (2**1 + 2**2)
        expected = torch.tensor([[[w0 * 1.0 + w1 * 3.0, w0 * 2.0 + w1 * 4.0]]])

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            global_lse,
            torch.tensor([[expected_global_lse]]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_lse_pack_dim(self):
        """Packed A2A stores one fp32 LSE in output-dtype lanes."""
        from vllm.v1.attention.ops.dcp_alltoall import _dcp_a2a_lse_pack_dim

        assert _dcp_a2a_lse_pack_dim(torch.bfloat16) == 2
        assert _dcp_a2a_lse_pack_dim(torch.float16) == 2
        assert _dcp_a2a_lse_pack_dim(torch.float32) == 1


class TestA2AValidation:
    def test_rejects_mismatched_lse_shape(self):
        from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

        with pytest.raises(ValueError, match="LSE shape"):
            dcp_a2a_lse_reduce(
                torch.empty(2, 4, 8),
                torch.empty(2, 3),
                _FakeCPGroup(2, None),  # type: ignore[arg-type]
            )

    def test_rejects_non_fp32_lse(self):
        from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

        with pytest.raises(ValueError, match="requires fp32 LSE"):
            dcp_a2a_lse_reduce(
                torch.empty(2, 4, 8),
                torch.empty(2, 4, dtype=torch.float16),
                _FakeCPGroup(2, None),  # type: ignore[arg-type]
            )


class TestPackedA2AKernels:
    @pytest.mark.skipif(
        torch.accelerator.device_count() < 1, reason="CUDA is required."
    )
    @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
    @pytest.mark.parametrize("return_lse", [False, True])
    @pytest.mark.parametrize("is_lse_base_on_e", [False, True])
    def test_pack_unpack_combine_matches_reference(
        self,
        dtype_name: str,
        return_lse: bool,
        is_lse_base_on_e: bool,
    ):
        from vllm.v1.attention.ops.dcp_alltoall import (
            _dcp_a2a_lse_pack_dim,
            _dcp_a2a_pack_send,
            _dcp_a2a_unpack_combine,
        )

        torch.manual_seed(0)
        dtype = _dtype_from_name(dtype_name)
        device = torch.device("cuda")
        world_size, B, h_per_rank, D = 4, 7, 2, 32
        H = world_size * h_per_rank
        cp_attn_out = torch.randn(B, H, D, device=device, dtype=dtype)
        cp_attn_lse = torch.randn(B, H, device=device, dtype=torch.float32)
        lse_pack_dim = _dcp_a2a_lse_pack_dim(dtype)
        send_buffer = torch.empty(
            (world_size, B, h_per_rank, D + lse_pack_dim),
            device=device,
            dtype=dtype,
        )

        _dcp_a2a_pack_send(
            cp_attn_out,
            cp_attn_lse,
            send_buffer,
            world_size,
            h_per_rank,
            D,
            lse_pack_dim,
        )
        actual = _dcp_a2a_unpack_combine(
            send_buffer, D, lse_pack_dim, return_lse, is_lse_base_on_e
        )
        expected_out, expected_lse = _packed_a2a_reference(
            cp_attn_out, cp_attn_lse, world_size, h_per_rank, is_lse_base_on_e
        )

        if return_lse:
            actual_out, actual_lse = actual
            _assert_packed_a2a_close(actual_out, expected_out, dtype)
            torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-4, atol=1e-4)
        else:
            _assert_packed_a2a_close(actual, expected_out, dtype)

    @pytest.mark.skipif(
        torch.accelerator.device_count() < 1, reason="CUDA is required."
    )
    def test_empty_shards_produce_zero_output_and_negative_infinite_lse(self):
        from vllm.v1.attention.ops.dcp_alltoall import (
            _dcp_a2a_pack_send,
            _dcp_a2a_unpack_combine,
        )

        world_size, batch, h_per_rank, head_dim = 2, 3, 2, 32
        outputs = torch.randn(
            batch,
            world_size * h_per_rank,
            head_dim,
            device="cuda",
            dtype=torch.float16,
        )
        lses = torch.full(
            (batch, world_size * h_per_rank),
            -float("inf"),
            device="cuda",
            dtype=torch.float32,
        )
        packed = torch.empty(
            (world_size, batch, h_per_rank, head_dim + 2),
            device="cuda",
            dtype=torch.float16,
        )
        _dcp_a2a_pack_send(outputs, lses, packed, world_size, h_per_rank, head_dim, 2)
        actual_out, actual_lse = _dcp_a2a_unpack_combine(
            packed, head_dim, 2, True, True
        )
        assert torch.count_nonzero(actual_out) == 0
        assert torch.isneginf(actual_lse).all()

    @pytest.mark.skipif(
        torch.accelerator.device_count() < 1, reason="CUDA is required."
    )
    def test_call_workspace_buffers_are_stable_and_non_aliasing(self):
        from vllm.v1.attention.ops.dcp_alltoall import _dcp_a2a_call_buffers
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            reset_workspace_manager,
        )

        reset_workspace_manager()
        init_workspace_manager(torch.device("cuda"))
        try:
            args = ((2, 5, 3, 258), (5, 3, 256))
            first = _dcp_a2a_call_buffers(
                *args,
                device=torch.device("cuda"),
                dtype=torch.float16,
                return_lse=True,
            )
            second = _dcp_a2a_call_buffers(
                *args,
                device=torch.device("cuda"),
                dtype=torch.float16,
                return_lse=True,
            )
            assert all(a is not None and b is not None for a, b in zip(first, second))
            assert [tensor.data_ptr() for tensor in first if tensor is not None] == [
                tensor.data_ptr() for tensor in second if tensor is not None
            ]
            spans = sorted(
                (
                    tensor.data_ptr(),
                    tensor.data_ptr() + tensor.numel() * tensor.element_size(),
                )
                for tensor in first
                if tensor is not None
            )
            assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
        finally:
            reset_workspace_manager()

    @pytest.mark.skipif(
        torch.accelerator.device_count() < 1, reason="CUDA is required."
    )
    def test_persistent_decode_buffers_are_stable_and_non_aliasing(self):
        from vllm.v1.attention.ops.dcp_alltoall import (
            _dcp_a2a_persistent_buffer_cache,
            _dcp_a2a_persistent_call_buffers,
        )

        _dcp_a2a_persistent_buffer_cache.clear()
        args = ((2, 32, 3, 258), (32, 3, 256))
        first = _dcp_a2a_persistent_call_buffers(
            *args,
            device=torch.device("cuda"),
            dtype=torch.float16,
            return_lse=False,
        )
        second = _dcp_a2a_persistent_call_buffers(
            *args,
            device=torch.device("cuda"),
            dtype=torch.float16,
            return_lse=False,
        )
        assert all(left is right for left, right in zip(first, second))
        spans = sorted(
            (
                tensor.data_ptr(),
                tensor.data_ptr() + tensor.numel() * tensor.element_size(),
            )
            for tensor in first
            if tensor is not None
        )
        assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
        _dcp_a2a_persistent_buffer_cache.clear()

    @pytest.mark.skipif(
        torch.accelerator.device_count() < 1, reason="CUDA is required."
    )
    def test_locked_workspace_allows_explicit_eager_prefill_buffers(self):
        from vllm.v1.attention.ops.dcp_alltoall import _dcp_a2a_call_buffers
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            lock_workspace,
            reset_workspace_manager,
        )

        reset_workspace_manager()
        init_workspace_manager(torch.device("cuda"))
        try:
            _dcp_a2a_call_buffers(
                (2, 1, 3, 258),
                (1, 3, 256),
                device=torch.device("cuda"),
                dtype=torch.float16,
                return_lse=False,
            )
            lock_workspace()
            buffers = _dcp_a2a_call_buffers(
                (2, 8192, 3, 258),
                (8192, 3, 256),
                device=torch.device("cuda"),
                dtype=torch.float16,
                return_lse=True,
                allow_unmanaged_buffers=True,
            )
            assert all(buffer is not None for buffer in buffers)
        finally:
            reset_workspace_manager()


def _distributed_packed_a2a_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="nccl")
    use_workspace = env.get("USE_WORKSPACE") == "1"
    if use_workspace:
        from vllm.v1.worker.workspace import init_workspace_manager

        init_workspace_manager(torch.device(f"cuda:{local_rank}"))
    try:
        from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

        dtype = _dtype_from_name(env["TEST_DTYPE"])
        return_lse = env["RETURN_LSE"] == "1"
        is_lse_base_on_e = env["LSE_BASE_E"] == "1"
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        B = int(env.get("BATCH_TOKENS", "5"))
        h_per_rank, D = 2, 32
        H = world_size * h_per_rank

        generator = torch.Generator(device=f"cuda:{local_rank}")
        generator.manual_seed(1234 + rank)
        cp_attn_out = torch.randn(
            B,
            H,
            D,
            device=f"cuda:{local_rank}",
            dtype=dtype,
            generator=generator,
        )
        cp_attn_lse = torch.randn(
            B,
            H,
            device=f"cuda:{local_rank}",
            dtype=torch.float32,
            generator=generator,
        )
        cp_group = _FakeCPGroup(world_size, dist.group.WORLD)
        use_graph = env.get("USE_CUDA_GRAPH") == "1"
        if use_graph:
            capture_stream = torch.cuda.Stream(device=local_rank)
            capture_stream.wait_stream(torch.cuda.current_stream(local_rank))
            with torch.cuda.stream(capture_stream):
                # Compile Triton and populate the per-stream persistent buffers
                # before capture. The MTP4 c32 shape is 32 requests x 5 tokens.
                dcp_a2a_lse_reduce(
                    cp_attn_out,
                    cp_attn_lse,
                    cp_group,
                    return_lse=return_lse,
                    is_lse_base_on_e=is_lse_base_on_e,
                    use_persistent_buffers=True,
                )
            dist.barrier()
            capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                actual = dcp_a2a_lse_reduce(
                    cp_attn_out,
                    cp_attn_lse,
                    cp_group,
                    return_lse=return_lse,
                    is_lse_base_on_e=is_lse_base_on_e,
                    use_persistent_buffers=True,
                )
            graph.replay()
            torch.cuda.synchronize()
        else:
            actual = dcp_a2a_lse_reduce(
                cp_attn_out,
                cp_attn_lse,
                cp_group,
                return_lse=return_lse,
                is_lse_base_on_e=is_lse_base_on_e,
            )

        gathered_out = [torch.empty_like(cp_attn_out) for _ in range(world_size)]
        gathered_lse = [torch.empty_like(cp_attn_lse) for _ in range(world_size)]
        dist.all_gather(gathered_out, cp_attn_out)
        dist.all_gather(gathered_lse, cp_attn_lse)
        outputs = torch.stack(
            [
                t[:, rank * h_per_rank : (rank + 1) * h_per_rank, :]
                for t in gathered_out
            ],
            dim=0,
        ).float()
        lses = torch.stack(
            [t[:, rank * h_per_rank : (rank + 1) * h_per_rank] for t in gathered_lse],
            dim=0,
        )
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        expected_out, expected_lse = _lse_weighted_combine(
            outputs,
            lses,
            return_lse=True,
            is_lse_base_on_e=is_lse_base_on_e,
        )

        if return_lse:
            actual_out, actual_lse = actual
            _assert_packed_a2a_close(actual_out, expected_out, dtype)
            torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-4, atol=1e-4)
        else:
            _assert_packed_a2a_close(actual, expected_out, dtype)

        if use_graph:
            from vllm.v1.attention.ops.dcp_alltoall import (
                _dcp_a2a_persistent_buffer_cache,
            )

            del actual, graph, capture_stream
            _dcp_a2a_persistent_buffer_cache.clear()
            torch.cuda.synchronize()
    finally:
        if use_workspace:
            from vllm.v1.worker.workspace import reset_workspace_manager

            reset_workspace_manager()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2, reason="Need at least 2 GPUs."
)
def test_distributed_packed_a2a_dcp2_workspace_matches_reference():
    _distributed_run(
        _distributed_packed_a2a_worker,
        world_size=2,
        extra_env={
            "TEST_DTYPE": "float16",
            "RETURN_LSE": "1",
            "LSE_BASE_E": "1",
            "USE_WORKSPACE": "1",
        },
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2, reason="Need at least 2 GPUs."
)
def test_distributed_packed_a2a_dcp2_mtp4_c32_cuda_graph():
    _distributed_run(
        _distributed_packed_a2a_worker,
        world_size=2,
        extra_env={
            "TEST_DTYPE": "float16",
            "RETURN_LSE": "1",
            "LSE_BASE_E": "1",
            "BATCH_TOKENS": "160",
            "USE_CUDA_GRAPH": "1",
        },
    )


def _distributed_direct_query_gather_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="gloo")
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.v1.attention.ops.dcp_query_gather import (
        _dcp_query_gather_buffer_cache,
        dcp_query_all_gather,
    )

    communicator = PyNcclCommunicator(dist.group.WORLD, device=local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = SimpleNamespace(
        world_size=world_size,
        rank_in_group=rank,
        unique_name="dcp:query-gather-test",
        device_communicator=SimpleNamespace(pynccl_comm=communicator),
    )
    graph = None
    capture_stream = None
    try:
        tokens, local_heads, head_dim = 160, 3, 256
        token_values = torch.arange(
            tokens, device=f"cuda:{local_rank}", dtype=torch.float16
        ).view(tokens, 1, 1)
        head_values = torch.arange(
            local_heads, device=f"cuda:{local_rank}", dtype=torch.float16
        ).view(1, local_heads, 1)
        dim_values = (
            torch.arange(head_dim, device=f"cuda:{local_rank}") % 7
        ).to(torch.float16).view(1, 1, head_dim)
        local = token_values + 10 * head_values + dim_values + 1000 * rank
        backing = torch.empty(
            tokens,
            local_heads,
            head_dim * 2,
            device=f"cuda:{local_rank}",
            dtype=torch.float16,
        )
        query = backing[..., ::2]
        query.copy_(local)
        assert not query.is_contiguous()

        first, direct = dcp_query_all_gather(query, group)
        second, direct_second = dcp_query_all_gather(query, group)
        assert direct and direct_second
        assert first.data_ptr() == second.data_ptr()
        expected = torch.cat(
            [local + 1000 * (other_rank - rank) for other_rank in range(world_size)],
            dim=1,
        )
        torch.testing.assert_close(second, expected)

        # A graph replay must read new strided input through the persistent local
        # staging tensor and retain the direct collective/output addresses.
        capture_stream = torch.cuda.Stream(device=local_rank)
        capture_stream.wait_stream(torch.cuda.current_stream(local_rank))
        with torch.cuda.stream(capture_stream):
            warm, warm_direct = dcp_query_all_gather(query, group)
        assert warm_direct
        capture_stream.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            captured, captured_direct = dcp_query_all_gather(query, group)
        assert captured_direct
        query.add_(1)
        graph.replay()
        capture_stream.synchronize()
        torch.testing.assert_close(captured, expected + 1)

        different_shape, shape_direct = dcp_query_all_gather(query[:80], group)
        assert shape_direct
        assert different_shape.data_ptr() != first.data_ptr()

        other_stream = torch.cuda.Stream(device=local_rank)
        with torch.cuda.stream(other_stream):
            stream_output, stream_direct = dcp_query_all_gather(query, group)
        other_stream.synchronize()
        assert stream_direct
        assert stream_output.data_ptr() != first.data_ptr()
        torch.testing.assert_close(stream_output, expected + 1)

        spans_by_key = []
        for buffers in _dcp_query_gather_buffer_cache.values():
            spans = sorted(
                (
                    tensor.data_ptr(),
                    tensor.data_ptr() + tensor.numel() * tensor.element_size(),
                )
                for tensor in buffers
            )
            assert all(
                left[1] <= right[0] for left, right in zip(spans, spans[1:])
            )
            spans_by_key.append(spans)
        assert len(spans_by_key) >= 3
    finally:
        del graph, capture_stream
        _dcp_query_gather_buffer_cache.clear()
        torch.cuda.synchronize()
        communicator.destroy()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2, reason="Need at least 2 GPUs."
)
def test_distributed_direct_query_gather_dcp2_cuda_graph():
    _distributed_run(
        _distributed_direct_query_gather_worker,
        world_size=2,
        extra_env={},
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4, reason="Need at least 4 GPUs."
)
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
def test_distributed_packed_a2a_matches_reference(dtype_name: str):
    _distributed_run(
        _distributed_packed_a2a_worker,
        world_size=4,
        extra_env={
            "TEST_DTYPE": dtype_name,
            "RETURN_LSE": "1",
            "LSE_BASE_E": "1",
        },
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4, reason="Need at least 4 GPUs."
)
def test_distributed_packed_a2a_with_workspace_matches_reference():
    _distributed_run(
        _distributed_packed_a2a_worker,
        world_size=4,
        extra_env={
            "TEST_DTYPE": "bfloat16",
            "RETURN_LSE": "1",
            "LSE_BASE_E": "1",
            "USE_WORKSPACE": "1",
        },
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
