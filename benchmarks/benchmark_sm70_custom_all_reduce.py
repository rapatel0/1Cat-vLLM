# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate generic custom allreduce on SM70.

This is intentionally model-free.  It isolates the production custom TP
allreduce path, including CUDA graph registered-buffer capture, and compares
it against NCCL/PyTorch all_reduce.

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    benchmarks/benchmark_sm70_custom_all_reduce.py \
    --json-out bench_results/sm70_migration_20260602/custom_ar.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _make_input(
    size: int,
    dtype: torch.dtype,
    rank: int,
    pattern: str,
    seed: int,
) -> torch.Tensor:
    base = torch.arange(size, device="cuda", dtype=torch.float32)
    if pattern == "exact_int":
        tensor = ((base % 23) - 11 + rank).to(dtype)
    elif pattern == "rank_marker":
        tensor = torch.full((size,), rank + 1, device="cuda", dtype=dtype)
    elif pattern == "random_small":
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + rank)
        tensor = (torch.rand(size, device="cuda", generator=generator) - 0.5).to(
            dtype
        )
    elif pattern == "model_like":
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + rank)
        tensor = (torch.randn(size, device="cuda", generator=generator) * 0.03).to(
            dtype
        )
    else:
        raise ValueError(f"unsupported pattern: {pattern}")
    return tensor


def _reference_all_reduce(inp: torch.Tensor) -> torch.Tensor:
    ref = inp.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    return ref


def _custom_allreduce_algo(
    world_size: int,
    fully_connected: bool,
    num_bytes: int,
) -> str:
    env_algo = os.environ.get("VLLM_CUSTOM_ALLREDUCE_ALGO")
    if env_algo in ("1stage", "oneshot"):
        return "1stage"
    if env_algo in ("2stage", "twoshot"):
        return "2stage"
    if env_algo:
        raise ValueError(f"unsupported VLLM_CUSTOM_ALLREDUCE_ALGO={env_algo!r}")
    if world_size == 2:
        return "1stage"
    if fully_connected:
        if (world_size <= 4 and num_bytes < 512 * 1024) or (
            world_size <= 8 and num_bytes < 256 * 1024
        ):
            return "1stage"
        return "2stage"
    raise RuntimeError("custom allreduce should not be active without full P2P")


def _downcast_like(acc: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return acc if dtype == torch.float32 else acc.to(dtype)


def _reference_custom_order(
    inp: torch.Tensor,
    algo: str,
    world_size: int,
) -> torch.Tensor:
    gathered = [torch.empty_like(inp) for _ in range(world_size)]
    dist.all_gather(gathered, inp)
    torch.cuda.synchronize()
    if algo == "1stage":
        acc = gathered[0].float()
        for item in gathered[1:]:
            acc = acc + item.float()
        return _downcast_like(acc, inp.dtype)

    if algo != "2stage":
        raise ValueError(f"unsupported custom allreduce algo: {algo}")

    ref = torch.empty_like(inp)
    part = inp.numel() // world_size
    for partition in range(world_size):
        start = partition * part
        end = inp.numel() if partition == world_size - 1 else start + part
        acc = gathered[partition][start:end].float()
        for offset in range(1, world_size):
            rank = (partition + offset) % world_size
            acc = acc + gathered[rank][start:end].float()
        ref[start:end] = _downcast_like(acc, inp.dtype)
    return ref


def _compare(out: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    diff = (out.float() - ref.float()).abs()
    mismatch = out != ref
    nonzero = int(mismatch.sum().item())
    first_mismatch: dict[str, Any] | None = None
    if nonzero:
        idx = int(torch.nonzero(mismatch, as_tuple=False)[0].item())
        first_mismatch = {
            "index": idx,
            "out": float(out.flatten()[idx].float().item()),
            "ref": float(ref.flatten()[idx].float().item()),
            "diff": float(diff.flatten()[idx].item()),
        }
    return {
        "equal": bool(torch.equal(out, ref)),
        "max_diff": float(diff.max().item()),
        "mean_diff": float(diff.mean().item()),
        "nonzero_count": nonzero,
        "out_checksum": float(out.float().sum().item()),
        "ref_checksum": float(ref.float().sum().item()),
        "first_mismatch": first_mismatch,
    }


def _run_eager(ca: CustomAllreduce, inp: torch.Tensor) -> torch.Tensor:
    torch.cuda.synchronize()
    dist.barrier()
    out = ca.custom_all_reduce(inp)
    if out is None:
        raise RuntimeError("custom allreduce rejected eager input")
    torch.cuda.synchronize()
    dist.barrier()
    return out


def _run_graph(
    ca: CustomAllreduce,
    inputs: list[torch.Tensor],
    replays: int,
) -> list[torch.Tensor]:
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with ca.capture(), torch.cuda.graph(graph):
        outputs = [ca.custom_all_reduce(inp) for inp in inputs]
    torch.cuda.synchronize()
    dist.barrier()
    if any(out is None for out in outputs):
        raise RuntimeError("custom allreduce rejected graph input")
    concrete_outputs = [out for out in outputs if out is not None]
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    return concrete_outputs


def _run_graph_latency(
    ca: CustomAllreduce,
    inputs: list[torch.Tensor],
    warmup_replays: int,
    timed_replays: int,
) -> tuple[list[torch.Tensor], float]:
    """Time a graph containing several production-shaped reductions.

    Putting several reductions in one graph amortizes the Python graph-replay
    call and CUDA event overhead.  The reported value is milliseconds per
    custom all-reduce, not milliseconds per graph replay.
    """
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with ca.capture(), torch.cuda.graph(graph):
        outputs = [ca.custom_all_reduce(inp) for inp in inputs]
    if any(out is None for out in outputs):
        raise RuntimeError("custom allreduce rejected graph input")
    concrete_outputs = [out for out in outputs if out is not None]

    for _ in range(warmup_replays):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timed_replays):
        graph.replay()
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)
    latency_ms = elapsed_ms / (timed_replays * len(inputs))
    dist.barrier()
    return concrete_outputs, latency_ms


def _run_warmup(ca: CustomAllreduce, inp: torch.Tensor) -> torch.Tensor:
    torch.cuda.synchronize()
    dist.barrier()
    with ca.capture():
        out = ca.custom_all_reduce(inp)
    if out is None:
        raise RuntimeError("custom allreduce rejected warmup input")
    torch.cuda.synchronize()
    dist.barrier()
    return out


def _run_compile_graph(
    ca: CustomAllreduce,
    inp: torch.Tensor,
    replays: int,
) -> torch.Tensor:
    def _compiled_fn(x: torch.Tensor) -> torch.Tensor:
        return ca.all_reduce(x, registered=False)

    torch.cuda.synchronize()
    dist.barrier()
    compiled_fn = torch.compile(_compiled_fn, fullgraph=True, backend="inductor")
    out = compiled_fn(inp)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with ca.capture(), torch.cuda.graph(graph):
        out = compiled_fn(inp)
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    return out


def _check_single(
    ca: CustomAllreduce,
    mode: str,
    size: int,
    dtype: torch.dtype,
    rank: int,
    pattern: str,
    seed: int,
    graph_replays: int,
) -> dict[str, Any]:
    inp = _make_input(size, dtype, rank, pattern, seed)
    num_bytes = size * inp.element_size()
    algo = _custom_allreduce_algo(ca.world_size, ca.fully_connected, num_bytes)
    nccl_ref = _reference_all_reduce(inp)
    custom_order_ref = _reference_custom_order(inp, algo, ca.world_size)
    if mode == "eager":
        out = _run_eager(ca, inp)
    elif mode == "warmup":
        out = _run_warmup(ca, inp)
    elif mode == "graph":
        out = _run_graph(ca, [inp], graph_replays)[0]
    elif mode == "compile_graph":
        out = _run_compile_graph(ca, inp, graph_replays)
    else:
        raise ValueError(f"unsupported single mode: {mode}")
    return {
        "mode": mode,
        "size": size,
        "bytes": num_bytes,
        "dtype": str(dtype).replace("torch.", ""),
        "pattern": pattern,
        "custom_allreduce_algo": algo,
        "comparison": _compare(out, nccl_ref),
        "custom_order_comparison": _compare(out, custom_order_ref),
    }


def _check_graph_multi(
    ca: CustomAllreduce,
    size: int,
    dtype: torch.dtype,
    rank: int,
    pattern: str,
    seed: int,
    graph_replays: int,
    multi_count: int,
    latency_warmup_replays: int,
    latency_replays: int,
    cooperative_ctas_per_row: int,
    cooperative_threads: int,
    measure_latency: bool = False,
) -> dict[str, Any]:
    inputs = [
        _make_input(size, dtype, rank, pattern, seed + i * 1009)
        for i in range(multi_count)
    ]
    refs = [_reference_all_reduce(inp) for inp in inputs]
    if measure_latency:
        outputs, latency_ms = _run_graph_latency(
            ca,
            inputs,
            latency_warmup_replays,
            latency_replays,
        )
    else:
        outputs = _run_graph(ca, inputs, graph_replays)
        latency_ms = None
    comparisons = [_compare(out, ref) for out, ref in zip(outputs, refs)]
    max_diff = max(item["max_diff"] for item in comparisons)
    nonzero_count = sum(item["nonzero_count"] for item in comparisons)
    first_bad = next(
        (
            {"op_index": i, "comparison": item}
            for i, item in enumerate(comparisons)
            if not item["equal"]
        ),
        None,
    )
    return {
        "mode": "graph_multi_latency" if measure_latency else "graph_multi",
        "size": size,
        "bytes": size * torch.empty((), dtype=dtype).element_size(),
        "dtype": str(dtype).replace("torch.", ""),
        "pattern": pattern,
        "multi_count": multi_count,
        "all_equal": all(item["equal"] for item in comparisons),
        "max_diff": float(max_diff),
        "nonzero_count": int(nonzero_count),
        "first_bad": first_bad,
        "latency_ms_per_allreduce": latency_ms,
        "block_limit": os.environ.get(
            "VLLM_CUSTOM_ALLREDUCE_BLOCK_LIMIT", "default"
        ),
    }


def _check_graph_fused_gemma(
    ca: CustomAllreduce,
    size: int,
    rank: int,
    input_pattern: str,
    residual_pattern: str,
    seed: int,
    multi_count: int,
    latency_warmup_replays: int,
    latency_replays: int,
    cooperative_ctas_per_row: int,
    cooperative_threads: int,
) -> dict[str, Any]:
    if size not in (5120, 10240, 20480):
        raise ValueError(
            "fused Gemma mode supports only 5120, 10240, or 20480 elements"
        )
    rows = size // 5120
    inputs = [
        _make_input(
            size,
            torch.float16,
            rank,
            input_pattern,
            seed + i * 1009,
        )
        .reshape(rows, 5120)
        .contiguous()
        for i in range(multi_count)
    ]
    residuals = []
    gammas = []
    reference_norms = []
    reference_residuals = []
    row_sum_workspaces = []
    for index, inp in enumerate(inputs):
        reduced = _reference_custom_order(inp.flatten(), "1stage", ca.world_size)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + index * 2017)
        if residual_pattern == "random_model":
            residual = torch.randn(
                inp.shape,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            )
        elif residual_pattern == "zero":
            residual = torch.zeros_like(inp, dtype=torch.float32)
        elif residual_pattern == "large":
            residual = (
                torch.randn(
                    inp.shape,
                    device="cuda",
                    dtype=torch.float32,
                    generator=generator,
                )
                * 128.0
            )
        elif residual_pattern == "cancellation":
            delta = (
                (torch.arange(size, device="cuda", dtype=torch.float32) % 17)
                - 8.0
            ).reshape_as(inp) * 1e-4
            residual = -reduced.reshape_as(inp).float() + delta
        else:
            raise ValueError(f"unsupported fused residual pattern: {residual_pattern}")
        generator.manual_seed(seed + index * 3019)
        gamma = (
            torch.randn(
                (5120,), device="cuda", dtype=torch.float32, generator=generator
            )
            * 0.05
        ).half()
        reference_residual = reduced.reshape_as(inp).float() + residual
        inverse_rms = torch.rsqrt(
            reference_residual.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        reference_norm = (
            reference_residual * inverse_rms * (gamma.float() + 1.0)
        ).half()
        residuals.append(residual)
        gammas.append(gamma)
        reference_norms.append(reference_norm)
        reference_residuals.append(reference_residual)
        row_sum_workspaces.append(
            torch.empty((rows, 4), device="cuda", dtype=torch.float32)
        )

    torch.cuda.synchronize()
    dist.barrier()
    baseline_graph = torch.cuda.CUDAGraph()
    fused_graph = torch.cuda.CUDAGraph()

    def baseline_call(
        inp: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduced = ca.custom_all_reduce(inp)
        if reduced is None:
            raise RuntimeError("custom all-reduce rejected baseline input")
        residual_out = reduced.float() + residual
        inverse_rms = torch.rsqrt(
            residual_out.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        norm_out = (
            residual_out * inverse_rms * (gamma.float() + 1.0)
        ).half()
        return norm_out, residual_out

    def fused_call(
        inp: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(
            torch.ops.qwen38_c2_ar,
            "all_reduce_gemma_rms_norm_sm70_cooperative",
        ):
            norm_out = torch.empty_like(inp)
            residual_out = torch.empty_like(inp, dtype=torch.float32)
            registered = ca._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
            reg_buffer = 0 if registered else ca.buffer_ptrs[ca.rank]
            reg_buffer_size = 0 if registered else ca.max_size
            workspace = row_sum_workspaces[len(fused_outputs_pending)]
            torch.ops.qwen38_c2_ar.all_reduce_gemma_rms_norm_sm70_cooperative(
                ca._ptr,
                inp,
                residual,
                gamma,
                norm_out,
                residual_out,
                workspace,
                1e-6,
                reg_buffer,
                reg_buffer_size,
                cooperative_ctas_per_row,
                cooperative_threads,
            )
            fused_outputs_pending.append(None)
            return norm_out, residual_out
        if hasattr(ca, "custom_all_reduce_gemma_rms_norm_sm70"):
            output = ca.custom_all_reduce_gemma_rms_norm_sm70(
                inp, residual, gamma, 1e-6
            )
            if output is None:
                raise RuntimeError("fused Gemma custom all-reduce rejected input")
            return output
        norm_out = torch.empty_like(inp)
        residual_out = torch.empty_like(inp, dtype=torch.float32)
        registered = ca._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        reg_buffer = 0 if registered else ca.buffer_ptrs[ca.rank]
        reg_buffer_size = 0 if registered else ca.max_size
        torch.ops.qwen38_c2_ar.all_reduce_gemma_rms_norm_sm70(
            ca._ptr,
            inp,
            residual,
            gamma,
            norm_out,
            residual_out,
            1e-6,
            reg_buffer,
            reg_buffer_size,
        )
        return norm_out, residual_out

    with ca.capture(), torch.cuda.graph(baseline_graph):
        baseline_outputs = [
            baseline_call(inp, residual, gamma)
            for inp, residual, gamma in zip(inputs, residuals, gammas)
        ]
    fused_outputs_pending: list[None] = []
    with ca.capture(), torch.cuda.graph(fused_graph):
        fused_outputs = [
            fused_call(inp, residual, gamma)
            for inp, residual, gamma in zip(inputs, residuals, gammas)
        ]

    for _ in range(latency_warmup_replays):
        baseline_graph.replay()
        fused_graph.replay()
    torch.cuda.synchronize()
    dist.barrier()

    def measure(graph: torch.cuda.CUDAGraph) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(latency_replays):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / (latency_replays * multi_count)

    baseline_latency_ms = measure(baseline_graph)
    fused_latency_ms = measure(fused_graph)
    dist.barrier()

    norm_max_diff = 0.0
    residual_max_diff = 0.0
    norm_allclose = True
    residual_equal = True
    for (norm, residual), ref_norm, ref_residual in zip(
        fused_outputs, reference_norms, reference_residuals
    ):
        norm_max_diff = max(
            norm_max_diff, float((norm.float() - ref_norm.float()).abs().max().item())
        )
        residual_max_diff = max(
            residual_max_diff,
            float((residual - ref_residual).abs().max().item()),
        )
        norm_allclose &= bool(torch.allclose(norm, ref_norm, atol=1e-3, rtol=1e-3))
        residual_equal &= bool(torch.equal(residual, ref_residual))

    passed = norm_allclose and residual_equal
    return {
        "mode": "graph_fused_gemma",
        "size": size,
        "bytes": size * 2,
        "dtype": "float16",
        "pattern": input_pattern,
        "residual_pattern": residual_pattern,
        "multi_count": multi_count,
        "cooperative_ctas_per_row": cooperative_ctas_per_row,
        "cooperative_threads": cooperative_threads,
        "baseline_latency_ms_per_fusion": baseline_latency_ms,
        "fused_latency_ms_per_fusion": fused_latency_ms,
        "speedup": baseline_latency_ms / fused_latency_ms,
        "comparison": {
            "equal": passed,
            "norm_allclose": norm_allclose,
            "norm_max_diff": norm_max_diff,
            "residual_equal": residual_equal,
            "residual_max_diff": residual_max_diff,
        },
    }


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_csv(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default="5120,10240,20480,65536,262144,524288",
        help="Comma-separated tensor numel values. Each must be vector-packable.",
    )
    parser.add_argument("--dtypes", default="float16,float32")
    parser.add_argument(
        "--patterns",
        default="exact_int,rank_marker,random_small,model_like",
    )
    parser.add_argument("--modes", default="eager,graph,graph_multi")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--graph-replays", type=int, default=3)
    parser.add_argument("--multi-count", type=int, default=160)
    parser.add_argument("--latency-warmup-replays", type=int, default=25)
    parser.add_argument("--latency-replays", type=int, default=500)
    parser.add_argument("--cooperative-ctas-per-row", type=int, default=4)
    parser.add_argument("--cooperative-threads", type=int, default=128)
    parser.add_argument(
        "--fused-residual-pattern",
        choices=("random_model", "zero", "large", "cancellation"),
        default="random_model",
    )
    parser.add_argument("--max-size-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument(
        "--require-exact-patterns",
        default="exact_int,rank_marker",
        help="Patterns that must be bitwise equal in every requested mode.",
    )
    parser.add_argument("--json-out")
    parser.add_argument(
        "--standalone-fused-library",
        help="Temporary extension containing the C2 fused operator.",
    )
    args = parser.parse_args()

    if args.standalone_fused_library:
        torch.ops.load_library(args.standalone_fused_library)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    gloo_group = dist.new_group(backend="gloo")

    ca = CustomAllreduce(
        group=gloo_group,
        device=local_rank,
        max_size=args.max_size_bytes,
    )
    results: list[dict[str, Any]] = []
    try:
        if ca.disabled:
            payload = {
                "world_size": world_size,
                "rank": rank,
                "custom_allreduce_disabled": True,
            }
            if rank == 0:
                print(json.dumps(payload, indent=2, sort_keys=True))
            raise SystemExit(2)

        sizes = _parse_csv_ints(args.sizes)
        dtypes = [_dtype(name) for name in _parse_csv(args.dtypes)]
        patterns = _parse_csv(args.patterns)
        modes = _parse_csv(args.modes)
        require_exact_patterns = set(_parse_csv(args.require_exact_patterns))

        for dtype in dtypes:
            pack = 16 // torch.empty((), dtype=dtype).element_size()
            for size in sizes:
                if size % pack != 0:
                    raise ValueError(
                        f"size {size} is not a multiple of vector pack {pack}"
                    )
                for pattern in patterns:
                    for mode in modes:
                        if mode in ("eager", "warmup", "graph", "compile_graph"):
                            results.append(
                                _check_single(
                                    ca,
                                    mode,
                                    size,
                                    dtype,
                                    rank,
                                    pattern,
                                    args.seed,
                                    args.graph_replays,
                                )
                            )
                        elif mode in ("graph_multi", "graph_multi_latency"):
                            results.append(
                                _check_graph_multi(
                                    ca,
                                    size,
                                    dtype,
                                    rank,
                                    pattern,
                                    args.seed,
                                    args.graph_replays,
                                    args.multi_count,
                                    args.latency_warmup_replays,
                                    args.latency_replays,
                                    measure_latency=mode == "graph_multi_latency",
                                )
                            )
                        elif mode == "graph_fused_gemma":
                            if dtype != torch.float16:
                                continue
                            results.append(
                                _check_graph_fused_gemma(
                                    ca,
                                    size,
                                    rank,
                                    pattern,
                                    args.fused_residual_pattern,
                                    args.seed,
                                    args.multi_count,
                                    args.latency_warmup_replays,
                                    args.latency_replays,
                                    args.cooperative_ctas_per_row,
                                    args.cooperative_threads,
                                )
                            )
                        else:
                            raise ValueError(f"unsupported mode: {mode}")

        gathered: list[list[dict[str, Any]]] = [None] * world_size  # type: ignore
        dist.all_gather_object(gathered, results)
        all_results = [
            {"rank": rank_id, "results": rank_results}
            for rank_id, rank_results in enumerate(gathered)
        ]
        required_failures = []
        for rank_id, rank_results in enumerate(gathered):
            for item in rank_results:
                if item["pattern"] not in require_exact_patterns:
                    continue
                if item["mode"] in ("graph_multi", "graph_multi_latency"):
                    equal = item["all_equal"]
                    comparison = item
                else:
                    equal = item["comparison"]["equal"]
                    comparison = item["comparison"]
                if not equal:
                    required_failures.append(
                        {
                            "rank": rank_id,
                            "mode": item["mode"],
                            "size": item["size"],
                            "dtype": item["dtype"],
                            "pattern": item["pattern"],
                            "comparison": comparison,
                        }
                    )

        payload = {
            "world_size": world_size,
            "custom_allreduce_disabled": False,
            "fully_connected": ca.fully_connected,
            "max_size_bytes": ca.max_size,
            "required_exact_patterns": sorted(require_exact_patterns),
            "required_exact_passed": not required_failures,
            "required_failures": required_failures[:16],
            "rank_results": all_results,
        }
        if rank == 0:
            text = json.dumps(payload, indent=2, sort_keys=True)
            print(text)
            if args.json_out:
                path = Path(args.json_out)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text + "\n")
        raise SystemExit(0 if not required_failures else 1)
    finally:
        ca.close()
        dist.destroy_process_group(gloo_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
