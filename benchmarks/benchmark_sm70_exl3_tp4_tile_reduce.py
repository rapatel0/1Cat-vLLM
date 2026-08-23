#!/usr/bin/env python3
"""Validate and time TP4 EXL3 down-projection tile reduction on SM70."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from safetensors import safe_open

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


DOWN = "model.language_model.layers.0.mlp.down_proj"


def load_tensor(root: Path, index: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(root / index[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def ordered_reference(local: torch.Tensor) -> torch.Tensor:
    rank_inputs = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(rank_inputs, local)
    total = rank_inputs[0].float()
    for rank_input in rank_inputs[1:]:
        total.add_(rank_input.float())
    return total.to(local.dtype)


def median_graph_ms(graph: torch.cuda.CUDAGraph, warmup: int, iterations: int, repeats: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    samples: list[float] = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        dist.barrier()
        samples.append(begin.elapsed_time(end) / iterations)
    return statistics.median(samples)


def capture(ca: CustomAllreduce, body: Callable[[], torch.Tensor]) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    graph = torch.cuda.CUDAGraph()
    with ca.capture():
        with torch.cuda.graph(graph):
            output = body()
    torch.cuda.synchronize()
    dist.barrier()
    return graph, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4, choices=(1, 4))
    parser.add_argument("--splits", type=int, default=6)
    parser.add_argument("--swizzle", type=int, default=2)
    parser.add_argument("--reducer-blocks", default="4,8,16")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError("EXL3 tile-reduce benchmark requires TP4")
    torch.cuda.set_device(local_rank)
    torch.ops.load_library(str(args.library))
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    gloo_group = dist.new_group(backend="gloo")
    ca = CustomAllreduce(group=gloo_group, device=local_rank)

    try:
        if ca.disabled or not ca.fully_connected:
            raise RuntimeError("TP4 custom allreduce is unavailable")
        with (args.checkpoint / "model.safetensors.index.json").open() as handle:
            index = json.load(handle)["weight_map"]
        trellis = load_tensor(args.checkpoint, index, f"{DOWN}.trellis")
        suh = load_tensor(args.checkpoint, index, f"{DOWN}.suh")
        svh = load_tensor(args.checkpoint, index, f"{DOWN}.svh")
        if trellis.shape[0] % world_size or suh.numel() % world_size:
            raise RuntimeError("down projection cannot be row-sharded across TP4")
        trellis = trellis.chunk(world_size, dim=0)[rank].contiguous().cuda()
        suh = suh.chunk(world_size, dim=0)[rank].contiguous().cuda()
        svh = svh.contiguous().cuda()
        state = torch.ops._C.exl3_sm70_tm_state_repack(trellis)
        bits = trellis.shape[2] // 16
        k = state.shape[0] * 16
        n = state.shape[1] * 32
        rows = args.rows

        generator = torch.Generator(device="cuda").manual_seed(380000 + rank)
        x = torch.randn((rows, k), generator=generator, device="cuda", dtype=torch.float16).mul_(0.125)
        x_had = torch.empty_like(x)
        staging = torch.empty((rows, n), device="cuda", dtype=torch.float16)
        reduced = torch.empty_like(staging)
        partials = torch.empty((8, n), device="cuda", dtype=torch.float32)
        locks = torch.zeros(n // 128, device="cuda", dtype=torch.int32)

        def local_projection(out: torch.Tensor) -> torch.Tensor:
            torch.ops._C.exl3_sm70_tm_state_gemm_hadamard_out(
                out,
                x,
                state,
                suh,
                svh,
                x_had,
                partials,
                locks,
                bits,
                args.splits,
                args.swizzle,
            )
            return out

        local_projection(staging)
        torch.cuda.synchronize()
        local_baseline = staging.clone()
        reference = ordered_reference(local_baseline)

        local_graph, local_output = capture(ca, lambda: local_projection(staging))
        local_graph.replay()
        torch.cuda.synchronize()
        local_mismatch = int(
            torch.count_nonzero(
                local_output.view(torch.int16) != local_baseline.view(torch.int16)
            ).item()
        )
        if local_mismatch:
            raise RuntimeError(f"local graph mismatch count={local_mismatch}")
        local_ms = median_graph_ms(
            local_graph, args.warmup, args.iterations, args.repeats
        )

        def copied_allreduce() -> torch.Tensor:
            local_projection(staging)
            return ca.all_reduce(staging)

        baseline_graph, baseline_output = capture(ca, copied_allreduce)
        baseline_graph.replay()
        torch.cuda.synchronize()
        baseline_mismatch = int(
            torch.count_nonzero(
                baseline_output.view(torch.int16) != reference.view(torch.int16)
            ).item()
        )
        if baseline_mismatch:
            raise RuntimeError(
                f"copy+allreduce baseline mismatch count={baseline_mismatch}"
            )
        copied_allreduce_ms = median_graph_ms(
            baseline_graph, args.warmup, args.iterations, args.repeats
        )

        records: list[dict[str, float | int]] = []
        for reducer_blocks in (int(v) for v in args.reducer_blocks.split(",")):
            def candidate() -> torch.Tensor:
                torch.ops._C.exl3_sm70_tm_state_gemm_hadamard_tile_reduce_out(
                    reduced,
                    staging,
                    x,
                    state,
                    suh,
                    svh,
                    x_had,
                    partials,
                    locks,
                    bits,
                    args.splits,
                    args.swizzle,
                    ca._ptr,
                    reducer_blocks,
                )
                return reduced

            graph, output = capture(ca, candidate)
            graph.replay()
            torch.cuda.synchronize()
            mismatch = int(torch.count_nonzero(output.view(torch.int16) != reference.view(torch.int16)).item())
            max_abs = float((output.float() - reference.float()).abs().max().item())
            if mismatch:
                raise RuntimeError(
                    f"tile reducer mismatch blocks={reducer_blocks} mismatch={mismatch} max_abs={max_abs}"
                )
            elapsed = median_graph_ms(
                graph, args.warmup, args.iterations, args.repeats
            )
            graph.replay()
            torch.cuda.synchronize()
            final_mismatch = int(
                torch.count_nonzero(
                    output.view(torch.int16) != reference.view(torch.int16)
                ).item()
            )
            if final_mismatch:
                raise RuntimeError(
                    f"tile reducer final replay mismatch blocks={reducer_blocks} count={final_mismatch}"
                )
            records.append(
                {
                    "reducer_blocks": reducer_blocks,
                    "median_ms": elapsed,
                    "mismatch": mismatch,
                    "max_abs": max_abs,
                    "final_mismatch": final_mismatch,
                }
            )

        gathered: list[list[dict[str, float | int]]] = [None] * world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered, records)
        if rank == 0:
            payload = {
                "rows": rows,
                "k": k,
                "n": n,
                "bits": bits,
                "splits": args.splits,
                "swizzle": args.swizzle,
                "world_size": world_size,
                "local_projection_ms": local_ms,
                "copy_allreduce_ms": copied_allreduce_ms,
                "copy_allreduce_overhead_ms": copied_allreduce_ms - local_ms,
                "rank_results": gathered,
            }
            text = json.dumps(payload, indent=2, sort_keys=True)
            print(text)
            if args.json_out:
                args.json_out.write_text(text + "\n")
    finally:
        ca.close()
        dist.destroy_process_group(gloo_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
