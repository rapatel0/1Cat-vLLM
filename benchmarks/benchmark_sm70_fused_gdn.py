#!/usr/bin/env python3
"""Falsify the standalone SM70/FP16 fused Qwen3.8 GDN MTP kernel."""

from __future__ import annotations

import argparse
import json
import math

import torch


H = 4
HV = 12
D = 128
TOKENS = 4
CONV_WIDTH = 4
CONV_STATE_LEN = CONV_WIDTH - 1 + TOKENS - 1


def reference(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    selector: torch.Tensor,
    state: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    direct_state_slot: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_out = state.clone()
    output = torch.empty_like(gate)
    selected = int(selector[0])
    source_slot = int(state_indices[0, selected - 1]) if selected > 0 else 0
    ratio = HV // H
    scale = D**-0.5
    for value_head in range(HV):
        key_head = value_head // ratio
        q = mixed_qkv[:, key_head * D : (key_head + 1) * D].float()
        k_start = H * D + key_head * D
        k = mixed_qkv[:, k_start : k_start + D].float()
        v_start = 2 * H * D + value_head * D
        v = mixed_qkv[:, v_start : v_start + D].float()
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1.0e-6) * scale
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1.0e-6)
        decay = torch.exp(
            -torch.exp(a_log[value_head])
            * torch.nn.functional.softplus(a[:, value_head].float() + dt_bias[value_head])
        )
        beta = torch.sigmoid(b[:, value_head].float())
        recurrent = state_out[source_slot, value_head].clone()
        core_rows: list[torch.Tensor] = []
        for token in range(mixed_qkv.size(0)):
            recurrent = recurrent * decay[token]
            hk = recurrent @ k[token]
            delta = (v[token] - hk) * beta[token]
            recurrent = recurrent + delta[:, None] * k[token][None, :]
            core_rows.append((recurrent @ q[token]).half())
            destination = int(state_indices[0, token])
            should_commit = (
                destination >= 0 if direct_state_slot else destination > 0
            )
            if should_commit:
                state_out[destination, value_head] = recurrent
        core = torch.stack(core_rows).float()
        inverse_rms = torch.rsqrt(core.square().mean(-1, keepdim=True) + eps)
        output[:, value_head] = (
            core
            * inverse_rms
            * weight.float()[None, :]
            * torch.nn.functional.silu(gate[:, value_head].float())
        ).half()
    return output, state_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--replays", type=int, default=300)
    parser.add_argument("--calls-per-graph", type=int, default=48)
    parser.add_argument("--tokens", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument(
        "--selector",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help=(
            "State selector used by the q=4 verifier; accepted-prefix lengths "
            "0..3 map to selectors 1..4."
        ),
    )
    parser.add_argument("--zero-state", action="store_true")
    parser.add_argument(
        "--state-slot",
        type=int,
        choices=range(6),
        default=2,
        help="Live cache slot for the q=1 direct-slot validation.",
    )
    parser.add_argument("--no-conv-bias", action="store_true")
    parser.add_argument(
        "--op",
        choices=(
            "fused_gdn_sm70",
            "fused_gdn_sm70_48block",
            "fused_gdn_sm70_48block_full",
            "fused_gdn_sm70_48block_full_q1",
        ),
        default="fused_gdn_sm70",
    )
    args = parser.parse_args()
    torch.ops.load_library(args.library)
    fused_op = getattr(torch.ops.qwen38_u1, args.op)
    torch.manual_seed(20260815)
    device = torch.device("cuda")
    tokens = args.tokens
    raw_storage = (0.05 * torch.randn(tokens, 4096, device=device)).half()
    mixed_qkv = raw_storage[:, :2560]
    ba_storage = (0.05 * torch.randn(tokens, 2 * HV, device=device)).half()
    b = ba_storage[:, :HV]
    a = ba_storage[:, HV:]
    a_log = 0.05 * torch.randn(HV, device=device)
    dt_bias = (0.05 * torch.randn(HV, device=device)).to(torch.bfloat16)
    first_state_slot = (
        args.state_slot
        if args.op == "fused_gdn_sm70_48block_full_q1"
        else 2
    )
    reference_state_indices = torch.tensor(
        [[first_state_slot, 3, 4, 5]], dtype=torch.int32, device=device
    )
    state_indices = (
        reference_state_indices[0, :1].contiguous()
        if args.op == "fused_gdn_sm70_48block_full_q1"
        else reference_state_indices
    )
    query_start_loc = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    selector = torch.tensor([args.selector], dtype=torch.int32, device=device)
    state_seed = 0.01 * torch.randn(6, HV, D, D, device=device)
    if args.zero_state:
        state_seed.zero_()
    gate_storage = (0.05 * torch.randn(tokens, 4096, device=device)).half()
    gate = gate_storage[:, : HV * D].reshape(tokens, HV, D)
    weight = (1.0 + 0.05 * torch.randn(D, device=device)).half()
    eps = 1.0e-6

    conv_state_len = (
        CONV_WIDTH - 1
        if args.op == "fused_gdn_sm70_48block_full_q1"
        else CONV_STATE_LEN
    )
    conv_state_storage = (
        0.02 * torch.randn(6, conv_state_len, 2560, device=device)
    ).half()
    conv_state_seed = conv_state_storage.transpose(1, 2)
    conv_weight = (0.05 * torch.randn(2560, 4, device=device)).half()
    conv_bias = None
    if not args.no_conv_bias:
        conv_bias = (0.05 * torch.randn(2560, device=device)).half()
    conv_bias_arg = (
        conv_bias if conv_bias is not None else conv_weight.reshape(-1)[:1]
    )
    expected_conv_state = conv_state_seed.clone()
    reference_qkv = mixed_qkv
    if args.op in (
        "fused_gdn_sm70_48block_full",
        "fused_gdn_sm70_48block_full_q1",
    ):
        if args.op == "fused_gdn_sm70_48block_full" and tokens != TOKENS:
            raise SystemExit("the speculative full conv fusion is fixed to q=4")
        if args.op == "fused_gdn_sm70_48block_full_q1" and tokens != 1:
            raise SystemExit("the non-spec full conv fusion is fixed to q=1")
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_update,
        )

        if args.op == "fused_gdn_sm70_48block_full":
            reference_qkv = causal_conv1d_update(
                mixed_qkv.clone(),
                expected_conv_state,
                conv_weight,
                conv_bias,
                "silu",
                conv_state_indices=state_indices[:1, 0],
                num_accepted_tokens=selector,
                query_start_loc=query_start_loc,
                max_query_len=TOKENS,
                validate_data=False,
            )
        else:
            reference_qkv = causal_conv1d_update(
                mixed_qkv.clone(),
                expected_conv_state,
                conv_weight,
                conv_bias,
                "silu",
                conv_state_indices=state_indices,
                validate_data=False,
            )

    expected_out, expected_state = reference(
        reference_qkv,
        a,
        b,
        a_log,
        dt_bias,
        reference_state_indices,
        selector,
        state_seed,
        gate,
        weight,
        eps,
        direct_state_slot=args.op == "fused_gdn_sm70_48block_full_q1",
    )
    actual_state = state_seed.clone()
    actual_out = torch.empty_like(gate)
    actual_conv_state = conv_state_seed.clone()

    def invoke(
        recurrent_state: torch.Tensor,
        conv_state: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if args.op == "fused_gdn_sm70_48block_full":
            fused_op(
                mixed_qkv,
                conv_state,
                conv_weight,
                conv_bias_arg,
                a,
                b,
                a_log,
                dt_bias,
                state_indices,
                query_start_loc,
                selector,
                recurrent_state,
                gate,
                weight,
                output,
                D**-0.5,
                eps,
            )
        elif args.op == "fused_gdn_sm70_48block_full_q1":
            fused_op(
                mixed_qkv,
                conv_state,
                conv_weight,
                conv_bias_arg,
                a,
                b,
                a_log,
                dt_bias,
                state_indices,
                query_start_loc,
                recurrent_state,
                gate,
                weight,
                output,
                D**-0.5,
                eps,
            )
        else:
            fused_op(
                mixed_qkv.contiguous(),
                a.contiguous(),
                b.contiguous(),
                a_log,
                dt_bias,
                state_indices,
                query_start_loc,
                selector,
                recurrent_state,
                gate.contiguous(),
                weight,
                output,
                D**-0.5,
                eps,
            )

    invoke(actual_state, actual_conv_state, actual_out)
    torch.cuda.synchronize()
    output_diff = (actual_out.float() - expected_out.float()).abs()
    touched = torch.tensor(
        [args.state_slot]
        if args.op == "fused_gdn_sm70_48block_full_q1"
        else [2, 3, 4, 5],
        device=device,
    )
    state_diff = (
        actual_state.index_select(0, touched)
        - expected_state.index_select(0, touched)
    ).abs()
    output_flat_index = int(output_diff.argmax())
    state_flat_index = int(state_diff.argmax())
    untouched = torch.tensor(
        [slot for slot in range(state_seed.size(0)) if slot not in touched.tolist()],
        device=device,
    )
    untouched_equal = torch.equal(
        actual_state.index_select(0, untouched),
        state_seed.index_select(0, untouched),
    )
    conv_state_diff = (
        actual_conv_state.float() - expected_conv_state.float()
    ).abs()

    timing_state = state_seed.clone()
    timing_conv_state = conv_state_seed.clone()
    timing_out = torch.empty_like(gate)
    graph = torch.cuda.CUDAGraph()
    for _ in range(10):
        invoke(timing_state, timing_conv_state, timing_out)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(args.calls_per_graph):
            invoke(timing_state, timing_conv_state, timing_out)
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.replays):
        graph.replay()
    end.record()
    end.synchronize()
    graph_ms = start.elapsed_time(end) / args.replays

    result = {
        "output_max_abs": output_diff.max().item(),
        "output_mean_abs": output_diff.mean().item(),
        "output_max_index": output_flat_index,
        "output_actual_at_max": actual_out.float().flatten()[output_flat_index].item(),
        "output_expected_at_max": expected_out.float()
        .flatten()[output_flat_index]
        .item(),
        "output_max_magnitude": expected_out.float().abs().max().item(),
        "output_close_3e2": torch.allclose(
            actual_out, expected_out, atol=3.0e-2, rtol=3.0e-2
        ),
        "output_close_5e5": output_diff.max().item() <= 5.0e-5,
        "state_max_abs": state_diff.max().item(),
        "state_mean_abs": state_diff.mean().item(),
        "state_max_index": state_flat_index,
        "state_actual_at_max": actual_state.index_select(0, touched)
        .flatten()[state_flat_index]
        .item(),
        "state_expected_at_max": expected_state.index_select(0, touched)
        .flatten()[state_flat_index]
        .item(),
        "state_max_magnitude": expected_state.index_select(0, touched)
        .abs()
        .max()
        .item(),
        "state_close_3e2": torch.allclose(
            actual_state.index_select(0, touched),
            expected_state.index_select(0, touched),
            atol=3.0e-2,
            rtol=3.0e-2,
        ),
        "state_close_1e5": state_diff.max().item() <= 1.0e-5,
        "untouched_state_bitwise_equal": untouched_equal,
        "conv_state_max_abs": conv_state_diff.max().item(),
        "conv_state_bitwise_equal": torch.equal(
            actual_conv_state, expected_conv_state
        ),
        "calls_per_graph": args.calls_per_graph,
        "op": args.op,
        "tokens": tokens,
        "selector": args.selector,
        "state_slot": args.state_slot,
        "graph_ms": graph_ms,
        "us_per_call": graph_ms * 1000 / args.calls_per_graph,
        "finite": bool(
            math.isfinite(actual_out.float().sum().item())
            and math.isfinite(actual_state.float().sum().item())
        ),
    }
    if args.zero_state and tokens == 1:
        key_head = 6 // (HV // H)
        k_start = H * D + key_head * D
        k_row = mixed_qkv[0, k_start : k_start + D].float()
        k_row = k_row * torch.rsqrt(k_row.square().sum() + 1.0e-6)
        beta_value = torch.sigmoid(b[0, 6].float())
        inferred_v = result["state_actual_at_max"] / (
            k_row[71].item() * beta_value.item()
        )
        all_v = mixed_qkv[0, 2 * H * D :].float().reshape(HV, D)
        closest = int((all_v - inferred_v).abs().argmin())
        result.update(
            {
                "debug_k_col71": k_row[71].item(),
                "debug_beta_h6": beta_value.item(),
                "debug_expected_v_h6_r43": all_v[6, 43].item(),
                "debug_inferred_v": inferred_v,
                "debug_closest_v_head": closest // D,
                "debug_closest_v_row": closest % D,
                "debug_closest_v": all_v.flatten()[closest].item(),
            }
        )
    print(json.dumps(result, sort_keys=True))
    if (
        not untouched_equal
        or not result["finite"]
        or not result["conv_state_bitwise_equal"]
    ):
        raise SystemExit(1)
    if not result["output_close_3e2"] or not result["state_close_3e2"]:
        raise SystemExit(1)
    if args.op == "fused_gdn_sm70_48block_full_q1" and (
        not result["output_close_5e5"] or not result["state_close_1e5"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
