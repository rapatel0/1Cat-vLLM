# SPDX-License-Identifier: Apache-2.0
"""Tests for the dynamic power-of-two FP16 range guard.

The bug these exist to prevent: a BF16-range activation cast to FP16 overflows
to +inf on one TP rank, a collective fans the inf out to every rank, and the
next layer turns it into NaN logits, which argmax reads as token 0. The model
then emits a wall of "!!!!" with nothing in the log. Assertions here are
therefore mostly hard invariants rather than golden values.
"""

import pytest
import torch

from vllm.model_executor.layers import fp16_range

TARGET = fp16_range.DEFAULT_TARGET

# The eight per-rank maxima measured at Inkling layer 40 on a 3-tool prompt.
# Rank 3 sat 1.25x past FP16_MAX under the old fixed 1/64 scale and produced
# the +inf. Every other rank had 20x of room, which is why a single shared
# scale is wasteful here.
MEASURED_SHARDS = [3137.0, 2740.0, 2119.0, 8.16e4, 3058.0, 3676.0, 2034.0, 1.73e4]

# Powers of two and values just off them: exp2(floor(log2(x))) lands on the
# wrong side of a binade boundary for some of these, which is why the
# implementation uses frexp.
BINADES = [2.0**k for k in range(-40, 40)]
NEAR_BINADES = [2.0**k * 1.0000001 for k in range(-20, 20)]
PEAKS = [1.9e5, 1.26e6, 5.2e6, 1.0, 1e-8, 1e8, 6.1e-5] + MEASURED_SHARDS


def _stack_gather(scales):
    """Stand-in for an all-gather with (tensor, dim) concat semantics."""

    def all_gather(x, dim):
        return torch.stack(scales).reshape(-1)

    return all_gather


@pytest.mark.parametrize("peak", PEAKS + BINADES + NEAR_BINADES)
def test_pow2_scale_invariants(peak):
    t = torch.tensor([peak, -peak / 3, 0.0], dtype=torch.float32)
    s = fp16_range.pow2_scale(t, TARGET)

    assert s.dtype == torch.float32 and s.dim() == 0
    assert torch.log2(s).item().is_integer(), "scale must be a power of two"

    scaled = (t * s).abs().max().item()
    assert scaled <= TARGET, "scale left the tensor above target"
    assert scaled > TARGET / 2, "scale threw away a whole binade of range"
    assert torch.isfinite((t * s).to(torch.float16)).all()
    # A power of two is exponent-only, so the round trip must not touch a
    # single mantissa bit -- that is the entire reason for the constraint.
    assert torch.equal((t * s) / s, t)


def test_measured_shards_no_longer_overflow():
    for v in MEASURED_SHARDS:
        t = torch.tensor([v], dtype=torch.float32)
        s = fp16_range.pow2_scale(t, TARGET)
        assert torch.isfinite((t * s).to(torch.float16)).all(), v


def test_unscaled_outlier_really_does_overflow():
    """Guards the test above from passing vacuously."""
    assert not torch.isfinite(torch.tensor([8.16e4]).to(torch.float16)).all()


@pytest.mark.parametrize(
    "t",
    [
        torch.zeros(4),
        torch.empty(0),  # zero-token shapes of the startup profiling pass
        torch.tensor([float("inf")]),
        torch.tensor([float("nan")]),
    ],
)
def test_degenerate_inputs_scale_by_one(t):
    """Leave the call site's numerics exactly as if scaling were off."""
    assert fp16_range.pow2_scale(t, TARGET).item() == 1.0


def test_narrowest_is_the_min_and_covers_every_rank():
    scales = [fp16_range.pow2_scale(torch.tensor([v]), TARGET) for v in MEASURED_SHARDS]
    agreed = fp16_range.narrowest(scales[0], _stack_gather(scales))

    assert agreed.item() == min(s.item() for s in scales)
    for v in MEASURED_SHARDS:
        assert v * agreed.item() <= TARGET


def test_shared_scale_survives_the_reduce_scatter_sum():
    """A reduce-scatter sums tp_size addends *after* the cast."""
    tp = 8
    target = fp16_range.FP16_MAX / (2 * tp)
    scales = [fp16_range.pow2_scale(torch.tensor([v]), target) for v in MEASURED_SHARDS]
    agreed = fp16_range.narrowest(scales[0], _stack_gather(scales))

    worst_sum = max(MEASURED_SHARDS) * agreed.item() * tp
    assert worst_sum <= fp16_range.FP16_MAX


def test_per_rank_beats_shared_on_the_measured_data():
    """The reason the all-gather uses per-rank scales rather than one shared.

    FP16 has 10 mantissa bits; a shared scale set by rank 3's outlier costs the
    quieter ranks most of them.
    """
    scales = [fp16_range.pow2_scale(torch.tensor([v]), TARGET) for v in MEASURED_SHARDS]
    agreed = fp16_range.narrowest(scales[0], _stack_gather(scales))
    gain = max(torch.log2(s / agreed).item() for s in scales)
    assert gain >= 5.0


def test_unscale_blocks_is_exact_and_in_place():
    world, block, tokens = 8, 5, 3
    # Decades apart per rank, so a rank-order mistake cannot pass.
    blocks = [torch.randn(tokens, block) * (10.0**i) for i in range(world)]
    truth = torch.cat(blocks, dim=-1)
    scales = torch.stack([fp16_range.pow2_scale(b, TARGET) for b in blocks])

    gathered = torch.cat([b * scales[i] for i, b in enumerate(blocks)], dim=-1)
    out = fp16_range.unscale_blocks(gathered, scales.reciprocal(), world)

    assert torch.equal(out, truth)
    assert out.data_ptr() == gathered.data_ptr()


def test_unscale_blocks_folds_a_second_constant_scale():
    """model.py folds the conv-cache unscale into the per-rank inverses so both
    come off in one pass over the gathered tensor."""
    world, block, cache = 8, 4, 1.0 / 64.0
    blocks = [torch.randn(2, block) * (10.0**i) for i in range(world)]
    truth = torch.cat(blocks, dim=-1)
    scales = torch.stack([fp16_range.pow2_scale(b, TARGET) for b in blocks])

    gathered = torch.cat([b * scales[i] for i, b in enumerate(blocks)], dim=-1)
    inv = scales.reciprocal().div_(cache)
    out = fp16_range.unscale_blocks(gathered, inv, world)

    assert torch.equal(out, truth / cache)


def test_unscale_blocks_tolerates_zero_tokens():
    world, block = 8, 4
    empty = torch.empty(0, world * block)
    scales = torch.ones(world)
    assert fp16_range.unscale_blocks(empty, scales, world).shape == (0, world * block)


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("VLLM_FP16_RANGE_DYNAMIC", "0")
    assert not fp16_range.enabled()
    monkeypatch.setenv("VLLM_FP16_RANGE_DYNAMIC", "1")
    assert fp16_range.enabled()
    monkeypatch.delenv("VLLM_FP16_RANGE_DYNAMIC")
    assert fp16_range.enabled(), "dynamic scaling is the default"
