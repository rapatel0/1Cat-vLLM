# SPDX-License-Identifier: Apache-2.0
"""Carrying BF16-range activations through an FP16 store or wire.

Why this exists
---------------
A BF16-native checkpoint run on hardware without BF16 -- Volta (sm_70) is the
case that motivated this -- has to keep its activations somewhere. FP16 has the
same 5-bit exponent budget spent differently: BF16 reaches ~3.4e38, FP16 stops
at 65504. Mantissa is not the problem (FP16 has 10 bits against BF16's 7, so an
FP16 store is *more* precise where it fits); range is.

The split is structural rather than incidental. RMSNorm forces its output to
O(1), so every normalised tensor has a thousandfold of headroom and can be
narrowed for free. It is the unnormalised residual accumulator and the partial
sums feeding it that carry the model's dynamic range, and those are exactly the
tensors worth narrowing because they are the ones on the wire.

The trick is to multiply by a power of two before the cast and divide it back
out afterwards. A power of two shifts the exponent and leaves the mantissa
bit-identical, so the round trip costs nothing beyond the FP16 rounding that
narrowing already implies. Any linear operator in between -- a copy, a sum, a
GEMM, a convolution -- commutes with it exactly.

Static scales and what is wrong with them
-----------------------------------------
The obvious implementation is a module constant picked from a profiling run.
That is a bet that the tensor's maximum never moves, and it is a bet that gets
lost quietly: overflow produces +inf, a collective fans the +inf out to every
rank, and the next layer turns it into NaN logits. Argmax over NaN is token 0,
so the model emits a wall of "!!!!" with nothing in the log.

Widening the constant does not fix the class of bug, it just moves the
threshold, and it is not free at the other end: every extra factor lifts the
FP16 denormal cliff (6.1e-5) by the same factor, and these tensors get added
into a residual stream channel by channel, where a small contribution to a
non-outlier channel is signal rather than noise.

A scale derived from the data has no threshold to exceed. :func:`pow2_scale`
computes the largest power of two that fits the tensor under ``target``, on
device, without a host synchronisation. It is the same idea as dynamic loss
scaling, pointed the other way: loss scaling lifts small gradients off the
denormal floor, this pushes large activations under the overflow ceiling.

When it applies -- and when it does not
---------------------------------------
The scale has to be computed from the tensor that actually gets narrowed. That
sounds trivial and is the whole difficulty, because it splits call sites into
three kinds:

*Transport.* The tensor being measured is the tensor being cast: a wire format
for an all-gather or a reduce-scatter, or a plain store. Measure it, scale it,
cast it, undo the scale on the far side. This is the case this module handles
and the only one where a dynamic scale is unconditionally sound.

*Operator output.* The store that overflows is a GEMM's output, produced inside
a fused kernel, but the only tensor available before the kernel runs is its
input. An input maximum does not bound an output maximum without knowing the
weight. This is not fixable by measuring harder. The principled version is a
load-time bound -- ``|out| <= |in| * max_row_sum(|W|)``, the induced infinity
norm, computed once per weight -- which is guaranteed rather than measured, but
is a worst-case bound and so buys its safety with headroom that a static
constant tuned to the observed maximum may not have to spend. Prefer a static
constant here unless the bound turns out to be tight.

*Stateful store.* The scaled values persist across forward passes -- a paged
cache read back on a later step. A per-call scale silently corrupts those
reads, since step N+1 unscales step N's values with the wrong factor. Either
keep a static constant, or store the scale alongside the data and rescale on
change. A static constant is almost always the right answer.

Cross-rank agreement
--------------------
If a collective sits between the scale and the unscale, ranks that disagree
about the scale produce silently wrong numerics -- a worse failure than the
overflow this replaces. Two shapes work:

*Shared scale.* Required whenever the collective **sums** (reduce-scatter,
all-reduce): the addends must share an exponent for the sum to mean anything.
Use :func:`narrowest`. Note that the post-scale ceiling then has to leave room
for the sum itself, so divide ``target`` by the number of contributions.

*Per-rank scale.* Available when the collective only **moves** data
(all-gather). Each rank scales by its own factor and the factors ride the same
collective, so the far side can undo them block by block. This is both cheaper
in headroom and strictly more precise: one rank's outlier no longer forces every
other rank onto a coarse exponent. Measured on Inkling layer 40, seven ranks sat
near 3e3 while one sat at 8.2e4, so a shared scale would have cost the other
seven about five bits of mantissa for nothing.

Both shapes need one extra small collective per call. Route it through the same
primitive the payload uses (rather than a raw ``torch.distributed`` call) so it
inherits the framework's CUDA-graph capture path and its rank-ordering contract.

Determinism
-----------
A data-dependent scale means the same prompt can round differently depending on
what else is in the batch. That is a real cost for bit-exact reproducibility and
the reason for the :func:`enabled` kill switch: set
``VLLM_FP16_RANGE_DYNAMIC=0`` to fall back to whatever static constants the call
site kept, which also makes an A/B against the static path a one-variable
experiment.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

FP16_MAX = 65504.0

# Where a scaled tensor is aimed. Two binades under FP16_MAX: enough that a
# consumer doing a little arithmetic before the unscale cannot walk off the
# end, and cheap, since the cost of an unused binade is one bit of exponent
# rather than anything mantissa-side.
DEFAULT_TARGET = FP16_MAX / 4.0

# Denormal-ish maxima would make target/amax overflow FP32 on the way to the
# exponent. Anything under this is indistinguishable from an all-zero tensor
# for scaling purposes.
_AMAX_FLOOR = 1e-30


def enabled() -> bool:
    """Whether dynamic scaling is on. See the determinism note above."""
    return os.getenv("VLLM_FP16_RANGE_DYNAMIC", "1").lower() not in (
        "0",
        "false",
        "no",
    )


def pow2_scale(
    t: torch.Tensor,
    target: float = DEFAULT_TARGET,
) -> torch.Tensor:
    """Largest power of two ``s`` with ``t.abs().max() * s <= target``.

    Returns a 0-d FP32 tensor on ``t``'s device. Deliberately never calls
    ``.item()``: the scale stays on the card, so this adds no host
    synchronisation and survives CUDA-graph capture.

    Degenerate inputs all return 1, which leaves the call site's numerics
    exactly as they would have been without scaling:

    - empty ``t`` (the zero-token shapes of the startup profiling pass, which
      make ``amax()`` raise rather than return anything)
    - all-zero ``t``, where no finite scale is determined
    - non-finite ``t``, where something upstream is already broken and a scale
      chosen from inf/NaN would only obscure it
    """
    if t.numel():
        amax = t.detach().abs().amax().to(torch.float32)
    else:
        amax = torch.zeros((), dtype=torch.float32, device=t.device)

    usable = torch.isfinite(amax) & (amax > 0.0)
    # Substituting `target` on the degenerate branch makes the ratio exactly 1,
    # hence a scale of exactly 1, with no second torch.where at the end.
    denom = torch.where(
        usable, amax.clamp_min(_AMAX_FLOOR), torch.full_like(amax, target)
    )
    ratio = target / denom
    # frexp gives ratio = m * 2**exp with m in [0.5, 1), so 2**(exp-1) is the
    # largest power of two <= ratio. Integer arithmetic on the exponent, where
    # exp2(floor(log2(x))) can land on the wrong side of a binade boundary and
    # hand back a scale that overflows by one bit.
    _, exp = torch.frexp(ratio)
    return torch.ldexp(torch.ones_like(ratio), exp - 1)


def narrowest(
    scale: torch.Tensor,
    all_gather: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Reduce per-rank scales to the one every rank can safely use.

    The minimum scale is the one belonging to the rank with the largest
    maximum, so it is the only choice that keeps every rank's values under
    ``target``. Equivalent to a MAX-all-reduce over the maxima, but expressed
    over the scales so it needs only an all-gather -- which is generally the
    primitive a framework has already made CUDA-graph-safe.

    ``all_gather`` is passed in rather than imported so this module stays
    usable outside any one parallelism stack; it must have the usual
    ``(tensor, dim)`` concat semantics.
    """
    return all_gather(scale.reshape(1), -1).amin()


def unscale_blocks(
    gathered: torch.Tensor,
    inv_scales: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Undo per-rank scales on a tensor all-gathered along its last dim.

    ``gathered``'s last dim is the concatenation of ``world_size`` equal blocks
    in rank order, so ``inv_scales[r]`` applies to block ``r``. Mutates
    ``gathered`` in place and returns it.

    The rank-major layout is not an assumption this adds: the payload is only
    correct in the first place if the framework's all-gather concatenates rank
    blocks in rank order along that dim, and gathering the scales through the
    same call inherits the same ordering.
    """
    block = gathered.shape[-1] // world_size
    gathered.view(*gathered.shape[:-1], world_size, block).mul_(
        inv_scales.view(world_size, 1)
    )
    return gathered
