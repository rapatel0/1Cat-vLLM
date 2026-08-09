# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 stand-in for the upstream CuTe-DSL low-latency BF16 GEMM.

Upstream's ``ll_bf16`` is a CUTLASS-Python-DSL skinny GEMM used for small-M
router/gate projections. It requires SM90+ (and the ``cutlass`` DSL package),
neither of which this Volta fork has. Volta additionally has no native BF16
arithmetic at all.

Every upstream caller guards the fast path with ``ll_bf16.is_available()``
alongside an explicit ``has_device_capability(90)`` check and falls back to
``torch.mm`` -- e.g. ``vllm/models/inkling/nvidia/moe.py::_linear_with_fp32_out``.
Reporting unavailable therefore keeps the vendored code on the correct,
numerically equivalent path rather than diverging from it.

``ll_bf16_gemm`` is kept as a loud failure rather than omitted: if a future
vendored model calls it without checking availability first, we want a clear
error naming the reason, not an AttributeError.
"""

from __future__ import annotations

import torch

__all__ = ["is_available", "ll_bf16_gemm"]


def is_available() -> bool:
    """Always False on this fork: the kernel is SM90+ CuTe-DSL only."""
    return False


def ll_bf16_gemm(a: torch.Tensor, b: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    raise RuntimeError(
        "ll_bf16_gemm is unavailable in the SM70 fork (CuTe-DSL kernel requires "
        "SM90+). Guard the call with ll_bf16.is_available() and use the "
        "torch.mm fallback."
    )
