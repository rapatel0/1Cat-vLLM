# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-module mixed-format lookup for NVFP4/FP8/BF16 checkpoints."""

from __future__ import annotations

from fnmatch import fnmatch

import torch

FORMAT_NVFP4 = "nvfp4"
FORMAT_FP8_BLOCK128 = "fp8_block128"
FORMAT_FP8_RUNTIME_AMAX = "fp8_block128_runtime"
FORMAT_BF16 = "bf16"

_KNOWN = {
    FORMAT_NVFP4,
    FORMAT_FP8_BLOCK128,
    FORMAT_FP8_RUNTIME_AMAX,
    FORMAT_BF16,
}


def lookup_mixed_format(mixed_modules: dict[str, str], prefix: str) -> str | None:
    """Return the unique format for ``prefix``, or None.

    Fail closed when two patterns match the same prefix with different formats.
    """
    if not mixed_modules:
        return None
    hits: list[str] = []
    for pattern, fmt in mixed_modules.items():
        if fmt not in _KNOWN:
            raise ValueError(
                f"unknown mixed_modules format {fmt!r} for pattern {pattern!r}"
            )
        if fnmatch(prefix, pattern):
            hits.append(fmt)
    unique = set(hits)
    if len(unique) > 1:
        raise ValueError(
            f"ambiguous mixed_modules formats for {prefix!r}: {sorted(unique)}"
        )
    if unique:
        return next(iter(unique))
    return None


def validate_serialized_fp8_experts(w13: torch.Tensor, w2: torch.Tensor) -> None:
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "serialized block-FP8 experts must be float8_e4m3fn, "
            f"got w13={w13.dtype} w2={w2.dtype}"
        )
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError(
            f"serialized block-FP8 experts must be rank-3, got {w13.shape} {w2.shape}"
        )


def validate_runtime_amax_experts(w13: torch.Tensor, w2: torch.Tensor) -> None:
    if w13.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            "runtime-amax MTP experts must load as BF16/FP16, "
            f"got w13={w13.dtype}"
        )
    if w2.dtype != w13.dtype:
        raise ValueError(f"runtime-amax w2 dtype {w2.dtype} != w13 {w13.dtype}")
