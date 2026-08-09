# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe-DSL linear kernels.

This fork targets SM70 (Volta), where none of the CuTe-DSL kernels are
usable: they are written against the CUTLASS Python DSL and gated on SM90+.
Only the availability probe is provided, so callers take their fallback path
instead of failing at import time.
"""

from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16

__all__ = ["ll_bf16"]
