# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel page-size reporting for the Qwen4Exp QSA backend.

Source under test:
  vllm/models/qwen4_exp/nvidia/qsa.py
    Qwen4ExpQSAFlashAttentionBackend.get_supported_kernel_block_sizes
"""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_cache_interface import (
    INT8_BLOCK32_CHANNEL_SIZE,
    FullAttentionSpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.utils import select_common_block_size

# The hybrid Mamba/attention alignment for Qwen3.8 Flash Next on SM70 produces
# this attention page. It is the value the engine logs as
# "Setting attention block size to 816 tokens".
HYBRID_ALIGNED_PAGE = 816


def _supported_sizes():
    return Qwen4ExpQSAFlashAttentionBackend.get_supported_kernel_block_sizes()


def test_qsa_admits_the_hybrid_aligned_page():
    """QSA must be able to run the aligned hybrid page as one kernel page.

    Regression guard. FlashAttentionBackend narrows hybrid float32-Mamba models
    to [16, 32, 64]. If QSA inherited that list again, the only admissible
    factor of 816 would be 16, the scheduler page would be virtual-split, and
    the INT8 equal-page guard would reject the configuration.
    """
    selected = select_common_block_size(
        HYBRID_ALIGNED_PAGE, [Qwen4ExpQSAFlashAttentionBackend]
    )
    assert selected == HYBRID_ALIGNED_PAGE


def test_qsa_does_not_inherit_the_flash_attention_nan_list():
    """QSA must not report the FlashAttention hybrid float32-Mamba list.

    That list exists for FlashAttention kernels. QSA runs its own Triton
    kernels, so inheriting it would only break page alignment.
    """
    sizes = _supported_sizes()
    assert sizes != [16, 32, 64]
    assert any(isinstance(size, MultipleOf) for size in sizes)
    for size in sizes:
        if isinstance(size, MultipleOf):
            assert HYBRID_ALIGNED_PAGE % size.base == 0


def test_flash_attention_backend_is_unchanged():
    """The FlashAttention backend keeps its own reporting.

    QSA overrides only its own class. Any change to FlashAttentionBackend
    would weaken the NaN protection on the real FlashAttention path.
    """
    assert (
        Qwen4ExpQSAFlashAttentionBackend.get_supported_kernel_block_sizes
        is not FlashAttentionBackend.get_supported_kernel_block_sizes
    )


def test_int8_block32_page_stays_one_to_one():
    """The INT8 equal-page contract must hold at the aligned hybrid page.

    An int8_block32 page carries per-page FP16 K and V block scales and an
    int32 publication owner, so one scheduler page must map to exactly one
    kernel page.
    """
    num_kv_heads, head_size = 1, 256
    spec = FullAttentionSpec(
        block_size=HYBRID_ALIGNED_PAGE,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size,
        dtype=torch.int8,
        kv_quant_mode=get_kv_quant_mode("int8_block32"),
    )
    kernel_block_size = select_common_block_size(
        spec.block_size, [Qwen4ExpQSAFlashAttentionBackend]
    )
    # This equality is exactly what _reshape_kv_cache enforces for INT8.
    assert kernel_block_size == spec.block_size

    channel_blocks = head_size // INT8_BLOCK32_CHANNEL_SIZE
    payload = HYBRID_ALIGNED_PAGE * num_kv_heads * head_size
    expected = 2 * payload + 2 * num_kv_heads * channel_blocks * 2 + 4
    assert spec.page_size_bytes == expected


@pytest.mark.parametrize("block_size", [16, 32, 64, 128, 816])
def test_qsa_reports_multiples_of_sixteen(block_size):
    """Every multiple of 16 must remain admissible, including small pages.

    FP16 and BF16 QSA routes rely on the same reporting, so narrowing it would
    change non-INT8 behavior.
    """
    selected = select_common_block_size(block_size, [Qwen4ExpQSAFlashAttentionBackend])
    assert selected == block_size
