# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility shim for the upstream ``routed_experts`` module split.

Upstream vLLM split ``FusedMoE`` out of ``layer.py`` into a standalone
``RoutedExperts`` class living here, with ``layer.py`` retaining only the
``FusedMoEFactory`` assembly function. This fork predates that split: the
weight-holding module is still ``FusedMoE`` and it constructs its own
``MoERunner`` internally.

Rather than replay the refactor across the whole tree -- which would touch
every MoE model and the SM70 quant paths hanging off them -- this module
re-exports the fork's equivalents under the upstream names so vendored model
code (Inkling) imports cleanly. ``RoutedExperts`` is already aliased to
``FusedMoE`` in ``fused_moe/__init__.py``; this just gives it the module path
upstream code expects.
"""

from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)

# Upstream's RoutedExperts is this fork's FusedMoE: the module that owns the
# w13/w2 expert parameters and the ExpertMapManager.
RoutedExperts = FusedMoE

__all__ = ["FusedMoeWeightScaleSupported", "RoutedExperts"]
