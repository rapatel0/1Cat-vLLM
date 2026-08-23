# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""EXL3 (ExLlamaV3 trellis) quantization support.

Dense-loader provenance: ``local-inference-lab/vllm`` commit
``fa033bd4e1b16d9d729ad94be2d87da5a13210ce``. The SM70 execution adapter is
fork-local; artifact validation and tensor-parallel sharding retain the
Gilded Gnosis behavior.

Rank-sliced routed-expert checkpoints use B12X's unified planned
``fused_moe`` API for the Trellis decode/prefill windows and the ExLlamaV3
extension for the small eager parity window. Generic dense and non-rank-sliced
MoE checkpoints use the
bit-faithful ``exllamav3_ext.exl3_gemm`` parity path. Every logical checkpoint
matrix is dispatched independently: vLLM's packed QKV and gate/up modules are
not treated as one EXL3 matrix because each source matrix owns its Hadamard
vectors and codebook marker.

Both dependencies are imported lazily. Importing this module, parsing
checkpoint metadata, or compiling it with ``py_compile`` does not load either
one or initialize CUDA.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import re
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from transformers import PretrainedConfig

from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    MoEActivation,
    RoutedExperts,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_MCG_SENTINEL = 0xCBAC1FED
_MUL1_SENTINEL = 0x83DCD12D
_HADAMARD_BLOCK = 128
# Shapes qualified by real-checkpoint component tests, graph replay, live
# throughput, and the full stochastic quality gate.  Their decode path needs
# only the persistent INT8 operand plus the original trellis large-M fallback;
# retaining a second expanded exact state is diagnostic duplication.
_SM70_INT8_SHAPES = {
    # Qwen3.8 TP4 GDN q/k, GDN v/z, and full-attention k/v.  These execute as
    # equal-shape grouped pairs; the pair path is qualified separately from
    # the single-matrix dispatcher below.
    (5, 5120, 256),
    (5, 5120, 512),
    (5, 5120, 1536),
    (5, 1536, 5120),
    (5, 5120, 3072),
    (5, 5120, 4352),
    (6, 4352, 5120),
    (6, 5120, 62080),
}
# Full real-checkpoint projection timings for the destination-local raw
# trellis decoder.  Keep this deliberately narrower than the INT8 allowlist:
# these are the shapes where the exact ~5 bpw operand is faster after both
# Hadamards, split-K, and persistent-lock reset are included.
_SM70_RAW_TRELLIS_SHAPES: set[tuple[int, int, int]] = {
    # Empty after 2026-08-18 census: INT8 fused beat raw on (5,1536,5120)
    # (23.5 vs 28.0 us, L2 ~0.0075). Keep the set so RAW_TRELLIS=1 stays
    # fail-closed rather than deleting the override path.
}
_EXL3_EXT: Any | None = None
_B12X_FUSED_MOE_API: Any | None = None
_B12X_MIXED_TRELLIS_API: Any | None = None
_RANK_SLICED_RUNTIMES: dict[tuple[Any, ...], dict[str, Any]] = {}
_MIXED_TRELLIS_RUNTIMES: dict[tuple[Any, ...], dict[str, Any]] = {}
_NEXT_RUNTIME_SCOPE_ID = 0
_SM70_LOADED_LIBRARIES: set[str] = set()


def _ensure_sm70_operator(required_op: str = "exl3_sm70_gemm") -> None:
    """Optionally load the iterate-without-rebuilding SM70 sidecar library."""

    if hasattr(torch.ops._C, required_op):
        return
    if required_op == "exl3_sm70_gemm":
        candidates = (
            os.getenv("VLLM_EXL3_SM70_BASE_LIBRARY"),
            os.getenv("VLLM_EXL3_SM70_LIBRARY"),
        )
    else:
        candidates = (os.getenv("VLLM_EXL3_SM70_LIBRARY"),)
    for library in candidates:
        if library and library not in _SM70_LOADED_LIBRARIES:
            torch.ops.load_library(library)
            _SM70_LOADED_LIBRARIES.add(library)
        if hasattr(torch.ops._C, required_op):
            return


_MIXED_TRELLIS_ROUTE_BLOCK_SIZE = 8


# Smallest m the Trellis kernel path can service, and therefore the smallest
# row count an EXL3 rank-sliced MoE layer can be CUDA-graph captured at. A
# capture-size selector may read this to align its sizes with the backend
# instead of failing at capture time.
MIN_CAPTURABLE_TRELLIS_M = 1

# Target execution also reaches m=1..3 during profiling and small-batch decode.
# Keep every supported row count on the native path by default; operators may
# still raise the threshold explicitly as a diagnostic kill switch.
_DEFAULT_TRELLIS_MIN_M = MIN_CAPTURABLE_TRELLIS_M


def _is_draft_layer(layer: Any) -> bool:
    """True for a rank-sliced MTP/nextn/eagle draft MoE layer.

    The role is stamped (``exl3_is_draft = True``) on every draft-owned module
    by ``load_eagle_model`` at draft construction, which is the single funnel
    every speculator draft passes through. It is NOT inferable here: a
    GLM-5.2-style MTP head is an extra decoder layer named exactly like a
    target layer (``model.layers.78.*`` with ``num_hidden_layers = 78``), and
    this function runs at plan/capture/forward time, when
    ``set_current_vllm_config`` has already exited and
    ``get_current_vllm_config_or_none()`` returns None -- so both name and
    layer-index inference silently fail. The substring fallback below covers
    drafts built outside ``load_eagle_model``.
    """
    stamped = getattr(layer, "exl3_is_draft", None)
    if stamped is not None:
        return bool(stamped)
    name = str(getattr(layer, "layer_name", "") or getattr(layer, "prefix", ""))
    return any(
        token in name
        for token in (".mtp", "mtp.", "nextn", "eagle", "draft", "speculator")
    )


def _runtime_owner_token(quant_config: Any, layer: Any) -> tuple[int, bool]:
    """Runtime-cache owner identity: (config scope, is_draft).

    Adding the role makes target/draft isolation independent of whether the model
    file happened to mint a separate quant config.
    """
    return (_runtime_scope_id(quant_config), _is_draft_layer(layer))


def _runtime_scope_id(quant_config: Any) -> int:
    """Stable identity for the model that owns a rank-sliced runtime.

    A cached runtime owns mutable Trellis/prefill scratch plus parity staging and
    sort buffers, so an entry must never be shared across models. A target MoE
    layer and a rank-sliced MTP draft layer have identical shapes, topk and
    planner settings -- both read ``max_num_batched_tokens`` from the same
    scheduler config -- so a shape-only key makes the draft reuse the target's
    scratch. That defeats the target/draft resource isolation their
    independently captured CUDA graphs rely on.

    Scoping by the owning quant config is deliberately coarser than per-layer:
    the draft is built with its own ``Exl3Config`` while every layer of one model
    shares a single config, so each model gets exactly one runtime. The prefill
    arena alone is ~1 GiB, so per-layer runtimes would cost tens of GiB per rank
    on a 75+ layer model and are not affordable.
    """
    global _NEXT_RUNTIME_SCOPE_ID
    scope = getattr(quant_config, "_exl3_runtime_scope_id", None)
    if scope is not None:
        return scope
    scope = _NEXT_RUNTIME_SCOPE_ID
    _NEXT_RUNTIME_SCOPE_ID += 1
    try:
        quant_config._exl3_runtime_scope_id = scope  # noqa: SLF001
    except AttributeError:
        # Frozen/slotted config: fall back to object identity. Configs live for
        # the process lifetime, so reuse-after-GC aliasing is not a concern here.
        return id(quant_config)
    return scope


_RANK_SLICED_FORMAT = "exl3-trellis"
_RANK_SLICED_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+)\.rank(?P<rank>\d+)\."
    r"(?P<field>trellis|suh|svh|mcg|mul1)$"
)

ShardId = str | int | tuple[int, ...] | None


def _load_exl3_ext() -> Any:
    """Load the existing ExLlamaV3 extension only from an actual CUDA call."""

    global _EXL3_EXT
    if _EXL3_EXT is not None:
        return _EXL3_EXT

    shim = os.environ.get("VLLM_EXL3_ABI_SHIM")
    if shim:
        ctypes.CDLL(shim, mode=ctypes.RTLD_GLOBAL)

    ext_path = os.environ.get("VLLM_EXL3_EXT_PATH")
    if ext_path:
        search_dir = ext_path if os.path.isdir(ext_path) else os.path.dirname(ext_path)
        if search_dir and search_dir not in sys.path:
            sys.path.insert(0, search_dir)

    try:
        ext = importlib.import_module("exllamav3_ext")
    except Exception as exc:
        hint = (
            "Set VLLM_EXL3_EXT_PATH to the directory containing "
            "exllamav3_ext*.so (and VLLM_EXL3_ABI_SHIM when the local "
            "PyTorch ABI shim is required)."
        )
        raise RuntimeError(f"Unable to import exllamav3_ext. {hint}") from exc

    if not hasattr(ext, "exl3_gemm"):
        raise RuntimeError(
            "The imported exllamav3_ext does not export exl3_gemm; rebuild the "
            "track_a_retile extension used by this overlay."
        )
    _EXL3_EXT = ext
    return ext


def _load_b12x_fused_moe() -> Any:
    """Resolve the public unified MoE API lazily."""

    global _B12X_FUSED_MOE_API
    if _B12X_FUSED_MOE_API is not None:
        return _B12X_FUSED_MOE_API
    try:
        from b12x.moe import fused_moe
    except Exception as exc:
        raise RuntimeError(
            "Rank-sliced EXL3 requires the exl3_trellis_mcg source in "
            "b12x.moe.fused_moe. Install a matching B12X build."
        ) from exc
    _B12X_FUSED_MOE_API = fused_moe
    return fused_moe


def _load_b12x_mixed_trellis() -> Any:
    """Resolve the one-grid mixed-bitrate Trellis API lazily."""

    global _B12X_MIXED_TRELLIS_API
    if _B12X_MIXED_TRELLIS_API is not None:
        return _B12X_MIXED_TRELLIS_API
    try:
        module = importlib.import_module(
            "b12x.moe._shared.kernels.w4a16.mixed_trellis"
        )
        prepare = importlib.import_module(
            "b12x.moe._shared.kernels.w4a16.prepare"
        )
        host = importlib.import_module("b12x.moe._shared.kernels.w4a16.host")
    except Exception as exc:
        raise RuntimeError(
            "Mixed-bitrate rank-sliced EXL3 requires the matching B12X "
            "mixed_trellis implementation."
        ) from exc
    api = SimpleNamespace(
        build_tiered_maps=module.build_tiered_maps,
        combine_trellis_rotations=module.combine_trellis_rotations,
        compile_mixed_trellis=module.compile_mixed_trellis,
        make_mixed_trellis_buffers=module.make_mixed_trellis_buffers,
        max_packed_route_slots=host.max_packed_route_slots,
        prepare_weights=prepare.prepare_trellis256_moe_weights,
        run_mixed_trellis=module.run_mixed_trellis,
    )
    _B12X_MIXED_TRELLIS_API = api
    return api


def _unique_tensor_storage_bytes(*buffers: Any) -> int:
    """Count unique tensor storage while ignoring buffer metadata fields."""

    total = 0
    seen: set[tuple[int, int]] = set()
    for buffers_ in buffers:
        for value in vars(buffers_).values():
            if not isinstance(value, torch.Tensor):
                continue
            storage = value.untyped_storage()
            storage_key = (storage.data_ptr(), storage.nbytes())
            if storage_key not in seen:
                seen.add(storage_key)
                total += storage.nbytes()
    return total


def _positive_env_int(name: str, default: int) -> int:
    # An env var that is present but blank means "unset". Compose and Kubernetes
    # both render an unset variable as the empty string, so int("") would abort
    # engine startup with a bare
    #   ValueError: invalid literal for int() with base 10: ''
    # that names neither the variable nor the fix.
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(
            f"{name} must be a positive integer or unset, got {raw!r}"
        ) from None
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@torch.library.custom_op(
    "vllm::exl3_gemm",
    mutates_args=(),
    device_types="cuda",
)
def _exl3_gemm(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    """Opaque torch op around the selected bit-faithful EXL3 dense call."""

    capability = torch.cuda.get_device_capability(x.device)
    if capability[0] == 7:
        _ensure_sm70_operator()
        if not hasattr(torch.ops._C, "exl3_sm70_gemm"):
            raise RuntimeError(
                "EXL3 on SM70 requires the fork-local _C::exl3_sm70_gemm "
                "operator; the loader is active but the V100 kernel was not "
                "built into this image."
            )
        return torch.ops._C.exl3_sm70_gemm(
            x, trellis, suh, svh, mcg, mul1
        )

    ext = _load_exl3_ext()
    output = torch.empty(
        (x.shape[0], trellis.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )
    x_had = torch.empty_like(x)
    ext.exl3_gemm(
        x,
        trellis,
        output,
        suh,
        x_had,
        svh,
        -1,
        mcg,
        mul1,
        0,
    )
    return output


@_exl3_gemm.register_fake
def _exl3_gemm_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    del suh, svh, mcg, mul1
    return torch.empty(
        (x.shape[0], trellis.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )


@torch.library.custom_op(
    "vllm::exl3_sm70_tm_state_gemm",
    mutates_args=(),
    device_types="cuda",
)
def _exl3_sm70_tm_state_gemm(
    x: torch.Tensor,
    state: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    bits: int,
    splits: int,
    swizzle: int,
) -> torch.Tensor:
    """Opaque complete projection using the TurboMind-derived state path."""

    _ensure_sm70_operator("exl3_sm70_tm_state_gemm")
    if not hasattr(torch.ops._C, "exl3_sm70_tm_state_gemm"):
        raise RuntimeError(
            "VLLM_EXL3_SM70_TM_STATE requires a sidecar with "
            "_C::exl3_sm70_tm_state_gemm"
        )
    return torch.ops._C.exl3_sm70_tm_state_gemm(
        x, state, suh, svh, bits, splits, swizzle
    )


@_exl3_sm70_tm_state_gemm.register_fake
def _exl3_sm70_tm_state_gemm_fake(
    x: torch.Tensor,
    state: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    bits: int,
    splits: int,
    swizzle: int,
) -> torch.Tensor:
    del suh, svh, bits, splits, swizzle
    return torch.empty(
        (x.shape[0], state.shape[1] * 32),
        dtype=torch.float16,
        device=x.device,
    )


@torch.library.custom_op(
    "vllm::exl3_sm70_tm_dispatch_gemm",
    mutates_args=(),
    device_types="cuda",
)
def _exl3_sm70_tm_dispatch_gemm(
    x: torch.Tensor,
    trellis: torch.Tensor,
    state: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    bits: int,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    """Runtime-M dispatch between TurboMind decode and exact prefill."""

    _ensure_sm70_operator("exl3_sm70_tm_dispatch_gemm")
    if not hasattr(torch.ops._C, "exl3_sm70_tm_dispatch_gemm"):
        raise RuntimeError(
            "VLLM_EXL3_SM70_TM_STATE requires a sidecar with "
            "_C::exl3_sm70_tm_dispatch_gemm"
        )
    return torch.ops._C.exl3_sm70_tm_dispatch_gemm(
        x, trellis, state, suh, svh, bits, mcg, mul1
    )


@_exl3_sm70_tm_dispatch_gemm.register_fake
def _exl3_sm70_tm_dispatch_gemm_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    state: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    bits: int,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    del state, suh, svh, bits, mcg, mul1
    return torch.empty(
        (x.shape[0], trellis.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )


@torch.library.custom_op(
    "vllm::exl3_sm70_tm_dispatch_gemm_persistent_locks",
    mutates_args=("locks",),
    device_types="cuda",
)
def _exl3_sm70_tm_dispatch_gemm_persistent_locks(
    x: torch.Tensor,
    trellis: torch.Tensor,
    state: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    """Runtime-M dispatch with graph-persistent split-K semaphores."""

    op_name = "exl3_sm70_tm_dispatch_gemm_persistent_locks"
    _ensure_sm70_operator(op_name)
    if not hasattr(torch.ops._C, op_name):
        raise RuntimeError(
            "VLLM_EXL3_SM70_TM_STATE requires a sidecar with "
            f"_C::{op_name}"
        )
    return torch.ops._C.exl3_sm70_tm_dispatch_gemm_persistent_locks(
        x, trellis, state, suh, svh, locks, bits, mcg, mul1
    )


@_exl3_sm70_tm_dispatch_gemm_persistent_locks.register_fake
def _exl3_sm70_tm_dispatch_gemm_persistent_locks_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    state: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    del state, suh, svh, locks, bits, mcg, mul1
    return torch.empty(
        (x.shape[0], trellis.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )


@torch.library.custom_op(
    "vllm::exl3_sm70_tm_raw_dispatch_gemm_persistent_locks",
    mutates_args=("locks",),
    device_types="cuda",
)
def _exl3_sm70_tm_raw_dispatch_gemm_persistent_locks(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    """Runtime-M dispatch for direct compact-trellis SM70 reconstruction."""

    op_name = "exl3_sm70_tm_raw_dispatch_gemm_persistent_locks"
    _ensure_sm70_operator(op_name)
    if not hasattr(torch.ops._C, op_name):
        raise RuntimeError(
            "VLLM_EXL3_SM70_RAW_TRELLIS requires a sidecar with "
            f"_C::{op_name}"
        )
    return torch.ops._C.exl3_sm70_tm_raw_dispatch_gemm_persistent_locks(
        x, trellis, suh, svh, locks, bits, mcg, mul1
    )


@_exl3_sm70_tm_raw_dispatch_gemm_persistent_locks.register_fake
def _exl3_sm70_tm_raw_dispatch_gemm_persistent_locks_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    del suh, svh, locks, bits, mcg, mul1
    return torch.empty(
        (x.shape[0], trellis.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )


@torch.library.custom_op(
    "vllm::exl3_sm70_tm_int8_dispatch_gemm_persistent_locks",
    mutates_args=("locks",),
    device_types="cuda",
)
def _exl3_sm70_tm_int8_dispatch_gemm_persistent_locks(
    x: torch.Tensor,
    trellis: torch.Tensor,
    packed_lane: torch.Tensor,
    tile_scales: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    """Runtime-M dispatch for the SM70 tile-scaled INT8 research path."""

    op_name = "exl3_sm70_tm_int8_dispatch_gemm_persistent_locks"
    _ensure_sm70_operator(op_name)
    if not hasattr(torch.ops._C, op_name):
        raise RuntimeError(
            "VLLM_EXL3_SM70_INT8_REPACK requires a sidecar with "
            f"_C::{op_name}"
        )
    return torch.ops._C.exl3_sm70_tm_int8_dispatch_gemm_persistent_locks(
        x,
        trellis,
        packed_lane,
        tile_scales,
        suh,
        svh,
        locks,
        bits,
        mcg,
        mul1,
    )


@_exl3_sm70_tm_int8_dispatch_gemm_persistent_locks.register_fake
def _exl3_sm70_tm_int8_dispatch_gemm_persistent_locks_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    packed_lane: torch.Tensor,
    tile_scales: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    del packed_lane, tile_scales, suh, svh, locks, bits, mcg, mul1
    return torch.empty(
        (x.shape[0], trellis.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )


@torch.library.custom_op(
    "vllm::exl3_sm70_tm_int8_pair_gemm",
    mutates_args=("locks",),
    device_types="cuda",
)
def _exl3_sm70_tm_int8_pair_gemm(
    x: torch.Tensor,
    trellis0: torch.Tensor,
    trellis1: torch.Tensor,
    packed0: torch.Tensor,
    scales0: torch.Tensor,
    packed1: torch.Tensor,
    scales1: torch.Tensor,
    suh0: torch.Tensor,
    suh1: torch.Tensor,
    svh0: torch.Tensor,
    svh1: torch.Tensor,
    metadata: torch.Tensor,
    offsets: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Run two equal-shape SM70 tile-scaled INT8 projections in one grid."""

    op_name = "exl3_sm70_tm_int8_pair_gemm"
    _ensure_sm70_operator(op_name)
    if not hasattr(torch.ops._C, op_name):
        raise RuntimeError(
            "VLLM_EXL3_SM70_INT8_REPACK requires a sidecar with "
            f"_C::{op_name}"
        )
    return torch.ops._C.exl3_sm70_tm_int8_pair_gemm(
        x,
        trellis0,
        trellis1,
        packed0,
        scales0,
        packed1,
        scales1,
        suh0,
        suh1,
        svh0,
        svh1,
        metadata,
        offsets,
        locks,
        bits,
    )


@_exl3_sm70_tm_int8_pair_gemm.register_fake
def _exl3_sm70_tm_int8_pair_gemm_fake(
    x: torch.Tensor,
    trellis0: torch.Tensor,
    trellis1: torch.Tensor,
    packed0: torch.Tensor,
    scales0: torch.Tensor,
    packed1: torch.Tensor,
    scales1: torch.Tensor,
    suh0: torch.Tensor,
    suh1: torch.Tensor,
    svh0: torch.Tensor,
    svh1: torch.Tensor,
    metadata: torch.Tensor,
    offsets: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    del (
        trellis1,
        packed0,
        scales0,
        packed1,
        scales1,
        suh0,
        suh1,
        svh0,
        svh1,
        metadata,
        offsets,
        locks,
        bits,
    )
    return torch.empty(
        (x.shape[0], 2 * trellis0.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )


@torch.library.custom_op(
    "vllm::exl3_sm70_tm_state_pair_gemm",
    mutates_args=("locks",),
    device_types="cuda",
)
def _exl3_sm70_tm_state_pair_gemm(
    x: torch.Tensor,
    trellis0: torch.Tensor,
    trellis1: torch.Tensor,
    state0: torch.Tensor,
    state1: torch.Tensor,
    suh0: torch.Tensor,
    suh1: torch.Tensor,
    svh0: torch.Tensor,
    svh1: torch.Tensor,
    metadata: torch.Tensor,
    offsets: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """One-grid exact pair of equal-shape SM70 EXL3 projections."""

    op_name = "exl3_sm70_tm_state_pair_gemm"
    _ensure_sm70_operator(op_name)
    if not hasattr(torch.ops._C, op_name):
        raise RuntimeError(
            "VLLM_EXL3_SM70_GROUPED_QKV requires a sidecar with "
            f"_C::{op_name}"
        )
    return torch.ops._C.exl3_sm70_tm_state_pair_gemm(
        x,
        trellis0,
        trellis1,
        state0,
        state1,
        suh0,
        suh1,
        svh0,
        svh1,
        metadata,
        offsets,
        locks,
        bits,
    )


@_exl3_sm70_tm_state_pair_gemm.register_fake
def _exl3_sm70_tm_state_pair_gemm_fake(
    x: torch.Tensor,
    trellis0: torch.Tensor,
    trellis1: torch.Tensor,
    state0: torch.Tensor,
    state1: torch.Tensor,
    suh0: torch.Tensor,
    suh1: torch.Tensor,
    svh0: torch.Tensor,
    svh1: torch.Tensor,
    metadata: torch.Tensor,
    offsets: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    del (
        trellis0,
        trellis1,
        state1,
        suh0,
        suh1,
        svh0,
        svh1,
        metadata,
        offsets,
        locks,
        bits,
    )
    return torch.empty(
        (x.shape[0], 2 * state0.shape[1] * 32),
        dtype=torch.float16,
        device=x.device,
    )


@torch.library.custom_op(
    "vllm::exl3_sm70_tm_state_gate_up_silu_mul",
    mutates_args=("locks",),
    device_types="cuda",
)
def _exl3_sm70_tm_state_gate_up_silu_mul(
    x: torch.Tensor,
    gate_trellis: torch.Tensor,
    up_trellis: torch.Tensor,
    gate_state: torch.Tensor,
    up_state: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    gate_svh: torch.Tensor,
    up_svh: torch.Tensor,
    metadata: torch.Tensor,
    offsets: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """One-grid exact gate/up projection for the SM70 state executor."""

    op_name = "exl3_sm70_tm_state_gate_up_silu_mul"
    _ensure_sm70_operator(op_name)
    if not hasattr(torch.ops._C, op_name):
        raise RuntimeError(
            "VLLM_EXL3_SM70_FUSED_GATE_UP_SILU requires a sidecar with "
            f"_C::{op_name}"
        )
    return torch.ops._C.exl3_sm70_tm_state_gate_up_silu_mul(
        x,
        gate_trellis,
        up_trellis,
        gate_state,
        up_state,
        gate_suh,
        up_suh,
        gate_svh,
        up_svh,
        metadata,
        offsets,
        locks,
        bits,
    )


@_exl3_sm70_tm_state_gate_up_silu_mul.register_fake
def _exl3_sm70_tm_state_gate_up_silu_mul_fake(
    x: torch.Tensor,
    gate_trellis: torch.Tensor,
    up_trellis: torch.Tensor,
    gate_state: torch.Tensor,
    up_state: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    gate_svh: torch.Tensor,
    up_svh: torch.Tensor,
    metadata: torch.Tensor,
    offsets: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    del (
        gate_trellis,
        up_trellis,
        up_state,
        gate_suh,
        up_suh,
        gate_svh,
        up_svh,
        metadata,
        offsets,
        locks,
        bits,
    )
    return torch.empty(
        (x.shape[0], gate_state.shape[1] * 32),
        dtype=torch.float16,
        device=x.device,
    )


@torch.library.custom_op(
    "vllm::exl3_sm70_tm_state_mlp",
    mutates_args=("gate_locks", "down_locks"),
    device_types="cuda",
)
def _exl3_sm70_tm_state_mlp(
    x: torch.Tensor,
    gate_trellis: torch.Tensor,
    up_trellis: torch.Tensor,
    down_trellis: torch.Tensor,
    gate_state: torch.Tensor,
    up_state: torch.Tensor,
    down_state: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    down_suh: torch.Tensor,
    gate_svh: torch.Tensor,
    up_svh: torch.Tensor,
    down_svh: torch.Tensor,
    down_packed_lane: torch.Tensor,
    down_tile_scales: torch.Tensor,
    gate_metadata: torch.Tensor,
    gate_offsets: torch.Tensor,
    gate_locks: torch.Tensor,
    down_locks: torch.Tensor,
    gate_bits: int,
    down_bits: int,
    int8_gate: bool,
    int8_down: bool,
) -> torch.Tensor:
    """SM70 EXL3 gate/up, activation, and local down projection."""

    op_name = "exl3_sm70_tm_state_mlp"
    _ensure_sm70_operator(op_name)
    if not hasattr(torch.ops._C, op_name):
        raise RuntimeError(
            "VLLM_EXL3_SM70_FUSED_MLP requires a sidecar with " f"_C::{op_name}"
        )
    return torch.ops._C.exl3_sm70_tm_state_mlp(
        x,
        gate_trellis,
        up_trellis,
        down_trellis,
        gate_state,
        up_state,
        down_state,
        gate_suh,
        up_suh,
        down_suh,
        gate_svh,
        up_svh,
        down_svh,
        down_packed_lane,
        down_tile_scales,
        gate_metadata,
        gate_offsets,
        gate_locks,
        down_locks,
        gate_bits,
        down_bits,
        int8_gate,
        int8_down,
    )


@_exl3_sm70_tm_state_mlp.register_fake
def _exl3_sm70_tm_state_mlp_fake(
    x: torch.Tensor,
    gate_trellis: torch.Tensor,
    up_trellis: torch.Tensor,
    down_trellis: torch.Tensor,
    gate_state: torch.Tensor,
    up_state: torch.Tensor,
    down_state: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    down_suh: torch.Tensor,
    gate_svh: torch.Tensor,
    up_svh: torch.Tensor,
    down_svh: torch.Tensor,
    down_packed_lane: torch.Tensor,
    down_tile_scales: torch.Tensor,
    gate_metadata: torch.Tensor,
    gate_offsets: torch.Tensor,
    gate_locks: torch.Tensor,
    down_locks: torch.Tensor,
    gate_bits: int,
    down_bits: int,
    int8_gate: bool,
    int8_down: bool,
) -> torch.Tensor:
    del (
        gate_trellis,
        up_trellis,
        gate_state,
        up_state,
        down_state,
        gate_suh,
        up_suh,
        down_suh,
        gate_svh,
        up_svh,
        down_svh,
        down_packed_lane,
        down_tile_scales,
        gate_metadata,
        gate_offsets,
        gate_locks,
        down_locks,
        gate_bits,
        down_bits,
        int8_gate,
        int8_down,
    )
    return torch.empty(
        (x.shape[0], down_trellis.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )


@torch.library.custom_op(
    "vllm::exl3_sm70_gate_up_silu_mul",
    mutates_args=(),
    device_types="cuda",
)
def _exl3_sm70_gate_up_silu_mul(
    x: torch.Tensor,
    gate_trellis: torch.Tensor,
    up_trellis: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    gate_svh: torch.Tensor,
    up_svh: torch.Tensor,
    gate_mcg: bool,
    up_mcg: bool,
    gate_mul1: bool,
    up_mul1: bool,
) -> torch.Tensor:
    """Opaque paired EXL3 epilogue for the specialized SM70 MLP path."""

    _ensure_sm70_operator("exl3_sm70_gate_up_silu_mul")
    if not hasattr(torch.ops._C, "exl3_sm70_gate_up_silu_mul"):
        raise RuntimeError(
            "VLLM_EXL3_SM70_FUSED_GATE_UP_SILU requires a sidecar with "
            "_C::exl3_sm70_gate_up_silu_mul"
        )
    return torch.ops._C.exl3_sm70_gate_up_silu_mul(
        x,
        gate_trellis,
        up_trellis,
        gate_suh,
        up_suh,
        gate_svh,
        up_svh,
        gate_mcg,
        up_mcg,
        gate_mul1,
        up_mul1,
    )


@_exl3_sm70_gate_up_silu_mul.register_fake
def _exl3_sm70_gate_up_silu_mul_fake(
    x: torch.Tensor,
    gate_trellis: torch.Tensor,
    up_trellis: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    gate_svh: torch.Tensor,
    up_svh: torch.Tensor,
    gate_mcg: bool,
    up_mcg: bool,
    gate_mul1: bool,
    up_mul1: bool,
) -> torch.Tensor:
    del (
        up_trellis,
        gate_suh,
        up_suh,
        gate_svh,
        up_svh,
        gate_mcg,
        up_mcg,
        gate_mul1,
        up_mul1,
    )
    return torch.empty(
        (x.shape[0], gate_trellis.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )


class Exl3Config(QuantizationConfig):
    """Configuration for modern and legacy EXL3 trellis checkpoints."""

    def __init__(
        self,
        bits: float | None = None,
        head_bits: float | None = None,
        codebook: str | None = None,
        version: str | None = None,
        tensor_storage: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.head_bits = head_bits
        self.codebook = codebook
        self.version = version
        self.tensor_storage = tensor_storage or {}
        self._eager_checked = False
        self.rank_sliced_metadata: dict[str, Any] | None = None
        self.rank_sliced_k_values: tuple[int, ...] | None = None
        self.rank_sliced_bits_by_layer: dict[int, tuple[int, ...]] = {}

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # The kernel boundary is always fp16.  BF16 model activations are cast
        # in apply() and converted back after the fp16 bias addition.
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # SM70 uses the fork-local operator above. SM80+ retains the proven
        # ExLlamaV3 extension path from the Gilded Gnosis implementation.
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Exl3Config:
        return cls(
            bits=config.get("bits"),
            head_bits=config.get("head_bits"),
            codebook=config.get("codebook"),
            version=config.get("version"),
            tensor_storage=config.get("tensor_storage"),
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: PretrainedConfig | None = None,
    ) -> str | None:
        del hf_quant_cfg
        if user_quant is not None and user_quant != "exl3":
            return None
        metadata = getattr(hf_config, "hybrid_tr3_tail", None)
        if isinstance(metadata, dict) and metadata.get("format") == _RANK_SLICED_FORMAT:
            return "exl3"
        return None

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ) -> None:
        rank_sliced = getattr(hf_config, "hybrid_tr3_tail", None)
        if (
            isinstance(rank_sliced, dict)
            and rank_sliced.get("format") == _RANK_SLICED_FORMAT
        ):
            self._configure_rank_sliced(rank_sliced)
            if self.rank_sliced_k_values is not None:
                resolved_revision = revision
                if resolved_revision is None and hf_config is not None:
                    resolved_revision = getattr(hf_config, "_commit_hash", None)
                self._load_rank_sliced_bitrates(
                    model_name,
                    revision=resolved_revision,
                )
            return

        # vLLM returns the summary embedded in config.json without consulting
        # get_config_filenames().  Hydrate the per-module records explicitly.
        if not self.tensor_storage:
            resolved_revision = revision
            if resolved_revision is None and hf_config is not None:
                resolved_revision = getattr(hf_config, "_commit_hash", None)
            config = get_hf_file_to_dict(
                "quantization_config.json",
                model_name,
                revision=resolved_revision,
            )
            if not config or not config.get("tensor_storage"):
                raise ValueError(
                    "EXL3 requires quantization_config.json with a non-empty "
                    "tensor_storage map. For branch-indexed Hugging Face repos, "
                    "download/serve an actual bpw revision rather than main."
                )
            self.bits = config.get("bits", self.bits)
            self.head_bits = config.get("head_bits", self.head_bits)
            self.codebook = config.get("codebook", self.codebook)
            self.version = config.get("version", self.version)
            self.tensor_storage = config["tensor_storage"]

        self._validate_storage_metadata()
        self._force_independent_lm_head(hf_config)

    def _configure_rank_sliced(self, metadata: dict[str, Any]) -> None:
        required = {
            "bits",
            "codebook",
            "experts_per_layer",
            "moe_layers",
            "tensor_schema",
            "tp",
        }
        missing = sorted(required.difference(metadata))
        if missing:
            raise ValueError(
                "rank-sliced EXL3 metadata is missing: " + ", ".join(missing)
            )
        if metadata["codebook"] != "mcg":
            raise ValueError(
                "rank-sliced EXL3 currently requires the MCG codebook, got "
                f"{metadata['codebook']!r}"
            )
        layers = metadata["moe_layers"]
        if (
            not isinstance(layers, list)
            or len(layers) != 2
            or int(layers[0]) < 0
            or int(layers[1]) < int(layers[0])
        ):
            raise ValueError("rank-sliced EXL3 moe_layers must be [first, last]")
        expected_schema = (
            "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"
        )
        if metadata["tensor_schema"] != expected_schema:
            raise ValueError(
                "unsupported rank-sliced EXL3 tensor schema: "
                f"{metadata['tensor_schema']!r}"
            )
        self.rank_sliced_metadata = dict(metadata)
        bits_field = metadata["bits"]
        if isinstance(bits_field, str) and bits_field.strip().lower() == "mixed":
            k_values = tuple(
                sorted({int(value) for value in metadata.get("k_values", ())})
            )
            if not k_values or any(value not in (3, 4, 5, 6) for value in k_values):
                raise ValueError(
                    "mixed rank-sliced EXL3 requires k_values within 3..6, got "
                    f"{metadata.get('k_values')!r}"
                )
            if not isinstance(metadata.get("bits_per_expert"), str):
                raise ValueError(
                    "mixed rank-sliced EXL3 requires a bits_per_expert JSON reference"
                )
            self.bits = None
            self.rank_sliced_k_values = k_values
        else:
            self.bits = float(bits_field)
            self.rank_sliced_k_values = None
        self.codebook = str(metadata["codebook"])
        self.version = str(metadata.get("exllamav3_version", "rank-sliced"))

    def _load_rank_sliced_bitrates(
        self,
        model_name: str,
        *,
        revision: str | None,
    ) -> None:
        assert self.rank_sliced_metadata is not None
        reference = str(self.rank_sliced_metadata["bits_per_expert"])
        try:
            filename, field = reference.rsplit(":", 1)
        except ValueError as exc:
            raise ValueError(
                "rank-sliced EXL3 bits_per_expert must use 'file.json:field' syntax, "
                f"got {reference!r}"
            ) from exc
        payload = get_hf_file_to_dict(filename, model_name, revision=revision)
        if not isinstance(payload, dict):
            raise ValueError(f"rank-sliced EXL3 could not load {filename!r}")

        experts = int(self.rank_sliced_metadata["experts_per_layer"])
        first, last = (int(value) for value in self.rank_sliced_metadata["moe_layers"])
        allowed = set(self.rank_sliced_k_values or ())
        by_layer: dict[int, tuple[int, ...]] = {}
        for layer_index in range(first, last + 1):
            entry = payload.get(str(layer_index))
            if not isinstance(entry, dict):
                raise ValueError(
                    f"rank-sliced EXL3 bitrate map is missing layer {layer_index}"
                )
            raw = entry.get(field)
            # The GLM-5.2 MTP overlay records all routed experts under tail_tr3
            # instead of repeating a 256-entry K3 vector.
            if raw is None and len(entry.get("tail_tr3", ())) == experts:
                raw = [3] * experts
            if not isinstance(raw, list) or len(raw) != experts:
                raise ValueError(
                    "rank-sliced EXL3 bitrate map must contain one entry per expert: "
                    f"layer={layer_index}, field={field!r}, expected={experts}"
                )
            bitrates = tuple(int(value) for value in raw)
            unexpected = sorted(set(bitrates).difference(allowed))
            if unexpected:
                raise ValueError(
                    f"rank-sliced EXL3 layer {layer_index} uses undeclared bitrates "
                    f"{unexpected}; declared={sorted(allowed)}"
                )
            by_layer[layer_index] = bitrates
        self.rank_sliced_bits_by_layer = by_layer

    def rank_sliced_layer_bitrates(self, layer_name: str) -> tuple[int, ...]:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
        if match is None:
            raise ValueError(
                f"cannot resolve rank-sliced EXL3 layer index from {layer_name!r}"
            )
        layer_index = int(match.group(1))
        if self.rank_sliced_k_values is None:
            if self.bits is None or float(self.bits) != int(self.bits):
                raise ValueError(f"invalid uniform EXL3 bitrate {self.bits!r}")
            experts = int(self.rank_sliced_metadata["experts_per_layer"])
            return (int(self.bits),) * experts
        try:
            return self.rank_sliced_bits_by_layer[layer_index]
        except KeyError as exc:
            raise ValueError(
                f"rank-sliced EXL3 bitrate map has no layer {layer_index}"
            ) from exc

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        # Keep both spellings: loader prefixes use vLLM names, while packed
        # source-matrix discovery intentionally refers to the unstacked HF name.
        mapped = hf_to_vllm_mapper.apply_dict(self.tensor_storage)
        self.tensor_storage = {**self.tensor_storage, **mapped}

    def _validate_storage_metadata(self) -> None:
        bad: list[str] = []
        exl3_count = 0
        for prefix, entry in self.tensor_storage.items():
            if entry.get("quant_format") != "exl3":
                continue
            exl3_count += 1
            stored = entry.get("stored_tensors", {})
            suffixes = {name.rsplit(".", 1)[-1] for name in stored}
            required = {"trellis"}
            if not ({"suh", "su"} & suffixes):
                required.add("suh|su")
            if not ({"svh", "sv"} & suffixes):
                required.add("svh|sv")
            missing = [name for name in required if name not in suffixes]
            if missing:
                bad.append(f"{prefix}: missing {','.join(sorted(missing))}")
            if {"mcg", "mul1"} <= suffixes:
                bad.append(f"{prefix}: both mcg and mul1 are present")
        if not exl3_count:
            raise ValueError("quantization_config.json has no EXL3 tensor records")
        if bad:
            raise ValueError("Invalid EXL3 tensor metadata: " + "; ".join(bad[:16]))

    def _force_independent_lm_head(self, hf_config: PretrainedConfig | None) -> None:
        if hf_config is None or not self.has_quantized_lm_head():
            return
        configs: list[Any] = [hf_config]
        try:
            text_config = hf_config.get_text_config()
        except (AttributeError, TypeError):
            text_config = None
        if text_config is not None and text_config is not hf_config:
            configs.append(text_config)
        changed = False
        for config in configs:
            if getattr(config, "tie_word_embeddings", False):
                config.tie_word_embeddings = False
                changed = True
        if changed:
            logger.warning_once(
                "EXL3 metadata contains an independently quantized lm_head; "
                "overriding tie_word_embeddings so vLLM instantiates it."
            )

    def _require_enforce_eager(self) -> None:
        if self.rank_sliced_metadata is not None:
            # The routed-expert fast path is eagerly planned before graph
            # capture. Only its large-M parity fallback remains eager.
            return
        # The fork-local SM70 implementation has a single static kernel per
        # EXL3 bit width.  It neither autotunes nor performs timing launches,
        # so it is safe to warm up and capture.  Keep the eager-only guard for
        # exllamav3_ext on other architectures.
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 7:
            _ensure_sm70_operator()
            if hasattr(torch.ops._C, "exl3_sm70_gemm"):
                return
        # exllamav3_ext's exl3_gemm autotunes with timing launches on the first
        # call per (m-bucket, k, n, K) shape hash; under CUDA-graph capture
        # those launches fault, and m-bucketing means a warmup pass cannot
        # reliably cover every bucket. Fail fast at build time instead of
        # faulting mid-capture.
        if self._eager_checked:
            return
        self._eager_checked = True
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return
        if not vllm_config.model_config.enforce_eager:
            raise ValueError(
                "The EXL3 quantization backend requires eager execution: "
                "pass --enforce-eager (enforce_eager=True). exl3_gemm "
                "autotunes with timing launches on first use per shape "
                "bucket, which is incompatible with CUDA-graph capture."
            )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        self._require_enforce_eager()
        is_lm_head = layer.__class__.__name__ == "ParallelLMHead"
        if is_lm_head and not prefix:
            prefix = "lm_head"
        if isinstance(layer, LinearBase) or is_lm_head:
            if not self._linear_prefix_is_exl3(prefix):
                return UnquantizedLinearMethod()
            return Exl3LinearMethod(self)
        if isinstance(layer, RoutedExperts):
            if not self._moe_prefix_is_exl3(prefix, layer):
                return None
            raise NotImplementedError(
                "This fork currently imports the dense EXL3 path only; "
                "routed-expert EXL3 requires the Gilded Gnosis MoE backend."
            )
        return None

    def _storage_entry(self, prefix: str) -> dict[str, Any] | None:
        candidates = [prefix]
        if prefix.startswith("model."):
            candidates.append(prefix.removeprefix("model."))
        else:
            candidates.append(f"model.{prefix}")

        # Multimodal wrappers often add an extra `model` or `language_model`
        # segment relative to vLLM's text-only module — interior
        # (`model.language_model.layers...`) or leading
        # (`language_model.lm_head`), so leading segments collapse too.
        parts = prefix.split(".")
        for removable in ("model", "language_model"):
            for idx in range(0, len(parts) - 1):
                if parts[idx] != removable:
                    continue
                collapsed = ".".join(parts[:idx] + parts[idx + 1 :])
                candidates.extend((collapsed, f"model.{collapsed}"))
                if collapsed.startswith("model."):
                    candidates.append(collapsed.removeprefix("model."))

        for candidate in dict.fromkeys(candidates):
            entry = self.tensor_storage.get(candidate)
            if entry is not None:
                return entry
        return None

    def _is_exl3_prefix(self, prefix: str) -> bool:
        entry = self._storage_entry(prefix)
        return entry is not None and entry.get("quant_format") == "exl3"

    def _linear_prefix_is_exl3(self, prefix: str) -> bool:
        if self._is_exl3_prefix(prefix):
            return True
        leaf = prefix.rsplit(".", 1)[-1]
        source_leaves = self.packed_modules_mapping.get(leaf)
        if not source_leaves:
            return False
        base = prefix.rsplit(".", 1)[0] if "." in prefix else ""
        return all(
            self._is_exl3_prefix(f"{base}.{source}" if base else source)
            for source in source_leaves
        )

    def _moe_prefix_is_exl3(
        self, prefix: str, layer: torch.nn.Module | None = None
    ) -> bool:
        if self.rank_sliced_metadata is not None:
            match = re.search(r"layers\.(\d+)\b", prefix)
            if match is None:
                return False
            first, last = (int(v) for v in self.rank_sliced_metadata["moe_layers"])
            return first <= int(match.group(1)) <= last
        # Use the layer's checkpoint projection names (the same fields
        # _validate_codebooks keys off) so remapped-projection MoE
        # checkpoints are still detected; fall back to the defaults when the
        # layer variant does not carry them.
        projections = tuple(
            getattr(layer, attr, default)
            for attr, default in (
                ("ckpt_gate_proj_name", "gate_proj"),
                ("ckpt_up_proj_name", "up_proj"),
                ("ckpt_down_proj_name", "down_proj"),
            )
        )
        expert_prefixes = (f"{prefix}.0", f"{prefix}.experts.0")
        return any(
            all(
                self._is_exl3_prefix(f"{expert}.{projection}")
                for projection in projections
            )
            for expert in expert_prefixes
        )

    def codebook_for_prefix(self, prefix: str) -> str | None:
        if self.rank_sliced_metadata is not None:
            match = re.search(r"layers\.(\d+)\b", prefix)
            if match is None:
                return None
            first, last = (int(v) for v in self.rank_sliced_metadata["moe_layers"])
            return "mcg" if first <= int(match.group(1)) <= last else None
        entry = self._storage_entry(prefix)
        if entry is None:
            return None
        suffixes = {name.rsplit(".", 1)[-1] for name in entry.get("stored_tensors", {})}
        if "mcg" in suffixes:
            return "mcg"
        if "mul1" in suffixes:
            return "mul1"
        return None

    def has_quantized_lm_head(self) -> bool:
        return self._is_exl3_prefix("lm_head")

    def normalize_rank_sliced_weight_name(self, name: str) -> str | None:
        """Drop non-local TP payloads and remove the serialized rank segment."""
        if self.rank_sliced_metadata is None:
            return name
        match = _RANK_SLICED_WEIGHT_RE.match(name)
        if match is None:
            return name
        if int(match.group("rank")) != get_tensor_model_parallel_rank():
            return None
        return f"{match.group('prefix')}.{match.group('field')}"


class Exl3Parameter(BasevLLMParameter):
    """Zero-sized parameter holding independently shaped EXL3 components."""

    def __new__(cls, *, weight_loader):
        data = torch.empty(0, dtype=torch.uint8)
        return super().__new__(cls, data=data, weight_loader=weight_loader)

    def __init__(self, *, weight_loader):
        self.exl3_tensors: dict[ShardId, torch.Tensor] = {}
        super().__init__(data=self.data, weight_loader=weight_loader)

    def load_exl3_weight(
        self,
        loaded_weight: torch.Tensor,
        shard_id: ShardId = None,
    ) -> None:
        self.exl3_tensors[shard_id] = loaded_weight.contiguous()


def _exl3_weight_loader(
    param: Exl3Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: ShardId = None,
) -> None:
    param.load_exl3_weight(loaded_weight, loaded_shard_id)


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Exl3Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype, extra_weight_attrs
        if layer.__class__.__name__ == "ParallelLMHead":
            org = getattr(layer, "org_vocab_size", None)
            total = getattr(layer, "num_embeddings", None)
            if org is not None and total is not None and org != total:
                raise NotImplementedError(
                    "EXL3 lm_head with added vocabulary is unsupported: the "
                    f"trellis tensor covers the original {org} rows but the "
                    f"layer allocates {total}; TP slicing would silently "
                    "misalign. Strip --lora-extra-vocab-size / added tokens "
                    "or leave lm_head unquantized."
                )
        # Respect the layer's effective topology. disable_tp linears set their
        # own tp_size=1, while ReplicatedLinear carries full weights even when
        # the process-wide tensor group is larger than one.
        if isinstance(layer, ReplicatedLinear):
            layer.exl3_tp_rank = 0
            layer.exl3_tp_size = 1
        else:
            layer.exl3_tp_rank = getattr(
                layer, "tp_rank", get_tensor_model_parallel_rank()
            )
            layer.exl3_tp_size = getattr(
                layer, "tp_size", get_tensor_model_parallel_world_size()
            )
        layer.exl3_input_size = input_size
        layer.exl3_input_size_per_partition = input_size_per_partition
        layer.exl3_output_size = output_size
        layer.exl3_output_partition_sizes = output_partition_sizes
        layer.exl3_shard_ids = self._shard_ids_for_layer(layer, output_partition_sizes)
        layer.exl3_parallel_mode = (
            "row" if input_size_per_partition != input_size else "column"
        )
        source_prefixes = self._source_prefixes_for_layer(layer, layer.exl3_shard_ids)
        layer.exl3_expected_codebooks = {
            shard_id: self.quant_config.codebook_for_prefix(source_prefix)
            for shard_id, source_prefix in zip(
                layer.exl3_shard_ids, source_prefixes, strict=True
            )
        }

        # su/sv are legacy packed sign bitfields.  Modern checkpoints load
        # suh/svh directly.
        for name in ("suh", "svh", "su", "sv", "trellis", "mcg", "mul1"):
            layer.register_parameter(
                name,
                Exl3Parameter(weight_loader=_exl3_weight_loader),
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._materialize_legacy_hadamard(layer)
        missing: list[str] = []
        for attr in ("suh", "svh", "trellis"):
            param = getattr(layer, attr)
            for shard_id in layer.exl3_shard_ids:
                if shard_id not in param.exl3_tensors:
                    missing.append(f"{attr}[{shard_id!r}]")
        for shard_id in layer.exl3_shard_ids:
            expected = layer.exl3_expected_codebooks[shard_id]
            has_mcg = shard_id in layer.mcg.exl3_tensors
            has_mul1 = shard_id in layer.mul1.exl3_tensors
            if has_mcg and has_mul1:
                missing.append(f"codebook[{shard_id!r}]=both mcg and mul1")
            elif expected == "mcg" and not has_mcg:
                missing.append(f"mcg[{shard_id!r}]")
            elif expected == "mul1" and not has_mul1:
                missing.append(f"mul1[{shard_id!r}]")
            elif expected is None and (has_mcg or has_mul1):
                missing.append(f"unexpected codebook[{shard_id!r}]")
        if missing:
            prefix = getattr(layer, "prefix", layer.__class__.__name__)
            raise ValueError(
                f"Missing or inconsistent EXL3 tensors for {prefix}: "
                + ", ".join(missing)
            )

        self._validate_loaded_tensors(layer)
        self._shard_tensors_for_tensor_parallel(layer)
        self._validate_loaded_tensors(layer)

        # device_loading_context has moved the zero-sized registered parameter
        # to the model target device.  Its device is the safest destination for
        # the tensors kept in the side dictionaries.
        device = layer.trellis.device
        moved_tensors: dict[int, torch.Tensor] = {}
        for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
            param = getattr(layer, attr)
            for shard_id, tensor in list(param.exl3_tensors.items()):
                # Expanded physical QKV shards intentionally share SUH and
                # codebook tensors.  Preserve that alias when moving to CUDA;
                # separate .to() calls would duplicate the vector and hide the
                # q/k pair's one-transform opportunity from pointer-stable
                # graph capture.
                tensor_key = id(tensor)
                moved = moved_tensors.get(tensor_key)
                if moved is None:
                    moved = tensor.to(
                        device=device, non_blocking=True
                    ).contiguous()
                    moved_tensors[tensor_key] = moved
                param.exl3_tensors[shard_id] = moved

        layer.exl3_tm_states = {}
        layer.exl3_tm_int8 = {}
        layer.exl3_tm_locks = {}
        layer.exl3_tm_empty_state = torch.empty(
            (0,), dtype=torch.int32, device=device
        )
        layer.exl3_tm_gate_up_metadata = None
        layer.exl3_tm_gate_up_int8_metadata = None
        layer.exl3_tm_gate_up_offsets = None
        layer.exl3_tm_gate_up_locks = None
        layer.exl3_tm_pair_metadata = {}
        layer.exl3_tm_pair_int8_metadata = {}
        layer.exl3_tm_pair_offsets = {}
        layer.exl3_tm_pair_locks = {}
        if (
            os.getenv("VLLM_EXL3_SM70_TM_STATE", "0") == "1"
            and device.type == "cuda"
            and torch.cuda.get_device_capability(device) == (7, 0)
        ):
            _ensure_sm70_operator("exl3_sm70_tm_state_repack")
            if not hasattr(torch.ops._C, "exl3_sm70_tm_state_repack"):
                raise RuntimeError(
                    "VLLM_EXL3_SM70_TM_STATE=1 but the state-repack operator "
                    "is unavailable"
                )
            for shard_id in layer.exl3_shard_ids:
                trellis = layer.trellis.exl3_tensors[shard_id]
                bits = trellis.shape[2] // 16
                if (
                    shard_id in layer.mcg.exl3_tensors
                    and bits in (4, 5, 6)
                    and trellis.shape[1] % 2 == 0
                ):
                    k = trellis.shape[0] * 16
                    n = trellis.shape[1] * 16
                    int8_shape = (bits, k, n)
                    use_int8 = (
                        os.getenv("VLLM_EXL3_SM70_INT8_REPACK", "0") == "1"
                        and int8_shape in _SM70_INT8_SHAPES
                    )
                    if use_int8:
                        op_name = "exl3_sm70_tm_int8_repack"
                        _ensure_sm70_operator(op_name)
                        if not hasattr(torch.ops._C, op_name):
                            raise RuntimeError(
                                "VLLM_EXL3_SM70_INT8_REPACK=1 but the "
                                "tile-scaled INT8 repack operator is unavailable"
                            )
                        layer.exl3_tm_int8[shard_id] = (
                            torch.ops._C.exl3_sm70_tm_int8_repack(trellis)
                        )
                    else:
                        layer.exl3_tm_states[shard_id] = (
                            torch.ops._C.exl3_sm70_tm_state_repack(trellis)
                        )
                    # TurboMind's split-K epilogue returns every semaphore to
                    # zero after the last split.  Keep one max-M<=8 lock set
                    # per logical matrix so graph replay does not launch a
                    # redundant zero-fill before all 497 decode projections.
                    layer.exl3_tm_locks[shard_id] = torch.zeros(
                        (n // _HADAMARD_BLOCK,),
                        dtype=torch.int32,
                        device=device,
                    )

            if (
                getattr(layer, "prefix", "").endswith("gate_up_proj")
                and list(layer.exl3_shard_ids) == [0, 1]
            ):
                gate_trellis = layer.trellis.exl3_tensors[0]
                up_trellis = layer.trellis.exl3_tensors[1]
                if gate_trellis.shape != up_trellis.shape:
                    raise ValueError(
                        "EXL3 paired gate/up trellis shapes must match, got "
                        f"{gate_trellis.shape} and {up_trellis.shape}"
                    )
                packed_n = gate_trellis.shape[1] * 16
                gate_state = layer.exl3_tm_states.get(0)
                up_state = layer.exl3_tm_states.get(1)
                if gate_state is not None and up_state is not None:
                    _ensure_sm70_operator("exl3_sm70_tm_gate_up_metadata")
                    if not hasattr(
                        torch.ops._C, "exl3_sm70_tm_gate_up_metadata"
                    ):
                        raise RuntimeError(
                            "The SM70 state sidecar is missing paired gate/up "
                            "metadata support"
                        )
                    if gate_state.shape != up_state.shape:
                        raise ValueError(
                            "EXL3 paired gate/up state shapes must match, got "
                            f"{gate_state.shape} and {up_state.shape}"
                        )
                    layer.exl3_tm_gate_up_metadata = (
                        torch.ops._C.exl3_sm70_tm_gate_up_metadata(
                            gate_state,
                            up_state,
                            layer.svh.exl3_tensors[0],
                            layer.svh.exl3_tensors[1],
                        )
                    )
                gate_int8 = layer.exl3_tm_int8.get(0)
                up_int8 = layer.exl3_tm_int8.get(1)
                if gate_int8 is not None and up_int8 is not None:
                    op_name = "exl3_sm70_tm_int8_pair_metadata"
                    _ensure_sm70_operator(op_name)
                    if not hasattr(torch.ops._C, op_name):
                        raise RuntimeError(
                            "The SM70 sidecar is missing grouped INT8 "
                            "gate/up metadata support"
                        )
                    layer.exl3_tm_gate_up_int8_metadata = (
                        torch.ops._C.exl3_sm70_tm_int8_pair_metadata(
                            gate_int8[0],
                            gate_int8[1],
                            up_int8[0],
                            up_int8[1],
                            layer.svh.exl3_tensors[0],
                            layer.svh.exl3_tensors[1],
                        )
                    )
                # First triplet: grouped N boundaries for the scheduler.
                # Second: fixed row bases for two max-M=8 workspaces.
                layer.exl3_tm_gate_up_offsets = torch.tensor(
                    [0, packed_n, 2 * packed_n, 0, 8, 16],
                    dtype=torch.int32,
                    device=device,
                )
                # INT8 swizzles can cover a wider grid than the exact state
                # policy, so preserve the 64-tile headroom proven by the
                # grouped-pair graph replay test.
                layer.exl3_tm_gate_up_locks = torch.zeros(
                    (
                        2
                        * (
                            packed_n // _HADAMARD_BLOCK
                            + (
                                64
                                if layer.exl3_tm_gate_up_int8_metadata is not None
                                else 4
                            )
                        ),
                    ),
                    dtype=torch.int32,
                    device=device,
                )

            # Qwen3.8 keeps GDN q/k and v/z, plus full-attention k/v, as
            # equal-shape adjacent shards.  They use the same bit width and
            # split-K policy, so the already-qualified grouped scheduler can
            # execute each pair without changing either branch's accumulation
            # order.  Keep the pair metadata graph-static at model load.
            if os.getenv("VLLM_EXL3_SM70_GROUPED_QKV", "0") == "1":
                prefix = getattr(layer, "prefix", "")
                shard_ids = list(layer.exl3_shard_ids)
                candidate_pairs: list[tuple[ShardId, ShardId]] = []
                if prefix.endswith("in_proj_qkvz") and shard_ids == [0, 1, 2, 3]:
                    candidate_pairs = [(0, 1), (2, 3)]
                elif prefix.endswith("qkv_proj") and shard_ids == ["q", "k", "v"]:
                    candidate_pairs = [("k", "v")]

                for pair in candidate_pairs:
                    first, second = pair
                    trellis0 = layer.trellis.exl3_tensors[first]
                    trellis1 = layer.trellis.exl3_tensors[second]
                    if trellis0.shape != trellis1.shape:
                        continue
                    bits0 = trellis0.shape[2] // 16
                    bits1 = trellis1.shape[2] // 16
                    if bits0 != bits1 or bits0 not in (4, 5, 6):
                        continue
                    if not (
                        first in layer.mcg.exl3_tensors
                        and second in layer.mcg.exl3_tensors
                        and first not in layer.mul1.exl3_tensors
                        and second not in layer.mul1.exl3_tensors
                    ):
                        continue
                    packed_n = trellis0.shape[1] * 16
                    logical0 = self._output_shard_size(layer, first)
                    logical1 = self._output_shard_size(layer, second)
                    if logical0 != packed_n or logical1 != packed_n:
                        continue

                    state0 = layer.exl3_tm_states.get(first)
                    state1 = layer.exl3_tm_states.get(second)
                    if (
                        state0 is not None
                        and state1 is not None
                        and state0.shape == state1.shape
                    ):
                        layer.exl3_tm_pair_metadata[pair] = (
                            torch.ops._C.exl3_sm70_tm_gate_up_metadata(
                                state0,
                                state1,
                                layer.svh.exl3_tensors[first],
                                layer.svh.exl3_tensors[second],
                            )
                        )
                    shared_suh = (
                        layer.suh.exl3_tensors[first].data_ptr()
                        == layer.suh.exl3_tensors[second].data_ptr()
                    )
                    # Scheduler N bounds, input-workspace row bases, then
                    # independent split-K partial row bases.
                    offsets = torch.tensor(
                        [
                            0,
                            packed_n,
                            2 * packed_n,
                            0,
                            0 if shared_suh else 8,
                            16,
                            0,
                            8,
                            16,
                        ],
                        dtype=torch.int32,
                        device=device,
                    )
                    int8_first = layer.exl3_tm_int8.get(first)
                    int8_second = layer.exl3_tm_int8.get(second)
                    if int8_first is not None and int8_second is not None:
                        op_name = "exl3_sm70_tm_int8_pair_metadata"
                        _ensure_sm70_operator(op_name)
                        if not hasattr(torch.ops._C, op_name):
                            raise RuntimeError(
                                "The SM70 sidecar is missing grouped INT8 "
                                "metadata support"
                            )
                        layer.exl3_tm_pair_int8_metadata[pair] = (
                            torch.ops._C.exl3_sm70_tm_int8_pair_metadata(
                                int8_first[0],
                                int8_first[1],
                                int8_second[0],
                                int8_second[1],
                                layer.svh.exl3_tensors[first],
                                layer.svh.exl3_tensors[second],
                            )
                        )
                    if (
                        pair not in layer.exl3_tm_pair_metadata
                        and pair not in layer.exl3_tm_pair_int8_metadata
                    ):
                        continue
                    layer.exl3_tm_pair_offsets[pair] = offsets
                    layer.exl3_tm_pair_locks[pair] = torch.zeros(
                        (2 * (packed_n // _HADAMARD_BLOCK + 64),),
                        dtype=torch.int32,
                        device=device,
                    )

        if (
            os.getenv("VLLM_EXL3_MTP_DENSE_F16", "0") == "1"
            and getattr(layer, "_sm70_f16_force_enable", False)
            and getattr(layer, "prefix", "").rsplit(".", 1)[-1] == "fc"
        ):
            self._materialize_mtp_dense_f16(layer)

    def _materialize_mtp_dense_f16(self, layer: torch.nn.Module) -> None:
        """One-time exact EXL3 reconstruct of the tiny MTP layer into TM FP16."""
        if getattr(layer, "_sm70_exl3_mtp_dense_prepared", False):
            return
        if not hasattr(torch.ops._C, "sm70_f16_prepare"):
            return

        first_trellis = next(iter(layer.trellis.exl3_tensors.values()))
        packed_k = int(first_trellis.shape[0] * 16)
        device = first_trellis.device
        tile_m = 8
        rows: list[torch.Tensor] = []
        eye = torch.zeros((tile_m, packed_k), dtype=torch.float16, device=device)
        with torch.inference_mode():
            for start in range(0, packed_k, tile_m):
                width = min(tile_m, packed_k - start)
                eye.zero_()
                for row in range(width):
                    eye[row, start + row] = 1
                y = self.apply(layer, eye[:width])
                rows.append(y.detach())
        dense = torch.cat(rows, dim=0).contiguous()
        weight = dense.t().contiguous()
        if weight.shape[0] % 32 != 0 or weight.shape[1] % 16 != 0:
            logger.warning_once(
                "MTP dense F16 skip for %s: shape %s not TM-aligned",
                getattr(layer, "prefix", "<unknown>"),
                tuple(weight.shape),
            )
            return
        from vllm import _sm70_ops as sm70_ops

        prepared = sm70_ops.sm70_f16_prepare(weight)
        layer._sm70_exl3_mtp_dense_weight = weight
        layer._sm70_exl3_mtp_tm_weight = prepared[0]
        layer._sm70_exl3_mtp_k_ld = int(prepared[1][0].item())
        layer._sm70_exl3_mtp_dense_prepared = True
        logger.info_once(
            "SM70 MTP draft dense FP16 materialized for %s weight=%s",
            getattr(layer, "prefix", "<unknown>"),
            tuple(weight.shape),
        )

    @staticmethod
    def _apply_state_pair(
        layer: torch.nn.Module,
        x: torch.Tensor,
        pair: tuple[ShardId, ShardId],
    ) -> torch.Tensor | None:
        metadata = getattr(layer, "exl3_tm_pair_metadata", {}).get(pair)
        int8_metadata = getattr(
            layer, "exl3_tm_pair_int8_metadata", {}
        ).get(pair)
        offsets = getattr(layer, "exl3_tm_pair_offsets", {}).get(pair)
        locks = getattr(layer, "exl3_tm_pair_locks", {}).get(pair)
        if (
            (metadata is None and int8_metadata is None)
            or offsets is None
            or locks is None
        ):
            return None

        first, second = pair
        trellis0 = layer.trellis.exl3_tensors[first]
        trellis1 = layer.trellis.exl3_tensors[second]
        packed_k = trellis0.shape[0] * 16
        if trellis1.shape[0] * 16 != packed_k or x.shape[-1] > packed_k:
            return None
        if x.shape[-1] < packed_k:
            x = torch.nn.functional.pad(x, (0, packed_k - x.shape[-1]))
        bits = trellis0.shape[2] // 16
        if trellis1.shape[2] // 16 != bits:
            return None

        int8_first = getattr(layer, "exl3_tm_int8", {}).get(first)
        int8_second = getattr(layer, "exl3_tm_int8", {}).get(second)
        if (
            int8_metadata is not None
            and int8_first is not None
            and int8_second is not None
        ):
            return _exl3_sm70_tm_int8_pair_gemm(
                x,
                trellis0,
                trellis1,
                int8_first[0],
                int8_first[1],
                int8_second[0],
                int8_second[1],
                layer.suh.exl3_tensors[first],
                layer.suh.exl3_tensors[second],
                layer.svh.exl3_tensors[first],
                layer.svh.exl3_tensors[second],
                int8_metadata,
                offsets,
                locks,
                bits,
            )

        state0 = getattr(layer, "exl3_tm_states", {}).get(first)
        state1 = getattr(layer, "exl3_tm_states", {}).get(second)
        if metadata is None or state0 is None or state1 is None:
            return None
        return _exl3_sm70_tm_state_pair_gemm(
            x,
            trellis0,
            trellis1,
            state0,
            state1,
            layer.suh.exl3_tensors[first],
            layer.suh.exl3_tensors[second],
            layer.svh.exl3_tensors[first],
            layer.svh.exl3_tensors[second],
            metadata,
            offsets,
            locks,
            bits,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = x.shape[:-1]
        original_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()
        if getattr(layer, "_sm70_exl3_mtp_dense_prepared", False):
            from vllm import _sm70_ops as sm70_ops

            tm_weight = getattr(layer, "_sm70_exl3_mtp_tm_weight", None)
            k_ld = getattr(layer, "_sm70_exl3_mtp_k_ld", None)
            if tm_weight is not None and k_ld is not None:
                out = torch.empty(
                    (x_2d.size(0), tm_weight.shape[0]),
                    dtype=x_2d.dtype,
                    device=x_2d.device,
                )
                sm70_ops.sm70_f16_gemm_out(out, x_2d, tm_weight, k_ld, False)
                if bias is not None:
                    out = out + bias.to(dtype=out.dtype)
                out = out.reshape(*original_shape, out.shape[-1])
                return out if out.dtype == original_dtype else out.to(
                    original_dtype
                )
        outputs: list[torch.Tensor] | None = None
        if os.getenv("VLLM_EXL3_SM70_GROUPED_QKV", "0") == "1":
            prefix = getattr(layer, "prefix", "")
            shard_ids = list(layer.exl3_shard_ids)
            if prefix.endswith("in_proj_qkvz") and shard_ids == [0, 1, 2, 3]:
                qk = self._apply_state_pair(layer, x_2d, (0, 1))
                vz = self._apply_state_pair(layer, x_2d, (2, 3))
                if qk is not None and vz is not None:
                    outputs = [qk, vz]
            elif prefix.endswith("qkv_proj") and shard_ids == ["q", "k", "v"]:
                kv = self._apply_state_pair(layer, x_2d, ("k", "v"))
                if kv is not None:
                    outputs = [self._apply_one(layer, x_2d, "q"), kv]
        if outputs is None:
            outputs = [
                self._apply_one(layer, x_2d, shard_id)
                for shard_id in layer.exl3_shard_ids
            ]
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if bias is not None:
            output = output + bias.to(dtype=output.dtype)
        output = output.reshape(*original_shape, output.shape[-1])
        return output if output.dtype == original_dtype else output.to(original_dtype)

    def apply_fused_mlp(
        self,
        down_layer: torch.nn.Module,
        gate_up_layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run the exact SM70 EXL3 local MLP projection before TP reduction."""

        if os.getenv("VLLM_EXL3_SM70_FUSED_MLP", "0") != "1":
            return None
        if os.getenv("VLLM_SM70_DUMP_QWEN_MLP_INTERNALS", "0") == "1":
            return None
        if (
            not x.is_cuda
            or x.dtype != torch.float16
            or torch.cuda.get_device_capability(x.device) != (7, 0)
        ):
            return None
        gate_prefix = getattr(gate_up_layer, "prefix", "")
        down_prefix = getattr(down_layer, "prefix", "")
        if not gate_prefix.endswith("gate_up_proj") or not down_prefix.endswith(
            "down_proj"
        ):
            return None
        if gate_prefix.rsplit(".", 1)[0] != down_prefix.rsplit(".", 1)[0]:
            return None
        if list(getattr(gate_up_layer, "exl3_shard_ids", ())) != [0, 1]:
            return None
        down_ids = list(getattr(down_layer, "exl3_shard_ids", ()))
        if len(down_ids) != 1:
            return None
        down_id = down_ids[0]

        gate_state = getattr(gate_up_layer, "exl3_tm_states", {}).get(0)
        up_state = getattr(gate_up_layer, "exl3_tm_states", {}).get(1)
        down_state = getattr(down_layer, "exl3_tm_states", {}).get(down_id)
        gate_metadata = getattr(gate_up_layer, "exl3_tm_gate_up_metadata", None)
        gate_int8_metadata = getattr(
            gate_up_layer, "exl3_tm_gate_up_int8_metadata", None
        )
        gate_offsets = getattr(gate_up_layer, "exl3_tm_gate_up_offsets", None)
        gate_locks = getattr(gate_up_layer, "exl3_tm_gate_up_locks", None)
        down_locks = getattr(down_layer, "exl3_tm_locks", {}).get(down_id)
        down_int8 = getattr(down_layer, "exl3_tm_int8", {}).get(down_id)
        gate_exact = (
            gate_state is not None
            and up_state is not None
            and gate_metadata is not None
        )
        gate_int8 = gate_int8_metadata is not None
        down_exact = down_state is not None
        if (
            not (gate_exact or gate_int8)
            or not (down_exact or down_int8 is not None)
            or any(
                tensor is None
                for tensor in (gate_offsets, gate_locks, down_locks)
            )
        ):
            return None
        if not (
            all(shard_id in gate_up_layer.mcg.exl3_tensors for shard_id in (0, 1))
            and down_id in down_layer.mcg.exl3_tensors
            and all(
                shard_id not in gate_up_layer.mul1.exl3_tensors
                for shard_id in (0, 1)
            )
            and down_id not in down_layer.mul1.exl3_tensors
        ):
            return None

        gate_trellis = gate_up_layer.trellis.exl3_tensors[0]
        up_trellis = gate_up_layer.trellis.exl3_tensors[1]
        down_trellis = down_layer.trellis.exl3_tensors[down_id]
        gate_bits = gate_trellis.shape[2] // 16
        up_bits = up_trellis.shape[2] // 16
        down_bits = down_trellis.shape[2] // 16
        if gate_bits != up_bits or gate_bits not in (4, 5, 6):
            return None
        if down_bits not in (4, 5, 6):
            return None

        gate_logical_n = self._output_shard_size(gate_up_layer, 0)
        up_logical_n = self._output_shard_size(gate_up_layer, 1)
        gate_packed_n = gate_trellis.shape[1] * 16
        up_packed_n = up_trellis.shape[1] * 16
        down_packed_k = down_trellis.shape[0] * 16
        down_packed_n = down_trellis.shape[1] * 16
        # The ordinary module boundary slices gate/up to logical N and then
        # zero-pads to down packed K.  Fusing across it is exact only when that
        # operation is an identity.  Keep this deliberately shape-specific to
        # the qualified TP4 Qwen3.8 MLP (also used by each TP4 PP2 stage).
        if not (
            gate_logical_n
            == up_logical_n
            == gate_packed_n
            == up_packed_n
            == down_packed_k
            == 4352
            and gate_trellis.shape[0] * 16 == 5120
            and down_packed_n == 5120
            and gate_bits == 5
            and down_bits == 6
        ):
            return None

        original_shape = x.shape[:-1]
        original_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()
        if x_2d.shape[0] == 0:
            return None
        packed_gate_k = gate_trellis.shape[0] * 16
        if x_2d.shape[-1] > packed_gate_k:
            return None
        if x_2d.shape[-1] < packed_gate_k:
            x_2d = torch.nn.functional.pad(
                x_2d, (0, packed_gate_k - x_2d.shape[-1])
            )

        intermediate_n = gate_packed_n
        if gate_trellis.shape != up_trellis.shape or (
            down_trellis.shape[0] * 16 != intermediate_n
        ):
            return None
        empty_state = getattr(gate_up_layer, "exl3_tm_empty_state", None)
        if empty_state is None:
            return None
        gate_state_arg = gate_state if gate_state is not None else empty_state
        up_state_arg = up_state if up_state is not None else empty_state
        down_state_arg = down_state if down_state is not None else empty_state
        op_name = "exl3_sm70_tm_state_mlp"
        _ensure_sm70_operator(op_name)
        if not hasattr(torch.ops._C, op_name):
            raise RuntimeError(
                "VLLM_EXL3_SM70_FUSED_MLP=1 but the specialized SM70 EXL3 "
                "MLP operator is unavailable"
            )
        output = _exl3_sm70_tm_state_mlp(
            x_2d,
            gate_trellis,
            up_trellis,
            down_trellis,
            gate_state_arg,
            up_state_arg,
            down_state_arg,
            gate_up_layer.suh.exl3_tensors[0],
            gate_up_layer.suh.exl3_tensors[1],
            down_layer.suh.exl3_tensors[down_id],
            gate_up_layer.svh.exl3_tensors[0],
            gate_up_layer.svh.exl3_tensors[1],
            down_layer.svh.exl3_tensors[down_id],
            down_int8[0] if down_int8 is not None else down_state_arg,
            (
                down_int8[1]
                if down_int8 is not None
                else down_layer.svh.exl3_tensors[down_id]
            ),
            gate_int8_metadata if gate_int8 else gate_metadata,
            gate_offsets,
            gate_locks,
            down_locks,
            gate_bits,
            down_bits,
            gate_int8,
            down_int8 is not None,
        )
        logical_n = self._output_shard_size(down_layer, down_id)
        output = output[..., :logical_n].reshape(*original_shape, logical_n)
        return output if output.dtype == original_dtype else output.to(original_dtype)

    def apply_fused_silu_and_mul(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor | None:
        # Linear.forward_fused_silu_and_mul rejects bias-bearing layers before
        # calling this hook; Qwen3.8 gate_up_proj is bias-free.
        if os.getenv("VLLM_EXL3_SM70_FUSED_GATE_UP_SILU", "0") != "1":
            return None

        if not x.is_cuda or torch.cuda.get_device_capability(x.device) != (7, 0):
            return None
        if not getattr(layer, "prefix", "").endswith("gate_up_proj"):
            return None
        if list(getattr(layer, "exl3_shard_ids", ())) != [0, 1]:
            return None

        original_shape = x.shape[:-1]
        original_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()

        gate_id, up_id = 0, 1
        gate_trellis = layer.trellis.exl3_tensors[gate_id]
        up_trellis = layer.trellis.exl3_tensors[up_id]
        packed_k = gate_trellis.shape[0] * 16
        if up_trellis.shape[0] * 16 != packed_k or x_2d.shape[-1] > packed_k:
            return None
        if x_2d.shape[-1] < packed_k:
            x_2d = torch.nn.functional.pad(x_2d, (0, packed_k - x_2d.shape[-1]))

        gate_logical_n = self._output_shard_size(layer, gate_id)
        up_logical_n = self._output_shard_size(layer, up_id)
        gate_packed_n = gate_trellis.shape[1] * 16
        up_packed_n = up_trellis.shape[1] * 16
        if (
            gate_logical_n != up_logical_n
            or gate_packed_n != up_packed_n
            or gate_packed_n < gate_logical_n
            or gate_packed_n >= 32768
        ):
            return None

        gate_mcg = gate_id in layer.mcg.exl3_tensors
        up_mcg = up_id in layer.mcg.exl3_tensors
        gate_mul1 = gate_id in layer.mul1.exl3_tensors
        up_mul1 = up_id in layer.mul1.exl3_tensors
        if not (gate_mcg and up_mcg) or gate_mul1 or up_mul1:
            return None

        gate_state = getattr(layer, "exl3_tm_states", {}).get(gate_id)
        up_state = getattr(layer, "exl3_tm_states", {}).get(up_id)
        metadata = getattr(layer, "exl3_tm_gate_up_metadata", None)
        offsets = getattr(layer, "exl3_tm_gate_up_offsets", None)
        locks = getattr(layer, "exl3_tm_gate_up_locks", None)
        if (
            gate_state is not None
            and up_state is not None
            and metadata is not None
            and offsets is not None
            and locks is not None
        ):
            gate_bits = gate_trellis.shape[2] // 16
            up_bits = up_trellis.shape[2] // 16
            if gate_bits != up_bits:
                return None
            op_name = "exl3_sm70_tm_state_gate_up_silu_mul"
            _ensure_sm70_operator(op_name)
            if not hasattr(torch.ops._C, op_name):
                raise RuntimeError(
                    "VLLM_EXL3_SM70_FUSED_GATE_UP_SILU=1 but the paired "
                    "TurboMind state operator is unavailable"
                )
            output = _exl3_sm70_tm_state_gate_up_silu_mul(
                x_2d,
                gate_trellis,
                up_trellis,
                gate_state,
                up_state,
                layer.suh.exl3_tensors[gate_id],
                layer.suh.exl3_tensors[up_id],
                layer.svh.exl3_tensors[gate_id],
                layer.svh.exl3_tensors[up_id],
                metadata,
                offsets,
                locks,
                gate_bits,
            )[..., :gate_logical_n]
            output = output.reshape(*original_shape, gate_logical_n)
            return output if output.dtype == original_dtype else output.to(
                original_dtype
            )

        _ensure_sm70_operator("exl3_sm70_gate_up_silu_mul")
        if not hasattr(torch.ops._C, "exl3_sm70_gate_up_silu_mul"):
            logger.warning_once(
                "VLLM_EXL3_SM70_FUSED_GATE_UP_SILU=1 but the paired SM70 "
                "EXL3 operator is unavailable; using the unfused MLP path."
            )
            return None
        output = _exl3_sm70_gate_up_silu_mul(
            x_2d,
            gate_trellis,
            up_trellis,
            layer.suh.exl3_tensors[gate_id],
            layer.suh.exl3_tensors[up_id],
            layer.svh.exl3_tensors[gate_id],
            layer.svh.exl3_tensors[up_id],
            gate_mcg,
            up_mcg,
            gate_mul1,
            up_mul1,
        )[..., :gate_logical_n]
        output = output.reshape(*original_shape, gate_logical_n)
        return output if output.dtype == original_dtype else output.to(original_dtype)

    @staticmethod
    def _unpack_signs(bitfield: torch.Tensor) -> torch.Tensor:
        words = bitfield.contiguous().view(torch.uint16).to(torch.int32)
        masks = 1 << torch.arange(16, device=words.device, dtype=torch.int32)
        negative = (words.unsqueeze(-1) & masks) != 0
        return (
            torch.where(
                negative,
                torch.tensor(-1.0, device=words.device, dtype=torch.float16),
                torch.tensor(1.0, device=words.device, dtype=torch.float16),
            )
            .flatten()
            .contiguous()
        )

    @classmethod
    def _materialize_legacy_hadamard(cls, layer: torch.nn.Module) -> None:
        for packed_name, half_name in (("su", "suh"), ("sv", "svh")):
            packed = getattr(layer, packed_name).exl3_tensors
            half = getattr(layer, half_name).exl3_tensors
            for shard_id in layer.exl3_shard_ids:
                if shard_id not in half and shard_id in packed:
                    half[shard_id] = cls._unpack_signs(packed[shard_id])

    @staticmethod
    def _validate_marker(tensor: torch.Tensor, expected: int, name: str) -> None:
        if tensor.dtype != torch.int32 or tensor.numel() != 1:
            raise ValueError(f"EXL3 {name} must be a scalar int32 sentinel")
        value = int(tensor.reshape(()).item()) & 0xFFFFFFFF
        if value != expected:
            raise ValueError(
                f"Invalid EXL3 {name} sentinel 0x{value:08x}; expected 0x{expected:08x}"
            )

    @classmethod
    def _validate_loaded_tensors(cls, layer: torch.nn.Module) -> None:
        for shard_id in layer.exl3_shard_ids:
            trellis = layer.trellis.exl3_tensors[shard_id]
            suh = layer.suh.exl3_tensors[shard_id]
            svh = layer.svh.exl3_tensors[shard_id]
            if trellis.dtype != torch.int16 or trellis.ndim != 3:
                raise ValueError("EXL3 trellis must be rank-3 int16")
            if trellis.shape[2] % 16 or not 1 <= trellis.shape[2] // 16 <= 8:
                raise ValueError(
                    f"Invalid EXL3 trellis bit width {trellis.shape[2]} / 16"
                )
            if suh.dtype != torch.float16 or suh.ndim != 1:
                raise ValueError("EXL3 suh must be rank-1 float16")
            if svh.dtype != torch.float16 or svh.ndim != 1:
                raise ValueError("EXL3 svh must be rank-1 float16")
            k = trellis.shape[0] * 16
            n = trellis.shape[1] * 16
            if suh.numel() != k or svh.numel() != n:
                raise ValueError(
                    "EXL3 dimensions disagree: "
                    f"trellis={tuple(trellis.shape)}, suh={suh.numel()}, "
                    f"svh={svh.numel()}"
                )
            if k % _HADAMARD_BLOCK or n % _HADAMARD_BLOCK:
                raise ValueError(
                    f"EXL3 kernel dimensions must be {_HADAMARD_BLOCK}-aligned, "
                    f"got K={k}, N={n}"
                )
            if shard_id in layer.mcg.exl3_tensors:
                cls._validate_marker(
                    layer.mcg.exl3_tensors[shard_id], _MCG_SENTINEL, "mcg"
                )
            if shard_id in layer.mul1.exl3_tensors:
                cls._validate_marker(
                    layer.mul1.exl3_tensors[shard_id], _MUL1_SENTINEL, "mul1"
                )

    @staticmethod
    def _slice_exl3_tensor(
        tensor: torch.Tensor,
        *,
        dim: int,
        start: int,
        size: int,
    ) -> torch.Tensor:
        if start % _HADAMARD_BLOCK or size % _HADAMARD_BLOCK:
            axis = "output" if dim == 1 else "input"
            raise ValueError(
                f"EXL3 TP {axis} slice must be {_HADAMARD_BLOCK}-aligned, "
                f"got start={start}, size={size}"
            )
        return tensor.narrow(dim, start // 16, size // 16).contiguous()

    @staticmethod
    def _output_shard_size(layer: torch.nn.Module, shard_id: ShardId) -> int:
        if shard_id is None:
            return layer.exl3_output_partition_sizes[0]
        if isinstance(shard_id, str) and shard_id in ("q", "k", "v"):
            return layer.exl3_output_partition_sizes[{"q": 0, "k": 1, "v": 2}[shard_id]]
        if isinstance(shard_id, tuple):
            return sum(layer.exl3_output_partition_sizes[idx] for idx in shard_id)
        if isinstance(shard_id, int):
            return layer.exl3_output_partition_sizes[shard_id]
        return layer.exl3_output_partition_sizes[layer.exl3_shard_ids.index(shard_id)]

    @staticmethod
    def _qkv_output_start(
        layer: torch.nn.Module, shard_id: ShardId, shard_size: int
    ) -> int:
        if shard_id in ("k", "v"):
            shard_rank = layer.exl3_tp_rank // layer.num_kv_head_replicas
        else:
            shard_rank = layer.exl3_tp_rank
        return shard_rank * shard_size

    @classmethod
    def _shard_tensors_for_tensor_parallel(cls, layer: torch.nn.Module) -> None:
        if layer.exl3_tp_size == 1:
            return
        if layer.exl3_parallel_mode == "row":
            start = layer.exl3_tp_rank * layer.exl3_input_size_per_partition
            size = layer.exl3_input_size_per_partition
            for shard_id in layer.exl3_shard_ids:
                layer.suh.exl3_tensors[shard_id] = (
                    layer.suh.exl3_tensors[shard_id].narrow(0, start, size).contiguous()
                )
                layer.trellis.exl3_tensors[shard_id] = cls._slice_exl3_tensor(
                    layer.trellis.exl3_tensors[shard_id],
                    dim=0,
                    start=start,
                    size=size,
                )
            return

        already_sharded = cls._expand_tuple_output_shards(layer)
        for shard_id in layer.exl3_shard_ids:
            if shard_id in already_sharded:
                continue
            size = cls._output_shard_size(layer, shard_id)
            start = cls._qkv_output_start(layer, shard_id, size)
            layer.svh.exl3_tensors[shard_id] = (
                layer.svh.exl3_tensors[shard_id].narrow(0, start, size).contiguous()
            )
            layer.trellis.exl3_tensors[shard_id] = cls._slice_exl3_tensor(
                layer.trellis.exl3_tensors[shard_id],
                dim=1,
                start=start,
                size=size,
            )

    @classmethod
    def _expand_tuple_output_shards(cls, layer: torch.nn.Module) -> set[int]:
        tuples = [sid for sid in layer.exl3_shard_ids if isinstance(sid, tuple)]
        if not tuples:
            return set()

        expanded_ids: list[ShardId] = []
        component_ids: set[int] = set()
        for shard_id in layer.exl3_shard_ids:
            if isinstance(shard_id, tuple):
                expanded_ids.extend(shard_id)
                component_ids.update(shard_id)
            else:
                expanded_ids.append(shard_id)

        for tuple_id in tuples:
            full_offsets: dict[int, int] = {}
            offset = 0
            for idx in tuple_id:
                full_offsets[idx] = offset
                offset += layer.exl3_output_partition_sizes[idx] * layer.exl3_tp_size
            for idx in tuple_id:
                size = layer.exl3_output_partition_sizes[idx]
                start = full_offsets[idx] + layer.exl3_tp_rank * size
                layer.suh.exl3_tensors[idx] = layer.suh.exl3_tensors[tuple_id]
                layer.svh.exl3_tensors[idx] = (
                    layer.svh.exl3_tensors[tuple_id].narrow(0, start, size).contiguous()
                )
                layer.trellis.exl3_tensors[idx] = cls._slice_exl3_tensor(
                    layer.trellis.exl3_tensors[tuple_id],
                    dim=1,
                    start=start,
                    size=size,
                )
                layer.exl3_expected_codebooks[idx] = layer.exl3_expected_codebooks[
                    tuple_id
                ]
                for marker in ("mcg", "mul1"):
                    tensors = getattr(layer, marker).exl3_tensors
                    if tuple_id in tensors:
                        tensors[idx] = tensors[tuple_id]
            for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
                getattr(layer, attr).exl3_tensors.pop(tuple_id, None)
            layer.exl3_expected_codebooks.pop(tuple_id, None)

        layer.exl3_shard_ids = expanded_ids
        return component_ids

    @staticmethod
    def _shard_ids_for_layer(
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
    ) -> list[ShardId]:
        if len(output_partition_sizes) == 1:
            return [None]
        prefix = getattr(layer, "prefix", "")
        if isinstance(layer, QKVParallelLinear) and len(output_partition_sizes) == 3:
            return ["q", "k", "v"]
        if prefix.endswith("in_proj_qkvz"):
            if len(output_partition_sizes) != 4:
                raise ValueError(
                    "EXL3 Qwen GDN in_proj_qkvz requires split qkv/z and b/a "
                    "projections (four qkvz output partitions), got "
                    f"{len(output_partition_sizes)}. Check EXL3 split-projection "
                    "metadata detection."
                )
            return [(0, 1, 2), 3]
        return list(range(len(output_partition_sizes)))

    def _source_prefixes_for_layer(
        self, layer: torch.nn.Module, shard_ids: list[ShardId]
    ) -> list[str]:
        prefix = getattr(layer, "prefix", "")
        if len(shard_ids) == 1:
            return [prefix or "lm_head"]
        leaf = prefix.rsplit(".", 1)[-1]
        base = prefix.rsplit(".", 1)[0] if "." in prefix else ""
        sources = self.quant_config.packed_modules_mapping.get(leaf)
        if sources and len(sources) == len(shard_ids):
            return [f"{base}.{source}" if base else source for source in sources]
        raise ValueError(
            f"EXL3 does not know the source matrices for packed layer {prefix}; "
            "add it to the model's packed_modules_mapping."
        )

    @staticmethod
    def _apply_one(
        layer: torch.nn.Module, x: torch.Tensor, shard_id: ShardId
    ) -> torch.Tensor:
        trellis = layer.trellis.exl3_tensors[shard_id]
        packed_k = trellis.shape[0] * 16
        if x.shape[-1] > packed_k:
            raise ValueError(
                f"EXL3 input width {x.shape[-1]} exceeds packed K={packed_k}"
            )
        if x.shape[-1] < packed_k:
            x = torch.nn.functional.pad(x, (0, packed_k - x.shape[-1]))
        bits = trellis.shape[2] // 16
        raw_shape = (bits, packed_k, trellis.shape[1] * 16)
        use_raw_trellis = (
            os.getenv("VLLM_EXL3_SM70_RAW_TRELLIS", "0") == "1"
            and raw_shape in _SM70_RAW_TRELLIS_SHAPES
            and shard_id in layer.mcg.exl3_tensors
            and shard_id not in layer.mul1.exl3_tensors
        )
        int8_weights = getattr(layer, "exl3_tm_int8", {}).get(shard_id)
        state = getattr(layer, "exl3_tm_states", {}).get(shard_id)
        if use_raw_trellis:
            locks = layer.exl3_tm_locks[shard_id]
            output = _exl3_sm70_tm_raw_dispatch_gemm_persistent_locks(
                x,
                trellis,
                layer.suh.exl3_tensors[shard_id],
                layer.svh.exl3_tensors[shard_id],
                locks,
                bits,
                True,
                False,
            )
        elif int8_weights is not None:
            packed_lane, tile_scales = int8_weights
            locks = layer.exl3_tm_locks[shard_id]
            output = _exl3_sm70_tm_int8_dispatch_gemm_persistent_locks(
                x,
                trellis,
                packed_lane,
                tile_scales,
                layer.suh.exl3_tensors[shard_id],
                layer.svh.exl3_tensors[shard_id],
                locks,
                bits,
                shard_id in layer.mcg.exl3_tensors,
                shard_id in layer.mul1.exl3_tensors,
            )
        elif state is not None:
            locks = layer.exl3_tm_locks[shard_id]
            output = _exl3_sm70_tm_dispatch_gemm_persistent_locks(
                x,
                trellis,
                state,
                layer.suh.exl3_tensors[shard_id],
                layer.svh.exl3_tensors[shard_id],
                locks,
                bits,
                shard_id in layer.mcg.exl3_tensors,
                shard_id in layer.mul1.exl3_tensors,
            )
        else:
            output = _exl3_gemm(
                x,
                trellis,
                layer.suh.exl3_tensors[shard_id],
                layer.svh.exl3_tensors[shard_id],
                shard_id in layer.mcg.exl3_tensors,
                shard_id in layer.mul1.exl3_tensors,
            )
        logical_n = Exl3LinearMethod._output_shard_size(layer, shard_id)
        if output.shape[-1] < logical_n:
            raise ValueError(
                f"EXL3 packed N={output.shape[-1]} is below logical N={logical_n}"
            )
        return output[..., :logical_n]
