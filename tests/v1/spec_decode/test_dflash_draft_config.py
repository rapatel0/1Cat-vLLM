# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace

from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


@dataclass
class _CacheConfig:
    cache_dtype: str = "fp8_ds_mla"
    kv_cache_dtype_skip_layers: list[str] = field(default_factory=lambda: ["layer"])


@dataclass
class _AttentionConfig:
    use_non_causal: bool = False


@dataclass
class _VllmConfig:
    cache_config: _CacheConfig = field(default_factory=_CacheConfig)
    attention_config: _AttentionConfig = field(default_factory=_AttentionConfig)


def _proposer(*, use_dspark: bool) -> DFlashProposer:
    proposer = object.__new__(DFlashProposer)
    proposer.speculative_config = SimpleNamespace(use_dspark=lambda: use_dspark)
    proposer.dflash_causal = False
    return proposer


def test_dspark_preserves_target_cache_config(monkeypatch) -> None:
    base = _VllmConfig()
    monkeypatch.setattr(
        SpecDecodeBaseProposer,
        "_create_draft_vllm_config",
        lambda self: base,
    )

    draft = _proposer(use_dspark=True)._create_draft_vllm_config()

    assert draft is base
    assert draft.cache_config.cache_dtype == "fp8_ds_mla"
    assert draft.cache_config.kv_cache_dtype_skip_layers == ["layer"]
    assert not draft.attention_config.use_non_causal


def test_dflash_uses_model_dtype_non_causal_cache(monkeypatch) -> None:
    base = _VllmConfig()
    monkeypatch.setattr(
        SpecDecodeBaseProposer,
        "_create_draft_vllm_config",
        lambda self: base,
    )

    draft = _proposer(use_dspark=False)._create_draft_vllm_config()

    assert draft is not base
    assert draft.cache_config.cache_dtype == "auto"
    assert draft.cache_config.kv_cache_dtype_skip_layers == []
    assert draft.attention_config.use_non_causal
    assert base.cache_config.cache_dtype == "fp8_ds_mla"
    assert not base.attention_config.use_non_causal
