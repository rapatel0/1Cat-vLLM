# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU branch tests for final-stage-local Qwen3.5 native MTP."""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor

_HIDDEN_SIZE = 4


def _stub_predictor(standalone: bool) -> Qwen3_5MultiTokenPredictor:
    predictor = object.__new__(Qwen3_5MultiTokenPredictor)
    torch.nn.Module.__init__(predictor)
    predictor.standalone_draft = standalone
    predictor.num_mtp_layers = 1
    predictor.embed_tokens = lambda ids: torch.zeros(ids.shape[0], _HIDDEN_SIZE)
    predictor.pre_fc_norm_embedding = lambda value: value
    predictor.pre_fc_norm_hidden = lambda value: value
    predictor.fc = lambda value: value[..., :_HIDDEN_SIZE]
    predictor.layers = [
        lambda positions, hidden_states, residual: (hidden_states, residual)
    ]
    predictor.norm = lambda hidden_states, residual: (hidden_states, None)
    return predictor


def _last_rank_of_pp2():
    group = type(
        "PipelineGroup",
        (),
        {"is_first_rank": False, "is_last_rank": True},
    )()
    return patch(
        "vllm.model_executor.models.qwen3_5_mtp.get_pp_group",
        lambda: group,
    )


def test_standalone_draft_executes_complete_layer_on_last_target_stage() -> None:
    predictor = _stub_predictor(standalone=True)
    input_ids = torch.zeros(3, dtype=torch.long)
    positions = torch.zeros(3, dtype=torch.long)
    hidden_states = torch.zeros(3, _HIDDEN_SIZE)

    with _last_rank_of_pp2():
        output = predictor.forward(input_ids, positions, hidden_states)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, _HIDDEN_SIZE)


def test_pipeline_sharded_draft_still_requires_intermediate_tensors() -> None:
    predictor = _stub_predictor(standalone=False)
    input_ids = torch.zeros(3, dtype=torch.long)
    positions = torch.zeros(3, dtype=torch.long)
    hidden_states = torch.zeros(3, _HIDDEN_SIZE)

    with _last_rank_of_pp2(), pytest.raises(AssertionError):
        predictor.forward(input_ids, positions, hidden_states)
