# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.transformers_utils.configs.speculators.algos import update_dflash


def test_dflash_infers_multistream_target_hidden_size() -> None:
    config = {
        "draft_vocab_size": 32000,
        "target_hidden_size": None,
        "mask_token_id": 1,
        "aux_hidden_state_layer_ids": [3, 13, 23, 32, 42],
    }
    pretrained_config = {"hidden_size": 4096, "hc_mult": 4}

    update_dflash(config, pretrained_config)

    assert pretrained_config["target_hidden_size"] == 16384


def test_dflash_preserves_explicit_target_hidden_size() -> None:
    config = {
        "draft_vocab_size": 32000,
        "target_hidden_size": 7168,
        "mask_token_id": 1,
        "aux_hidden_state_layer_ids": [3, 13, 23, 32, 42],
    }
    pretrained_config = {"hidden_size": 4096, "hc_mult": 4}

    update_dflash(config, pretrained_config)

    assert pretrained_config["target_hidden_size"] == 7168
