from unittest.mock import patch

from vllm import envs
from vllm.model_executor.layers.fused_moe import fused_moe


def test_hy3_mtp_sm70_config_is_exact_shape_only(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_SM70_HY3_MTP_MOE_CONFIG", True)
    with (
        patch.object(fused_moe.current_platform, "is_cuda", return_value=True),
        patch.object(
            fused_moe.current_platform,
            "has_device_capability",
            side_effect=lambda capability: capability == 70,
        ),
    ):
        config = fused_moe.get_default_config(2, 192, 192, 4096, 8, None)
        non_hy3_config = fused_moe.get_default_config(2, 192, 256, 4096, 8, None)

    assert config == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
    }
    assert non_hy3_config != config
