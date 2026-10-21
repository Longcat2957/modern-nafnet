import json
from pathlib import Path

import pytest

from modern_nafnet.config import DEFAULT_CONFIG, load_config


def test_default_and_example_model_configuration_preserve_nafnet_contract():
    assert load_config()["architecture"] == "nafnet"
    assert "architecture" not in DEFAULT_CONFIG["model"]
    path = Path(__file__).parents[1] / "configs/trt-r2r-light.json"
    config = load_config(path)
    assert config["architecture"] == "trt-r2r-light"
    assert config["training"]["checkpoint"] is None
    assert config["model"] == DEFAULT_CONFIG["model"]
    assert config["trt_r2r_light"]["tasks"] == ["denoise", "derain", "dehaze", "deblur", "lowlight"]


@pytest.mark.parametrize(
    "document",
    [
        {"architecture": "r2r"},
        {"trt_r2r_light": {"tasks": []}},
        {"trt_r2r_light": {"tasks": ["derain", "derain"]}},
        {"trt_r2r_light": {"tasks": ["clean"]}},
        {"trt_r2r_light": {"tasks": [3]}},
        {"trt_r2r_light": {"slots_per_task": 0}},
        {"trt_r2r_light": {"query_dim": True}},
        {"trt_r2r_light": {"channels": [32, 64, 128]}},
        {"trt_r2r_light": {"channels": [32, 64, 128, 193]}},
        {"trt_r2r_light": {"encoder_blocks": [1, 1, 2]}},
        {"trt_r2r_light": {"decoder_blocks": [1, 1, -1]}},
        {"trt_r2r_light": {"bank_enabled": "false"}},
        {"trt_r2r_light": {"prior_scale_init": float("inf")}},
        {"restoration_loss": {"charbonnier_eps": 0}},
        {"restoration_loss": {"edge_weight": -1}},
        {"task_sampling": {"mode": "unknown"}},
        {"architecture": "trt-r2r-light", "data": {"crop_size": 255}},
        {"architecture": "trt-r2r-light", "export": {"shape": [1, 3, 1080, 1920]}},
        {"architecture": "trt-r2r-light", "export": {"shape": [2, 3, 256, 256]}},
        {"architecture": "trt-r2r-light", "data": {"synthetic_blur": {"enabled": True}}},
    ],
)
def test_r2r_configuration_rejects_invalid_semantics(tmp_path, document):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Configuration"):
        load_config(path)


def test_zero_loss_and_disabled_bank_are_valid_overrides():
    config = load_config(
        overrides={
            "trt_r2r_light": {"bank_enabled": False, "prior_scale_init": 0},
            "restoration_loss": {"edge_weight": 0, "task_weight": 0},
            "task_sampling": {"mode": "proportional"},
        }
    )
    assert config["trt_r2r_light"]["bank_enabled"] is False
    assert config["trt_r2r_light"]["prior_scale_init"] == 0
    assert config["restoration_loss"]["task_weight"] == 0
