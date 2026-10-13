import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from drone_deblur.config import DEFAULT_CONFIG
from drone_deblur.data.images import read_rgb, write_rgb
from drone_deblur.evaluation.quality import run_evaluate
from drone_deblur.inference import run_infer
from drone_deblur.models.nafnet import build_model


@pytest.fixture
def inference_config(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["model"].update(width=2, enc_blk_nums=[1], dec_blk_nums=[1], train_size=[1, 3, 8, 8])
    checkpoint = tmp_path / "model.pt"
    model = build_model(config["model"])
    torch.save({"state_dict": model.state_dict(), "config": config}, checkpoint)
    source = tmp_path / "source.png"
    image = np.random.default_rng(42).integers(0, 256, (11, 13, 3), dtype=np.uint8)
    write_rgb(source, image)
    config["inference"].update(
        checkpoint=str(checkpoint),
        input=str(source),
        output=str(tmp_path / "result.png"),
        device="cpu",
    )
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps([{"blur": "source.png", "sharp": "source.png"}]))
    config["evaluation"].update(
        checkpoint=str(checkpoint),
        manifest=str(manifest),
        output=str(tmp_path / "quality.json"),
        device="cpu",
    )
    return config


def test_infer_and_quality_workflows(inference_config):
    result = run_infer(inference_config)
    assert read_rgb(result["output"]).shape == (11, 13, 3)
    assert len(result["checkpoint_sha256"]) == 64
    quality = run_evaluate(inference_config)
    assert quality["count"] == 1
    assert np.isfinite(quality["psnr"])
    assert np.isfinite(quality["ssim"])
    assert quality["backend"] == "pytorch_fp32"


def test_original_image_cannot_be_overwritten(inference_config):
    inference_config["inference"]["output"] = inference_config["inference"]["input"]
    with pytest.raises(ValueError, match="overwrite"):
        run_infer(inference_config)


def test_evaluation_cannot_overwrite_ground_truth(inference_config):
    inference_config["evaluation"]["output"] = inference_config["inference"]["input"]
    with pytest.raises(ValueError, match="dataset image"):
        run_evaluate(inference_config)


def test_inference_sidecar_cannot_overwrite_checkpoint(inference_config, tmp_path):
    checkpoint = tmp_path / "result.png.json"
    original = inference_config["inference"]["checkpoint"]
    checkpoint.write_bytes(Path(original).read_bytes())
    inference_config["inference"]["checkpoint"] = str(checkpoint)
    with pytest.raises(ValueError, match="overwrite"):
        run_infer(inference_config)


def test_checkpoint_config_mismatch_fails(inference_config):
    inference_config["model"]["train_size"] = [1, 3, 10, 10]
    with pytest.raises(ValueError, match="config differs"):
        run_infer(inference_config)
