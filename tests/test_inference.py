import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from modern_nafnet.config import DEFAULT_CONFIG
from modern_nafnet.data.images import read_rgb, write_rgb
from modern_nafnet.evaluation.quality import run_evaluate
from modern_nafnet.inference import run_infer
from modern_nafnet.models.nafnet import build_model


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
    root = tmp_path / "data"
    write_rgb(root / "val/blur/image.png", image)
    write_rgb(root / "val/sharp/image.png", image)
    config["data"]["path"] = str(root)
    config["evaluation"].update(
        checkpoint=str(checkpoint),
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
    root = Path(inference_config["data"]["path"])
    assert quality["dataset"]["source"] == str(root / "val")
    assert len(quality["dataset"]["sha256"]) == 64
    assert quality["dataset"]["samples"][0]["sharp"] == str(root / "val/sharp/image.png")
    assert "manifest_sha256" not in quality


def test_original_image_cannot_be_overwritten(inference_config):
    inference_config["inference"]["output"] = inference_config["inference"]["input"]
    with pytest.raises(ValueError, match="overwrite"):
        run_infer(inference_config)


@pytest.mark.parametrize("kind", ["blur", "sharp"])
def test_evaluation_cannot_overwrite_dataset_image(inference_config, kind):
    image = Path(inference_config["data"]["path"]) / "val" / kind / "image.png"
    original = image.read_bytes()
    inference_config["evaluation"]["output"] = str(image)
    with pytest.raises(ValueError, match="dataset image"):
        run_evaluate(inference_config)
    assert image.read_bytes() == original


def test_evaluation_selects_test_split_with_gopro_sequence_layout(inference_config):
    root = Path(inference_config["data"]["path"])
    image = np.full((11, 13, 3), 90, dtype=np.uint8)
    write_rgb(root / "test/sequence/blur/image.png", image)
    write_rgb(root / "test/sequence/sharp/image.png", image)
    inference_config["evaluation"]["split"] = "test"
    result = run_evaluate(inference_config)
    assert result["count"] == 1
    assert result["dataset"]["source"] == str(root / "test")
    assert result["dataset"]["samples"][0]["blur"] == str(root / "test/sequence/blur/image.png")
    assert result["dataset"]["samples"][0]["sharp"] == str(root / "test/sequence/sharp/image.png")


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
