import ast
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from drone_deblur.evaluation.metrics import calculate_psnr, evaluate_image


def _metric_fixture():
    rng = np.random.default_rng(123)
    target = rng.integers(0, 256, size=(13, 15, 3), dtype=np.uint8)
    pred = target.copy()
    pred[1::2, 2::3, 0] = 100
    pred[0, :, 2] = 255
    return pred, target


def test_imagewise_psnr_and_ssim_identity():
    image = np.arange(13 * 15 * 3, dtype=np.uint16).reshape(13, 15, 3).astype(np.uint8)
    result = evaluate_image(image, image)
    assert math.isinf(result["psnr"])
    assert result["ssim"] == pytest.approx(1.0, abs=1e-6)


def test_official_channel_inclusive_metric_fixture():
    # Frozen from the pinned upstream CUDA evaluator; CPU Conv3D roundoff differs.
    result = evaluate_image(*_metric_fixture())
    assert result["psnr"] == pytest.approx(19.199172796349284, abs=1e-12)
    assert result["ssim"] == pytest.approx(0.9220057129859924, abs=2e-5)


def test_uint8_psnr_and_original_dark_prediction_range():
    pred = np.full((3, 4, 3), 20, dtype=np.uint8)
    target = np.full_like(pred, 30)
    assert calculate_psnr(pred, target) == pytest.approx(20 * math.log10(25.5))
    # The upstream metric selects range 1 for any prediction with max <= 1.
    assert calculate_psnr(np.zeros_like(pred), target) == pytest.approx(-20 * math.log10(30))


def test_metrics_reject_misalignment_and_unrounded_float():
    image = np.full((13, 15, 3), 20, dtype=np.uint8)
    with pytest.raises(ValueError, match="shapes differ"):
        evaluate_image(image, image[:-1])
    with pytest.raises(ValueError, match="uint8"):
        evaluate_image(image.astype(np.float32) / 255, image.astype(np.float32) / 255)


@pytest.mark.reference
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_metrics_match_pinned_upstream(monkeypatch, device):
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/reference/NAFNet/2b4af71ebe098a92a75910c233a3965a3e93ede4"
        / "basicsr/metrics/psnr_ssim.py"
    )
    if not path.is_file():
        pytest.skip("Pinned upstream metric snapshot is not present")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    names = {
        "calculate_psnr",
        "_3d_gaussian_calculator",
        "_generate_3d_gaussian_kernel",
        "_ssim_3d",
    }
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "cv2": cv2,
        "np": np,
        "torch": torch,
        "reorder_image": lambda image, input_order: image,
    }
    # Only adapt the upstream hardcoded CUDA placement; keep its arithmetic intact.
    if device == "cpu":
        monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    exec(compile(tree, str(path), "exec"), namespace)
    pred, target = _metric_fixture()
    result = evaluate_image(pred, target, device=device)
    assert result["psnr"] == pytest.approx(namespace["calculate_psnr"](pred, target, 0))
    expected_ssim = namespace["_ssim_3d"](pred, target, 255)
    assert result["ssim"] == pytest.approx(expected_ssim, abs=1e-7)
