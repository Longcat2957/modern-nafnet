import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from drone_deblur.training.losses import PSNRLoss


def test_psnr_loss_averages_imagewise_and_has_finite_gradient():
    # Two images have MSE 1/4 and 1: averaging MSE first would give a different loss.
    pred = torch.stack((torch.full((3, 2, 2), 0.5), torch.ones(3, 2, 2))).requires_grad_()
    target = torch.zeros_like(pred)
    loss = PSNRLoss()(pred, target)
    assert loss.item() == pytest.approx(5 * math.log10(0.25 + 1e-8) + 5 * math.log10(1 + 1e-8))
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    expected = (
        (10 / math.log(10))
        * 2
        * pred.detach()
        / (pred.detach().square().mean(dim=(1, 2, 3), keepdim=True) + 1e-8)
        / pred.numel()
    )
    torch.testing.assert_close(pred.grad, expected)


def test_psnr_loss_rgb_default_and_original_y_coefficients():
    pred = torch.full((1, 3, 2, 2), 0.25)
    target = torch.zeros_like(pred)
    rgb_loss = PSNRLoss()(pred, target)
    y_loss = PSNRLoss(toY=True)(pred, target)
    expected_mse = (0.25 * (65.481 + 128.553 + 24.966) / 255) ** 2
    assert y_loss.item() == pytest.approx(10 * math.log10(expected_mse + 1e-8), abs=2e-6)
    assert not torch.isclose(rgb_loss, y_loss)
    with pytest.raises(ValueError, match="mean"):
        PSNRLoss(reduction="sum")
    with pytest.raises(ValueError, match="matching NCHW"):
        PSNRLoss()(pred[0], target[0])


def test_half_loss_reduction_does_not_overflow_or_underflow():
    pred = torch.full((1, 3, 2, 2), 1e-4, dtype=torch.float16, requires_grad=True)
    loss = PSNRLoss()(pred, torch.zeros_like(pred))
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pred.grad).all()


@pytest.mark.reference
def test_loss_forward_and_gradient_match_pinned_upstream():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/reference/NAFNet/2b4af71ebe098a92a75910c233a3965a3e93ede4"
        / "basicsr/models/losses/losses.py"
    )
    if not path.is_file():
        pytest.skip("Pinned upstream loss snapshot is not present")
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PSNRLoss"
    ]
    namespace = {"torch": torch, "nn": torch.nn, "np": np}
    exec(compile(tree, str(path), "exec"), namespace)
    pred = torch.rand(2, 3, 5, 7, generator=torch.Generator().manual_seed(15)).requires_grad_()
    target = torch.rand(pred.shape, generator=torch.Generator().manual_seed(16))
    original = pred.detach().clone().requires_grad_()
    loss = PSNRLoss()(pred, target)
    reference = namespace["PSNRLoss"]()(original, target)
    loss.backward()
    reference.backward()
    torch.testing.assert_close(loss, reference, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(pred.grad, original.grad, atol=1e-6, rtol=1e-6)
