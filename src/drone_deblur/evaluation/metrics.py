"""NAFNet's RGB, border-zero, imagewise PSNR and channel-inclusive SSIM3d.

Reference: NAFNet 2b4af71ebe098a92a75910c233a3965a3e93ede4,
basicsr/metrics/psnr_ssim.py. Reimplemented with no BasicSR imports.
Copyright (c) 2022 megvii-model; Copyright 2018-2020 BasicSR Authors.
Upstream SSIM attribution: modified from
https://github.com/mayorx/matlab_ssim_pytorch_implementation/blob/main/calc_ssim.py
See THIRD_PARTY_NOTICES.md for original licenses and the upstream SSIM attribution.
"""

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _check_pair(pred: np.ndarray, target: np.ndarray) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"Image shapes differ: {pred.shape}, {target.shape}")
    if pred.ndim != 3 or pred.shape[2] != 3 or min(pred.shape[:2]) < 1:
        raise ValueError("Metrics expect nonempty HWC RGB images")
    if pred.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError("Official image metrics require rounded uint8 RGB images")


def calculate_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    _check_pair(pred, target)
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return math.inf
    # Preserve upstream's range heuristic, including its dark-image edge case.
    max_value = 1 if pred.max() <= 1 else 255
    return float(20 * np.log10(max_value / np.sqrt(mse)))


@torch.no_grad()
def calculate_ssim(
    pred: np.ndarray, target: np.ndarray, *, device: str | torch.device = "cpu"
) -> float:
    _check_pair(pred, target)
    g = cv2.getGaussianKernel(11, 1.5)
    spatial = np.outer(g, g.transpose())
    kernel = np.stack([spatial * weight for weight in g], axis=0)
    weight = torch.as_tensor(kernel, dtype=torch.float32, device=device)[None, None]
    x = torch.as_tensor(np.ascontiguousarray(pred), dtype=torch.float32, device=device)[None, None]
    y = torch.as_tensor(np.ascontiguousarray(target), dtype=torch.float32, device=device)[
        None, None
    ]

    def smooth(value: torch.Tensor) -> torch.Tensor:
        # HWC stays in this order: the Gaussian also spans the RGB dimension.
        return F.conv3d(F.pad(value, (5, 5, 5, 5, 5, 5), mode="replicate"), weight)

    mu_x, mu_y = smooth(x), smooth(y)
    var_x = smooth(x.square()) - mu_x.square()
    var_y = smooth(y.square()) - mu_y.square()
    covariance = smooth(x * y) - mu_x * mu_y
    max_value = 1 if pred.max() <= 1 else 255
    c1, c2 = (0.01 * max_value) ** 2, (0.03 * max_value) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * covariance + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    return float((numerator / denominator).mean().item())


def evaluate_image(
    pred_rgb_uint8: np.ndarray, gt_rgb_uint8: np.ndarray, *, device: str | torch.device = "cpu"
) -> dict[str, float]:
    return {
        "psnr": calculate_psnr(pred_rgb_uint8, gt_rgb_uint8),
        "ssim": calculate_ssim(pred_rgb_uint8, gt_rgb_uint8, device=device),
    }
