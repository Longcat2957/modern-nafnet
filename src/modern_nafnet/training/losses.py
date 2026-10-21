"""NAFNet PSNRLoss formula, reimplemented without a BasicSR dependency.

Reference: NAFNet 2b4af71ebe098a92a75910c233a3965a3e93ede4.
Copyright (c) 2022 megvii-model; Copyright 2018-2020 BasicSR Authors.
Original notices are in THIRD_PARTY_NOTICES.md.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PSNRLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0, reduction: str = "mean", toY: bool = False):
        super().__init__()
        if reduction != "mean":
            raise ValueError("PSNRLoss supports only mean reduction")
        self.loss_weight = loss_weight
        self.scale = 10 / math.log(10)
        self.toY = toY
        self.register_buffer(
            "coef",
            torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 4 or pred.shape != target.shape:
            raise ValueError("PSNRLoss expects matching NCHW prediction and target")
        if self.toY:
            if pred.shape[1] != 3:
                raise ValueError("Y-channel PSNRLoss requires RGB inputs")
            coef = self.coef.to(device=pred.device)
            pred = ((pred * coef).sum(dim=1, keepdim=True) + 16) / 255
            target = ((target * coef).sum(dim=1, keepdim=True) + 16) / 255
        # Keep the loss reduction stable when the model uses AMP.
        delta = (
            pred.float() - target.float()
            if pred.dtype in (torch.float16, torch.bfloat16)
            else pred - target
        )
        mse = delta.square().mean(dim=(1, 2, 3))
        return self.loss_weight * self.scale * torch.log(mse + 1e-8).mean()


class MultiTaskRestorationLoss(nn.Module):
    """Pixel/edge/multiscale restoration, raw-logit BCE and clean identity."""

    def __init__(
        self,
        charbonnier_eps: float = 0.001,
        edge_weight: float = 0.05,
        multiscale_weight: float = 0.1,
        task_weight: float = 0.1,
        clean_weight: float = 0.5,
    ):
        super().__init__()
        values = (charbonnier_eps, edge_weight, multiscale_weight, task_weight, clean_weight)
        if any(not math.isfinite(value) or value < 0 for value in values) or not charbonnier_eps:
            raise ValueError(
                "Restoration loss requires positive eps and nonnegative finite weights"
            )
        self.eps = charbonnier_eps
        self.edge_weight, self.multiscale_weight = edge_weight, multiscale_weight
        self.task_weight, self.clean_weight = task_weight, clean_weight

    def _charbonnier(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((prediction - target).square() + self.eps**2).sqrt().mean()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        inputs: torch.Tensor,
        task_logits: torch.Tensor | None,
        task_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.ndim != 4 or prediction.shape != target.shape or inputs.shape != target.shape:
            raise ValueError("Restoration loss expects matching NCHW prediction, target and input")
        if (
            task_labels.ndim != 2
            or task_labels.shape[0] != prediction.shape[0]
            or task_labels.shape[1] < 1
            or not torch.all((task_labels == 0) | (task_labels == 1))
        ):
            raise ValueError("Restoration loss expects binary [batch, tasks] multi-hot labels")
        prediction, target, inputs = prediction.float(), target.float(), inputs.float()
        zero = prediction.sum() * 0
        pixel = self._charbonnier(prediction, target)
        edge_terms = []
        if prediction.shape[-1] > 1:
            edge_terms.append(
                self._charbonnier(
                    prediction[..., 1:] - prediction[..., :-1],
                    target[..., 1:] - target[..., :-1],
                )
            )
        if prediction.shape[-2] > 1:
            edge_terms.append(
                self._charbonnier(
                    prediction[..., 1:, :] - prediction[..., :-1, :],
                    target[..., 1:, :] - target[..., :-1, :],
                )
            )
        edge = sum(edge_terms) / len(edge_terms) if edge_terms else zero
        scales = [scale for scale in (2, 4) if min(prediction.shape[-2:]) >= scale]
        multiscale = (
            sum(
                self._charbonnier(F.avg_pool2d(prediction, scale), F.avg_pool2d(target, scale))
                for scale in scales
            )
            / len(scales)
            if scales
            else zero
        )
        if task_logits is not None:
            if task_logits.shape != task_labels.shape:
                raise ValueError("Task logits and labels must have the same [batch, tasks] shape")
            task = F.binary_cross_entropy_with_logits(task_logits.float(), task_labels.float())
        else:
            task = zero
        clean_mask = task_labels.sum(dim=1) == 0
        clean = (
            (prediction[clean_mask] - inputs[clean_mask]).abs().mean() if clean_mask.any() else zero
        )
        restore = pixel + self.edge_weight * edge + self.multiscale_weight * multiscale
        total = restore + self.task_weight * task + self.clean_weight * clean
        return total, {
            "total": total,
            "restore": restore,
            "charbonnier": pixel,
            "edge": edge,
            "multiscale": multiscale,
            "task": task,
            "clean": clean,
        }
