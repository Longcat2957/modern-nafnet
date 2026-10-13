"""NAFNet PSNRLoss formula, reimplemented without a BasicSR dependency.

Reference: NAFNet 2b4af71ebe098a92a75910c233a3965a3e93ede4.
Copyright (c) 2022 megvii-model; Copyright 2018-2020 BasicSR Authors.
Original notices are in THIRD_PARTY_NOTICES.md.
"""

import math

import torch
from torch import nn


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
