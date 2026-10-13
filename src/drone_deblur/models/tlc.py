# Reference: NAFNet local_arch.py, commit 2b4af71ebe098a92a75910c233a3965a3e93ede4.
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# Reimplemented with fixed kernels and non-inplace sums; see THIRD_PARTY_NOTICES.md.
"""Exact local pooling used by NAFNet's test-time local converter."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LocalAvgPool2d(nn.Module):
    """Fixed TLC window, global fallback, and original asymmetric edge padding."""

    def __init__(self, kernel_size: tuple[int, int], auto_pad: bool = True) -> None:
        super().__init__()
        if len(kernel_size) != 2 or min(kernel_size) <= 0:
            raise ValueError("TLC kernel_size must contain two positive integers")
        self.kernel_size = tuple(kernel_size)
        self.auto_pad = auto_pad

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        if self.kernel_size[0] >= height and self.kernel_size[1] >= width:
            return F.adaptive_avg_pool2d(x, 1)

        kernel_h = min(height, self.kernel_size[0])
        kernel_w = min(width, self.kernel_size[1])
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        integral = F.pad(values.cumsum(dim=-1).cumsum(dim=-2), (1, 0, 1, 0))
        sums = (
            integral[:, :, kernel_h:, kernel_w:]
            + integral[:, :, :-kernel_h, :-kernel_w]
            - integral[:, :, :-kernel_h, kernel_w:]
            - integral[:, :, kernel_h:, :-kernel_w]
        )
        pooled = sums / (kernel_h * kernel_w)
        if self.auto_pad:
            pooled = F.pad(
                pooled,
                (
                    (kernel_w - 1) // 2,
                    kernel_w // 2,
                    (kernel_h - 1) // 2,
                    kernel_h // 2,
                ),
                mode="replicate",
            )
        return pooled.to(dtype=x.dtype)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, auto_pad={self.auto_pad}"
