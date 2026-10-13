# Reference: NAFNet arch_util.py, commit 2b4af71ebe098a92a75910c233a3965a3e93ede4.
# Copyright (c) 2022 megvii-model. Copyright 2018-2020 BasicSR Authors.
# Reimplemented with standard autograd; see THIRD_PARTY_NOTICES.md and licenses/.
"""Channel normalization with the original NAFNet parameter names."""

import torch
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    """Normalize channels independently at each pixel, with biased variance."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Half inputs use FP32 statistics; FP32/FP64 reference arithmetic is unchanged.
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        centered = values - values.mean(dim=1, keepdim=True)
        variance = centered.square().mean(dim=1, keepdim=True)
        normalized = centered / (variance + self.eps).sqrt()
        result = normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return result.to(dtype=x.dtype)


class SimpleGate(nn.Module):
    """Multiply the two channel halves without changing the NAFNet gate."""

    def forward(self, x: Tensor) -> Tensor:
        left, right = x.chunk(2, dim=1)
        return left * right
