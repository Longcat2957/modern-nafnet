# Reference: NAFNet_arch.py/local_arch.py, commit 2b4af71ebe098a92a75910c233a3965a3e93ede4.
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# Reimplemented with current PyTorch and fixed TLC initialization; see THIRD_PARTY_NOTICES.md.
"""Checkpoint-compatible NAFNet implemented without a BasicSR dependency."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from drone_deblur.models.layers import LayerNorm2d, SimpleGate
from drone_deblur.models.tlc import LocalAvgPool2d

UPSTREAM_COMMIT = "2b4af71ebe098a92a75910c233a3965a3e93ede4"


class NAFBlock(nn.Module):
    """Original two residual branches and simplified channel attention."""

    def __init__(
        self, c: int, DW_Expand: int = 2, FFN_Expand: int = 2, drop_out_rate: float = 0.0
    ) -> None:
        super().__init__()
        expanded = c * DW_Expand
        ffn = c * FFN_Expand
        self.conv1 = nn.Conv2d(c, expanded, 1)
        self.conv2 = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.conv3 = nn.Conv2d(expanded // 2, c, 1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(expanded // 2, expanded // 2, 1)
        )
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: Tensor) -> Tensor:
        gated = self.sg(self.conv2(self.conv1(self.norm1(inp))))
        attended = gated * self.sca(gated)
        first = inp + self.dropout1(self.conv3(attended)) * self.beta
        feed_forward = self.conv5(self.sg(self.conv4(self.norm2(first))))
        return first + self.dropout2(feed_forward) * self.gamma


class NAFNet(nn.Module):
    """NAFNet width32 defaults; constructor names match the official network."""

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 1,
        enc_blk_nums: Sequence[int] = (1, 1, 1, 28),
        dec_blk_nums: Sequence[int] = (1, 1, 1, 1),
    ) -> None:
        super().__init__()
        if img_channel < 1 or width < 1 or len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("Positive channels/width and matching encoder/decoder depths required")
        if min((middle_blk_num, *enc_blk_nums, *dec_blk_nums)) < 0:
            raise ValueError("Block counts cannot be negative")
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = width
        for count in enc_blk_nums:
            self.encoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(count))))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2
        self.middle_blks = nn.Sequential(*(NAFBlock(channels) for _ in range(middle_blk_num)))
        for count in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            channels //= 2
            self.decoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(count))))
        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        return F.pad(x, (0, -width % self.padder_size, 0, -height % self.padder_size))

    def forward(self, inp: Tensor) -> Tensor:
        height, width = inp.shape[-2:]
        padded = self.check_image_size(inp)
        features = self.intro(padded)
        skips = []
        for encoder, down in zip(self.encoders, self.downs, strict=True):
            features = encoder(features)
            skips.append(features)
            features = down(features)
        features = self.middle_blks(features)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips), strict=True):
            features = decoder(up(features) + skip)
        return (self.ending(features) + padded)[:, :, :height, :width]


class NAFNetLocal(NAFNet):
    """TLC kernels equal the official train-size dummy forward, without running it."""

    def __init__(
        self,
        *args: Any,
        train_size: Sequence[int] = (1, 3, 256, 256),
        fast_imp: bool = False,
        **kwargs: Any,
    ) -> None:
        if fast_imp:
            raise ValueError("fast_imp=True is an approximate TLC path; use fast_imp=False")
        super().__init__(*args, **kwargs)
        if len(train_size) != 4 or min(train_size) <= 0 or train_size[1] != self.intro.in_channels:
            raise ValueError("train_size must be positive NCHW and match img_channel")
        self.train_size = tuple(train_size)
        self.fast_imp = fast_imp
        train_h, train_w = self.train_size[-2:]
        base_h, base_w = int(train_h * 1.5), int(train_w * 1.5)
        padded_h = train_h + (-train_h % self.padder_size)
        padded_w = train_w + (-train_w % self.padder_size)

        def set_pool(blocks: nn.Sequential, level: int) -> None:
            # Shapes are deterministic: the reference forward only initializes these kernels.
            kernel = (
                (padded_h // 2**level) * base_h // train_h,
                (padded_w // 2**level) * base_w // train_w,
            )
            for block in blocks:
                block.sca[0] = LocalAvgPool2d(kernel)

        for level, blocks in enumerate(self.encoders):
            set_pool(blocks, level)
        set_pool(self.middle_blks, len(self.encoders))
        for index, blocks in enumerate(self.decoders):
            set_pool(blocks, len(self.encoders) - index - 1)
        self.eval()


def build_model(model_config: Mapping[str, Any]) -> NAFNetLocal:
    """Build the configured official-shape local model without changing the input dict."""
    options = dict(model_config)
    if "train_size" in options:
        options["train_size"] = tuple(options["train_size"])
    return NAFNetLocal(**options)


def tlc_kernels(model: nn.Module) -> dict[str, list[int]]:
    """Return initialized TLC windows for checkpoint and export metadata."""
    return {
        name: list(module.kernel_size)
        for name, module in model.named_modules()
        if isinstance(module, LocalAvgPool2d)
    }


def check_checkpoint_metadata(
    model: nn.Module, checkpoint: Mapping[str, Any], model_config: Mapping[str, Any]
) -> None:
    """Reject own checkpoints whose weight-compatible TLC/config state differs."""

    def normalize(values: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: tuple(value) if isinstance(value, (list, tuple)) else value
            for key, value in values.items()
        }

    if "config" in checkpoint:
        config = checkpoint["config"]
        if not isinstance(config, Mapping):
            raise ValueError("Checkpoint config must be a mapping")
        if "model" in config:
            saved_model = config["model"]
            if not isinstance(saved_model, Mapping):
                raise ValueError("Checkpoint model config must be a mapping")
            if normalize(saved_model) != normalize(model_config):
                raise ValueError("Checkpoint model config differs from the requested config")
    if "tlc_kernels" in checkpoint:
        saved_kernels = checkpoint["tlc_kernels"]
        if not isinstance(saved_kernels, Mapping):
            raise ValueError("Checkpoint TLC kernels must be a mapping")
        if normalize(saved_kernels) != normalize(tlc_kernels(model)):
            raise ValueError("Checkpoint TLC kernels differ from the initialized model")


def load_checkpoint(model: nn.Module, path: str | Path) -> dict[str, Any]:
    """Strictly load raw/official/FP-training weights and return non-weight fields."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a state dict or a dict containing params/state_dict")
    if "modelopt_state" in checkpoint:
        raise ValueError(
            "Restore ModelOpt checkpoints with the QAT workflow before loading weights"
        )
    weight_keys = [name for name in ("params", "state_dict") if name in checkpoint]
    if len(weight_keys) > 1:
        raise ValueError("Checkpoint contains both params and state_dict; select one explicitly")
    if weight_keys:
        state_key = weight_keys[0]
        weights = checkpoint[state_key]
        metadata = {key: value for key, value in checkpoint.items() if key != state_key}
    else:
        weights, metadata = checkpoint, {}
    if not isinstance(weights, Mapping):
        raise ValueError("Checkpoint weights must be a state dict")
    model.load_state_dict(weights, strict=True)
    return metadata
