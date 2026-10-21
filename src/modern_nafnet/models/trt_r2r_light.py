"""R2R-inspired residual CNN with a static, trainable degradation bank.

This is an independent design, not an implementation of the R2R architecture.
Inputs are externally padded RGB NCHW tensors with spatial dimensions divisible by 16.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modern_nafnet.config import validate_trt_r2r_light_config

TASKS = ("denoise", "derain", "dehaze", "deblur", "lowlight")


class LiteResidual(nn.Module):
    """A dense Conv3/ReLU/Conv1 residual block without normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 1)

    def forward(self, image: Tensor) -> Tensor:
        return image + self.conv1(F.relu(self.conv3(image)))


class CompactDegradationBank(nn.Module):
    """Fixed-size, task-local softmax retrieval with independent task gates."""

    def __init__(
        self,
        tasks: Sequence[str],
        slots_per_task: int,
        query_dim: int,
        value_dim: int,
    ) -> None:
        super().__init__()
        if (
            not tasks
            or any(task not in TASKS for task in tasks)
            or len(set(tasks)) != len(tasks)
            or any(type(n) is not int or n < 1 for n in (slots_per_task, query_dim, value_dim))
        ):
            raise ValueError("Bank needs unique supported tasks and positive dimensions")
        self.tasks = tuple(tasks)
        self.slots_per_task = slots_per_task
        self.query_dim = query_dim
        self.key = nn.Parameter(torch.empty(len(tasks) * slots_per_task, query_dim))
        self.value = nn.Parameter(torch.empty(len(tasks) * slots_per_task, value_dim))
        self.task_gate = nn.Linear(query_dim, len(tasks))
        nn.init.normal_(self.key, std=0.02)
        nn.init.normal_(self.value, std=0.02)

    def validate_bank(self) -> None:
        """Check stored prototypes before calibration, save, restore, or export."""
        for name, parameter in (("key", self.key), ("value", self.value)):
            if not torch.isfinite(parameter).all().item():
                raise ValueError(f"Bank {name} contains NaN or Inf")

    def freeze_bank(self) -> None:
        self.validate_bank()
        self.key.requires_grad_(False)
        self.value.requires_grad_(False)

    def forward(self, query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        scores = (query @ self.key.transpose(0, 1)) * (self.query_dim**-0.5)
        attention = scores.reshape(-1, len(self.tasks), self.slots_per_task).softmax(dim=-1)
        task_logits = self.task_gate(query)
        weights = (attention * task_logits.sigmoid().unsqueeze(-1)).flatten(1)
        return weights @ self.value, task_logits, attention


class TrtR2RLight(nn.Module):
    """Shared restoration backbone with an optional V0 prototype bank."""

    architecture = "trt-r2r-light"
    padder_size = 16

    def __init__(
        self,
        tasks: Sequence[str] = TASKS,
        slots_per_task: int = 8,
        query_dim: int = 64,
        channels: Sequence[int] = (32, 64, 128, 192),
        encoder_blocks: Sequence[int] = (1, 1, 2, 2),
        decoder_blocks: Sequence[int] = (1, 1, 1),
        bank_enabled: bool = True,
        prior_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_spec = {
            "tasks": list(tasks),
            "slots_per_task": slots_per_task,
            "query_dim": query_dim,
            "channels": list(channels),
            "encoder_blocks": list(encoder_blocks),
            "decoder_blocks": list(decoder_blocks),
            "bank_enabled": bank_enabled,
            "prior_scale_init": prior_scale_init,
        }
        validate_trt_r2r_light_config(self.model_spec)
        self.tasks = tuple(tasks)
        self.bank_enabled = bank_enabled
        self.prior_scale = prior_scale_init
        self.unshuffle = nn.PixelUnshuffle(2)
        self.stem = nn.Conv2d(12, channels[0], 3, padding=1)
        self.encoders = nn.ModuleList(
            nn.Sequential(*(LiteResidual(c) for _ in range(count)))
            for c, count in zip(channels, encoder_blocks, strict=True)
        )
        self.downs = nn.ModuleList(
            nn.Conv2d(source, target, 3, stride=2, padding=1)
            for source, target in zip(channels[:-1], channels[1:], strict=True)
        )
        self.ups = nn.ModuleList(
            nn.Conv2d(source, target, 1)
            for source, target in zip(reversed(channels[1:]), reversed(channels[:-1]), strict=True)
        )
        self.decoders = nn.ModuleList(
            nn.Sequential(*(LiteResidual(c) for _ in range(count)))
            for c, count in zip(reversed(channels[:-1]), decoder_blocks, strict=True)
        )
        self.query_head = nn.Linear(channels[-1], query_dim) if bank_enabled else None
        self.bank = (
            CompactDegradationBank(tasks, slots_per_task, query_dim, channels[-1])
            if bank_enabled
            else None
        )
        self.tail = nn.Conv2d(channels[0], 12, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def validate_bank(self) -> None:
        if self.bank is not None:
            self.bank.validate_bank()

    def freeze_bank(self) -> None:
        """Freeze prototypes for PTQ/QAT; keep query and task heads trainable."""
        if self.bank is not None:
            self.bank.freeze_bank()

    def _forward_impl(self, image: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("TRT-R2R-Light input must be RGB NCHW")
        if min(image.shape[-2:]) < 16 or any(size % 16 for size in image.shape[-2:]):
            raise ValueError("TRT-R2R-Light input height/width must be positive multiples of 16")
        features = self.stem(self.unshuffle(image))
        skips = []
        for encoder, down in zip(self.encoders[:-1], self.downs, strict=True):
            features = encoder(features)
            skips.append(features)
            features = down(features)
        features = self.encoders[-1](features)
        task_logits, attention = None, None
        if self.bank is not None:
            query = self.query_head(features.mean(dim=(-2, -1)))
            prior, task_logits, attention = self.bank(query)
            features = features + self.prior_scale * prior[:, :, None, None]
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips), strict=True):
            features = decoder(F.interpolate(up(features), scale_factor=2, mode="nearest") + skip)
        return self.shuffle(self.tail(features)) + image, task_logits, attention

    def forward(self, image: Tensor) -> Tensor:
        return self._forward_impl(image)[0]

    def forward_with_aux(self, image: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None]:
        return self._forward_impl(image)
