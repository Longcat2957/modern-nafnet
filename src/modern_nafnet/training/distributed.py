"""One CUDA process per GPU, launched with torchrun."""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TrainingWorld:
    device: torch.device
    rank: int = 0
    world_size: int = 1

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def run(self, operation):
        """Propagate local input/loss/save failures before the next DDP collective."""
        error = None
        value = None
        try:
            value = operation()
        except Exception as exc:
            error = exc
        if not self.distributed:
            if error is not None:
                raise error
            return value
        failed = torch.tensor(int(error is not None), device=self.device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            messages = self.gather(
                f"rank {self.rank}: {type(error).__name__}: {error}" if error else None
            )
            if error is not None:
                raise error
            raise RuntimeError("Distributed training failed: " + "; ".join(filter(None, messages)))
        return value

    def gather(self, value) -> list:
        if not self.distributed:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def require_same(self, value, message: str) -> None:
        if self.distributed:
            values = self.gather(value)
            if any(item != values[0] for item in values[1:]):
                raise ValueError(message)

    def mean(self, value: torch.Tensor) -> float:
        value = value.detach().clone()
        if self.distributed:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= self.world_size
        return float(value)


@contextmanager
def training_world(requested_device: str):
    """Select LOCAL_RANK before initializing NCCL and always release our group."""
    device = torch.device(requested_device)
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as exc:
        raise ValueError("torchrun WORLD_SIZE, RANK and LOCAL_RANK must be integers") from exc
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid torchrun WORLD_SIZE, RANK or LOCAL_RANK")
    if world_size > 1:
        if any(key not in os.environ for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK")):
            raise ValueError("Launch multi-GPU training with torchrun (WORLD_SIZE/RANK/LOCAL_RANK)")
        if device.type != "cuda":
            raise ValueError("Multi-GPU training requires training.device=cuda")
        if device.index is not None and device.index != local_rank:
            raise ValueError("Use --device cuda with torchrun; LOCAL_RANK selects each GPU")
        if not dist.is_available() or not dist.is_nccl_available():
            raise RuntimeError("Multi-GPU training requires PyTorch with NCCL support")
        if local_rank >= torch.cuda.device_count():
            raise ValueError("torchrun LOCAL_RANK exceeds the visible CUDA device count")
        device = torch.device("cuda", local_rank)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is not available")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
    if world_size == 1:
        yield TrainingWorld(device)
        return
    if dist.is_initialized():
        raise RuntimeError("Training expects to own the torchrun process group")
    dist.init_process_group(
        "nccl", init_method="env://", device_id=device, timeout=timedelta(minutes=2)
    )
    try:
        yield TrainingWorld(device, rank, world_size)
    finally:
        # No barrier here: a failed peer must not leave healthy ranks waiting in cleanup.
        dist.destroy_process_group()
