"""Task-labelled restoration pairs and reproducible balanced epoch sampling."""

import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from modern_nafnet.data.images import read_rgb
from modern_nafnet.data.paired import PairedImageDataset
from modern_nafnet.data.sources import load_restoration_samples


class RestorationDataset(Dataset):
    def __init__(self, source: str | Path, tasks: list[str], transform: Any = None):
        self.samples = load_restoration_samples(source, tasks)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        target = read_rgb(sample["target"])
        image = target if sample["input"] == sample["target"] else read_rgb(sample["input"])
        if image.shape != target.shape:
            raise ValueError(
                f"Unaligned pair {sample['name']}: input {image.shape}, target {target.shape}"
            )
        if self.transform is not None:
            transformed = self.transform(image=image, sharp=target)
            image, target = transformed["image"], transformed["sharp"]
        return {
            "input": PairedImageDataset._to_tensor(image),
            "target": PairedImageDataset._to_tensor(target),
            "task_label": torch.tensor(sample["task_label"], dtype=torch.float32),
            "name": sample["name"],
            "task": sample["task"],
        }


class BalancedTaskSampler(Sampler[int]):
    """Oversample small task folders; each DDP rank gets the same task counts."""

    def __init__(
        self,
        dataset: RestorationDataset,
        seed: int = 0,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
    ):
        if type(num_replicas) is not int or num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid distributed sampler world size or rank")
        self.groups: dict[str, list[int]] = {}
        for index, sample in enumerate(dataset.samples):
            self.groups.setdefault(sample["task"], []).append(index)
        if not self.groups:
            raise ValueError("Balanced sampling requires a nonempty dataset")
        self.groups = dict(sorted(self.groups.items()))
        self.seed, self.rank, self.num_replicas = seed, rank, num_replicas
        self.shuffle, self.epoch = shuffle, 0
        self.per_task = math.ceil(max(map(len, self.groups.values())) / num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.per_task * len(self.groups)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        ranks = []
        needed = self.per_task * self.num_replicas
        for indices in self.groups.values():
            draws = []
            while len(draws) < needed:
                order = (
                    torch.randperm(len(indices), generator=generator).tolist()
                    if self.shuffle
                    else range(len(indices))
                )
                draws.extend(indices[index] for index in order)
            ranks.append(draws[:needed][self.rank :: self.num_replicas])
        result = []
        for offset in range(self.per_task):
            order = (
                torch.randperm(len(ranks), generator=generator).tolist()
                if self.shuffle
                else range(len(ranks))
            )
            result.extend(ranks[index][offset] for index in order)
        return iter(result)

    def task_counts(self) -> dict[str, int]:
        return dict.fromkeys(self.groups, self.per_task)
