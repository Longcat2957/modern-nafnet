"""Manifest paths are relative to the manifest, with no implicit resizing."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from drone_deblur.data.images import read_rgb


def load_manifest(manifest: str | Path) -> list[dict[str, Any]]:
    manifest = Path(manifest).resolve()
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("A paired manifest must contain a nonempty JSON array")
    samples = []
    names: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry {index} must be an object")
        sample: dict[str, Any] = {}
        for key in ("blur", "sharp"):
            value = entry.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Manifest entry {index} needs a nonempty {key} path")
            path = (manifest.parent / value).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Manifest entry {index} {key}: {path}")
            sample[key] = path
        name = entry.get("name", entry["blur"])
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"Manifest entry {index} needs a unique nonempty name")
        names.add(name)
        sample["name"] = name
        samples.append(sample)
    return samples


class PairedImageDataset(Dataset):
    def __init__(self, manifest: str | Path, transform: Any = None):
        self.samples = load_manifest(manifest)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        blur, sharp = read_rgb(sample["blur"]), read_rgb(sample["sharp"])
        if blur.shape != sharp.shape:
            raise ValueError(
                f"Unaligned pair {sample['name']}: blur {blur.shape}, sharp {sharp.shape}"
            )
        if self.transform is not None:
            transformed = self.transform(image=blur, sharp=sharp)
            blur, sharp = transformed["image"], transformed["sharp"]
        return {
            "blur": self._to_tensor(blur),
            "sharp": self._to_tensor(sharp),
            "name": sample["name"],
        }

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.uint8:
            raise ValueError("The geometry pipeline must preserve uint8 RGB values")
        chw = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
        return torch.from_numpy(chw).div_(255)
