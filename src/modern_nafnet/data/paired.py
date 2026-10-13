"""Directory image pairs, with no implicit resizing."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from modern_nafnet.data.images import read_rgb
from modern_nafnet.data.sources import load_samples


class PairedImageDataset(Dataset):
    def __init__(self, source: str | Path, transform: Any = None, *, blur_transform: Any = None):
        self.samples = load_samples(source, allow_sharp_only=blur_transform is not None)
        self.transform = transform
        self.blur_transform = blur_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        sharp = read_rgb(sample["sharp"])
        synthetic = "blur" not in sample
        blur = sharp if synthetic else read_rgb(sample["blur"])
        if blur.shape != sharp.shape:
            raise ValueError(
                f"Unaligned pair {sample['name']}: blur {blur.shape}, sharp {sharp.shape}"
            )
        if self.transform is not None:
            transformed = self.transform(image=blur, sharp=sharp)
            blur, sharp = transformed["image"], transformed["sharp"]
        if synthetic:
            blur = self.blur_transform(image=blur)["image"]
        return {
            "blur": self._to_tensor(blur),
            "sharp": self._to_tensor(sharp),
            "name": sample["name"],
        }

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.uint8:
            raise ValueError("The augmentation pipeline must preserve uint8 RGB values")
        chw = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
        return torch.from_numpy(chw).div_(255)


def init_data_worker(_worker_id: int) -> None:
    """DataLoader processes handle parallelism; avoid nested OpenCV thread pools."""
    import cv2

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
