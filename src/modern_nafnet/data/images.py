"""The image boundary is uint8 HWC RGB; model floats use the range [0, 1]."""

from pathlib import Path

import cv2
import numpy as np


def _validate_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f"Expected a nonempty HWC RGB image, got {image.shape}")


def read_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    _validate_rgb(image)
    if image.dtype != np.uint8:
        raise ValueError(f"Expected an 8-bit RGB image: {path} ({image.dtype})")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_uint8(image: np.ndarray) -> np.ndarray:
    """Apply the official evaluation clip, multiply, round, and uint8 conversion."""
    image = np.asarray(image)
    _validate_rgb(image)
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if not np.issubdtype(image.dtype, np.floating) or not np.isfinite(image).all():
        raise ValueError("Expected finite floating RGB values or uint8 RGB")
    return np.ascontiguousarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))


def write_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    rgb = rgb_to_uint8(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image: {path}")
