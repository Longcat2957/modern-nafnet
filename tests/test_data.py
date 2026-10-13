import json

import cv2
import numpy as np
import pytest
import torch

from drone_deblur.data.augment import build_train_transform
from drone_deblur.data.images import read_rgb, rgb_to_uint8, write_rgb
from drone_deblur.data.paired import PairedImageDataset


def _manifest(tmp_path, blur, sharp):
    write_rgb(tmp_path / "blur.png", blur)
    write_rgb(tmp_path / "sharp.png", sharp)
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([{"blur": "blur.png", "sharp": "sharp.png"}]))
    return path


def test_rgb_io_and_official_rounding(tmp_path):
    rgb = np.array([[[4, 129, 250], [255, 0, 31]]], dtype=np.uint8)
    path = tmp_path / "rgb.png"
    assert cv2.imwrite(str(path), rgb[..., ::-1])
    np.testing.assert_array_equal(read_rgb(path), rgb)
    write_rgb(tmp_path / "written.png", rgb)
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / "written.png")), rgb[..., ::-1])
    values = np.array([[[-1.0, 0.5, 2.0]]], dtype=np.float32)
    np.testing.assert_array_equal(rgb_to_uint8(values), [[[0, 128, 255]]])
    with pytest.raises(ValueError, match="finite"):
        rgb_to_uint8(np.full((2, 2, 3), np.nan))


def test_paired_crop_orientation_and_seed(tmp_path):
    blur = np.arange(18 * 20 * 3, dtype=np.uint16).reshape(18, 20, 3).astype(np.uint8)
    manifest = _manifest(tmp_path, blur, 255 - blur)
    dataset_a = PairedImageDataset(manifest, build_train_transform(12, seed=42))
    dataset_b = PairedImageDataset(manifest, build_train_transform(12, seed=42))
    first = dataset_a[0]
    repeated = dataset_b[0]
    assert first["blur"].shape == (3, 12, 12)
    assert first["blur"].is_contiguous()
    assert first["blur"].dtype == torch.float32
    assert torch.equal(first["blur"], repeated["blur"])
    assert torch.allclose(first["blur"] + first["sharp"], torch.ones_like(first["blur"]))
    assert 0 <= first["blur"].min() <= first["blur"].max() <= 1
    assert torch.equal(dataset_a[0]["blur"], dataset_b[0]["blur"])


def test_eval_dataset_is_deterministic_and_resolves_manifest_paths(tmp_path, monkeypatch):
    rgb = np.full((11, 13, 3), (20, 50, 200), dtype=np.uint8)
    manifest = _manifest(tmp_path, rgb, rgb)
    monkeypatch.chdir(tmp_path.parent)
    dataset = PairedImageDataset(manifest)
    first, second = dataset[0], dataset[0]
    assert first["blur"].shape == (3, 11, 13)
    assert torch.equal(first["blur"], second["blur"])
    assert torch.allclose(first["blur"][:, 0, 0], torch.tensor([20, 50, 200]) / 255)


def test_missing_corrupt_or_unaligned_pair_fails(tmp_path):
    manifest = tmp_path / "pairs.json"
    manifest.write_text('[{"blur":"missing.png","sharp":"missing.png"}]')
    with pytest.raises(FileNotFoundError, match="entry 0"):
        PairedImageDataset(manifest)
    manifest = _manifest(tmp_path, np.zeros((12, 12, 3), np.uint8), np.zeros((11, 12, 3), np.uint8))
    with pytest.raises(ValueError, match="Unaligned pair"):
        PairedImageDataset(manifest)[0]
    (tmp_path / "blur.png").write_text("invalid PNG")
    with pytest.raises(ValueError, match="decode"):
        PairedImageDataset(manifest)[0]


def test_reject_invalid_crop_and_duplicate_names(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        build_train_transform(0, seed=1)
    rgb = np.zeros((12, 12, 3), np.uint8)
    manifest = _manifest(tmp_path, rgb, rgb)
    entry = json.loads(manifest.read_text())[0]
    manifest.write_text(json.dumps([entry, entry]))
    with pytest.raises(ValueError, match="unique"):
        PairedImageDataset(manifest)
