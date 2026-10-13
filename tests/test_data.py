import json

import cv2
import numpy as np
import pytest
import torch

from modern_nafnet.data.augment import build_blur_transform, build_train_transform
from modern_nafnet.data.images import read_rgb, rgb_to_uint8, write_rgb
from modern_nafnet.data.paired import PairedImageDataset, init_data_worker


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


def test_sharp_only_blur_is_seeded_crop_first_and_preserves_gt(tmp_path, monkeypatch):
    from modern_nafnet.data import paired

    sharp = np.random.default_rng(4).integers(0, 256, (40, 48, 3), dtype=np.uint8)
    write_rgb(tmp_path / "sharp.png", sharp)
    manifest = tmp_path / "sharp.json"
    manifest.write_text('[{"sharp":"sharp.png"}]')
    geometry = build_train_transform(16, seed=42)
    degradation = build_blur_transform([7, 7], seed=43)
    expected_gt = geometry(image=sharp, sharp=sharp)["sharp"]
    expected_blur = degradation(image=expected_gt)["image"]
    reads, blur_shapes = [], []
    original_read = paired.read_rgb

    def read_once(path):
        reads.append(path)
        return original_read(path)

    def check_crop_first(**kwargs):
        blur_shapes.append(kwargs["image"].shape)
        return build_blur_transform([7, 7], seed=43)(**kwargs)

    monkeypatch.setattr(paired, "read_rgb", read_once)
    dataset = PairedImageDataset(
        manifest, build_train_transform(16, seed=42), blur_transform=check_crop_first
    )
    pair = dataset[0]
    assert len(reads) == 1 and blur_shapes == [(16, 16, 3)]
    assert pair["name"] == "sharp.png"
    assert torch.equal(pair["sharp"], torch.from_numpy(expected_gt.copy()).permute(2, 0, 1) / 255)
    assert torch.equal(pair["blur"], torch.from_numpy(expected_blur).permute(2, 0, 1) / 255)
    assert not torch.equal(pair["blur"], pair["sharp"])
    assert pair["blur"].is_contiguous() and pair["sharp"].is_contiguous()
    assert pair["blur"].dtype == torch.float32
    assert 0 <= pair["blur"].min() <= pair["blur"].max() <= 1
    np.testing.assert_array_equal(read_rgb(tmp_path / "sharp.png"), sharp)


def test_mixed_manifest_blurs_only_missing_inputs(tmp_path):
    rgb = np.random.default_rng(2).integers(0, 256, (12, 12, 3), dtype=np.uint8)
    manifest = _manifest(tmp_path, rgb, 255 - rgb)
    entries = json.loads(manifest.read_text())
    entries.append({"name": "synthetic", "sharp": "sharp.png"})
    manifest.write_text(json.dumps(entries))
    dataset = PairedImageDataset(manifest, blur_transform=build_blur_transform([7, 7], seed=3))
    real, synthetic = dataset[0], dataset[1]
    assert torch.allclose(real["blur"] + real["sharp"], torch.ones_like(real["sharp"]))
    assert torch.equal(real["sharp"], synthetic["sharp"])
    assert not torch.equal(synthetic["blur"], synthetic["sharp"])
    with pytest.raises(ValueError, match="blur path"):
        PairedImageDataset(manifest)
    entries[1]["blur"] = None
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="blur path"):
        PairedImageDataset(manifest, blur_transform=build_blur_transform([7, 7], seed=3))


def test_worker_init_disables_nested_opencv_threads(monkeypatch):
    calls = []
    monkeypatch.setattr(cv2, "setNumThreads", lambda n: calls.append(("threads", n)))
    monkeypatch.setattr(cv2.ocl, "setUseOpenCL", lambda enabled: calls.append(("opencl", enabled)))
    init_data_worker(0)
    assert calls == [("threads", 0), ("opencl", False)]


@pytest.mark.parametrize("blur_range", [[2, 7], [3, 4], [7, 3], [3], [True, 7], [3, 7.0]])
def test_invalid_blur_range_is_not_silently_adjusted(blur_range):
    with pytest.raises(ValueError, match="blur_range"):
        build_blur_transform(blur_range, seed=1)
