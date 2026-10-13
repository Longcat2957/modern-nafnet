import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from modern_nafnet.data.augment import build_blur_transform, build_train_transform
from modern_nafnet.data.images import read_rgb, rgb_to_uint8, write_rgb
from modern_nafnet.data.paired import PairedImageDataset, init_data_worker
from modern_nafnet.data.sources import load_samples, resolve_data_source, source_record


def _pair(tmp_path, blur, sharp):
    write_rgb(tmp_path / "blur" / "image.png", blur)
    write_rgb(tmp_path / "sharp" / "image.png", sharp)
    return tmp_path


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
    source = _pair(tmp_path, blur, 255 - blur)
    dataset_a = PairedImageDataset(source, build_train_transform(12, seed=42))
    dataset_b = PairedImageDataset(source, build_train_transform(12, seed=42))
    first = dataset_a[0]
    repeated = dataset_b[0]
    assert first["blur"].shape == (3, 12, 12)
    assert first["blur"].is_contiguous()
    assert first["blur"].dtype == torch.float32
    assert torch.equal(first["blur"], repeated["blur"])
    assert torch.allclose(first["blur"] + first["sharp"], torch.ones_like(first["blur"]))
    assert 0 <= first["blur"].min() <= first["blur"].max() <= 1
    assert torch.equal(dataset_a[0]["blur"], dataset_b[0]["blur"])


def test_eval_dataset_is_deterministic_and_resolves_directory_paths(tmp_path, monkeypatch):
    rgb = np.full((11, 13, 3), (20, 50, 200), dtype=np.uint8)
    source = _pair(tmp_path, rgb, rgb)
    monkeypatch.chdir(tmp_path.parent)
    dataset = PairedImageDataset(source.name)
    assert dataset.samples[0]["blur"] == source / "blur" / "image.png"
    first, second = dataset[0], dataset[0]
    assert first["blur"].shape == (3, 11, 13)
    assert torch.equal(first["blur"], second["blur"])
    assert torch.allclose(first["blur"][:, 0, 0], torch.tensor([20, 50, 200]) / 255)


def test_missing_corrupt_or_unaligned_pair_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="source"):
        PairedImageDataset(tmp_path / "missing")
    source = _pair(tmp_path, np.zeros((12, 12, 3), np.uint8), np.zeros((11, 12, 3), np.uint8))
    with pytest.raises(ValueError, match="Unaligned pair"):
        PairedImageDataset(source)[0]
    (tmp_path / "blur" / "image.png").write_text("invalid PNG")
    with pytest.raises(ValueError, match="decode"):
        PairedImageDataset(source)[0]


def test_reject_invalid_crop_and_duplicate_sample_paths(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        build_train_transform(0, seed=1)
    rgb = np.zeros((12, 12, 3), np.uint8)
    _pair(tmp_path, rgb, rgb)
    write_rgb(tmp_path / "blur" / "sequence" / "frame.png", rgb)
    write_rgb(tmp_path / "sequence" / "blur" / "frame.png", rgb)
    with pytest.raises(ValueError, match="Duplicate blur"):
        PairedImageDataset(tmp_path)


def test_sharp_only_blur_is_seeded_crop_first_and_preserves_gt(tmp_path, monkeypatch):
    from modern_nafnet.data import paired

    sharp = np.random.default_rng(4).integers(0, 256, (40, 48, 3), dtype=np.uint8)
    write_rgb(tmp_path / "sharp.png", sharp)
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
        tmp_path, build_train_transform(16, seed=42), blur_transform=check_crop_first
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


def test_actual_pairs_are_not_degraded_when_synthetic_blur_is_enabled(tmp_path):
    rgb = np.random.default_rng(2).integers(0, 256, (12, 12, 3), dtype=np.uint8)
    source = _pair(tmp_path, rgb, 255 - rgb)

    def unexpected_blur(**_kwargs):
        pytest.fail("An actual image pair must not receive synthetic blur")

    real = PairedImageDataset(source, blur_transform=unexpected_blur)[0]
    assert torch.allclose(real["blur"] + real["sharp"], torch.ones_like(real["sharp"]))
    (source / "blur" / "image.png").unlink()
    with pytest.raises(ValueError, match="Missing blur.*synthetic fallback"):
        PairedImageDataset(source, blur_transform=unexpected_blur)


@pytest.mark.parametrize("layout", ["simple", "gopro"])
def test_exact_pair_matching_and_stable_sequence_order(tmp_path, layout, monkeypatch):
    rgb = np.zeros((8, 8, 3), np.uint8)
    for sequence in ("z", "a"):
        for role in ("blur", "sharp"):
            folder = (
                tmp_path / role / sequence if layout == "simple" else tmp_path / sequence / role
            )
            write_rgb(folder / "0001.png", rgb)
    (tmp_path / "README.txt").write_text("Non-image files are ignored")
    monkeypatch.setattr(
        cv2, "imread", lambda *_args: pytest.fail("Scanning must not decode images")
    )
    samples = load_samples(tmp_path)
    assert [sample["name"] for sample in samples] == ["a/0001.png", "z/0001.png"]
    assert all(
        isinstance(sample["blur"], Path) and sample["blur"].is_absolute() for sample in samples
    )
    assert load_samples(tmp_path) == samples


def test_dataset_scans_once_and_decodes_only_when_indexed(tmp_path, monkeypatch):
    from modern_nafnet.data import paired

    rgb = np.zeros((8, 8, 3), np.uint8)
    source = _pair(tmp_path, rgb, rgb)
    scans, reads = [], []
    original_scan = Path.rglob

    def scan(path, pattern):
        scans.append(path)
        return original_scan(path, pattern)

    def decode(path):
        reads.append(path)
        return rgb

    monkeypatch.setattr(Path, "rglob", scan)
    monkeypatch.setattr(paired, "read_rgb", decode)
    dataset = PairedImageDataset(source)
    assert scans == [source] and not reads
    dataset[0]
    dataset[0]
    assert scans == [source] and len(reads) == 4


def test_pair_extensions_must_match_exactly(tmp_path):
    rgb = np.zeros((8, 8, 3), np.uint8)
    source = _pair(tmp_path, rgb, rgb)
    (source / "sharp" / "image.png").rename(source / "sharp" / "image.jpg")
    with pytest.raises(ValueError, match="Missing .* pair"):
        load_samples(source, allow_sharp_only=True)


@pytest.mark.parametrize("extra", ["loose.png", "blur/sharp/ambiguous.png"])
def test_unclassified_and_ambiguous_images_are_errors(tmp_path, extra):
    rgb = np.zeros((8, 8, 3), np.uint8)
    source = _pair(tmp_path, rgb, rgb)
    write_rgb(source / extra, rgb)
    with pytest.raises(ValueError, match="Unclassified or ambiguous"):
        load_samples(source, allow_sharp_only=True)


@pytest.mark.parametrize(
    "image_path, expected_name",
    [
        ("frame.png", "frame.png"),
        ("scene/frame.png", "scene/frame.png"),
        ("sharp/frame.png", "frame.png"),
        ("scene/sharp/frame.png", "scene/frame.png"),
    ],
)
def test_sharp_only_directory_layouts_require_opt_in(tmp_path, image_path, expected_name):
    rgb = np.zeros((8, 8, 3), np.uint8)
    write_rgb(tmp_path / image_path, rgb)
    with pytest.raises(ValueError, match="Paired directory|Missing blur"):
        load_samples(tmp_path)
    samples = load_samples(tmp_path, allow_sharp_only=True)
    assert samples == [{"name": expected_name, "sharp": (tmp_path / image_path).resolve()}]
    if "sharp" in Path(image_path).parts[:-1]:
        folder = (tmp_path / image_path).parent
        assert load_samples(folder, allow_sharp_only=True)[0]["name"] == "frame.png"


def test_empty_blur_directory_prevents_synthetic_fallback(tmp_path):
    write_rgb(tmp_path / "sharp" / "image.png", np.zeros((8, 8, 3), np.uint8))
    (tmp_path / "blur").mkdir()
    with pytest.raises(ValueError, match="Missing blur.*synthetic fallback"):
        load_samples(tmp_path, allow_sharp_only=True)


def test_mixed_sharp_roles_and_unclassified_clean_images_are_rejected(tmp_path):
    rgb = np.zeros((8, 8, 3), np.uint8)
    write_rgb(tmp_path / "sharp" / "image.png", rgb)
    write_rgb(tmp_path / "loose.png", rgb)
    with pytest.raises(ValueError, match="Unclassified"):
        load_samples(tmp_path, allow_sharp_only=True)


@pytest.mark.parametrize("options", [{"allow_sharp_only": True}, {"blur_only": True}])
def test_flat_directory_rejects_duplicate_symlink_images(tmp_path, options):
    original = tmp_path / "frame.png"
    write_rgb(original, np.zeros((8, 8, 3), np.uint8))
    (tmp_path / "alias.png").symlink_to(original)
    with pytest.raises(ValueError, match="Duplicate .* image"):
        load_samples(tmp_path, **options)


def test_calibration_selects_blur_without_validating_sharp(tmp_path):
    rgb = np.zeros((8, 8, 3), np.uint8)
    write_rgb(tmp_path / "sequence" / "blur" / "frame.png", rgb)
    ignored = tmp_path / "sequence" / "sharp" / "blur" / "unmatched.png"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("Sharp images are neither matched nor decoded for calibration")
    samples = load_samples(tmp_path, blur_only=True)
    assert samples == [
        {"name": "sequence/frame.png", "blur": tmp_path / "sequence" / "blur" / "frame.png"}
    ]
    assert load_samples(tmp_path / "sequence" / "blur", blur_only=True)[0]["name"] == "frame.png"
    with pytest.raises(ValueError, match="not a sharp directory"):
        load_samples(tmp_path / "sequence" / "sharp", blur_only=True)


def test_flat_calibration_and_empty_directories(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no supported images"):
        load_samples(empty, blur_only=True)
    source = tmp_path / "calibration"
    write_rgb(source / "frame.png", np.zeros((8, 8, 3), np.uint8))
    assert load_samples(source, blur_only=True) == [
        {"name": "frame.png", "blur": source / "frame.png"}
    ]


def test_json_files_are_not_data_inputs(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="must be a directory"):
        load_samples(path)
    with pytest.raises(ValueError, match="must be a directory"):
        source_record(path, [])


def test_source_record_is_json_safe_and_detects_file_metadata_changes(tmp_path, monkeypatch):
    rgb = np.zeros((8, 8, 3), np.uint8)
    _pair(tmp_path, rgb, rgb)
    for role in ("blur", "sharp"):
        write_rgb(tmp_path / role / "second.png", rgb)
    samples = load_samples(tmp_path)
    monkeypatch.setattr(
        Path, "open", lambda *_args, **_kwargs: pytest.fail("Metadata must not read images")
    )
    record = source_record(tmp_path, samples)
    assert json.loads(json.dumps(record)) == record
    assert record["source"] == str(tmp_path)
    assert len(record["samples"]) == 2
    assert record["samples"][0] == {
        "name": "image.png",
        "blur": str(tmp_path / "blur" / "image.png"),
        "sharp": str(tmp_path / "sharp" / "image.png"),
    }
    assert record["sha256"] == source_record(tmp_path, list(reversed(samples)))["sha256"]
    path = samples[0]["blur"]
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    assert record["sha256"] != source_record(tmp_path, samples)["sha256"]
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    samples[0]["name"] = "renamed.png"
    assert record["sha256"] != source_record(tmp_path, samples)["sha256"]


def test_resolve_data_source_uses_cwd_root_and_absolute_split(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = {"path": "dataset", "train": "trainset", "val": str(tmp_path / "external")}
    assert resolve_data_source(data, "train") == tmp_path / "dataset" / "trainset"
    assert resolve_data_source(data, "val") == tmp_path / "external"
    assert resolve_data_source(data, "test") == tmp_path / "dataset" / "test"
    with pytest.raises(ValueError, match="data.path"):
        resolve_data_source({"path": None}, "train")


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
