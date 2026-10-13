# ruff: noqa: E402
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("albumentations")

from modern_nafnet.config import DEFAULT_CONFIG
from modern_nafnet.data.augment import build_blur_transform, build_train_transform
from modern_nafnet.data.images import write_rgb
from modern_nafnet.training.loop import restore_rng, rng_state, run_train


@pytest.fixture
def training_case(tmp_path, monkeypatch):
    from modern_nafnet.models import nafnet

    def factory(**kwargs):
        return torch.nn.Conv2d(3, 3, 1)

    monkeypatch.setattr(nafnet, "NAFNetLocal", factory)
    source = tmp_path / "source.pt"
    torch.save(factory().state_dict(), source)
    root = tmp_path / "data"
    write_rgb(root / "train/blur/image.png", np.full((8, 8, 3), 70, dtype=np.uint8))
    write_rgb(root / "train/sharp/image.png", np.full((8, 8, 3), 120, dtype=np.uint8))
    config = deepcopy(DEFAULT_CONFIG)
    config["model"] = {}
    config["data"].update(path=str(root), crop_size=8)
    config["training"].update(
        checkpoint=source, output_dir=str(tmp_path / "run"), steps=1, device="cpu"
    )
    return config


def test_fp32_training_checkpoint_is_safe_and_contains_training_state(training_case):
    result = run_train(training_case)
    saved = torch.load(result["checkpoint"], weights_only=True)
    source = torch.load(training_case["training"]["checkpoint"], weights_only=True)
    assert result["step"] == 1 and result["resumable"]
    assert saved["training_state"]["optimizer"]["state"]
    assert saved["training_state"]["scheduler"]["last_epoch"] == 1
    assert saved["training_state"]["rng"]["augmentation"]["children"]
    snapshot = saved["training_state"]["dataset"]
    root = Path(training_case["data"]["path"])
    assert snapshot["source"] == str(root / "train")
    assert len(snapshot["sha256"]) == 64
    assert len(snapshot["samples"]) == 1
    assert snapshot["samples"][0]["blur"] == str(root / "train/blur/image.png")
    assert snapshot["samples"][0]["sharp"] == str(root / "train/sharp/image.png")
    assert "manifest_sha256" not in saved["training_state"]
    assert isinstance(saved["config"]["training"]["checkpoint"], str)
    assert not torch.equal(saved["state_dict"]["weight"], source["weight"])
    assert all(torch.isfinite(tensor).all() for tensor in saved["state_dict"].values())


@pytest.mark.parametrize("steps", [0, 1])
def test_ptq_and_qat_save_exportable_modelopt_checkpoints(training_case, steps):
    pytest.importorskip("modelopt")
    pytest.importorskip("onnx")
    from modern_nafnet.deployment.export import run_export

    source = Path(training_case["training"]["checkpoint"])
    root = Path(training_case["data"]["path"])
    calibration = root / "calibration"
    write_rgb(calibration / "first.png", np.full((8, 8, 3), 70, dtype=np.uint8))
    write_rgb(calibration / "second.png", np.full((8, 8, 3), 90, dtype=np.uint8))
    training_case["quantization"]["calibration_batches"] = 1
    training_case["training"]["steps"] = steps
    training_case["export"]["shape"] = [1, 3, 8, 8]
    result = run_train(training_case, qat=True)
    saved = torch.load(result["checkpoint"], weights_only=True)
    assert result["mode"] == ("qat" if steps else "ptq")
    assert saved["modelopt_state"]
    snapshot = saved["training_state"]["calibration_dataset"]
    assert snapshot["source"] == str(calibration)
    assert len(snapshot["sha256"]) == 64
    assert len(snapshot["samples"]) == 1
    assert snapshot["samples"][0]["blur"] == str(calibration / "first.png")
    assert "sharp" not in snapshot["samples"][0]
    assert "calibration_manifest_sha256" not in saved["training_state"]
    if steps:
        assert saved["training_state"]["optimizer"]["state"]
    else:
        assert not saved["training_state"]["resumable"]
    training_case["export"].update(
        checkpoint=result["checkpoint"],
        output=str(source.parent / "quantized.onnx"),
        quantized=True,
        device="cpu",
    )
    exported = run_export(training_case)
    assert exported["quantize_nodes"] and exported["dequantize_nodes"]
    assert Path(exported["manifest"]).is_file()


def test_partial_epoch_export_checkpoint_cannot_claim_exact_resume(training_case):
    root = Path(training_case["data"]["path"])
    write_rgb(root / "train/blur/second.png", np.full((8, 8, 3), 70, dtype=np.uint8))
    write_rgb(root / "train/sharp/second.png", np.full((8, 8, 3), 120, dtype=np.uint8))
    result = run_train(training_case)
    assert not result["resumable"]
    training_case["training"]["resume"] = result["checkpoint"]
    with pytest.raises(ValueError, match="epoch-boundary"):
        run_train(training_case)


@pytest.mark.parametrize("synthetic", [False, True])
def test_epoch_checkpoint_survives_failure_and_resumes(training_case, monkeypatch, synthetic):
    from modern_nafnet.training import losses

    original = losses.PSNRLoss

    class InterruptedLoss(original):
        calls = 0

        def forward(self, prediction, target):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated interruption")
            return super().forward(prediction, target)

    training_case["training"]["steps"] = 2
    if synthetic:
        training_case["data"]["train"] = "clean"
        write_rgb(
            Path(training_case["data"]["path"]) / "clean/image.png",
            np.random.default_rng(10).integers(0, 256, (12, 12, 3), dtype=np.uint8),
        )
        training_case["data"]["synthetic_blur"].update(enabled=True, blur_range=[3, 7])
    uninterrupted = deepcopy(training_case)
    uninterrupted["training"]["output_dir"] += "-uninterrupted"
    expected = torch.load(run_train(uninterrupted)["checkpoint"], weights_only=True)
    monkeypatch.setattr(losses, "PSNRLoss", InterruptedLoss)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_train(training_case)
    boundary = Path(training_case["training"]["output_dir"]) / "last.pt"
    checkpoint = torch.load(boundary, weights_only=True)
    assert checkpoint["training_state"]["step"] == 1
    assert checkpoint["training_state"]["batch_in_epoch"] == 0
    monkeypatch.setattr(losses, "PSNRLoss", original)
    training_case["training"]["resume"] = str(boundary)
    if synthetic:
        changed = deepcopy(training_case)
        changed["data"]["synthetic_blur"]["blur_range"] = [3, 9]
        with pytest.raises(ValueError, match="original data.synthetic_blur"):
            run_train(changed)
    result = run_train(training_case)
    assert result["step"] == 2 and result["epoch"] == 2
    actual = torch.load(result["checkpoint"], weights_only=True)
    for key, value in expected["state_dict"].items():
        assert torch.equal(value, actual["state_dict"][key])


def test_sharp_only_training_with_worker_and_saved_degradation_rng(training_case):
    root = Path(training_case["data"]["path"])
    training_case["data"]["train"] = "clean"
    write_rgb(root / "clean/image.png", np.full((8, 8, 3), 120, dtype=np.uint8))
    training_case["data"]["synthetic_blur"]["enabled"] = True
    training_case["training"]["num_workers"] = 1
    result = run_train(training_case)
    saved = torch.load(result["checkpoint"], weights_only=True)
    assert result["step"] == 1 and result["resumable"]
    assert saved["config"]["data"]["synthetic_blur"]["enabled"]
    assert saved["training_state"]["rng"]["synthetic_blur"]["children"]
    assert not saved["training_state"]["worker_state_saved"]
    samples = saved["training_state"]["dataset"]["samples"]
    assert samples[0]["sharp"] == str(root / "clean/image.png")
    assert "blur" not in samples[0]


def test_resume_rejects_changed_image_content(training_case):
    result = run_train(training_case)
    training_case["training"]["resume"] = result["checkpoint"]
    root = Path(training_case["data"]["path"])
    write_rgb(root / "train/sharp/image.png", np.full((8, 8, 3), 121, dtype=np.uint8))
    with pytest.raises(ValueError, match="dataset changed"):
        run_train(training_case)


def test_old_checkpoint_without_snapshot_requires_weights_only_fine_tuning(training_case):
    result = run_train(training_case)
    checkpoint = torch.load(result["checkpoint"], weights_only=True)
    checkpoint["training_state"].pop("dataset")
    old = Path(training_case["training"]["output_dir"]).parent / "old.pt"
    torch.save(checkpoint, old)
    training_case["training"]["resume"] = str(old)
    with pytest.raises(ValueError, match="no dataset snapshot.*--checkpoint.*--resume"):
        run_train(training_case)
    training_case["training"].update(resume=None, checkpoint=str(old))
    assert run_train(training_case)["step"] == 1


def test_independent_augmentation_and_data_rng_restore():
    transform = build_train_transform(8, 14)
    blur_transform = build_blur_transform([3, 7], 15)
    generator = torch.Generator().manual_seed(77)
    image = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    state = rng_state(generator, transform, blur_transform)
    expected_sharp = transform(image=image, sharp=image)["sharp"]
    expected = blur_transform(image=expected_sharp)["image"]
    expected_order = torch.randperm(9, generator=generator)
    transform(image=image, sharp=image)
    blur_transform(image=expected_sharp)
    restore_rng(state, generator, transform, blur_transform)
    actual_sharp = transform(image=image, sharp=image)["sharp"]
    actual = blur_transform(image=actual_sharp)["image"]
    assert np.array_equal(actual_sharp, expected_sharp)
    assert np.array_equal(actual, expected)
    assert torch.equal(torch.randperm(9, generator=generator), expected_order)


def test_synthetic_resume_rejects_missing_rng_state():
    transform = build_blur_transform([3, 7], 15)
    generator = torch.Generator().manual_seed(77)
    state = rng_state(generator)
    with pytest.raises(ValueError, match="missing the synthetic blur RNG"):
        restore_rng(state, generator, blur_transform=transform)
