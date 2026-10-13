# ruff: noqa: E402
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("albumentations")

from drone_deblur.config import DEFAULT_CONFIG
from drone_deblur.data.augment import build_train_transform
from drone_deblur.data.images import write_rgb
from drone_deblur.training.loop import restore_rng, rng_state, run_train


@pytest.fixture
def training_case(tmp_path, monkeypatch):
    from drone_deblur.models import nafnet

    def factory(**kwargs):
        return torch.nn.Conv2d(3, 3, 1)

    monkeypatch.setattr(nafnet, "NAFNetLocal", factory)
    source = tmp_path / "source.pt"
    torch.save(factory().state_dict(), source)
    write_rgb(tmp_path / "blur.png", np.full((8, 8, 3), 70, dtype=np.uint8))
    write_rgb(tmp_path / "sharp.png", np.full((8, 8, 3), 120, dtype=np.uint8))
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps([{"blur": "blur.png", "sharp": "sharp.png"}]))
    config = deepcopy(DEFAULT_CONFIG)
    config["model"] = {}
    config["data"].update(train_manifest=str(manifest), crop_size=8)
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
    assert isinstance(saved["config"]["training"]["checkpoint"], str)
    assert not torch.equal(saved["state_dict"]["weight"], source["weight"])
    assert all(torch.isfinite(tensor).all() for tensor in saved["state_dict"].values())


@pytest.mark.parametrize("steps", [0, 1])
def test_ptq_and_qat_save_exportable_modelopt_checkpoints(training_case, steps):
    pytest.importorskip("modelopt")
    pytest.importorskip("onnx")
    from drone_deblur.deployment.export import run_export

    source = Path(training_case["training"]["checkpoint"])
    calibration = source.parent / "calibration.json"
    calibration.write_text(json.dumps([{"blur": "blur.png"}]))
    training_case["data"]["calibration_manifest"] = str(calibration)
    training_case["training"]["steps"] = steps
    training_case["export"]["shape"] = [1, 3, 8, 8]
    if not steps:
        training_case["data"]["train_manifest"] = None
    result = run_train(training_case, qat=True)
    saved = torch.load(result["checkpoint"], weights_only=True)
    assert result["mode"] == ("qat" if steps else "ptq")
    assert saved["modelopt_state"]
    assert saved["training_state"]["calibration_manifest_sha256"]
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
    manifest = training_case["data"]["train_manifest"]
    entries = json.loads(Path(manifest).read_text())
    entries[0]["name"] = "first"
    entries.append({"blur": "blur.png", "sharp": "sharp.png", "name": "second"})
    with open(manifest, "w") as handle:
        json.dump(entries, handle)
    result = run_train(training_case)
    assert not result["resumable"]
    training_case["training"]["resume"] = result["checkpoint"]
    with pytest.raises(ValueError, match="epoch-boundary"):
        run_train(training_case)


def test_epoch_checkpoint_survives_failure_and_resumes(training_case, monkeypatch):
    from drone_deblur.training import losses

    original = losses.PSNRLoss

    class InterruptedLoss(original):
        calls = 0

        def forward(self, prediction, target):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated interruption")
            return super().forward(prediction, target)

    training_case["training"]["steps"] = 2
    monkeypatch.setattr(losses, "PSNRLoss", InterruptedLoss)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_train(training_case)
    boundary = Path(training_case["training"]["output_dir"]) / "last.pt"
    checkpoint = torch.load(boundary, weights_only=True)
    assert checkpoint["training_state"]["step"] == 1
    assert checkpoint["training_state"]["batch_in_epoch"] == 0
    monkeypatch.setattr(losses, "PSNRLoss", original)
    training_case["training"]["resume"] = str(boundary)
    result = run_train(training_case)
    assert result["step"] == 2 and result["epoch"] == 2


def test_independent_augmentation_and_data_rng_restore():
    transform = build_train_transform(8, 14)
    generator = torch.Generator().manual_seed(77)
    image = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    state = rng_state(generator, transform)
    expected = transform(image=image, sharp=image)["image"]
    expected_order = torch.randperm(9, generator=generator)
    transform(image=image, sharp=image)
    restore_rng(state, generator, transform)
    actual = transform(image=image, sharp=image)["image"]
    assert np.array_equal(actual, expected)
    assert torch.equal(torch.randperm(9, generator=generator), expected_order)
