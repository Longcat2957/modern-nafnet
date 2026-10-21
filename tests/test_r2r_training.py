# ruff: noqa: E402
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("albumentations")

from modern_nafnet.config import DEFAULT_CONFIG
from modern_nafnet.data.images import read_rgb, write_rgb
from modern_nafnet.inference import run_infer
from modern_nafnet.models.factory import build_model_from_config, load_checkpoint
from modern_nafnet.training.loop import run_train


@pytest.fixture
def restoration_case(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    config["architecture"] = "trt-r2r-light"
    config["trt_r2r_light"].update(
        tasks=["derain", "deblur"],
        channels=[32] * 4,
        query_dim=8,
        slots_per_task=2,
        encoder_blocks=[1, 0, 0, 1],
        decoder_blocks=[0, 0, 0],
    )
    root = tmp_path / "data"
    rng = np.random.default_rng(14)
    for split in ("train", "val", "calibration"):
        for task in ("derain", "deblur", "clean"):
            target = rng.integers(20, 230, (19, 23, 3), dtype=np.uint8)
            image = target // 2
            if split == "calibration":
                write_rgb(
                    root / split / task / "input/image.png", target if task == "clean" else image
                )
            else:
                write_rgb(root / split / task / "target/image.png", target)
                if task != "clean":
                    write_rgb(root / split / task / "input/image.png", image)
    config["data"].update(path=str(root), crop_size=16)
    config["training"].update(
        steps=2,
        batch_size=3,
        device="cpu",
        output_dir=str(tmp_path / "train"),
    )
    config["export"].update(shape=[1, 3, 32, 32], device="cpu")
    config["quantization"]["calibration_batches"] = 3
    return config


@pytest.mark.parametrize("bank_enabled", [False, True])
def test_train_from_scratch_strict_restore_and_external_inference_padding(
    restoration_case,
    bank_enabled,
):
    config = restoration_case
    config["trt_r2r_light"]["bank_enabled"] = bank_enabled
    result = run_train(config)
    payload = torch.load(result["checkpoint"], weights_only=True)
    assert result["resumable"] and result["step"] == 2
    assert payload["architecture"] == "trt-r2r-light"
    assert payload["training_state"]["task_counts"] == {"clean": 2, "deblur": 2, "derain": 2}
    assert payload["training_state"]["sampler"] == "BalancedTaskSampler"
    assert payload["training_state"]["rng"]["augmentation"]["children"]
    assert (
        payload["bank_fingerprint"] is not None
        if bank_enabled
        else payload["bank_fingerprint"] is None
    )
    assert (
        result["loss_components"]["task"] > 0
        if bank_enabled
        else result["loss_components"]["task"] == 0
    )
    model = build_model_from_config(config)
    load_checkpoint(model, result["checkpoint"], config)
    source = Path(config["data"]["path"]) / "val/derain/input/image.png"
    config["inference"].update(
        checkpoint=result["checkpoint"],
        input=str(source),
        device="cpu",
        output=str(source.parents[4] / "output.png"),
    )
    inferred = run_infer(config)
    assert read_rgb(inferred["output"]).shape == (19, 23, 3)
    assert inferred["preprocess"]["pad_multiple"] == 16
    assert inferred["bank_fingerprint"] == payload["bank_fingerprint"]


def test_balanced_epoch_resume_replays_bank_and_augmentation(restoration_case, monkeypatch):
    from modern_nafnet.training import losses

    original = losses.MultiTaskRestorationLoss

    class InterruptOnce(original):
        calls = 0

        def forward(self, *args):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("intentional interruption")
            return super().forward(*args)

    config = restoration_case
    expected_config = deepcopy(config)
    expected_config["training"]["output_dir"] += "-expected"
    expected = torch.load(run_train(expected_config)["checkpoint"], weights_only=True)
    monkeypatch.setattr(losses, "MultiTaskRestorationLoss", InterruptOnce)
    with pytest.raises(RuntimeError, match="intentional interruption"):
        run_train(config)
    boundary = Path(config["training"]["output_dir"]) / "last.pt"
    monkeypatch.setattr(losses, "MultiTaskRestorationLoss", original)
    config["training"]["resume"] = str(boundary)
    actual = torch.load(run_train(config)["checkpoint"], weights_only=True)
    for name, tensor in expected["state_dict"].items():
        torch.testing.assert_close(actual["state_dict"][name], tensor, rtol=0, atol=0)
    assert actual["training_state"]["task_counts"] == expected["training_state"]["task_counts"]


def test_ptq_qat_separate_artifacts_and_preserve_prototypes_scales(restoration_case):
    pytest.importorskip("modelopt")
    from modern_nafnet.training.quantize import restore_quantized_model

    config = restoration_case
    fp_result = run_train(config)
    fp_path = Path(fp_result["checkpoint"])
    fp_bytes = fp_path.read_bytes()
    config["training"].update(checkpoint=str(fp_path), steps=0)
    ptq_result = run_train(config, qat=True)
    ptq_path = Path(ptq_result["checkpoint"])
    ptq = torch.load(ptq_path, weights_only=True)
    assert ptq_path == fp_path.parent / "ptq/ptq.pt"
    assert ptq["training_state"]["calibration_shape"] == [1, 3, 32, 32]
    assert {row["task"] for row in ptq["training_state"]["calibration_dataset"]["samples"]} == {
        "clean",
        "derain",
        "deblur",
    }
    config["training"].update(checkpoint=str(ptq_path), steps=1)
    qat_result = run_train(config, qat=True)
    qat_payload = torch.load(qat_result["checkpoint"], weights_only=True)
    assert Path(qat_result["checkpoint"]) == fp_path.parent / "qat/last.pt"
    assert fp_path.read_bytes() == fp_bytes
    assert qat_payload["bank_fingerprint"] == ptq["bank_fingerprint"]
    for name, value in ptq["state_dict"].items():
        if name.endswith("._amax") or name in ("bank.key", "bank.value"):
            assert torch.equal(value, qat_payload["state_dict"][name])
    restored = restore_quantized_model(build_model_from_config(config), qat_payload, config=config)
    assert not restored.bank.key.requires_grad and not restored.bank.value.requires_grad
    restored(torch.rand(1, 3, 32, 32)).square().mean().backward()
    assert restored.tail.weight.grad is not None
    assert torch.isfinite(restored.tail.weight.grad).all()
    config["export"]["shape"] = [1, 3, 16, 16]
    with pytest.raises(ValueError, match="calibration_shape"):
        run_train(config, qat=True)


def test_stage_a_rejects_mixed_precision_training(restoration_case):
    restoration_case["training"]["amp"] = True
    with pytest.raises(ValueError, match="Stage A"):
        run_train(restoration_case)


def test_train_rejects_untrained_task_prototypes(restoration_case):
    restoration_case["trt_r2r_light"]["tasks"].append("denoise")
    with pytest.raises(ValueError, match="each configured task and clean.*denoise"):
        run_train(restoration_case)
