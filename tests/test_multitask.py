from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
import torch

from modern_nafnet.data.augment import build_train_transform
from modern_nafnet.data.images import write_rgb
from modern_nafnet.data.multitask import BalancedTaskSampler, RestorationDataset
from modern_nafnet.data.sources import (
    balanced_calibration_samples,
    load_restoration_samples,
    source_record,
    task_label,
)
from modern_nafnet.evaluation.quality import multitask_quality_summary
from modern_nafnet.training.losses import MultiTaskRestorationLoss

TASKS = ["derain", "deblur"]


def _sample(root, task, name="scene/frame.png", *, calibration=False, value=80):
    rgb = np.full((12, 14, 3), value, np.uint8)
    roles = ["input"] if calibration else (["target"] if task == "clean" else ["input", "target"])
    for role in roles:
        write_rgb(root / task / role / name, rgb)


def test_restoration_labels_pairs_clean_and_scan_only(tmp_path, monkeypatch):
    for task in ("deblur", "derain", "clean", "derain+deblur"):
        _sample(tmp_path, task)
    import cv2

    with monkeypatch.context() as patch:
        patch.setattr(cv2, "imread", lambda *_: pytest.fail("Indexing must not decode images"))
        samples = load_restoration_samples(tmp_path, TASKS)
    assert [sample["name"] for sample in samples] == [
        "clean/scene/frame.png",
        "deblur/scene/frame.png",
        "derain/scene/frame.png",
        "derain+deblur/scene/frame.png",
    ]
    assert [sample["task_label"] for sample in samples] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert samples[0]["input"] == samples[0]["target"]
    a = RestorationDataset(tmp_path, TASKS, build_train_transform(8, 3))
    b = RestorationDataset(tmp_path, TASKS, build_train_transform(8, 3))
    for index in range(len(a)):
        sample, same = a[index], b[index]
        assert set(sample) == {"input", "target", "task_label", "name", "task"}
        assert sample["input"].shape == (3, 8, 8)
        assert sample["input"].is_contiguous() and sample["input"].dtype == torch.float32
        assert torch.equal(sample["input"], same["input"])
        assert torch.equal(sample["input"], sample["target"])
        assert 0 <= sample["input"].min() <= sample["input"].max() <= 1


@pytest.mark.parametrize("task", ["unknown", "derain+clean", "derain+derain", "dehaze"])
def test_unknown_or_duplicate_task_labels_rejected(task):
    with pytest.raises(ValueError, match="Unknown or duplicate"):
        task_label(task, TASKS)


def test_restoration_missing_extensions_empty_clean_and_mixture_alias_fail(tmp_path):
    _sample(tmp_path, "deblur")
    path = tmp_path / "deblur/target/scene/frame.png"
    path.rename(path.with_suffix(".jpg"))
    with pytest.raises(ValueError, match="Missing .* pair"):
        load_restoration_samples(tmp_path, TASKS)
    path.with_suffix(".jpg").rename(path)
    (tmp_path / "clean/target").mkdir(parents=True)
    with pytest.raises(ValueError, match="no input/target pairs: clean"):
        load_restoration_samples(tmp_path, TASKS)
    _sample(tmp_path, "clean")
    _sample(tmp_path, "derain+deblur")
    _sample(tmp_path, "deblur+derain")
    with pytest.raises(ValueError, match="Duplicate restoration task combination"):
        load_restoration_samples(tmp_path, TASKS)


def test_restoration_corrupt_unaligned_and_duplicate_inputs_fail(tmp_path):
    _sample(tmp_path, "derain")
    target = tmp_path / "derain/target/scene/frame.png"
    write_rgb(target, np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError, match="Unaligned pair"):
        RestorationDataset(tmp_path, TASKS)[0]
    target.write_text("corrupt")
    with pytest.raises(ValueError, match="decode"):
        RestorationDataset(tmp_path, TASKS)[0]
    original = tmp_path / "derain/input/scene/frame.png"
    (original.parent / "alias.png").symlink_to(original)
    with pytest.raises(ValueError, match="Duplicate input"):
        load_restoration_samples(tmp_path, TASKS)


def test_clean_reads_once_and_refuses_extra_input(tmp_path, monkeypatch):
    _sample(tmp_path, "clean")
    from modern_nafnet.data import multitask

    original, reads = multitask.read_rgb, []

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(multitask, "read_rgb", read)
    sample = RestorationDataset(tmp_path, TASKS)[0]
    assert len(reads) == 1 and torch.equal(sample["input"], sample["target"])
    _sample(tmp_path, "clean", calibration=True)
    with pytest.raises(ValueError, match="input=target"):
        load_restoration_samples(tmp_path, TASKS)


def test_balanced_sampler_is_reproducible_and_balanced_on_every_rank(tmp_path):
    for index in range(5):
        _sample(tmp_path, "derain", f"{index}.png")
    _sample(tmp_path, "deblur")
    _sample(tmp_path, "clean")
    data = RestorationDataset(tmp_path, TASKS)
    samplers = [BalancedTaskSampler(data, 17, num_replicas=2, rank=rank) for rank in range(2)]
    for sampler in samplers:
        order = list(sampler)
        assert len(order) == len(sampler) == 9
        assert Counter(data.samples[index]["task"] for index in order) == sampler.task_counts()
        assert sampler.task_counts() == {"clean": 3, "deblur": 3, "derain": 3}
        assert order == list(sampler)
        sampler.set_epoch(1)
        assert list(sampler) != order
    for sampler in samplers:
        sampler.set_epoch(0)
    global_derain = {
        index
        for sampler in samplers
        for index in sampler
        if data.samples[index]["task"] == "derain"
    }
    assert len(global_derain) == 5


def test_calibration_balances_real_inputs_and_requires_coverage(tmp_path):
    for task in ("derain", "deblur", "clean", "derain+deblur"):
        _sample(tmp_path, task, calibration=True)
    for index in range(4):
        _sample(tmp_path, "derain", f"{index}.png", calibration=True)
    samples = load_restoration_samples(tmp_path, TASKS, calibration=True)
    chosen = balanced_calibration_samples(samples, TASKS, 4)
    assert Counter(sample["task"] for sample in chosen) == dict.fromkeys(
        ["derain", "deblur", "clean", "derain+deblur"], 1
    )
    assert all("target" not in sample for sample in samples)
    assert chosen == balanced_calibration_samples(list(reversed(samples)), TASKS, 4)
    with pytest.raises(ValueError, match="limit must cover"):
        balanced_calibration_samples(samples, TASKS, 3)
    with pytest.raises(ValueError, match="every configured task and clean"):
        balanced_calibration_samples(
            [sample for sample in samples if sample["task"] != "clean"], TASKS, 8
        )
    with pytest.raises(ValueError, match="single-task/clean folders.*deblur.*derain"):
        balanced_calibration_samples(
            [sample for sample in samples if sample["task"] in {"clean", "derain+deblur"}],
            TASKS,
            8,
        )


def test_snapshot_records_paths_and_label_order(tmp_path):
    _sample(tmp_path, "derain")
    samples = load_restoration_samples(tmp_path, TASKS)
    record = source_record(tmp_path, samples)
    assert record["samples"][0]["input"] == str(samples[0]["input"])
    assert record["samples"][0]["task_label"] == [1, 0]
    reverse_order = load_restoration_samples(tmp_path, list(reversed(TASKS)))
    assert record["sha256"] != source_record(tmp_path, reverse_order)["sha256"]


def test_restoration_loss_raw_logits_clean_subset_and_backward():
    prediction = torch.full((3, 3, 8, 8), 0.3, requires_grad=True)
    target, inputs = torch.full_like(prediction, 0.5), torch.full_like(prediction, 0.2)
    labels = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    logits = torch.tensor([[-1.0, -2.0], [2.0, -1.0], [3.0, 2.0]], requires_grad=True)
    total, parts = MultiTaskRestorationLoss()(prediction, target, inputs, logits, labels)
    assert parts["task"].item() == pytest.approx(
        torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    )
    assert parts["clean"].item() == pytest.approx(0.1)
    total.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    no_bank, parts = MultiTaskRestorationLoss()(prediction, target, inputs, None, labels)
    assert parts["task"].item() == 0 and torch.isfinite(no_bank)
    with pytest.raises(ValueError, match="multi-hot"):
        MultiTaskRestorationLoss()(prediction, target, inputs, logits, labels + 0.1)


def test_quality_summary_preserves_mixed_clean_and_gate_diagnostics():
    rows = [
        {
            "task": task,
            "task_label": label,
            "task_gates": gates,
            "psnr": score,
            "ssim": 0.9,
            "mean_absolute_change": 0.01,
            "max_absolute_change": 0.1,
            "slot_attention": [[0.25, 0.75], [0.5, 0.5]],
        }
        for task, label, gates, score in (
            ("clean", [0, 0], [0.1, 0.8], 40),
            ("derain", [1, 0], [0.9, 0.2], 30),
            ("derain+deblur", [1, 1], [0.7, 0.6], 20),
        )
    ]
    result = multitask_quality_summary(rows, TASKS)
    assert result["clean"]["psnr"] == 40 and result["mixed"]["psnr"] == 20
    assert result["per_task"]["derain"]["count"] == 1
    diagnostics = result["task_diagnostics"]
    assert diagnostics["clean_false_positive_rate"] == 1
    assert diagnostics["exact_match_accuracy"] == pytest.approx(2 / 3)
    assert diagnostics["per_task"]["deblur"]["false_positive"] == 1
    assert diagnostics["mean_slot_usage"] == [[0.25, 0.75], [0.5, 0.5]]


@pytest.mark.parametrize("training_combination", ["seen", "unseen", "unknown"])
def test_multitask_evaluation_loads_strict_checkpoint_and_crops_padding(
    tmp_path, training_combination
):
    from modern_nafnet.config import DEFAULT_CONFIG
    from modern_nafnet.evaluation.quality import run_evaluate
    from modern_nafnet.models.factory import build_model_from_config, checkpoint_metadata

    config = deepcopy(DEFAULT_CONFIG)
    config["architecture"] = "trt-r2r-light"
    config["trt_r2r_light"].update(
        tasks=TASKS,
        channels=[32] * 4,
        encoder_blocks=[1] * 4,
        decoder_blocks=[1] * 3,
        query_dim=8,
        slots_per_task=2,
    )
    model = build_model_from_config(config)
    checkpoint = tmp_path / "model.pt"
    payload = {"state_dict": model.state_dict(), **checkpoint_metadata(model, config)}
    calibration_shape = {
        "seen": [1, 3, 16, 16],
        "unseen": [1, 3, 32, 32],
        "unknown": None,
    }[training_combination]
    payload["training_state"] = {"calibration_shape": calibration_shape}
    if training_combination != "unknown":
        payload["training_state"]["dataset"] = {
            "samples": [
                {
                    "task_label": [1.0, 1.0] if training_combination == "seen" else [1.0, 0.0],
                }
            ]
        }
    torch.save(payload, checkpoint)
    root = tmp_path / "data"
    for task in ("derain", "clean", "derain+deblur"):
        _sample(root / "val", task)
    config["data"]["path"] = str(root)
    config["evaluation"].update(
        checkpoint=str(checkpoint), output=str(tmp_path / "quality.json"), device="cpu"
    )
    result = run_evaluate(config)
    assert result["count"] == 3 and result["mixed"]["count"] == result["clean"]["count"] == 1
    assert result["task_order"] == TASKS
    assert result["architecture"] == "trt-r2r-light" and result["bank_fingerprint"]
    assert result["mixed_by_training"][training_combination]["count"] == 1
    assert result["calibration_shape"] == calibration_shape
    expected_match = None if calibration_shape is None else calibration_shape == [1, 3, 16, 16]
    assert result["all_inputs_match_calibration_shape"] is expected_match
    assert result["evaluation_geometry"] == "native_resolution_external_zero_pad_crop"
    for row in result["images"]:
        assert row["original_shape"] == [1, 3, 12, 14]
        assert row["padded_shape"] == [1, 3, 16, 16]
        assert row["matches_calibration_shape"] is expected_match
        assert row["input_dtype"] == "float32"
    assert result["task_diagnostics"]["available"]
    assert result["bank_diagnostics"]["key_min_norm"] > 0
    assert result["dataset"]["samples"][0]["task"] == "clean"
    assert (tmp_path / "quality.json").is_file()
