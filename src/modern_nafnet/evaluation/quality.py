"""Imagewise quality evaluation with the original NAFNet image protocol."""

from pathlib import Path

import torch

from modern_nafnet.artifacts import environment_manifest, sha256, write_json
from modern_nafnet.data.images import rgb_to_uint8
from modern_nafnet.data.paired import PairedImageDataset
from modern_nafnet.data.sources import resolve_data_source, source_record
from modern_nafnet.evaluation.metrics import evaluate_image
from modern_nafnet.inference import load_inference_model
from modern_nafnet.models.factory import checkpoint_metadata


def _quality_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "psnr": None, "ssim": None}
    result = {
        "count": len(rows),
        **{key: sum(row[key] for row in rows) / len(rows) for key in ("psnr", "ssim")},
    }
    if "mean_absolute_change" in rows[0]:
        result["mean_absolute_change"] = sum(row["mean_absolute_change"] for row in rows) / len(
            rows
        )
        result["max_absolute_change"] = max(row["max_absolute_change"] for row in rows)
    return result


def multitask_quality_summary(rows: list[dict], tasks: list[str]) -> dict:
    """Keep each folder and clean/composite metrics visible beside the image mean."""
    result = {
        "task_order": list(tasks),
        "per_task": {
            task: _quality_summary([row for row in rows if row["task"] == task])
            for task in sorted({row["task"] for row in rows})
        },
        "clean": _quality_summary([row for row in rows if row["task"] == "clean"]),
        "single": _quality_summary(
            [row for row in rows if row["task"] != "clean" and "+" not in row["task"]]
        ),
        "mixed": _quality_summary([row for row in rows if "+" in row["task"]]),
    }
    aux_rows = [row for row in rows if "task_gates" in row]
    if not aux_rows:
        result["task_diagnostics"] = {"available": False}
        return result
    counts, means = {}, {}
    exact = 0
    clean_rows = [row for row in aux_rows if row["task"] == "clean"]
    clean_positive = sum(any(gate >= 0.5 for gate in row["task_gates"]) for row in clean_rows)
    for index, task in enumerate(tasks):
        tp = fp = fn = tn = 0
        for row in aux_rows:
            active = row["task_label"][index] == 1
            predicted = row["task_gates"][index] >= 0.5
            tp += active and predicted
            fp += not active and predicted
            fn += active and not predicted
            tn += not active and not predicted
        counts[task] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
        means[task] = sum(row["task_gates"][index] for row in aux_rows) / len(aux_rows)
    for row in aux_rows:
        exact += all(
            (gate >= 0.5) == bool(active)
            for gate, active in zip(row["task_gates"], row["task_label"], strict=True)
        )
    result["task_diagnostics"] = {
        "available": True,
        "threshold": 0.5,
        "count": len(aux_rows),
        "exact_match_accuracy": exact / len(aux_rows),
        "per_task": counts,
        "mean_gate": means,
        "clean_false_positive_rate": clean_positive / len(clean_rows) if clean_rows else None,
    }
    slots = [torch.tensor(row["slot_attention"]) for row in aux_rows if "slot_attention" in row]
    if slots:
        result["task_diagnostics"]["mean_slot_usage"] = torch.stack(slots).mean(dim=0).tolist()
    return result


@torch.inference_mode()
def run_evaluate(config: dict) -> dict:
    settings = config["evaluation"]
    if not settings.get("output"):
        raise ValueError("evaluation.output is required")
    split = settings.get("split", "val")
    if split not in ("val", "test"):
        raise ValueError("evaluation.split must be val or test")
    source = resolve_data_source(config["data"], split)
    output = Path(settings["output"])
    protected = [source, settings.get("checkpoint")]
    if any(value and output.resolve() == Path(value).resolve() for value in protected):
        raise ValueError("Evaluation output must not overwrite the data source or checkpoint")
    multitask = config.get("architecture", "nafnet") == "trt-r2r-light"
    if multitask:
        from modern_nafnet.data.multitask import RestorationDataset

        tasks = config["trt_r2r_light"]["tasks"]
        dataset = RestorationDataset(source, tasks)
    else:
        dataset = PairedImageDataset(source)
    record = source_record(source, dataset.samples)
    if any(
        output.resolve() == sample[key].resolve()
        for sample in dataset.samples
        for key in (("input", "target") if multitask else ("blur", "sharp"))
    ):
        raise ValueError("Evaluation output must not overwrite any dataset image")
    model, device = load_inference_model(config, "evaluation")
    rows = []
    snapshot = getattr(model, "training_dataset_snapshot", None)
    seen_labels = (
        {
            tuple(sample["task_label"])
            for sample in snapshot.get("samples", [])
            if "task_label" in sample
        }
        if snapshot
        else None
    )
    for sample in dataset:
        if multitask:
            from modern_nafnet.inference import restore_image

            inputs = sample["input"].unsqueeze(0).to(device)
            restored, logits, attention = restore_image(model, inputs, config, with_aux=True)
            target_tensor = sample["target"]
        else:
            restored = model(sample["blur"].unsqueeze(0).to(device))
            target_tensor = sample["sharp"]
        prediction = rgb_to_uint8(restored[0].permute(1, 2, 0).cpu().numpy())
        target = rgb_to_uint8(target_tensor.permute(1, 2, 0).numpy())
        row = {"name": sample["name"], **evaluate_image(prediction, target, device=device)}
        if multitask:
            row.update(task=sample["task"], task_label=sample["task_label"].tolist())
            original_shape = list(inputs.shape)
            padded_shape = [
                *original_shape[:2],
                *((size + 15) // 16 * 16 for size in original_shape[-2:]),
            ]
            calibration_shape = getattr(model, "calibration_shape", None)
            row.update(
                original_shape=original_shape,
                padded_shape=padded_shape,
                input_dtype=str(inputs.dtype).removeprefix("torch."),
                matches_calibration_shape=(
                    padded_shape == list(calibration_shape) if calibration_shape else None
                ),
            )
            if "+" in sample["task"]:
                row["training_combination"] = (
                    "unknown"
                    if seen_labels is None
                    else "seen"
                    if tuple(row["task_label"]) in seen_labels
                    else "unseen"
                )
            change = (restored[0].cpu() - sample["input"]).abs()
            row.update(
                mean_absolute_change=change.mean().item(), max_absolute_change=change.max().item()
            )
            if logits is not None:
                if not torch.isfinite(logits).all():
                    raise ValueError("Task diagnostic logits must be finite")
                row["task_gates"] = logits[0].sigmoid().cpu().tolist()
            if attention is not None:
                if not torch.isfinite(attention).all():
                    raise ValueError("Task slot attention must be finite")
                row["slot_attention"] = attention[0].cpu().tolist()
        rows.append(row)
    result = {
        **environment_manifest(),
        "config": config,
        "checkpoint_sha256": sha256(settings["checkpoint"]),
        "dataset": record,
        "protocol": "upstream-use_image-uint8-RGB-border0-ssim3d-imagewise",
        "backend": "pytorch_fake_int8" if settings.get("quantized") else "pytorch_fp32",
        **_quality_summary(rows),
        "images": rows,
    }
    if multitask:
        result.update(checkpoint_metadata(model, config))
        result.update(multitask_quality_summary(rows, tasks))
        calibration_shape = getattr(model, "calibration_shape", None)
        result.update(
            evaluation_geometry="native_resolution_external_zero_pad_crop",
            calibration_shape=calibration_shape,
            all_inputs_match_calibration_shape=(
                all(row["matches_calibration_shape"] for row in rows) if calibration_shape else None
            ),
        )
        result["mixed_by_training"] = {
            status: _quality_summary(
                [row for row in rows if row.get("training_combination") == status]
            )
            for status in ("seen", "unseen", "unknown")
        }
        bank = getattr(model, "bank", None)
        if bank is not None:
            diagnostics = {}
            for name in ("key", "value"):
                values = getattr(bank, name).detach().float().flatten(1)
                norms = values.norm(dim=1)
                unit = torch.nn.functional.normalize(values, dim=1)
                similarity = unit @ unit.T
                mask = ~torch.eye(len(values), device=values.device, dtype=torch.bool)
                diagnostics[f"{name}_mean_pair_cosine"] = (
                    similarity[mask].mean().item() if mask.any() else None
                )
                diagnostics[f"{name}_min_norm"] = norms.min().item()
            result["bank_diagnostics"] = diagnostics
    write_json(output, result)
    return result
