"""Imagewise quality evaluation with the original NAFNet image protocol."""

from pathlib import Path

import torch

from modern_nafnet.artifacts import environment_manifest, sha256, write_json
from modern_nafnet.data.images import rgb_to_uint8
from modern_nafnet.data.paired import PairedImageDataset
from modern_nafnet.data.sources import resolve_data_source, source_record
from modern_nafnet.evaluation.metrics import evaluate_image
from modern_nafnet.inference import load_inference_model


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
    dataset = PairedImageDataset(source)
    record = source_record(source, dataset.samples)
    if any(
        output.resolve() == sample[key].resolve()
        for sample in dataset.samples
        for key in ("blur", "sharp")
    ):
        raise ValueError("Evaluation output must not overwrite any dataset image")
    model, device = load_inference_model(config, "evaluation")
    rows = []
    for sample in dataset:
        restored = model(sample["blur"].unsqueeze(0).to(device))
        prediction = rgb_to_uint8(restored[0].permute(1, 2, 0).cpu().numpy())
        target = rgb_to_uint8(sample["sharp"].permute(1, 2, 0).numpy())
        rows.append({"name": sample["name"], **evaluate_image(prediction, target, device=device)})
    result = {
        **environment_manifest(),
        "config": config,
        "checkpoint_sha256": sha256(settings["checkpoint"]),
        "dataset": record,
        "protocol": "upstream-use_image-uint8-RGB-border0-ssim3d-imagewise",
        "backend": "pytorch_fake_int8" if settings.get("quantized") else "pytorch_fp32",
        "count": len(rows),
        "psnr": sum(row["psnr"] for row in rows) / len(rows),
        "ssim": sum(row["ssim"] for row in rows) / len(rows),
        "images": rows,
    }
    write_json(output, result)
    return result
