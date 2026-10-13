"""Imagewise quality evaluation with the original NAFNet image protocol."""

from pathlib import Path

import torch

from drone_deblur.artifacts import environment_manifest, sha256, write_json
from drone_deblur.data.images import rgb_to_uint8
from drone_deblur.data.paired import PairedImageDataset
from drone_deblur.evaluation.metrics import evaluate_image
from drone_deblur.inference import load_inference_model


@torch.inference_mode()
def run_evaluate(config: dict) -> dict:
    settings = config["evaluation"]
    if not settings.get("manifest") or not settings.get("output"):
        raise ValueError("evaluation.manifest and evaluation.output are required")
    output = Path(settings["output"])
    protected = [settings["manifest"], settings.get("checkpoint")]
    if any(value and output.resolve() == Path(value).resolve() for value in protected):
        raise ValueError("Evaluation output must not overwrite the manifest or checkpoint")
    dataset = PairedImageDataset(settings["manifest"])
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
        "manifest_sha256": sha256(settings["manifest"]),
        "protocol": "upstream-use_image-uint8-RGB-border0-ssim3d-imagewise",
        "backend": "pytorch_fake_int8" if settings.get("quantized") else "pytorch_fp32",
        "count": len(rows),
        "psnr": sum(row["psnr"] for row in rows) / len(rows),
        "ssim": sum(row["ssim"] for row in rows) / len(rows),
        "images": rows,
    }
    write_json(output, result)
    return result
