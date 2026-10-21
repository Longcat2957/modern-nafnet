"""Floating or fake-quantized PyTorch image inference on the training host."""

from pathlib import Path

import numpy as np
import torch

from modern_nafnet.artifacts import environment_manifest, sha256, write_json
from modern_nafnet.data.images import read_rgb, write_rgb
from modern_nafnet.models.factory import (
    build_model_from_config,
    check_checkpoint_metadata,
    checkpoint_metadata,
    load_checkpoint,
)


def load_inference_model(config: dict, section: str):
    settings = config[section]
    checkpoint = settings.get("checkpoint")
    if not checkpoint:
        raise ValueError(f"{section}.checkpoint is required")
    device = torch.device(settings.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or check the CUDA wheel/driver")
    model = build_model_from_config(config)
    if settings.get("quantized", False):
        from modern_nafnet.training.quantize import restore_quantized_model

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        check_checkpoint_metadata(model, payload, config)
        model = restore_quantized_model(model, payload, config=config)
        metadata = payload
    else:
        metadata = load_checkpoint(model, checkpoint, config=config)
    check_checkpoint_metadata(model, metadata, config)
    model.training_dataset_snapshot = metadata.get("training_state", {}).get("dataset")
    model.calibration_shape = metadata.get("training_state", {}).get("calibration_shape")
    return model.to(device).eval(), device


def restore_image(model, inputs: torch.Tensor, config: dict, *, with_aux: bool = False):
    """Apply the external R2R zero-padding/crop contract before a model call."""
    height, width = inputs.shape[-2:]
    if config.get("architecture", "nafnet") == "trt-r2r-light":
        inputs = torch.nn.functional.pad(inputs, (0, -width % 16, 0, -height % 16))
    if with_aux:
        restored, logits, attention = model.forward_with_aux(inputs)
        return restored[..., :height, :width], logits, attention
    return model(inputs)[..., :height, :width]


@torch.inference_mode()
def run_infer(config: dict) -> dict:
    settings = config["inference"]
    if not settings.get("input") or not settings.get("output"):
        raise ValueError("inference.input and inference.output are required")
    source, output = Path(settings["input"]), Path(settings["output"])
    checkpoint = Path(settings["checkpoint"]) if settings.get("checkpoint") else None
    outputs = {output.resolve(), output.with_suffix(output.suffix + ".json").resolve()}
    protected = {source.resolve(), checkpoint.resolve() if checkpoint else None}
    if outputs & protected:
        raise ValueError("Inference output must not overwrite the input or checkpoint")
    model, device = load_inference_model(config, "inference")
    image = read_rgb(source)
    inputs = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
    inputs = inputs.div_(255).unsqueeze(0).to(device)
    restored = restore_image(model, inputs, config)[0].permute(1, 2, 0).cpu().numpy()
    if restored.shape != image.shape:
        raise ValueError("Model output shape differs from the input image")
    write_rgb(output, restored)
    result = {
        **environment_manifest(),
        **checkpoint_metadata(model, config),
        "config": config,
        "checkpoint_sha256": sha256(settings["checkpoint"]),
        "input": str(source),
        "input_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "shape": list(inputs.shape),
        "preprocess": {
            "color": "RGB",
            "range": [0, 1],
            "pad_multiple": getattr(model, "padder_size", 16),
            "mode": "constant",
            "pad_value": 0,
            "crop_to_original_size": True,
        },
        "device": str(device),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    if config.get("architecture", "nafnet") == "trt-r2r-light":
        padded_shape = [
            *inputs.shape[:2],
            (inputs.shape[2] + 15) // 16 * 16,
            (inputs.shape[3] + 15) // 16 * 16,
        ]
        result["padded_shape"] = padded_shape
        result["calibration_shape"] = model.calibration_shape
        result["matches_calibration_shape"] = (
            padded_shape == model.calibration_shape if settings.get("quantized") else None
        )
    write_json(output.with_suffix(output.suffix + ".json"), result)
    return result
