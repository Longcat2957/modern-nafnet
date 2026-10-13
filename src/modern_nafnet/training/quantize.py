"""ModelOpt static symmetric INT8 calibration and state restoration."""

import json
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


def int8_config() -> dict:
    import modelopt.torch.quantization as mtq

    config = deepcopy(mtq.INT8_DEFAULT_CFG)
    config["quant_cfg"].extend(
        [
            {
                "quantizer_name": "*weight_quantizer",
                "parent_class": "nn.Conv2d",
                "cfg": {"num_bits": 8, "axis": 0, "unsigned": False, "type": "static"},
            },
            {
                "quantizer_name": "*input_quantizer",
                "parent_class": "nn.Conv2d",
                "cfg": {"num_bits": 8, "axis": None, "unsigned": False, "type": "static"},
            },
            {"quantizer_name": "*.sca.*", "enable": False},
            {"quantizer_name": "sca.*", "enable": False},
        ]
    )
    return config


def quantizer_report(model: torch.nn.Module) -> list[dict]:
    from modelopt.torch.quantization.nn import TensorQuantizer

    report = []
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            amax = module.amax
            report.append(
                {
                    "name": name,
                    "enabled": module.is_enabled,
                    "num_bits": module.num_bits,
                    "axis": module.axis,
                    "unsigned": module.unsigned,
                    "calibrated": amax is not None and bool(torch.isfinite(amax).all()),
                    "amax_shape": list(amax.shape) if amax is not None else None,
                }
            )
    return report


def check_int8_convs(model: torch.nn.Module) -> list[dict]:
    """Check the inserted modules, rather than counting ONNX Q/DQ nodes."""
    from modelopt.torch.quantization.nn import TensorQuantizer

    convs = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        attention = ".sca." in f".{name}."
        for kind, expected_axis in (("input", None), ("weight", 0)):
            quantizer = getattr(module, f"{kind}_quantizer", None)
            if not isinstance(quantizer, TensorQuantizer):
                raise RuntimeError(f"Missing {kind} quantizer on Conv2d {name}")
            if attention:
                if quantizer.is_enabled:
                    raise RuntimeError(f"Attention quantizer must stay disabled: {name}")
                continue
            if (
                not quantizer.is_enabled
                or quantizer.num_bits != 8
                or quantizer.axis != expected_axis
                or quantizer.unsigned
            ):
                raise RuntimeError(f"Conv2d {name}.{kind}_quantizer is not the W8A8 policy")
            if quantizer.amax is None or not bool(torch.isfinite(quantizer.amax).all()):
                raise RuntimeError(f"Uncalibrated quantizer: {name}.{kind}_quantizer")
        if not attention:
            convs += 1
    if not convs:
        raise RuntimeError("No W8A8 Conv2d modules were found")
    return quantizer_report(model)


def calibrate_int8(model: torch.nn.Module, inputs) -> torch.nn.Module:
    """Consume deterministic NCHW calibration inputs, then freeze their scales."""
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn import TensorQuantizer

    batches = 0

    def forward_loop(converted):
        nonlocal batches
        with torch.no_grad():
            for image in inputs:
                if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
                    raise ValueError("Calibration inputs must be batch-one RGB NCHW tensors")
                if not bool(torch.isfinite(image).all()) or image.min() < 0 or image.max() > 1:
                    raise ValueError("Calibration input must be finite and in [0, 1]")
                converted(image)
                batches += 1
        if not batches:
            raise ValueError("Calibration requires at least one image")

    model.eval()
    # ModelOpt status messages must not corrupt the CLI's JSON stdout.
    with redirect_stdout(sys.stderr):
        model = mtq.quantize(model, int8_config(), forward_loop=forward_loop)
    for module in model.modules():
        if isinstance(module, TensorQuantizer):
            module.disable_calib()
    check_int8_convs(model)
    return model


def calibration_inputs(manifest: str | Path, shape: list[int], device: str, limit: int):
    """Read blur-only or paired manifests without changing their image geometry."""
    from modern_nafnet.data.images import read_rgb

    if len(shape) != 4 or shape[:2] != [1, 3] or any(size <= 0 for size in shape):
        raise ValueError("Calibration shape must be [1, 3, height, width]")
    if limit < 1:
        raise ValueError("calibration_batches must be positive")
    manifest = Path(manifest).resolve()
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Calibration manifest must contain a nonempty JSON array")
    for index, entry in enumerate(entries[:limit]):
        if not isinstance(entry, dict) or not isinstance(entry.get("blur"), str):
            raise ValueError(f"Calibration entry {index} needs a blur path")
        image = read_rgb((manifest.parent / entry["blur"]).resolve())
        if list(image.shape[:2]) != shape[2:]:
            raise ValueError(
                f"Calibration image {index} has shape {image.shape[:2]}, expected {shape[2:]}; "
                "prepare deployment-sized images explicitly"
            )
        array = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
        yield torch.from_numpy(array).unsqueeze(0).div_(255).to(device)


def modelopt_state(model: torch.nn.Module) -> dict:
    import modelopt.torch.opt as mto

    return mto.modelopt_state(model)


def restore_quantized_model(model: torch.nn.Module, checkpoint: dict) -> torch.nn.Module:
    import modelopt.torch.opt as mto

    if not checkpoint.get("modelopt_state"):
        raise ValueError("Quantized checkpoint is missing modelopt_state")
    with redirect_stdout(sys.stderr):
        model = mto.restore_from_modelopt_state(model, checkpoint["modelopt_state"])
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
    if state is None:
        raise ValueError("Quantized checkpoint is missing state_dict")
    model.load_state_dict(state, strict=True)
    check_int8_convs(model)
    return model
