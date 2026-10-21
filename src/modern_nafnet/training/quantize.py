"""ModelOpt static symmetric INT8 calibration and state restoration."""

import sys
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

R2R_PRECISION_POLICY = "w8a8-conv-fp32-branch"


def _architecture(model: torch.nn.Module | None, architecture: str | None = None) -> str:
    if architecture is not None and hasattr(model, "architecture"):
        if architecture != model.architecture:
            raise ValueError("Quantization architecture differs from the model")
    architecture = architecture or getattr(model, "architecture", "nafnet")
    if architecture not in {"nafnet", "trt-r2r-light"}:
        raise ValueError(f"Unsupported quantization architecture: {architecture}")
    return architecture


def _check_precision_policy(architecture: str, precision_policy: str | None) -> None:
    if precision_policy not in {None, R2R_PRECISION_POLICY}:
        raise ValueError(f"Unsupported {architecture} precision policy: {precision_policy}")


def _fp_module(name: str, architecture: str) -> bool:
    if architecture == "trt-r2r-light":
        return name.split(".")[0] in {"stem", "tail", "query_head", "bank"}
    return ".sca." in f".{name}."


def int8_config(architecture: str = "nafnet", precision_policy: str | None = None) -> dict:
    import modelopt.torch.quantization as mtq

    architecture = _architecture(None, architecture)
    _check_precision_policy(architecture, precision_policy)
    config = deepcopy(mtq.INT8_DEFAULT_CFG)
    if architecture == "trt-r2r-light":
        # Only Conv input/weight quantizers belong to Stage A; Linear/Bank stay FP32.
        config["quant_cfg"].append({"quantizer_name": "*", "enable": False})
    config["quant_cfg"].extend(
        [
            {
                "quantizer_name": "*weight_quantizer",
                "parent_class": "nn.Conv2d",
                "enable": True,
                "cfg": {"num_bits": 8, "axis": 0, "unsigned": False, "type": "static"},
            },
            {
                "quantizer_name": "*input_quantizer",
                "parent_class": "nn.Conv2d",
                "enable": True,
                "cfg": {"num_bits": 8, "axis": None, "unsigned": False, "type": "static"},
            },
            {"quantizer_name": "*.sca.*", "enable": False},
            {"quantizer_name": "sca.*", "enable": False},
        ]
    )
    if architecture == "trt-r2r-light":
        config["quant_cfg"].extend(
            {"quantizer_name": f"{name}.*", "enable": False}
            for name in ("stem", "tail", "query_head", "bank")
        )
    return config


def quantizer_report(model: torch.nn.Module) -> list[dict]:
    from modelopt.torch.quantization.nn import TensorQuantizer

    architecture = _architecture(model)
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
                    "amax": amax.detach().cpu().tolist() if amax is not None else None,
                    "amax_finite": amax is not None and bool(torch.isfinite(amax).all()),
                    "precision": "w8a8" if module.is_enabled else "fp32",
                    "fp_excluded": _fp_module(name, architecture),
                }
            )
    return report


def check_int8_convs(model: torch.nn.Module, policy: str | None = None) -> list[dict]:
    """Check the inserted modules, rather than counting ONNX Q/DQ nodes."""
    from modelopt.torch.quantization.nn import TensorQuantizer

    architecture = _architecture(model)
    _check_precision_policy(architecture, policy)
    if architecture == "trt-r2r-light" and any(
        parameter.dtype != torch.float32 for parameter in model.parameters()
    ):
        raise RuntimeError("TRT-R2R-Light Stage A model parameters must be FP32")
    convs = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        excluded = _fp_module(name, architecture)
        for kind, expected_axis in (("input", None), ("weight", 0)):
            quantizer = getattr(module, f"{kind}_quantizer", None)
            if excluded:
                if isinstance(quantizer, TensorQuantizer) and quantizer.is_enabled:
                    raise RuntimeError(f"FP32 quantizer must stay disabled: {name}.{kind}")
                continue
            if not isinstance(quantizer, TensorQuantizer):
                raise RuntimeError(f"Missing {kind} quantizer on Conv2d {name}")
            if (
                not quantizer.is_enabled
                or quantizer.num_bits != 8
                or quantizer.axis != expected_axis
                or quantizer.unsigned
            ):
                raise RuntimeError(f"Conv2d {name}.{kind}_quantizer is not the W8A8 policy")
            if quantizer.amax is None or not bool(torch.isfinite(quantizer.amax).all()):
                raise RuntimeError(f"Uncalibrated quantizer: {name}.{kind}_quantizer")
            expected_shape = [] if kind == "input" else [module.out_channels, 1, 1, 1]
            if list(quantizer.amax.shape) != expected_shape:
                raise RuntimeError(
                    f"Invalid amax shape on {name}.{kind}_quantizer: "
                    f"{list(quantizer.amax.shape)}, expected {expected_shape}"
                )
        if not excluded:
            convs += 1
    if architecture == "trt-r2r-light":
        for name, module in model.named_modules():
            if not isinstance(module, TensorQuantizer) or not module.is_enabled:
                continue
            parent, _, kind = name.rpartition(".")
            owner = model.get_submodule(parent)
            if (
                _fp_module(name, architecture)
                or not isinstance(owner, torch.nn.Conv2d)
                or kind not in {"input_quantizer", "weight_quantizer"}
            ):
                raise RuntimeError(f"Unexpected Stage A quantizer: {name}")
        model.validate_bank()
        if model.bank is not None and (
            model.bank.key.requires_grad or model.bank.value.requires_grad
        ):
            raise RuntimeError("The degradation Bank must be frozen before PTQ/QAT")
    if not convs:
        raise RuntimeError("No W8A8 Conv2d modules were found")
    return quantizer_report(model)


def calibrate_int8(
    model: torch.nn.Module,
    inputs,
    *,
    architecture: str | None = None,
    precision_policy: str | None = None,
) -> torch.nn.Module:
    """Consume deterministic NCHW calibration inputs, then freeze their scales."""
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn import TensorQuantizer

    architecture = _architecture(model, architecture)
    _check_precision_policy(architecture, precision_policy)
    if architecture == "trt-r2r-light":
        model.validate_bank()
        model.freeze_bank()
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("TRT-R2R-Light Stage A calibration requires FP32 parameters")
    batches = 0

    def forward_loop(converted):
        nonlocal batches
        with torch.no_grad():
            for image in inputs:
                if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
                    raise ValueError("Calibration inputs must be batch-one RGB NCHW tensors")
                if architecture == "trt-r2r-light" and image.dtype != torch.float32:
                    raise ValueError("TRT-R2R-Light Stage A calibration requires FP32 inputs")
                if not bool(torch.isfinite(image).all()) or image.min() < 0 or image.max() > 1:
                    raise ValueError("Calibration input must be finite and in [0, 1]")
                converted(image)
                batches += 1
        if not batches:
            raise ValueError("Calibration requires at least one image")

    model.eval()
    # ModelOpt status messages must not corrupt the CLI's JSON stdout.
    with redirect_stdout(sys.stderr):
        model = mtq.quantize(
            model, int8_config(architecture, precision_policy), forward_loop=forward_loop
        )
    for module in model.modules():
        if isinstance(module, TensorQuantizer):
            module.disable_calib()
    check_int8_convs(model)
    return model


def calibration_inputs(
    source: str | Path,
    shape: list[int],
    device: str,
    limit: int,
    *,
    samples: list[dict] | None = None,
    architecture: str = "nafnet",
):
    """Read real calibration inputs; R2R uses the same external zero-pad as inference."""
    from modern_nafnet.data.images import read_rgb
    from modern_nafnet.data.sources import load_samples

    if len(shape) != 4 or shape[:2] != [1, 3] or any(size <= 0 for size in shape):
        raise ValueError("Calibration shape must be [1, 3, height, width]")
    if limit < 1:
        raise ValueError("calibration_batches must be positive")
    _architecture(None, architecture)
    if architecture == "trt-r2r-light" and (shape[2] % 16 or shape[3] % 16):
        raise ValueError("TRT-R2R-Light calibration shape must be padded to a multiple of 16")
    if samples is None:
        if architecture == "trt-r2r-light":
            raise ValueError("TRT-R2R-Light calibration requires balanced restoration samples")
        samples = load_samples(source, blur_only=True)
    for index, sample in enumerate(samples[:limit]):
        image = read_rgb(sample["input"] if architecture == "trt-r2r-light" else sample["blur"])
        if architecture == "trt-r2r-light":
            height, width = image.shape[:2]
            if [(height + 15) // 16 * 16, (width + 15) // 16 * 16] == shape[2:]:
                image = np.pad(image, ((0, shape[2] - height), (0, shape[3] - width), (0, 0)))
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


def restore_quantized_model(
    model: torch.nn.Module, checkpoint: dict, config: dict | None = None
) -> torch.nn.Module:
    import modelopt.torch.opt as mto

    if not checkpoint.get("modelopt_state"):
        raise ValueError("Quantized checkpoint is missing modelopt_state")
    architecture = _architecture(model)
    if architecture == "trt-r2r-light":
        from modern_nafnet.models.factory import check_checkpoint_metadata

        check_checkpoint_metadata(model, checkpoint, config or checkpoint.get("config", {}))
    with redirect_stdout(sys.stderr):
        model = mto.restore_from_modelopt_state(model, checkpoint["modelopt_state"])
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
    if state is None:
        raise ValueError("Quantized checkpoint is missing state_dict")
    model.load_state_dict(state, strict=True)
    if architecture == "trt-r2r-light":
        from modern_nafnet.models.factory import bank_fingerprint

        model.validate_bank()
        model.freeze_bank()
        if bank_fingerprint(model) != checkpoint["bank_fingerprint"]:
            raise ValueError("Restored Bank fingerprint differs from the checkpoint")
    check_int8_convs(model)
    return model
