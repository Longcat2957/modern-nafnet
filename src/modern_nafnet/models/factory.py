"""Explicit architecture selection and checkpoint identity checks.

Imports stay local so inspecting configuration and CLI help needs no Torch extra.
"""

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def build_model_from_config(config: Mapping[str, Any]):
    architecture = config.get("architecture", "nafnet")
    if architecture == "nafnet":
        from modern_nafnet.models.nafnet import build_model

        model = build_model(config["model"])
        model.architecture = "nafnet"
        model.model_spec = copy.deepcopy(dict(config["model"]))
        return model
    if architecture == "trt-r2r-light":
        from modern_nafnet.models.trt_r2r_light import TrtR2RLight

        return TrtR2RLight(**config["trt_r2r_light"])
    raise ValueError(f"Unsupported architecture: {architecture}")


def _normalized(value):
    if isinstance(value, Mapping):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    return value


def _bank_fingerprint(tasks, state_dict: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256(json.dumps(list(tasks), separators=(",", ":")).encode())
    for name in ("bank.key", "bank.value"):
        value = state_dict.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.numel() == 0:
            raise ValueError(f"Checkpoint {name} must be a nonempty matrix")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Bank {name} contains NaN or Inf")
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def bank_fingerprint(model) -> str | None:
    if getattr(model, "architecture", "nafnet") != "trt-r2r-light" or not model.bank_enabled:
        return None
    return _bank_fingerprint(model.tasks, model.state_dict())


def checkpoint_metadata(model, config: Mapping[str, Any]) -> dict[str, Any]:
    architecture = config.get("architecture", "nafnet")
    if architecture != getattr(model, "architecture", "nafnet"):
        raise ValueError("Model architecture differs from the requested config")
    spec = config["model" if architecture == "nafnet" else "trt_r2r_light"]
    if hasattr(model, "model_spec") and _normalized(spec) != _normalized(model.model_spec):
        raise ValueError("Model spec differs from the requested config")
    metadata = {
        "architecture": architecture,
        "task_order": list(model.tasks) if architecture == "trt-r2r-light" else [],
        "model_spec": copy.deepcopy(_normalized(spec)),
        "bank_fingerprint": bank_fingerprint(model),
    }
    if architecture == "nafnet":
        from modern_nafnet.models.nafnet import tlc_kernels

        metadata["tlc_kernels"] = tlc_kernels(model)
    return metadata


def check_checkpoint_metadata(model, checkpoint: Mapping[str, Any], config: Mapping[str, Any]):
    """Validate identity before mutation, including the serialized bank contents."""
    architecture = config.get("architecture", "nafnet")
    if architecture != getattr(model, "architecture", "nafnet"):
        raise ValueError("Model architecture differs from the requested config")
    saved_architecture = checkpoint.get("architecture", "nafnet")
    if architecture != saved_architecture:
        raise ValueError("Checkpoint architecture differs from the requested architecture")
    saved_config = checkpoint.get("config")
    if (
        isinstance(saved_config, Mapping)
        and saved_config.get("architecture", "nafnet") != architecture
    ):
        raise ValueError("Checkpoint config architecture differs from the requested architecture")
    if architecture == "nafnet":
        from modern_nafnet.models.nafnet import check_checkpoint_metadata as check_nafnet

        if checkpoint.get("task_order", []) != []:
            raise ValueError("NAFNet checkpoint task order must be empty")
        if checkpoint.get("bank_fingerprint") is not None:
            raise ValueError("NAFNet checkpoint must not contain a bank fingerprint")
        if "model_spec" in checkpoint and _normalized(checkpoint["model_spec"]) != _normalized(
            config["model"]
        ):
            raise ValueError("Checkpoint model spec differs from the requested config")
        check_nafnet(model, checkpoint, config["model"])
        return
    if architecture != "trt-r2r-light":
        raise ValueError(f"Unsupported architecture: {architecture}")
    spec = config["trt_r2r_light"]
    if _normalized(spec) != _normalized(model.model_spec):
        raise ValueError("Model spec differs from the requested config")
    for field in ("task_order", "model_spec", "bank_fingerprint"):
        if field not in checkpoint:
            raise ValueError(f"TRT-R2R-Light checkpoint is missing {field}")
    if _normalized(checkpoint["task_order"]) != list(model.tasks):
        raise ValueError("Checkpoint task order differs from the requested task order")
    if _normalized(checkpoint["model_spec"]) != _normalized(spec):
        raise ValueError("Checkpoint model spec differs from the requested config")
    if saved_config is not None:
        if not isinstance(saved_config, Mapping):
            raise ValueError("Checkpoint config must be a mapping")
        if saved_config.get("architecture") != architecture or _normalized(
            saved_config.get("trt_r2r_light")
        ) != _normalized(spec):
            raise ValueError("Checkpoint model config differs from the requested config")
    weights = checkpoint.get("state_dict")
    if weights is None:
        weights = model.state_dict()
    elif not isinstance(weights, Mapping):
        raise ValueError("TRT-R2R-Light checkpoint state_dict must be a mapping")
    expected = _bank_fingerprint(model.tasks, weights) if model.bank_enabled else None
    if checkpoint["bank_fingerprint"] != expected:
        raise ValueError("Checkpoint bank fingerprint does not match its stored bank")
    if model.bank_enabled:
        key, value = weights["bank.key"], weights["bank.value"]
        if key.shape != model.bank.key.shape or value.shape != model.bank.value.shape:
            raise ValueError("Checkpoint bank shape differs from the requested bank")


def load_checkpoint(model, path: str | Path, config: Mapping[str, Any] | None = None) -> dict:
    """Strictly restore the selected architecture; never reinterpret R2R as NAFNet."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping")
    if "modelopt_state" in checkpoint:
        raise ValueError(
            "Restore ModelOpt checkpoints with the QAT workflow before loading weights"
        )
    architecture = getattr(model, "architecture", "nafnet")
    if config is None:
        if not hasattr(model, "model_spec"):
            if checkpoint.get("architecture", "nafnet") != architecture:
                raise ValueError("Checkpoint architecture differs from the requested architecture")
            from modern_nafnet.models.nafnet import load_checkpoint as load_nafnet

            return load_nafnet(model, path)
        config = {
            "architecture": architecture,
            "model" if architecture == "nafnet" else "trt_r2r_light": model.model_spec,
        }
    check_checkpoint_metadata(model, checkpoint, config)
    if architecture == "nafnet":
        from modern_nafnet.models.nafnet import load_checkpoint as load_nafnet

        return load_nafnet(model, path)
    if "params" in checkpoint:
        raise ValueError("TRT-R2R-Light checkpoint must use state_dict, not params")
    if not isinstance(checkpoint.get("state_dict"), Mapping):
        raise ValueError("TRT-R2R-Light checkpoint must contain state_dict")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.validate_bank()
    return {key: value for key, value in checkpoint.items() if key != "state_dict"}
