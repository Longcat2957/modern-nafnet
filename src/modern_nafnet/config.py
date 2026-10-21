"""JSON configuration; paths are relative to the working directory."""

import copy
import json
import math
from pathlib import Path

DEFAULT_CONFIG = {
    "architecture": "nafnet",
    "model": {
        "img_channel": 3,
        "width": 32,
        "enc_blk_nums": [1, 1, 1, 28],
        "middle_blk_num": 1,
        "dec_blk_nums": [1, 1, 1, 1],
        "train_size": [1, 3, 256, 256],
        "fast_imp": False,
    },
    "trt_r2r_light": {
        "tasks": ["denoise", "derain", "dehaze", "deblur", "lowlight"],
        "slots_per_task": 8,
        "query_dim": 64,
        "channels": [32, 64, 128, 192],
        "encoder_blocks": [1, 1, 2, 2],
        "decoder_blocks": [1, 1, 1],
        "bank_enabled": True,
        "prior_scale_init": 0.1,
    },
    "restoration_loss": {
        "charbonnier_eps": 0.001,
        "edge_weight": 0.05,
        "multiscale_weight": 0.1,
        "task_weight": 0.1,
        "clean_weight": 0.5,
    },
    "task_sampling": {"mode": "balanced"},
    "data": {
        "path": None,
        "train": "train",
        "val": "val",
        "test": "test",
        "calibration": "calibration",
        "crop_size": 256,
        "synthetic_blur": {"enabled": False, "blur_range": [3, 15]},
    },
    "training": {
        "checkpoint": None,
        "resume": None,
        "output_dir": "artifacts/train",
        "steps": 1000,
        "batch_size": 1,
        "lr": 0.0001,
        "seed": 42,
        "num_workers": 0,
        "device": "cuda",
        "amp": False,
    },
    "quantization": {"checkpoint": None, "calibration_batches": 32},
    "export": {
        "checkpoint": None,
        "output": "artifacts/model.onnx",
        "shape": None,
        "opset": 18,
        "quantized": False,
        "device": "cuda",
    },
    "inference": {
        "checkpoint": None,
        "input": None,
        "output": "artifacts/restored.png",
        "device": "cuda",
        "quantized": False,
    },
    "evaluation": {
        "checkpoint": None,
        "split": "val",
        "output": "artifacts/evaluation.json",
        "device": "cuda",
        "quantized": False,
    },
    "benchmark": {"input_size": None, "target_fps": None, "power_mode": None},
}


def _validate_value(name: str, value) -> None:
    default = DEFAULT_CONFIG
    for part in name.split("."):
        default = default[part]
    key = name.rsplit(".", 1)[-1]
    if name == "architecture":
        if value not in ("nafnet", "trt-r2r-light"):
            raise ValueError("Configuration architecture must be nafnet or trt-r2r-light")
    elif name == "task_sampling.mode":
        if value not in ("balanced", "proportional"):
            raise ValueError("Configuration task_sampling.mode must be balanced or proportional")
    elif name == "trt_r2r_light.tasks":
        if (
            not isinstance(value, list)
            or not value
            or any(
                not isinstance(task, str)
                or task not in ("denoise", "derain", "dehaze", "deblur", "lowlight")
                for task in value
            )
            or len(set(value)) != len(value)
        ):
            raise ValueError("Configuration trt_r2r_light.tasks needs unique supported task names")
    elif name.startswith("restoration_loss.") or name == "trt_r2r_light.prior_scale_init":
        positive = name == "restoration_loss.charbonnier_eps"
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (positive and value == 0)
        ):
            raise ValueError(
                f"Configuration {name} must be finite and {'positive' if positive else 'nonnegative'}"
            )
    elif name == "data.synthetic_blur.blur_range":
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(type(n) is not int or n < 3 or n % 2 == 0 for n in value)
            or value[0] > value[1]
        ):
            raise ValueError(f"Configuration {name} needs two ordered odd integers >= 3")
    elif name == "evaluation.split":
        if value not in ("val", "test"):
            raise ValueError("Configuration evaluation.split must be val or test")
    elif key in {
        "checkpoint",
        "resume",
        "output_dir",
        "input",
        "output",
        "path",
        "train",
        "val",
        "test",
        "calibration",
    }:
        if value is None and default is None:
            return
        if not isinstance(value, (str, Path)) or not str(value):
            raise ValueError(f"Configuration {name} must be a nonempty path")
    elif default is None:
        if value is None:
            return
        if key in {"shape", "input_size"}:
            length = 4 if key == "shape" else 2
            if (
                not isinstance(value, list)
                or len(value) != length
                or any(type(n) is not int or n <= 0 for n in value)
            ):
                raise ValueError(f"Configuration {name} needs {length} positive integer dimensions")
        elif key == "target_fps":
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Configuration {name} must be a positive finite number or null")
        elif key == "power_mode" and type(value) not in (str, int):
            raise ValueError(f"Configuration {name} must be a power mode name/id or null")
    elif isinstance(default, bool):
        if type(value) is not bool:
            raise ValueError(f"Configuration {name} must be a boolean")
    elif isinstance(default, int):
        positive = key in {
            "width",
            "img_channel",
            "crop_size",
            "batch_size",
            "opset",
            "calibration_batches",
            "slots_per_task",
            "query_dim",
        }
        if type(value) is not int or value < (1 if positive else 0):
            raise ValueError(
                f"Configuration {name} must be a {'positive' if positive else 'nonnegative'} integer"
            )
    elif isinstance(default, float):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Configuration {name} must be a positive finite number")
    elif isinstance(default, list):
        if (
            not isinstance(value, list)
            or not value
            or any(type(n) is not int or n < 0 for n in value)
        ):
            raise ValueError(f"Configuration {name} must be a nonempty integer list")
        if key == "train_size" and (len(value) != 4 or min(value) < 1):
            raise ValueError(
                "Configuration model.train_size needs four positive integer dimensions"
            )
    elif isinstance(default, str) and (not isinstance(value, str) or not value):
        raise ValueError(f"Configuration {name} must be a nonempty string")


def _merge(target: dict, source: dict, *, cli: bool = False, prefix: str = "") -> None:
    for key, value in source.items():
        name = f"{prefix}{key}"
        if key not in target:
            raise ValueError(f"Unknown configuration key: {name}")
        if cli and value is None:
            continue
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Configuration {name} must be an object")
            _merge(target[key], value, cli=cli, prefix=f"{name}.")
        else:
            _validate_value(name, value)
            target[key] = value


def load_config(path: Path | None = None, overrides: dict | None = None) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            raise ValueError("Configuration root must be a JSON object")
        _merge(config, document)
    if overrides:
        _merge(config, overrides, cli=True)
    validate_trt_r2r_light_config(config["trt_r2r_light"])
    if config["architecture"] == "trt-r2r-light":
        if config["data"]["crop_size"] % 16:
            raise ValueError(
                "Configuration data.crop_size must be a multiple of 16 for trt-r2r-light"
            )
        shape = config["export"]["shape"]
        if shape is not None and (shape[:2] != [1, 3] or any(size % 16 for size in shape[-2:])):
            raise ValueError(
                "Configuration export.shape must be [1,3,H,W] with H/W multiples of 16"
            )
        if config["data"]["synthetic_blur"]["enabled"]:
            raise ValueError(
                "Configuration synthetic_blur is only supported for NAFNet paired data"
            )
    return config


def validate_trt_r2r_light_config(options: dict) -> None:
    """Validate fixed graph sizes without importing the ML dependencies."""
    for key in ("tasks", "slots_per_task", "query_dim", "bank_enabled", "prior_scale_init"):
        value = options[key]
        if key == "tasks" and isinstance(value, tuple):
            value = list(value)
        _validate_value(f"trt_r2r_light.{key}", value)
    channels = options["channels"]
    if (
        not isinstance(channels, (list, tuple))
        or len(channels) != 4
        or any(type(c) is not int or c <= 0 or c % 32 for c in channels)
    ):
        raise ValueError("Configuration trt_r2r_light.channels needs four positive multiples of 32")
    for name, length in (("encoder_blocks", 4), ("decoder_blocks", 3)):
        blocks = options[name]
        if (
            not isinstance(blocks, (list, tuple))
            or len(blocks) != length
            or any(type(count) is not int or count < 0 for count in blocks)
        ):
            raise ValueError(
                f"Configuration trt_r2r_light.{name} needs {length} nonnegative counts"
            )
