"""JSON configuration; paths are relative to the working directory."""

import copy
import json
import math
from pathlib import Path

DEFAULT_CONFIG = {
    "model": {
        "img_channel": 3,
        "width": 32,
        "enc_blk_nums": [1, 1, 1, 28],
        "middle_blk_num": 1,
        "dec_blk_nums": [1, 1, 1, 1],
        "train_size": [1, 3, 256, 256],
        "fast_imp": False,
    },
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
    if name == "data.synthetic_blur.blur_range":
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
    return config
