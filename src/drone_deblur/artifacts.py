"""Hashes and execution records without importing ML packages."""

import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
from pathlib import Path

UPSTREAM_COMMIT = "2b4af71ebe098a92a75910c233a3965a3e93ede4"


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path: Path | str, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def environment_manifest() -> dict:
    packages = {}
    for name in (
        "drone-deblur",
        "numpy",
        "opencv-python-headless",
        "typer",
        "rich",
        "torch",
        "albumentationsx",
        "nvidia-modelopt",
        "onnx",
        "onnxscript",
        "torchmetrics",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                gpu = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "nvidia_smi": gpu,
    }
