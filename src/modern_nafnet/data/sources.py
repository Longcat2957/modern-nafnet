"""Resolve explicit data sources and index image paths without decoding images."""

import hashlib
import json
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm"}
ROLES = {"blur", "sharp"}
RESTORATION_TASKS = {"denoise", "derain", "dehaze", "deblur", "lowlight"}


def task_label(task: str, tasks: list[str] | tuple[str, ...]) -> list[float]:
    """Encode a folder name against the checkpoint's fixed degradation order."""
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - RESTORATION_TASKS:
        raise ValueError("tasks must be unique supported degradation names")
    parts = task.split("+")
    if task == "clean":
        return [0.0] * len(tasks)
    if len(set(parts)) != len(parts) or set(parts) - set(tasks):
        raise ValueError(f"Unknown or duplicate restoration task: {task}")
    return [float(name in parts) for name in tasks]


def load_restoration_samples(
    source: str | Path, tasks: list[str] | tuple[str, ...], *, calibration: bool = False
) -> list[dict[str, Any]]:
    """Index task/input-target folders once; clean targets are also their inputs."""
    source = Path(source).resolve()
    if not source.is_dir():
        if not source.exists():
            raise FileNotFoundError(f"Data source does not exist: {source}")
        raise ValueError(f"Data source must be a directory: {source}")
    task_label("clean", tasks)
    labels, seen_labels = {}, set()
    for folder in sorted(source.iterdir()):
        if not folder.is_dir():
            continue
        label = task_label(folder.name, tasks)
        key = tuple(label)
        if key in seen_labels:
            raise ValueError(f"Duplicate restoration task combination: {folder.name}")
        seen_labels.add(key)
        labels[folder.name] = label
    indexed = {name: {"input": {}, "target": {}} for name in labels}
    used = {"input": set(), "target": set()}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(source)
        parts = relative.parts
        if len(parts) < 3 or parts[0] not in labels or parts[1] not in used:
            raise ValueError(f"Unclassified image in restoration directory: {relative}")
        task, role = parts[:2]
        if calibration and role == "target":
            continue
        if any(part in used for part in parts[2:-1]):
            raise ValueError(f"Ambiguous restoration image: {relative}")
        if not calibration and task == "clean" and role != "target":
            raise ValueError("Clean supervised data requires clean/target only; input=target")
        name = Path(*parts[2:]).as_posix()
        resolved = path.resolve()
        if name in indexed[task][role] or resolved in used[role]:
            raise ValueError(f"Duplicate {role} restoration image: {relative}")
        indexed[task][role][name] = resolved
        used[role].add(resolved)
    samples = []
    for task, roles in indexed.items():
        inputs, targets = roles["input"], roles["target"]
        if not calibration and task == "clean":
            inputs = targets
        if calibration:
            if not inputs:
                raise ValueError(f"Calibration task contains no input images: {task}")
        else:
            unmatched = set(inputs) ^ set(targets)
            if unmatched:
                name = min(unmatched)
                missing = "target" if name in inputs else "input"
                raise ValueError(f"Missing {missing} pair for {task}/{name}")
            if not inputs:
                raise ValueError(f"Restoration task contains no input/target pairs: {task}")
        for name, path in sorted(inputs.items()):
            sample = {
                "name": f"{task}/{name}",
                "task": task,
                "task_label": labels[task],
                "input": path,
            }
            if not calibration:
                sample["target"] = targets[name]
            samples.append(sample)
    if not samples:
        raise ValueError(f"Data directory contains no supported restoration images: {source}")
    return samples


def balanced_calibration_samples(
    samples: list[dict[str, Any]], tasks: list[str] | tuple[str, ...], limit: int
) -> list[dict[str, Any]]:
    """Select real inputs round-robin, retaining clean and every present mixture."""
    if type(limit) is not int or limit < 1:
        raise ValueError("Calibration sample limit must be a positive integer")
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        expected = task_label(sample["task"], tasks)
        if sample["task_label"] != expected:
            raise ValueError("Calibration task labels do not match the configured task order")
        groups.setdefault(sample["task"], []).append(sample)
    missing = (set(tasks) | {"clean"}) - set(groups)
    if missing:
        raise ValueError(
            f"Calibration must cover every configured task and clean; missing "
            f"single-task/clean folders {sorted(missing)}"
        )
    if limit < len(groups):
        raise ValueError("Calibration sample limit must cover every task/clean/composite group")
    groups = {
        key: sorted(value, key=lambda sample: sample["name"])
        for key, value in sorted(groups.items())
    }
    selected = []
    for offset in range(max(map(len, groups.values()))):
        for group in groups.values():
            if offset < len(group):
                selected.append(group[offset])
                if len(selected) == limit:
                    return selected
    return selected


def resolve_data_source(data: dict, split: str) -> Path:
    """data.path is relative to CWD; split directories are relative to that root."""
    root = data.get("path")
    if root is None:
        raise ValueError("Set data.path to the dataset root directory")
    return (Path(root) / data.get(split, split)).resolve()


def _scan_directory(
    source: Path, *, allow_sharp_only: bool, blur_only: bool
) -> list[dict[str, Any]]:
    paths = sorted(source.rglob("*"))
    roles = {path.name for path in paths if path.name in ROLES and path.is_dir()}
    images = [path for path in paths if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()]
    if not images:
        raise ValueError(f"Data directory contains no supported images: {source}")
    if not roles:
        if not (blur_only or allow_sharp_only):
            raise ValueError(f"Paired directory needs blur and sharp subdirectories: {source}")
        if not blur_only and source.name == "blur":
            raise ValueError(f"Sharp-only data cannot use a blur directory: {source}")
        role = "blur" if blur_only else "sharp"
        samples, used = [], set()
        for path in images:
            resolved = path.resolve()
            if resolved in used:
                raise ValueError(f"Duplicate {role} image: {path.relative_to(source)}")
            used.add(resolved)
            samples.append({"name": path.relative_to(source).as_posix(), role: resolved})
        return samples

    by_role: dict[str, dict[str, Path]] = {"blur": {}, "sharp": {}}
    used_paths: dict[str, set[Path]] = {"blur": set(), "sharp": set()}
    for path in images:
        relative = path.relative_to(source)
        positions = [index for index, part in enumerate(relative.parts[:-1]) if part in ROLES]
        if blur_only and positions and relative.parts[positions[0]] == "sharp":
            continue
        if len(positions) != 1:
            raise ValueError(f"Unclassified or ambiguous image in paired directory: {relative}")
        index = positions[0]
        role = relative.parts[index]
        name = Path(*relative.parts[:index], *relative.parts[index + 1 :]).as_posix()
        resolved = path.resolve()
        if name in by_role[role] or resolved in used_paths[role]:
            raise ValueError(f"Duplicate {role} image: {relative}")
        by_role[role][name] = resolved
        used_paths[role].add(resolved)

    if blur_only:
        if not by_role["blur"]:
            raise ValueError(f"Calibration directory contains no blur images: {source}")
        return [{"name": name, "blur": path} for name, path in sorted(by_role["blur"].items())]
    if allow_sharp_only and "blur" not in roles:
        return [{"name": name, "sharp": path} for name, path in sorted(by_role["sharp"].items())]
    blur_names, sharp_names = set(by_role["blur"]), set(by_role["sharp"])
    unmatched = blur_names ^ sharp_names
    if unmatched:
        name = min(unmatched)
        missing = "sharp" if name in blur_names else "blur"
        raise ValueError(f"Missing {missing} pair for {name}; synthetic fallback is not allowed")
    if not blur_names:
        raise ValueError(f"Paired directory contains no blur/sharp pairs: {source}")
    return [
        {"name": name, "blur": by_role["blur"][name], "sharp": by_role["sharp"][name]}
        for name in sorted(blur_names)
    ]


def load_samples(
    source: str | Path, *, allow_sharp_only: bool = False, blur_only: bool = False
) -> list[dict[str, Any]]:
    """Index a flat/GoPro directory, matching exact relative filenames and extensions."""
    source = Path(source).resolve()
    if not source.is_dir():
        if not source.exists():
            raise FileNotFoundError(f"Data source does not exist: {source}")
        raise ValueError(f"Data source must be a directory: {source}")
    if blur_only and source.name == "sharp":
        raise ValueError(f"Calibration requires real blur images, not a sharp directory: {source}")
    return _scan_directory(source, allow_sharp_only=allow_sharp_only, blur_only=blur_only)


def source_record(source: str | Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Record the selected paths and their size/mtime identity without reading image contents."""
    source = Path(source).resolve()
    if not source.is_dir():
        if not source.exists():
            raise FileNotFoundError(f"Data source does not exist: {source}")
        raise ValueError(f"Data source must be a directory: {source}")
    plain = [
        {
            "name": sample["name"],
            **{
                key: str(sample[key])
                for key in ("blur", "sharp", "input", "target")
                if key in sample
            },
            **{key: sample[key] for key in ("task", "task_label") if key in sample},
        }
        for sample in samples
    ]
    entries = []
    for sample in sorted(plain, key=lambda item: item["name"]):
        entry: dict[str, Any] = {"name": sample["name"]}
        for key in ("task", "task_label"):
            if key in sample:
                entry[key] = sample[key]
        for key in ("blur", "sharp", "input", "target"):
            if key in sample:
                path = Path(sample[key])
                info = path.stat()
                entry[key] = {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}
        entries.append(entry)
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"source": str(source), "sha256": digest, "samples": plain}
