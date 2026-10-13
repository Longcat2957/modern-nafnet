"""Resolve explicit data sources and index image paths without decoding images."""

import hashlib
import json
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm"}
ROLES = {"blur", "sharp"}


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
            **{key: str(sample[key]) for key in ("blur", "sharp") if key in sample},
        }
        for sample in samples
    ]
    entries = []
    for sample in sorted(plain, key=lambda item: item["name"]):
        entry: dict[str, Any] = {"name": sample["name"]}
        for key in ("blur", "sharp"):
            if key in sample:
                path = Path(sample[key])
                info = path.stat()
                entry[key] = {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}
        entries.append(entry)
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"source": str(source), "sha256": digest, "samples": plain}
