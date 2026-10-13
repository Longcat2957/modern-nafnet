"""FP32 fine-tuning and QAT share one loop; resume is at epoch boundaries."""

import logging
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from drone_deblur.artifacts import environment_manifest, json_safe, sha256, write_json


def _transform_state(transform):
    if transform is None:
        return None
    return {
        "python": transform.py_random.getstate(),
        "numpy": deepcopy(transform.random_generator.bit_generator.state),
        "children": [_transform_state(child) for child in getattr(transform, "transforms", [])],
    }


def _restore_transform(transform, state) -> None:
    if state is None:
        return
    transform.py_random.setstate(state["python"])
    transform.random_generator.bit_generator.state = state["numpy"]
    children = getattr(transform, "transforms", [])
    if len(children) != len(state["children"]):
        raise ValueError("The augmentation pipeline differs from the saved checkpoint")
    for child, child_state in zip(children, state["children"], strict=True):
        _restore_transform(child, child_state)


def rng_state(generator: torch.Generator, transform=None, blur_transform=None) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "data_generator": generator.get_state(),
        "augmentation": _transform_state(transform),
        "synthetic_blur": _transform_state(blur_transform),
    }


def restore_rng(
    state: dict, generator: torch.Generator, transform=None, blur_transform=None
) -> None:
    if blur_transform is not None and state.get("synthetic_blur") is None:
        raise ValueError("The checkpoint is missing the synthetic blur RNG state")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG restoration requires the original visible device count")
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["data_generator"])
    _restore_transform(transform, state["augmentation"])
    _restore_transform(blur_transform, state.get("synthetic_blur"))


def save_checkpoint(
    path: str | Path, model: torch.nn.Module, *, state: dict, config: dict, qat=False
):
    from drone_deblur.models.nafnet import tlc_kernels

    checkpoint = {
        "state_dict": model.state_dict(),
        "training_state": state,
        "config": json_safe(config),
        "tlc_kernels": tlc_kernels(model),
        "environment": environment_manifest(),
    }
    if qat:
        from drone_deblur.training.quantize import modelopt_state

        checkpoint["modelopt_state"] = modelopt_state(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_train(config: dict, *, qat: bool = False) -> dict:
    from torch.utils.data import DataLoader

    from drone_deblur.config import DEFAULT_CONFIG
    from drone_deblur.data.augment import build_blur_transform, build_train_transform
    from drone_deblur.data.paired import PairedImageDataset, init_data_worker
    from drone_deblur.models.nafnet import NAFNetLocal, check_checkpoint_metadata, load_checkpoint
    from drone_deblur.training.losses import PSNRLoss

    settings = config["training"]
    steps = settings["steps"]
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < (0 if qat else 1):
        raise ValueError("steps must be positive (QAT permits zero for calibration only)")
    device = torch.device(settings.get("device", "cuda"))
    amp = settings.get("amp", False)
    if amp and device.type != "cuda":
        raise ValueError("AMP training currently requires a CUDA device")
    seed = settings.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is not available")
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    model = NAFNetLocal(**config.get("model", {}))
    resume_path = settings.get("resume")
    checkpoint_path = (
        resume_path
        or (config.get("quantization", {}).get("checkpoint") if qat else None)
        or settings.get("checkpoint")
    )
    if not checkpoint_path:
        raise ValueError("training.checkpoint or training.resume is required")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source_hash = sha256(checkpoint_path)
    check_checkpoint_metadata(model, checkpoint, config.get("model", {}))
    saved_quantized = isinstance(checkpoint, dict) and bool(checkpoint.get("modelopt_state"))
    if saved_quantized:
        if not qat:
            raise ValueError("Use qat to restore a quantized training checkpoint")
        from drone_deblur.training.quantize import restore_quantized_model

        model = restore_quantized_model(model, checkpoint)
    else:
        load_checkpoint(model, checkpoint_path)
    model = model.to(device)
    if qat and not saved_quantized:
        from drone_deblur.training.quantize import calibrate_int8, calibration_inputs

        quantization = config["quantization"]
        manifest = config["data"].get("calibration_manifest")
        shape = config["export"].get("shape")
        if not manifest or not shape:
            raise ValueError("QAT initialization needs calibration_manifest and export.shape")
        if (
            len(shape) != 4
            or shape[:2] != [1, 3]
            or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in shape)
        ):
            raise ValueError("Calibration shape must be fixed [1, 3, height, width]")
        sample = torch.zeros(shape, device=device)
        with torch.inference_mode():
            model.eval()(sample)
        model = calibrate_int8(
            model,
            calibration_inputs(manifest, shape, str(device), quantization["calibration_batches"]),
        )
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if steps == 0:
        if resume_path:
            raise ValueError("Calibration-only execution does not resume a training loop")
        if saved_quantized:
            raise ValueError(
                "Calibration-only execution requires a floating-point source checkpoint"
            )
        path = output_dir / "ptq.pt"
        save_checkpoint(
            path,
            model,
            state={
                "step": 0,
                "resumable": False,
                "source_checkpoint_sha256": source_hash,
                "calibration_manifest_sha256": sha256(manifest),
            },
            config=config,
            qat=True,
        )
        from drone_deblur.training.quantize import quantizer_report

        result = {
            "checkpoint": str(path),
            "step": 0,
            "mode": "ptq",
            "quantizers": quantizer_report(model),
        }
        write_json(output_dir / "result.json", result)
        return result

    data = config["data"]
    if not data.get("train_manifest"):
        raise ValueError("data.train_manifest is required for supervised training/QAT")
    transform = build_train_transform(data["crop_size"], seed)
    synthetic_settings = data.get("synthetic_blur", DEFAULT_CONFIG["data"]["synthetic_blur"])
    blur_transform = (
        build_blur_transform(synthetic_settings["blur_range"], seed + 1)
        if synthetic_settings["enabled"]
        else None
    )
    dataset = PairedImageDataset(
        data["train_manifest"], transform=transform, blur_transform=blur_transform
    )
    batch_size = settings["batch_size"]
    workers = settings.get("num_workers", 0)
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=workers,
        worker_init_fn=init_data_worker,
        persistent_workers=False,
        drop_last=False,
    )
    manifest_hash = sha256(data["train_manifest"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, epoch, batch_in_epoch = 0, 0, 0
    if resume_path:
        state = checkpoint.get("training_state", {})
        if not state.get("resumable") or state.get("batch_in_epoch", 0):
            raise ValueError(
                "Only epoch-boundary checkpoints can resume; partial files remain valid for export"
            )
        previous = checkpoint["config"]
        for key in ("steps", "batch_size", "num_workers", "seed", "amp"):
            if previous["training"].get(key) != settings.get(key):
                raise ValueError(f"Resume requires the original training.{key}")
        if (
            previous["model"] != config["model"]
            or previous["data"]["crop_size"] != data["crop_size"]
        ):
            raise ValueError("Resume requires the original model and crop configuration")
        previous_synthetic = previous["data"].get(
            "synthetic_blur", DEFAULT_CONFIG["data"]["synthetic_blur"]
        )
        if previous_synthetic != synthetic_settings:
            raise ValueError("Resume requires the original data.synthetic_blur configuration")
        if state["manifest_sha256"] != manifest_hash:
            raise ValueError("The training manifest changed since the checkpoint")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, epoch = state["step"], state["epoch"]
        restore_rng(state["rng"], generator, transform, blur_transform)
        if step >= steps:
            raise ValueError("The checkpoint has already reached training.steps")
    loss_fn = PSNRLoss()
    losses = []

    def current_state() -> dict:
        return {
            "step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "resumable": batch_in_epoch == 0,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_state(generator, transform, blur_transform),
            "manifest_sha256": manifest_hash,
            "source_checkpoint_sha256": source_hash,
            "calibration_manifest_sha256": sha256(data["calibration_manifest"])
            if qat and data.get("calibration_manifest")
            else None,
            # shortcut: epoch-boundary replay with num_workers=0 only; add sampler/worker state before mid-step resume.
            "resume_policy": "epoch boundaries; exact augmentation replay only with num_workers=0",
            "worker_state_saved": False,
        }

    model.train()
    try:
        while step < steps:
            for batch_index, batch in enumerate(loader):
                batch_in_epoch = batch_index + 1
                blur, sharp = batch["blur"].to(device), batch["sharp"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    prediction = model(blur)
                loss = loss_fn(prediction.float(), sharp.float())
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite loss at step {step + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()
                ):
                    raise FloatingPointError(f"Non-finite gradient at step {step + 1}")
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                losses.append(float(loss.detach()))
                logging.getLogger(__name__).info("step %s/%s, loss %.6f", step, steps, losses[-1])
                if step == steps and batch_in_epoch < len(loader):
                    break
            if batch_in_epoch == len(loader):
                epoch += 1
                batch_in_epoch = 0
                save_checkpoint(
                    output_dir / "last.pt", model, state=current_state(), config=config, qat=qat
                )
    except BaseException:
        save_checkpoint(
            output_dir / "interrupted.pt", model, state=current_state(), config=config, qat=qat
        )
        raise
    path = output_dir / ("partial.pt" if batch_in_epoch else "last.pt")
    if batch_in_epoch:
        save_checkpoint(path, model, state=current_state(), config=config, qat=qat)
    result = {
        "checkpoint": str(path),
        "step": step,
        "epoch": epoch,
        "mode": "qat" if qat else "fp32",
        "last_loss": losses[-1],
        "resumable": batch_in_epoch == 0,
        "resume_policy": current_state()["resume_policy"],
    }
    write_json(output_dir / "result.json", result)
    return result
