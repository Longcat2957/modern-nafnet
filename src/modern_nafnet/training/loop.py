"""FP32 fine-tuning and QAT share one loop; resume is at epoch boundaries."""

import logging
import os
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from modern_nafnet.artifacts import environment_manifest, json_safe, sha256, write_json


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
    from modern_nafnet.models.factory import checkpoint_metadata
    from modern_nafnet.models.nafnet import tlc_kernels

    checkpoint = {
        **checkpoint_metadata(model, config),
        "state_dict": model.state_dict(),
        "training_state": state,
        "config": json_safe(config),
        "tlc_kernels": tlc_kernels(model),
        "environment": environment_manifest(),
    }
    if qat:
        from modern_nafnet.training.quantize import modelopt_state

        checkpoint["modelopt_state"] = modelopt_state(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class _TrainingForward(torch.nn.Module):
    """Keep auxiliary outputs inside the DDP forward boundary."""

    def __init__(self, model, multitask: bool):
        super().__init__()
        self.model = model
        self.multitask = multitask

    def forward(self, image):
        if self.multitask:
            return self.model.forward_with_aux(image)
        return self.model(image), None, None


def run_train(config: dict, *, qat: bool = False) -> dict:
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    initialized_here = world_size > 1 and not dist.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if initialized_here:
        cuda = torch.device(config["training"].get("device", "cuda")).type == "cuda"
        if cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA training was requested but CUDA is not available")
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if cuda else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    try:
        return _run_train(config, qat=qat, rank=rank, world_size=world_size, local_rank=local_rank)
    finally:
        if initialized_here:
            dist.destroy_process_group()


def _run_train(config: dict, *, qat: bool, rank: int, world_size: int, local_rank: int) -> dict:
    import torch.distributed as dist
    from torch.utils.data import DataLoader

    from modern_nafnet.config import DEFAULT_CONFIG
    from modern_nafnet.data.augment import build_blur_transform, build_train_transform
    from modern_nafnet.data.paired import PairedImageDataset, init_data_worker
    from modern_nafnet.data.sources import load_samples, resolve_data_source, source_record
    from modern_nafnet.models.factory import (
        build_model_from_config,
        check_checkpoint_metadata,
        load_checkpoint,
    )
    from modern_nafnet.training.losses import PSNRLoss

    settings = config["training"]
    steps = settings["steps"]
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < (0 if qat else 1):
        raise ValueError("steps must be positive (QAT permits zero for calibration only)")
    device = torch.device(settings.get("device", "cuda"))
    if world_size > 1 and device.type == "cuda":
        device = torch.device("cuda", local_rank)
    multitask = config.get("architecture", "nafnet") == "trt-r2r-light"
    amp = settings.get("amp", False)
    if multitask and amp:
        raise ValueError("TRT-R2R-Light Stage A currently requires FP32 training (amp=false)")
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
    generator = torch.Generator().manual_seed(seed + rank)
    data = config["data"]
    synthetic_settings = data.get("synthetic_blur", DEFAULT_CONFIG["data"]["synthetic_blur"])
    if multitask and synthetic_settings["enabled"]:
        raise ValueError("TRT-R2R-Light requires labeled input/target pairs, not synthetic_blur")
    transform = blur_transform = None
    if steps > 0:
        source = resolve_data_source(data, "train")
        transform = build_train_transform(data["crop_size"], seed + rank)
        if synthetic_settings["enabled"]:
            blur_transform = build_blur_transform(synthetic_settings["blur_range"], seed + rank + 1)
        if multitask:
            from modern_nafnet.data.multitask import RestorationDataset

            if data["crop_size"] % 16:
                raise ValueError("TRT-R2R-Light training crop_size must be a multiple of 16")
            dataset = RestorationDataset(source, config["trt_r2r_light"]["tasks"], transform)
            groups = {sample["task"] for sample in dataset.samples}
            required = {*config["trt_r2r_light"]["tasks"], "clean"}
            if missing := required - groups:
                raise ValueError(
                    f"Training must include each configured task and clean: {sorted(missing)}"
                )
        else:
            dataset = PairedImageDataset(source, transform=transform, blur_transform=blur_transform)
        dataset_record = source_record(source, dataset.samples)
        if world_size > 1:
            records = [None] * world_size
            dist.all_gather_object(records, dataset_record["sha256"])
            if len(set(records)) != 1:
                raise ValueError("DDP ranks must use the same training dataset snapshot")
    model = build_model_from_config(config)
    resume_path = settings.get("resume")
    checkpoint_path = (
        resume_path
        or (config.get("quantization", {}).get("checkpoint") if qat else None)
        or settings.get("checkpoint")
    )
    if not checkpoint_path and (not multitask or qat):
        raise ValueError("training.checkpoint or training.resume is required")
    checkpoint = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint_path
        else {}
    )
    calibration_record = checkpoint.get("training_state", {}).get("calibration_dataset")
    calibration_shape = checkpoint.get("training_state", {}).get("calibration_shape")
    source_hash = sha256(checkpoint_path) if checkpoint_path else None
    if checkpoint_path:
        check_checkpoint_metadata(model, checkpoint, config)
    saved_quantized = isinstance(checkpoint, dict) and bool(checkpoint.get("modelopt_state"))
    if saved_quantized:
        if not qat:
            raise ValueError("Use qat to restore a quantized training checkpoint")
        if multitask and calibration_shape != config["export"].get("shape"):
            raise ValueError("QAT requires the original calibration_shape in export.shape")
        from modern_nafnet.training.quantize import restore_quantized_model

        model = restore_quantized_model(model, checkpoint, config=config)
    elif checkpoint_path:
        load_checkpoint(model, checkpoint_path, config=config)
    model = model.to(device)
    if qat and not saved_quantized:
        from modern_nafnet.training.quantize import calibrate_int8, calibration_inputs

        quantization = config["quantization"]
        calibration_source = resolve_data_source(data, "calibration")
        shape = config["export"].get("shape")
        calibration_shape = shape
        if not shape:
            raise ValueError("QAT initialization needs export.shape")
        if (
            len(shape) != 4
            or shape[:2] != [1, 3]
            or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in shape)
        ):
            raise ValueError("Calibration shape must be fixed [1, 3, height, width]")
        if multitask:
            from modern_nafnet.data.sources import (
                balanced_calibration_samples,
                load_restoration_samples,
            )

            tasks = config["trt_r2r_light"]["tasks"]
            calibration_samples = balanced_calibration_samples(
                load_restoration_samples(calibration_source, tasks, calibration=True),
                tasks,
                quantization["calibration_batches"],
            )
        else:
            calibration_samples = load_samples(calibration_source, blur_only=True)[
                : quantization["calibration_batches"]
            ]
        calibration_record = source_record(calibration_source, calibration_samples)
        sample = torch.zeros(shape, device=device)
        with torch.inference_mode():
            model.eval()(sample)
        model = calibrate_int8(
            model,
            calibration_inputs(
                calibration_source,
                shape,
                str(device),
                quantization["calibration_batches"],
                samples=calibration_samples,
                architecture=config.get("architecture", "nafnet"),
            ),
        )
        if multitask:
            calibration_record.update(
                input={
                    "shape": shape,
                    "dtype": "float32",
                    "layout": "NCHW",
                    "color": "RGB",
                    "range": [0, 1],
                },
                task_counts=dict(Counter(row["task"] for row in calibration_samples)),
                preprocess={"pad_multiple": 16, "mode": "constant", "pad_value": 0},
            )
    if world_size > 1:
        # Calibration is deterministic on every rank; rank zero owns the final scale values.
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=0)
    output_dir = Path(settings["output_dir"])
    if multitask and qat:
        output_dir /= "ptq" if steps == 0 else "qat"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "config.json", config)
        write_json(output_dir / "environment.json", environment_manifest())
        if steps > 0:
            write_json(output_dir / "dataset.json", dataset_record)
        if calibration_record:
            if multitask:
                from modern_nafnet.training.quantize import quantizer_report

                calibration_record["quantizers"] = quantizer_report(model)
            write_json(output_dir / "calibration.json", calibration_record)
    if steps == 0:
        if resume_path:
            raise ValueError("Calibration-only execution does not resume a training loop")
        if saved_quantized:
            raise ValueError(
                "Calibration-only execution requires a floating-point source checkpoint"
            )
        path = output_dir / "ptq.pt"
        if rank == 0:
            save_checkpoint(
                path,
                model,
                state={
                    "step": 0,
                    "resumable": False,
                    "source_checkpoint_sha256": source_hash,
                    "calibration_dataset": calibration_record,
                    "dataset": checkpoint.get("training_state", {}).get("dataset"),
                    "calibration_shape": calibration_shape,
                    "world_size": world_size,
                },
                config=config,
                qat=True,
            )
        from modern_nafnet.training.quantize import quantizer_report

        result = {
            "checkpoint": str(path),
            "step": 0,
            "mode": "ptq",
            "quantizers": quantizer_report(model),
        }
        if rank == 0:
            write_json(output_dir / "result.json", result)
        return result

    batch_size = settings["batch_size"]
    workers = settings.get("num_workers", 0)
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    sampler = None
    sampling = config.get("task_sampling", {"mode": "balanced"})
    if multitask and sampling["mode"] == "balanced":
        from modern_nafnet.data.multitask import BalancedTaskSampler

        sampler = BalancedTaskSampler(dataset, seed=seed, num_replicas=world_size, rank=rank)
    elif world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        num_workers=workers,
        worker_init_fn=init_data_worker,
        persistent_workers=False,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, epoch, batch_in_epoch = 0, 0, 0
    task_counts = Counter()
    if resume_path:
        state = checkpoint.get("training_state", {})
        if not state.get("resumable") or state.get("batch_in_epoch", 0):
            raise ValueError(
                "Only epoch-boundary checkpoints can resume; partial files remain valid for export"
            )
        previous = checkpoint["config"]
        if state.get("world_size", 1) != world_size:
            raise ValueError("Resume requires the original world_size")
        for key in ("steps", "batch_size", "num_workers", "seed", "amp"):
            if previous["training"].get(key) != settings.get(key):
                raise ValueError(f"Resume requires the original training.{key}")
        spec_key = "trt_r2r_light" if multitask else "model"
        if (
            previous.get("architecture", "nafnet") != config.get("architecture", "nafnet")
            or previous[spec_key] != config[spec_key]
        ):
            raise ValueError("Resume requires the original model configuration")
        if (
            multitask
            and (
                previous.get("restoration_loss") != config.get("restoration_loss")
                or previous.get("task_sampling") != config.get("task_sampling")
            )
        ) or previous["data"]["crop_size"] != data["crop_size"]:
            raise ValueError("Resume requires the original loss, sampling and crop configuration")
        previous_synthetic = previous["data"].get(
            "synthetic_blur", DEFAULT_CONFIG["data"]["synthetic_blur"]
        )
        if previous_synthetic != synthetic_settings:
            raise ValueError("Resume requires the original data.synthetic_blur configuration")
        if not state.get("dataset"):
            raise ValueError(
                "The checkpoint has no dataset snapshot; use --checkpoint for fine-tuning "
                "instead of --resume"
            )
        if state["dataset"]["sha256"] != dataset_record["sha256"]:
            raise ValueError("The training dataset changed since the checkpoint")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, epoch = state["step"], state["epoch"]
        saved_rng = state["rank_states"][rank]["rng"] if "rank_states" in state else state["rng"]
        restore_rng(saved_rng, generator, transform, blur_transform)
        saved_counts = (
            state["rank_states"][rank].get("task_counts", {})
            if "rank_states" in state
            else state.get("task_counts", {})
        )
        task_counts.update(saved_counts)
        if step >= steps:
            raise ValueError("The checkpoint has already reached training.steps")
    if multitask:
        from modern_nafnet.training.losses import MultiTaskRestorationLoss

        loss_fn = MultiTaskRestorationLoss(**config["restoration_loss"])
    else:
        loss_fn = PSNRLoss()
    losses = []
    last_components = {}
    resume_policy = "epoch boundaries; exact augmentation replay only with num_workers=0"
    training_model = _TrainingForward(model, multitask)
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel

        training_model = DistributedDataParallel(
            training_model, device_ids=[device.index] if device.type == "cuda" else None
        )

    def current_state() -> dict:
        local_state = {
            "rng": rng_state(generator, transform, blur_transform),
            "task_counts": dict(task_counts),
        }
        rank_states = [local_state]
        if world_size > 1:
            rank_states = [None] * world_size
            dist.all_gather_object(rank_states, local_state)
        return {
            "step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "resumable": batch_in_epoch == 0,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": local_state["rng"],
            "rank_states": rank_states,
            "world_size": world_size,
            "sampler": type(sampler).__name__ if sampler is not None else "RandomSampler",
            "batch_size_per_rank": batch_size,
            "drop_last": False,
            "persistent_workers": False,
            "task_counts": dict(task_counts),
            "dataset": dataset_record,
            "source_checkpoint_sha256": source_hash,
            "calibration_dataset": calibration_record,
            "calibration_shape": calibration_shape,
            # shortcut: epoch-boundary replay with num_workers=0 only; add sampler/worker state before mid-step resume.
            "resume_policy": resume_policy,
            "worker_state_saved": False,
        }

    def save_training(path):
        state = current_state()
        if rank == 0:
            save_checkpoint(path, model, state=state, config=config, qat=qat)

    training_model.train()
    try:
        while step < steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(loader):
                batch_in_epoch = batch_index + 1
                input_key, target_key = ("input", "target") if multitask else ("blur", "sharp")
                blur, sharp = batch[input_key].to(device), batch[target_key].to(device)
                if multitask:
                    task_counts.update(batch["task"])
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    prediction, logits, _ = training_model(blur)
                if multitask:
                    loss, components = loss_fn(
                        prediction.float(),
                        sharp.float(),
                        blur.float(),
                        logits.float() if logits is not None else None,
                        batch["task_label"].to(device),
                    )
                    last_components = {
                        key: float(value.detach()) for key, value in components.items()
                    }
                else:
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
                if rank == 0:
                    logging.getLogger(__name__).info(
                        "step %s/%s, loss %.6f", step, steps, losses[-1]
                    )
                if step == steps and batch_in_epoch < len(loader):
                    break
            if batch_in_epoch == len(loader):
                epoch += 1
                batch_in_epoch = 0
                save_training(output_dir / "last.pt")
    except BaseException:
        if world_size == 1:
            save_training(output_dir / "interrupted.pt")
        raise
    path = output_dir / ("partial.pt" if batch_in_epoch else "last.pt")
    if batch_in_epoch:
        save_training(path)
    result = {
        "checkpoint": str(path),
        "step": step,
        "epoch": epoch,
        "mode": "qat" if qat else "fp32",
        "last_loss": losses[-1],
        "resumable": batch_in_epoch == 0,
        "resume_policy": resume_policy,
        "world_size": world_size,
        "architecture": config.get("architecture", "nafnet"),
        "loss_components": last_components,
        "task_counts": dict(task_counts),
    }
    if rank == 0:
        write_json(output_dir / "result.json", result)
    return result
