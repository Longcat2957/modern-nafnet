"""FP32 fine-tuning and QAT share one loop; resume is at epoch boundaries."""

import logging
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from modern_nafnet.artifacts import environment_manifest, json_safe, sha256, write_json
from modern_nafnet.training.distributed import TrainingWorld, training_world


def _transform_state(transform, *, root=True):
    if transform is None:
        return None
    # AlbumentationsX 2.5 keeps Compose children's RNGs dormant: they use the root stream.
    # Reading their generator properties would create unrelated, unseeded RNGs at save time.
    if not root and getattr(transform, "_rng_initialized", None) is False:
        return {"dormant": True}
    return {
        "python": transform.py_random.getstate(),
        "numpy": deepcopy(transform.random_generator.bit_generator.state),
        "children": [
            _transform_state(child, root=False) for child in getattr(transform, "transforms", [])
        ],
    }


def _restore_transform(transform, state) -> None:
    if state is None or state.get("dormant"):
        return
    transform.py_random.setstate(state["python"])
    transform.random_generator.bit_generator.state = state["numpy"]
    children = getattr(transform, "transforms", [])
    if len(children) != len(state["children"]):
        raise ValueError("The augmentation pipeline differs from the saved checkpoint")
    for child, child_state in zip(children, state["children"], strict=True):
        _restore_transform(child, child_state)


def rng_state(
    generator: torch.Generator, transform=None, blur_transform=None, *, device=None
) -> dict:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": (
            [torch.cuda.get_rng_state(device)]
            if device is not None
            else torch.cuda.get_rng_state_all()
            if torch.cuda.is_initialized()
            else []
        ),
        "data_generator": generator.get_state(),
        "augmentation": _transform_state(transform),
        "synthetic_blur": _transform_state(blur_transform),
    }
    if device is not None:
        state["cuda_device"] = device.index
    return state


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
        if "cuda_device" in state:
            if not torch.cuda.is_available() or state["cuda_device"] != torch.cuda.current_device():
                raise ValueError("CUDA RNG restoration requires the original rank's local device")
            torch.cuda.set_rng_state(state["cuda"][0], device=state["cuda_device"])
        elif not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG restoration requires the original visible device count")
        else:
            torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["data_generator"])
    _restore_transform(transform, state["augmentation"])
    _restore_transform(blur_transform, state.get("synthetic_blur"))


def save_checkpoint(
    path: str | Path, model: torch.nn.Module, *, state: dict, config: dict, qat=False
):
    from modern_nafnet.models.nafnet import tlc_kernels

    checkpoint = {
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


def run_train(config: dict, *, qat: bool = False) -> dict:
    with training_world(config["training"].get("device", "cuda")) as world:
        return _run_train(config, qat=qat, world=world)


def _run_train(config: dict, *, qat: bool, world: TrainingWorld) -> dict:
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from modern_nafnet.config import DEFAULT_CONFIG
    from modern_nafnet.data.augment import build_blur_transform, build_train_transform
    from modern_nafnet.data.paired import PairedImageDataset, init_data_worker
    from modern_nafnet.data.sources import load_samples, resolve_data_source, source_record
    from modern_nafnet.models.nafnet import NAFNetLocal, check_checkpoint_metadata, load_checkpoint
    from modern_nafnet.training.losses import PSNRLoss

    settings = config["training"]
    steps = settings["steps"]
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < (0 if qat else 1):
        raise ValueError("steps must be positive (QAT permits zero for calibration only)")
    device = world.device
    amp = settings.get("amp", False)
    if amp and device.type != "cuda":
        raise ValueError("AMP training currently requires a CUDA device")
    seed = settings.get("seed", 0)
    # Model weights come from the shared checkpoint; data and augmentation RNGs are per rank.
    rank_seed = seed + world.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(rank_seed)
    generator = torch.Generator().manual_seed(rank_seed)
    data = config["data"]
    synthetic_settings = data.get("synthetic_blur", DEFAULT_CONFIG["data"]["synthetic_blur"])
    transform = blur_transform = None
    if steps > 0:
        source = resolve_data_source(data, "train")
        transform = build_train_transform(data["crop_size"], rank_seed)
        if synthetic_settings["enabled"]:
            blur_transform = build_blur_transform(synthetic_settings["blur_range"], rank_seed + 1)
        dataset = PairedImageDataset(source, transform=transform, blur_transform=blur_transform)
        dataset_record = source_record(source, dataset.samples)
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
    calibration_record = checkpoint.get("training_state", {}).get("calibration_dataset")
    source_hash = sha256(checkpoint_path)
    check_checkpoint_metadata(model, checkpoint, config.get("model", {}))
    world.require_same(
        (source_hash, dataset_record["sha256"] if steps else None, json_safe(config)),
        "All ranks must use the same configuration, checkpoint and dataset",
    )
    saved_quantized = isinstance(checkpoint, dict) and bool(checkpoint.get("modelopt_state"))
    if resume_path:
        saved_world_size = (
            checkpoint.get("training_state", {}).get("distributed", {}).get("world_size", 1)
        )
        if saved_world_size != world.world_size:
            raise ValueError(
                "Resume requires the original distributed world_size; "
                "use --checkpoint for fine-tuning with a different GPU count"
            )
        if saved_quantized != qat:
            raise ValueError("Resume requires the original FP32/QAT training mode")
    if saved_quantized:
        if not qat:
            raise ValueError("Use qat to restore a quantized training checkpoint")
        from modern_nafnet.training.quantize import restore_quantized_model

        model = restore_quantized_model(model, checkpoint)
    else:
        load_checkpoint(model, checkpoint_path)
    model = model.to(device)
    if qat and not saved_quantized:
        from modern_nafnet.training.quantize import calibrate_int8, calibration_inputs

        quantization = config["quantization"]
        calibration_source = resolve_data_source(data, "calibration")
        shape = config["export"].get("shape")
        if not shape:
            raise ValueError("QAT initialization needs export.shape")
        if (
            len(shape) != 4
            or shape[:2] != [1, 3]
            or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in shape)
        ):
            raise ValueError("Calibration shape must be fixed [1, 3, height, width]")
        calibration_samples = load_samples(calibration_source, blur_only=True)[
            : quantization["calibration_batches"]
        ]
        calibration_record = source_record(calibration_source, calibration_samples)
        world.require_same(
            calibration_record["sha256"], "All ranks must use the same calibration dataset"
        )
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
            ),
        )
    output_dir = Path(settings["output_dir"])
    world.run(lambda: output_dir.mkdir(parents=True, exist_ok=True) if world.primary else None)
    if steps == 0:
        if resume_path:
            raise ValueError("Calibration-only execution does not resume a training loop")
        if saved_quantized:
            raise ValueError(
                "Calibration-only execution requires a floating-point source checkpoint"
            )
        path = output_dir / "ptq.pt"
        world.run(
            lambda: (
                save_checkpoint(
                    path,
                    model,
                    state={
                        "step": 0,
                        "resumable": False,
                        "source_checkpoint_sha256": source_hash,
                        "calibration_dataset": calibration_record,
                        "distributed": {
                            "world_size": world.world_size,
                            "backend": "nccl" if world.distributed else None,
                        },
                    },
                    config=config,
                    qat=True,
                )
                if world.primary
                else None
            )
        )
        from modern_nafnet.training.quantize import quantizer_report

        result = {
            "checkpoint": str(path),
            "step": 0,
            "mode": "ptq",
            "quantizers": quantizer_report(model),
        }
        world.run(lambda: write_json(output_dir / "result.json", result) if world.primary else None)
        return result

    batch_size = settings["batch_size"]
    workers = settings.get("num_workers", 0)
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    sampler = (
        DistributedSampler(
            dataset, num_replicas=world.world_size, rank=world.rank, seed=seed, drop_last=False
        )
        if world.distributed
        else None
    )
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
        **({"multiprocessing_context": "spawn"} if world.distributed and workers else {}),
    )
    distributed_record = {
        "world_size": world.world_size,
        "backend": "nccl" if world.distributed else None,
        "batch_size_per_rank": batch_size,
        "global_batch_size": batch_size * world.world_size,
        "num_workers_per_rank": workers,
        "sampler": "DistributedSampler" if sampler is not None else "RandomSampler",
        "sampler_seed": seed,
        "sampler_drop_last": False,
        "loader_drop_last": False,
        "persistent_workers": False,
        "multiprocessing_context": "spawn" if world.distributed and workers else None,
        "samples_per_rank": len(sampler) if sampler is not None else len(dataset),
        "padded_samples_per_epoch": (
            sampler.total_size - len(dataset) if sampler is not None else 0
        ),
    }
    # DDP's initial sync includes calibrated, frozen ModelOpt buffers. The loop does not mutate them.
    training_model = (
        DistributedDataParallel(model, device_ids=[device.index], forward_sync_buffers=False)
        if world.distributed
        else model
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, epoch, batch_in_epoch = 0, 0, 0
    amp_skipped_steps = 0
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
        amp_skipped_steps = state.get("amp_skipped_steps", 0)
        if world.distributed:
            rank_states = state.get("rng_by_rank", [])
            if len(rank_states) != world.world_size:
                raise ValueError("The checkpoint is missing per-rank RNG states")
            saved_rng = rank_states[world.rank]
        else:
            saved_rng = state["rng"]
        restore_rng(saved_rng, generator, transform, blur_transform)
        if step >= steps:
            raise ValueError("The checkpoint has already reached training.steps")
    loss_fn = PSNRLoss()
    losses = []

    def local_rng() -> dict:
        return rng_state(
            generator, transform, blur_transform, device=device if world.distributed else None
        )

    def current_state(rank_states=None) -> dict:
        state = {
            "step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "resumable": batch_in_epoch == 0,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "amp_skipped_steps": amp_skipped_steps,
            "rng": rank_states[0] if rank_states is not None else local_rng(),
            "dataset": dataset_record,
            "source_checkpoint_sha256": source_hash,
            "calibration_dataset": calibration_record,
            # shortcut: epoch-boundary replay with num_workers=0 only; add sampler/worker state before mid-step resume.
            "resume_policy": "epoch boundaries; exact augmentation replay only with num_workers=0",
            "worker_state_saved": False,
            "distributed": distributed_record,
        }
        if world.distributed and rank_states is not None:
            state["rng_by_rank"] = rank_states
        return state

    def coordinated_save(path: Path) -> None:
        rank_states = world.gather(local_rng())
        world.run(
            lambda: (
                save_checkpoint(
                    path, model, state=current_state(rank_states), config=config, qat=qat
                )
                if world.primary
                else None
            )
        )

    def forward_loss(blur, sharp):
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            prediction = training_model(blur)
        loss = loss_fn(prediction.float(), sharp.float())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite loss at step {step + 1}")
        return loss

    def check_gradients():
        finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
        if not amp and finite and not bool(torch.stack(finite).all()):
            raise FloatingPointError(f"Non-finite gradient at step {step + 1}")

    training_model.train()
    try:
        while step < steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = world.run(lambda: iter(loader))
            for batch_index in range(len(loader)):
                batch = world.run(lambda: next(iterator))
                batch_in_epoch = batch_index + 1
                blur, sharp = world.run(
                    lambda: (batch["blur"].to(device), batch["sharp"].to(device))
                )
                optimizer.zero_grad(set_to_none=True)
                loss = world.run(lambda: forward_loss(blur, sharp))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                world.run(check_gradients)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if amp and scaler.get_scale() < scale_before:
                    amp_skipped_steps += 1
                    if scaler.get_scale() <= 0:
                        raise FloatingPointError(
                            "AMP gradient scale exhausted after repeated overflows"
                        )
                    if world.primary:
                        logging.getLogger(__name__).warning(
                            "AMP gradient overflow: skipped update, scale reduced to %s",
                            scaler.get_scale(),
                        )
                    continue
                scheduler.step()
                step += 1
                losses.append(world.mean(loss))
                if world.primary:
                    logging.getLogger(__name__).info(
                        "step %s/%s, loss %.6f", step, steps, losses[-1]
                    )
                if step == steps and batch_in_epoch < len(loader):
                    break
            if batch_in_epoch == len(loader):
                epoch += 1
                batch_in_epoch = 0
                coordinated_save(output_dir / "last.pt")
    except BaseException:
        if world.primary:
            state = current_state()
            if world.distributed:
                # A failed peer may never reach this handler; do not collect RNGs here.
                state["resumable"] = False
            save_checkpoint(
                output_dir / "interrupted.pt", model, state=state, config=config, qat=qat
            )
        raise
    path = output_dir / ("partial.pt" if batch_in_epoch else "last.pt")
    if batch_in_epoch:
        coordinated_save(path)
    result = {
        "checkpoint": str(path),
        "step": step,
        "epoch": epoch,
        "mode": "qat" if qat else "fp32",
        "last_loss": losses[-1],
        "resumable": batch_in_epoch == 0,
        "resume_policy": current_state()["resume_policy"],
        "distributed": distributed_record,
        "amp_skipped_steps": amp_skipped_steps,
    }
    world.run(lambda: write_json(output_dir / "result.json", result) if world.primary else None)
    return result
