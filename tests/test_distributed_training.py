"""Exercise the real training loop with two independent CUDA processes.

This file also serves as the torchrun worker, so subprocesses use the same
locked environment as pytest without invoking uv concurrently.
"""

# ruff: noqa: E402
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("albumentations")

from modern_nafnet.config import DEFAULT_CONFIG
from modern_nafnet.data.images import write_rgb
from modern_nafnet.models.nafnet import NAFNetLocal
from modern_nafnet.training.distributed import training_world

TWO_CUDA_DEVICES = (
    torch.cuda.is_available()
    and torch.cuda.device_count() >= 2
    and torch.distributed.is_available()
    and torch.distributed.is_nccl_available()
)
SUMMARY_PREFIX = "DDP_TEST_RESULT "


@pytest.fixture
def clean_launch_environment(monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"WORLD_SIZE": "two"}, "must be integers"),
        ({"WORLD_SIZE": "0"}, "Invalid torchrun"),
        ({"WORLD_SIZE": "2", "RANK": "2", "LOCAL_RANK": "0"}, "Invalid torchrun"),
        ({"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "-1"}, "Invalid torchrun"),
        ({"WORLD_SIZE": "2", "RANK": "0"}, "Launch multi-GPU training with torchrun"),
    ],
)
def test_invalid_launch_environment_is_rejected(clean_launch_environment, environment, message):
    for name, value in environment.items():
        clean_launch_environment.setenv(name, value)
    with pytest.raises(ValueError, match=message), training_world("cpu"):
        pytest.fail("Invalid distributed launch was accepted")


def test_cpu_training_rejects_multi_process_launch(clean_launch_environment):
    for name, value in {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}.items():
        clean_launch_environment.setenv(name, value)
    with pytest.raises(ValueError, match="requires training.device=cuda"), training_world("cpu"):
        pytest.fail("A multi-process CPU launch was accepted")


def test_explicit_device_cannot_send_both_ranks_to_one_gpu(clean_launch_environment):
    for name, value in {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"}.items():
        clean_launch_environment.setenv(name, value)
    with pytest.raises(ValueError, match="LOCAL_RANK selects each GPU"), training_world("cuda:0"):
        pytest.fail("The requested device ignored LOCAL_RANK")


def test_single_process_cpu_requires_no_process_group(clean_launch_environment):
    def unexpected_init(*args, **kwargs):
        pytest.fail("Single-process training initialized a distributed process group")

    clean_launch_environment.setattr(torch.distributed, "init_process_group", unexpected_init)
    with training_world("cpu") as world:
        assert world.primary and not world.distributed
        assert world.device == torch.device("cpu")


def _hash_tensors(tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _worker(config_path: str, *, qat: bool, interrupt: bool, overflow: bool) -> None:
    """Observe rank assignment and parameters without changing training math."""
    from modern_nafnet.data.paired import PairedImageDataset
    from modern_nafnet.training import losses
    from modern_nafnet.training.loop import run_train

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rank = int(os.environ["RANK"])
    seen_indices = []
    synchronized_steps = []
    quantized = {}
    overflow_state = {"injected": False}
    original_getitem = PairedImageDataset.__getitem__
    original_step = torch.optim.AdamW.step

    def observed_getitem(self, index):
        seen_indices.append(index)
        return original_getitem(self, index)

    def observed_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        scales_hash = None
        if quantized:
            current = quantized["model"].state_dict()
            scales = quantized["scales"]
            assert scales, "The QAT test must exercise calibrated quantizers"
            assert all(
                torch.equal(current[name].detach().cpu(), value) for name, value in scales.items()
            )
            scales_hash = _hash_tensors(current[name] for name in sorted(scales))
        observation = {
            "rank": rank,
            "device": torch.cuda.current_device(),
            "indices": list(seen_indices),
            "parameter_sha256": _hash_tensors(
                parameter for group in self.param_groups for parameter in group["params"]
            ),
            "scales_sha256": scales_hash,
            "overflow_injected": overflow_state["injected"],
        }
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, observation)
        assert len({item["parameter_sha256"] for item in gathered}) == 1
        if qat:
            assert len({item["scales_sha256"] for item in gathered}) == 1
        synchronized_steps.append(gathered)
        return result

    PairedImageDataset.__getitem__ = observed_getitem
    torch.optim.AdamW.step = observed_step
    if overflow:
        from modern_nafnet.models import nafnet

        original_factory = nafnet.NAFNetLocal

        def model_with_one_rank_overflow(*args, **kwargs):
            model = original_factory(*args, **kwargs)

            def overflow_once(gradient):
                if not overflow_state["injected"]:
                    overflow_state["injected"] = True
                    return torch.full_like(gradient, float("inf"))
                return gradient

            if rank == 1:
                # Registered before DDP so this rank's inf is reduced to both GPUs.
                model.intro.weight.register_hook(overflow_once)
            return model

        nafnet.NAFNetLocal = model_with_one_rank_overflow
    if qat:
        from modern_nafnet.training import quantize

        def observe_quantized(function):
            def observed(*args, **kwargs):
                model = function(*args, **kwargs)
                quantized["model"] = model
                quantized["scales"] = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                    if name.endswith("_amax")
                }
                return model

            return observed

        quantize.calibrate_int8 = observe_quantized(quantize.calibrate_int8)
        quantize.restore_quantized_model = observe_quantized(quantize.restore_quantized_model)
    if interrupt:
        original_loss = losses.PSNRLoss

        class InterruptedLoss(original_loss):
            calls = 0

            def forward(self, prediction, target):
                self.calls += 1
                if rank == 1 and self.calls == 3:
                    raise RuntimeError("simulated rank-local interruption")
                return super().forward(prediction, target)

        losses.PSNRLoss = InterruptedLoss
    config = json.loads(Path(config_path).read_text())
    result = run_train(config, qat=qat)
    assert not torch.distributed.is_initialized(), "The loop must clean up its process group"
    if rank == 0:
        print(
            SUMMARY_PREFIX + json.dumps({"result": result, "steps": synchronized_steps}),
            flush=True,
        )


@pytest.fixture
def distributed_case(tmp_path):
    torch.manual_seed(17)
    config = deepcopy(DEFAULT_CONFIG)
    config["model"].update(
        width=2,
        enc_blk_nums=[1],
        middle_blk_num=1,
        dec_blk_nums=[1],
        train_size=[1, 3, 8, 8],
    )
    model = NAFNetLocal(**config["model"])
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("beta", "gamma")):
                parameter.fill_(0.1)
    source = tmp_path / "source.pt"
    torch.save(model.state_dict(), source)
    root = tmp_path / "data"
    random = np.random.default_rng(4)
    # An odd count exercises DistributedSampler's documented padding policy.
    for index in range(3):
        image = random.integers(0, 256, (12, 12, 3), dtype=np.uint8)
        write_rgb(root / f"train/sharp/{index}.png", image)
        write_rgb(root / f"train/blur/{index}.png", np.roll(image, 1, axis=1))
    write_rgb(
        root / "calibration/frame.png",
        random.integers(0, 256, (8, 8, 3), dtype=np.uint8),
    )
    config["data"].update(path=str(root), crop_size=8)
    config["training"].update(
        checkpoint=str(source),
        output_dir=str(tmp_path / "run"),
        steps=4,
        batch_size=1,
        num_workers=0,
        seed=31,
        device="cuda",
        amp=False,
    )
    config["quantization"]["calibration_batches"] = 1
    config["export"]["shape"] = [1, 3, 8, 8]
    return config


def _launch(config, tmp_path, *, qat=False, interrupt=False, overflow=False):
    config_file = tmp_path / f"{Path(config['training']['output_dir']).name}.json"
    config_file.write_text(json.dumps(config))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "--run-path",
        str(Path(__file__).resolve()),
        "--worker",
        str(config_file),
    ]
    if qat:
        command.append("--qat")
    if interrupt:
        command.append("--interrupt")
    if overflow:
        command.append("--overflow")
    environment = os.environ.copy()
    # Do not accidentally inherit an enclosing distributed launch in CI.
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
    environment.update(
        OMP_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        ALBUMENTATIONS_NO_TELEMETRY="1",
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
        check=False,
    )
    if interrupt:
        assert completed.returncode != 0
        assert "simulated rank-local interruption" in completed.stderr
        return None
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    summaries = [
        line.removeprefix(SUMMARY_PREFIX)
        for line in completed.stdout.splitlines()
        if line.startswith(SUMMARY_PREFIX)
    ]
    assert len(summaries) == 1, completed.stdout
    return json.loads(summaries[0])


def _assert_nested_equal(expected, actual):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _assert_nested_equal(left, right)
    else:
        assert expected == actual


@pytest.mark.skipif(not TWO_CUDA_DEVICES, reason="Two CUDA devices and NCCL are required")
@pytest.mark.parametrize(
    ("qat", "synthetic"),
    [(False, False), (True, False), (False, True)],
    ids=["fp32-paired", "qat-paired", "fp32-synthetic"],
)
def test_two_gpu_training_synchronizes_and_resumes_exactly(
    distributed_case, tmp_path, qat, synthetic
):
    if qat:
        pytest.importorskip("modelopt")
    if synthetic:
        root = Path(distributed_case["data"]["path"])
        random = np.random.default_rng(37)
        for index in range(3):
            write_rgb(
                root / f"clean/{index}.png",
                random.integers(0, 256, (12, 12, 3), dtype=np.uint8),
            )
        distributed_case["data"]["train"] = "clean"
        distributed_case["data"]["synthetic_blur"].update(enabled=True, blur_range=[3, 7])
    baseline = deepcopy(distributed_case)
    baseline["training"]["output_dir"] += "-baseline"
    summary = _launch(baseline, tmp_path, qat=qat)
    assert summary["result"]["step"] == 4
    assert summary["result"]["epoch"] == 2
    assert summary["result"]["resumable"]
    assert len(summary["steps"]) == 4
    ranks = summary["steps"][-1]
    assert [item["device"] for item in ranks] == [0, 1]
    for epoch in range(2):
        indices = [index for item in ranks for index in item["indices"][2 * epoch : 2 * epoch + 2]]
        assert sorted(set(indices)) == [0, 1, 2]
        assert len(indices) == 4
    expected = torch.load(summary["result"]["checkpoint"], weights_only=True)
    state = expected["training_state"]
    distributed = state["distributed"]
    assert distributed["world_size"] == 2
    assert distributed["backend"] == "nccl"
    assert distributed["batch_size_per_rank"] == 1
    assert distributed["global_batch_size"] == 2
    assert distributed["sampler"] == "DistributedSampler"
    assert distributed["samples_per_rank"] == 2
    assert distributed["padded_samples_per_epoch"] == 1
    assert not distributed["sampler_drop_last"] and not distributed["loader_drop_last"]
    assert len(state["rng_by_rank"]) == 2
    assert not torch.equal(
        state["rng_by_rank"][0]["data_generator"],
        state["rng_by_rank"][1]["data_generator"],
    )
    if synthetic:
        assert all(rank_rng["synthetic_blur"]["children"] for rank_rng in state["rng_by_rank"])
        assert all("blur" not in sample for sample in state["dataset"]["samples"])
    assert all(not name.startswith("module.") for name in expected["state_dict"])
    if qat:
        from modern_nafnet.training.quantize import restore_quantized_model

        assert expected["modelopt_state"]
        restore_quantized_model(NAFNetLocal(**distributed_case["model"]), expected)
    else:
        NAFNetLocal(**distributed_case["model"]).load_state_dict(
            expected["state_dict"], strict=True
        )
    assert sorted(path.name for path in Path(baseline["training"]["output_dir"]).iterdir()) == [
        "last.pt",
        "result.json",
    ]

    _launch(distributed_case, tmp_path, qat=qat, interrupt=True)
    boundary = Path(distributed_case["training"]["output_dir"]) / "last.pt"
    saved_boundary = torch.load(boundary, weights_only=True)
    assert saved_boundary["training_state"]["step"] == 2
    assert saved_boundary["training_state"]["epoch"] == 1
    assert saved_boundary["training_state"]["batch_in_epoch"] == 0
    assert saved_boundary["training_state"]["resumable"]
    assert len(saved_boundary["training_state"]["rng_by_rank"]) == 2
    distributed_case["training"]["resume"] = str(boundary)
    resumed = _launch(distributed_case, tmp_path, qat=qat)
    assert len(resumed["steps"]) == 2
    actual = torch.load(resumed["result"]["checkpoint"], weights_only=True)
    _assert_nested_equal(expected["state_dict"], actual["state_dict"])
    for name in ("optimizer", "scheduler", "scaler", "rng_by_rank"):
        _assert_nested_equal(state[name], actual["training_state"][name])
    assert actual["training_state"]["step"] == 4
    assert actual["training_state"]["epoch"] == 2
    if not qat:
        from modern_nafnet.training.loop import run_train

        changed_world = deepcopy(distributed_case)
        changed_world["training"]["device"] = "cpu"
        with pytest.raises(ValueError, match="world.size|world_size"):
            run_train(changed_world)


@pytest.mark.skipif(not TWO_CUDA_DEVICES, reason="Two CUDA devices and NCCL are required")
def test_two_gpu_training_with_spawned_data_workers(distributed_case, tmp_path):
    distributed_case["training"].update(num_workers=1, steps=2)
    summary = _launch(distributed_case, tmp_path)
    assert summary["result"]["step"] == 2
    assert len(summary["steps"]) == 2
    saved = torch.load(summary["result"]["checkpoint"], weights_only=True)
    assert saved["training_state"]["distributed"]["num_workers_per_rank"] == 1
    assert not saved["training_state"]["worker_state_saved"]


@pytest.mark.skipif(not TWO_CUDA_DEVICES, reason="Two CUDA devices and NCCL are required")
def test_two_gpu_ptq_saves_one_restorable_quantized_checkpoint(distributed_case, tmp_path):
    pytest.importorskip("modelopt")
    from modern_nafnet.training.quantize import check_int8_convs, restore_quantized_model

    distributed_case["training"]["steps"] = 0
    summary = _launch(distributed_case, tmp_path, qat=True)
    assert summary["result"]["mode"] == "ptq"
    assert summary["result"]["step"] == 0
    assert not summary["steps"], "PTQ must not run an optimizer update"
    output_dir = Path(distributed_case["training"]["output_dir"])
    assert sorted(path.name for path in output_dir.iterdir()) == ["ptq.pt", "result.json"]
    saved = torch.load(summary["result"]["checkpoint"], weights_only=True)
    assert saved["modelopt_state"]
    assert not saved["training_state"]["resumable"]
    assert saved["training_state"]["distributed"] == {"world_size": 2, "backend": "nccl"}
    assert len(saved["training_state"]["calibration_dataset"]["samples"]) == 1
    assert all(not name.startswith("module.") for name in saved["state_dict"])
    model = restore_quantized_model(NAFNetLocal(**distributed_case["model"]), saved)
    assert check_int8_convs(model)
    with torch.inference_mode():
        prediction = model.eval()(torch.rand(1, 3, 8, 8))
    assert prediction.shape == (1, 3, 8, 8)
    assert torch.isfinite(prediction).all()


@pytest.mark.skipif(not TWO_CUDA_DEVICES, reason="Two CUDA devices and NCCL are required")
def test_two_gpu_amp_backs_off_after_rank_local_gradient_overflow(distributed_case, tmp_path):
    distributed_case["training"].update(amp=True, steps=2)
    summary = _launch(distributed_case, tmp_path, overflow=True)
    assert summary["result"]["step"] == 2
    assert summary["result"]["amp_skipped_steps"] >= 1
    # The worker records real optimizer calls, so skipped iterations cannot count here.
    assert len(summary["steps"]) == 2
    ranks = summary["steps"][-1]
    assert not ranks[0]["overflow_injected"] and ranks[1]["overflow_injected"]
    assert len({item["parameter_sha256"] for item in ranks}) == 1
    saved = torch.load(summary["result"]["checkpoint"], weights_only=True)
    state = saved["training_state"]
    assert state["step"] == 2
    assert state["amp_skipped_steps"] == summary["result"]["amp_skipped_steps"]
    assert state["scheduler"]["last_epoch"] == 2
    assert 0 < state["scaler"]["scale"] < 65536
    assert all(torch.isfinite(value).all() for value in saved["state_dict"].values())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--qat", action="store_true")
    parser.add_argument("--interrupt", action="store_true")
    parser.add_argument("--overflow", action="store_true")
    arguments = parser.parse_args()
    _worker(
        arguments.worker,
        qat=arguments.qat,
        interrupt=arguments.interrupt,
        overflow=arguments.overflow,
    )
