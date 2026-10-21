"""Two-process CPU checks for the actual DDP training and calibration paths."""

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

from modern_nafnet.config import load_config  # noqa: E402
from modern_nafnet.data.images import write_rgb  # noqa: E402
from modern_nafnet.models.factory import (  # noqa: E402
    build_model_from_config,
    checkpoint_metadata,
)
from modern_nafnet.training.loop import run_train  # noqa: E402

WORKER = """\
import hashlib
import json
import os
import sys
from pathlib import Path
import torch
from modern_nafnet.cli import _run
from modern_nafnet.config import load_config
from modern_nafnet.models import factory
from modern_nafnet.training import loop

torch.set_num_threads(1)
config = load_config(Path(sys.argv[1]))
qat = sys.argv[2] == "qat"
probe = {}
models = []
original_build = factory.build_model_from_config
def build(config):
    model = original_build(config)
    probe["initial_bank"] = factory.bank_fingerprint(model)
    models.append(model)
    return model
factory.build_model_from_config = build

if qat:
    from modern_nafnet.training import quantize
    original_calibrate = quantize.calibrate_int8
    def calibrate(*args, **kwargs):
        model = original_calibrate(*args, **kwargs)
        probe["calibrated_bank"] = factory.bank_fingerprint(model)
        probe["bank_frozen"] = not model.bank.key.requires_grad and not model.bank.value.requires_grad
        digest = hashlib.sha256()
        count = 0
        for name, tensor in sorted(model.state_dict().items()):
            if name.endswith("._amax"):
                count += 1
                digest.update(name.encode())
                digest.update(tensor.cpu().contiguous().numpy().tobytes())
        probe["calibration_scale_count"] = count
        probe["calibration_scales"] = digest.hexdigest()
        return model
    quantize.calibrate_int8 = calibrate

_run(lambda: loop.run_train(config, qat=qat), "--extra train --extra qat")
probe["final_bank"] = factory.bank_fingerprint(models[0])
Path(sys.argv[3], f"rank-{os.environ['RANK']}.json").write_text(json.dumps(probe))
"""


@pytest.fixture
def distributed_case(tmp_path):
    data = tmp_path / "data"
    for task in ("derain", "deblur", "clean"):
        for index in range(2):
            target = np.full((16, 16, 3), 80 + index * 10, dtype=np.uint8)
            write_rgb(data / "train" / task / "target" / f"{index}.png", target)
            if task != "clean":
                write_rgb(data / "train" / task / "input" / f"{index}.png", target // 2)
        write_rgb(data / "calibration" / task / "input" / "0.png", target)
    config = load_config(
        overrides={
            "architecture": "trt-r2r-light",
            "trt_r2r_light": {
                "tasks": ["derain", "deblur"],
                "channels": [32, 32, 32, 32],
                "encoder_blocks": [1, 1, 1, 1],
            },
            "data": {"path": str(data), "crop_size": 16},
            "training": {
                "output_dir": str(tmp_path / "run"),
                "device": "cpu",
                "steps": 3,
                "batch_size": 1,
                "num_workers": 0,
            },
            "export": {"shape": [1, 3, 16, 16]},
            "quantization": {"calibration_batches": 3},
        }
    )
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER)
    return config, worker


def launch(config, worker, *, qat=False):
    path = worker.with_suffix(".json")
    path.write_text(json.dumps(config))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(worker),
            str(path),
            "qat" if qat else "train",
            str(worker.parent),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    record = json.loads(result.stdout)
    probes = [json.loads((worker.parent / f"rank-{rank}.json").read_text()) for rank in range(2)]
    return record, probes


def test_two_rank_training_bank_balance_rng_and_world_size_resume(distributed_case):
    config, worker = distributed_case
    result, probes = launch(config, worker)
    assert result["world_size"] == 2 and result["resumable"]
    assert probes[0]["initial_bank"] == probes[1]["initial_bank"]
    assert probes[0]["final_bank"] == probes[1]["final_bank"]
    assert probes[0]["initial_bank"] != probes[0]["final_bank"]
    saved = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    state = saved["training_state"]
    assert state["world_size"] == 2 and len(state["rank_states"]) == 2
    for rank_state in state["rank_states"]:
        assert rank_state["task_counts"] == {"clean": 1, "deblur": 1, "derain": 1}
        assert rank_state["rng"]["augmentation"]["children"]
    first, second = [rank_state["rng"] for rank_state in state["rank_states"]]
    assert not torch.equal(first["data_generator"], second["data_generator"])
    assert first["augmentation"]["python"] != second["augmentation"]["python"]
    files = {path.name for path in Path(config["training"]["output_dir"]).iterdir()}
    assert files == {"config.json", "environment.json", "dataset.json", "last.pt", "result.json"}
    resume = deepcopy(config)
    resume["training"]["resume"] = result["checkpoint"]
    with pytest.raises(ValueError, match="original world_size"):
        run_train(resume)


def test_two_rank_calibration_has_identical_frozen_bank_and_scales(distributed_case):
    pytest.importorskip("modelopt")
    config, worker = distributed_case
    source = worker.parent / "source.pt"
    model = build_model_from_config(config)
    torch.save(
        {**checkpoint_metadata(model, config), "config": config, "state_dict": model.state_dict()},
        source,
    )
    config["training"].update(checkpoint=str(source), steps=0)
    result, probes = launch(config, worker, qat=True)
    assert result["mode"] == "ptq"
    assert probes[0]["calibrated_bank"] == probes[1]["calibrated_bank"]
    assert all(probe["bank_frozen"] for probe in probes)
    assert all(probe["calibration_scale_count"] > 0 for probe in probes)
    assert probes[0]["calibration_scales"] == probes[1]["calibration_scales"]
    assert probes[0]["calibrated_bank"] == probes[0]["final_bank"]
    saved = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    assert saved["training_state"]["world_size"] == 2
    assert saved["modelopt_state"]
    assert saved["bank_fingerprint"] == probes[0]["calibrated_bank"]
    assert result["quantizers"] and all(
        item["calibrated"] for item in result["quantizers"] if item["enabled"]
    )
