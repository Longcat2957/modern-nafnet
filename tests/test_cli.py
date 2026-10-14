import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from modern_nafnet.cli import app
from modern_nafnet.config import load_config

runner = CliRunner()


@pytest.mark.parametrize("command", [[], ["train"], ["qat"], ["export"], ["infer"], ["eval"]])
def test_help_without_ml_imports(command):
    code = (
        "import sys; from typer.testing import CliRunner; from modern_nafnet.cli import app; "
        f"r = CliRunner().invoke(app, {command + ['--help']!r}); "
        "assert r.exit_code == 0, r.output; "
        "assert not {'torch', 'modelopt', 'albumentations', 'onnx'}.intersection(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_false_and_steps_override(monkeypatch):
    import types

    stub = types.ModuleType("modern_nafnet.training.loop")
    stub.run_train = lambda config, **kwargs: config["training"]
    monkeypatch.setitem(sys.modules, "modern_nafnet.training.loop", stub)
    result = runner.invoke(app, ["train", "--steps", "2", "--no-amp"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["amp"] is False
    assert json.loads(result.stdout)["steps"] == 2
    assert result.stderr == ""


def test_bad_argument():
    result = runner.invoke(app, ["train", "--steps", "0"])
    assert result.exit_code == 2


def test_qat_cli_zero_steps_selects_calibration_only(monkeypatch):
    import types

    stub = types.ModuleType("modern_nafnet.training.loop")
    stub.run_train = lambda config, **kwargs: config["training"]
    monkeypatch.setitem(sys.modules, "modern_nafnet.training.loop", stub)
    result = runner.invoke(app, ["qat", "--steps", "0"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["steps"] == 0


def test_workflow_status_does_not_corrupt_json_stdout(monkeypatch):
    import types

    stub = types.ModuleType("modern_nafnet.training.loop")

    def operation(config):
        print("library status")
        return {"step": 1}

    stub.run_train = operation
    monkeypatch.setitem(sys.modules, "modern_nafnet.training.loop", stub)
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"step": 1}
    assert result.stderr == "library status\n"


def test_module_entrypoint_exposes_help():
    result = subprocess.run(
        [sys.executable, "-m", "modern_nafnet.cli", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "train" in result.stdout


@pytest.mark.parametrize("command", ["train", "qat"])
@pytest.mark.parametrize("rank", ["0", "1"])
def test_distributed_training_only_primary_rank_prints_result(monkeypatch, command, rank):
    import types

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", rank)
    stub = types.ModuleType("modern_nafnet.training.loop")

    def operation(config, **kwargs):
        print(f"rank {rank} library status")
        return {"step": 1}

    stub.run_train = operation
    monkeypatch.setitem(sys.modules, "modern_nafnet.training.loop", stub)
    result = runner.invoke(app, [command])
    assert result.exit_code == 0, result.output
    assert result.stderr == f"rank {rank} library status\n"
    if rank == "0":
        assert json.loads(result.stdout) == {"step": 1}
    else:
        assert result.stdout == ""


def test_distributed_nonprimary_rank_still_reports_errors(monkeypatch):
    import types

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    stub = types.ModuleType("modern_nafnet.training.loop")

    def operation(config):
        raise RuntimeError("worker failure")

    stub.run_train = operation
    monkeypatch.setitem(sys.modules, "modern_nafnet.training.loop", stub)
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "worker failure" in result.stderr


def test_qat_cli_checkpoint_overrides_quantization_source(tmp_path, monkeypatch):
    import types

    source = tmp_path / "override.pt"
    source.write_bytes(b"fixture")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"quantization": {"checkpoint": "configured.pt"}}))
    stub = types.ModuleType("modern_nafnet.training.loop")
    stub.run_train = lambda config, **kwargs: config["quantization"]
    monkeypatch.setitem(sys.modules, "modern_nafnet.training.loop", stub)
    result = runner.invoke(app, ["qat", "--config", str(path), "--checkpoint", str(source)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["checkpoint"] == str(source)


@pytest.mark.parametrize("command", ["train", "qat"])
@pytest.mark.parametrize(
    "flag,enabled", [("--synthetic-blur", True), ("--no-synthetic-blur", False)]
)
def test_synthetic_blur_and_zero_workers_override_json(
    tmp_path, monkeypatch, command, flag, enabled
):
    import types

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {"data": {"synthetic_blur": {"enabled": not enabled}}, "training": {"num_workers": 3}}
        )
    )
    stub = types.ModuleType("modern_nafnet.training.loop")
    stub.run_train = lambda config, **kwargs: config
    monkeypatch.setitem(sys.modules, "modern_nafnet.training.loop", stub)
    result = runner.invoke(app, [command, "--config", str(path), flag, "--num-workers", "0"])
    assert result.exit_code == 0, result.output
    config = json.loads(result.stdout)
    assert config["data"]["synthetic_blur"]["enabled"] is enabled
    assert config["training"]["num_workers"] == 0


@pytest.mark.parametrize("command", ["train", "qat", "eval"])
@pytest.mark.parametrize("override", [False, True])
def test_data_root_cli_override_and_none_preserves_config(tmp_path, monkeypatch, command, override):
    import types

    configured = tmp_path / "configured"
    configured.mkdir()
    selected = tmp_path / "selected"
    selected.mkdir()
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "data": {"path": str(configured), "train": "clean"},
                "training": {"num_workers": 2},
                "evaluation": {"split": "test"},
            }
        )
    )
    module = (
        "modern_nafnet.evaluation.quality" if command == "eval" else "modern_nafnet.training.loop"
    )
    stub = types.ModuleType(module)
    operation = "run_evaluate" if command == "eval" else "run_train"
    setattr(stub, operation, lambda config, **kwargs: config)
    monkeypatch.setitem(sys.modules, module, stub)
    arguments = [command, "--config", str(path)]
    if override:
        arguments += ["--data", str(selected)]
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    config = json.loads(result.stdout)
    assert config["data"]["path"] == str(selected if override else configured)
    assert config["data"]["train"] == "clean"
    assert config["training"]["num_workers"] == 2
    assert config["evaluation"]["split"] == "test"


@pytest.mark.parametrize("split,expected", [(None, "test"), ("val", "val"), ("test", "test")])
def test_eval_split_cli_override_and_none_preserves_config(tmp_path, monkeypatch, split, expected):
    import types

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"evaluation": {"split": "test"}}))
    stub = types.ModuleType("modern_nafnet.evaluation.quality")
    stub.run_evaluate = lambda config: config["evaluation"]
    monkeypatch.setitem(sys.modules, "modern_nafnet.evaluation.quality", stub)
    arguments = ["eval", "--config", str(path)]
    if split is not None:
        arguments += ["--split", split]
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["split"] == expected
    assert load_config()["evaluation"]["split"] == "val"


@pytest.mark.parametrize(
    "removed",
    [
        {"data": {"train_manifest": "removed.json"}},
        {"data": {"calibration_manifest": "removed.json"}},
        {"evaluation": {"manifest": "removed.json"}},
        "--manifest",
    ],
)
def test_removed_input_manifest_configuration_and_option_are_rejected(tmp_path, removed):
    if isinstance(removed, dict):
        path = tmp_path / "invalid-config.json"
        path.write_text(json.dumps(removed))
        with pytest.raises(ValueError, match="Unknown configuration"):
            load_config(path)
    else:
        result = runner.invoke(app, ["eval", removed, "removed.json"])
        assert result.exit_code == 2
        assert "No such option" in result.output
        assert result.stdout == ""


def test_missing_dependency_reports_profile_on_stderr(monkeypatch):
    import types

    stub = types.ModuleType("modern_nafnet.training.loop")

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    stub.run_train = missing
    monkeypatch.setitem(sys.modules, "modern_nafnet.training.loop", stub)
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 1
    assert "--extra train" in result.stderr
    assert result.stdout == ""


def test_config_none_preserves_and_false_zero_override(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"training": {"amp": true, "steps": 5}}')
    config = load_config(path, {"training": {"amp": False, "steps": 0, "device": None}})
    assert config["training"]["amp"] is False
    assert config["training"]["steps"] == 0
    assert config["training"]["device"] == "cuda"
    assert load_config()["training"]["steps"] == 1000


def test_unknown_config_fails(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"training": {"stepz": 5}}')
    with pytest.raises(ValueError, match="training.stepz"):
        load_config(path)


@pytest.mark.parametrize(
    "document",
    [
        {"training": {"amp": "false"}},
        {"training": {"steps": True}},
        {"training": {"lr": float("nan")}},
        {"export": {"quantized": "false"}},
        {"export": {"shape": [1, 3, "8", 8]}},
        {"inference": {"checkpoint": {"path": "weights.pt"}}},
        {"model": {"enc_blk_nums": [1, 1.0]}},
        {"model": {"train_size": [1, 3, 0, 256]}},
        {"data": {"synthetic_blur": {"enabled": "true"}}},
        {"data": {"synthetic_blur": {"blur_range": [3, 8]}}},
        {"data": {"synthetic_blur": {"blur_range": [9, 3]}}},
        {"data": {"synthetic_blur": {"blur_range": [3]}}},
    ],
)
def test_json_configuration_rejects_wrong_leaf_types(tmp_path, document):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Configuration"):
        load_config(path)


def test_default_json_matches_defaults():
    from pathlib import Path

    from modern_nafnet.config import DEFAULT_CONFIG

    path = Path(__file__).parents[1] / "configs/default.json"
    assert json.loads(path.read_text()) == DEFAULT_CONFIG
