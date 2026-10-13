import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from drone_deblur.cli import app
from drone_deblur.config import load_config

runner = CliRunner()


@pytest.mark.parametrize("command", [[], ["train"], ["qat"], ["export"], ["infer"], ["eval"]])
def test_help_without_ml_imports(command):
    code = (
        "import sys; from typer.testing import CliRunner; from drone_deblur.cli import app; "
        f"r = CliRunner().invoke(app, {command + ['--help']!r}); "
        "assert r.exit_code == 0, r.output; "
        "assert not {'torch', 'modelopt', 'albumentations', 'onnx'}.intersection(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_false_and_steps_override(monkeypatch):
    import types

    stub = types.ModuleType("drone_deblur.training.loop")
    stub.run_train = lambda config, **kwargs: config["training"]
    monkeypatch.setitem(sys.modules, "drone_deblur.training.loop", stub)
    result = runner.invoke(app, ["train", "--steps", "2", "--no-amp"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["amp"] is False
    assert json.loads(result.stdout)["steps"] == 2
    assert result.stderr == ""


def test_bad_argument():
    result = runner.invoke(app, ["train", "--steps", "0"])
    assert result.exit_code == 2


def test_workflow_status_does_not_corrupt_json_stdout(monkeypatch):
    import types

    stub = types.ModuleType("drone_deblur.training.loop")

    def operation(config):
        print("library status")
        return {"step": 1}

    stub.run_train = operation
    monkeypatch.setitem(sys.modules, "drone_deblur.training.loop", stub)
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"step": 1}
    assert result.stderr == "library status\n"


def test_qat_cli_checkpoint_overrides_quantization_source(tmp_path, monkeypatch):
    import types

    source = tmp_path / "override.pt"
    source.write_bytes(b"fixture")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"quantization": {"checkpoint": "configured.pt"}}))
    stub = types.ModuleType("drone_deblur.training.loop")
    stub.run_train = lambda config, **kwargs: config["quantization"]
    monkeypatch.setitem(sys.modules, "drone_deblur.training.loop", stub)
    result = runner.invoke(app, ["qat", "--config", str(path), "--checkpoint", str(source)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["checkpoint"] == str(source)


def test_missing_dependency_reports_profile_on_stderr(monkeypatch):
    import types

    stub = types.ModuleType("drone_deblur.training.loop")

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    stub.run_train = missing
    monkeypatch.setitem(sys.modules, "drone_deblur.training.loop", stub)
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
    ],
)
def test_json_configuration_rejects_wrong_leaf_types(tmp_path, document):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Configuration"):
        load_config(path)


def test_default_json_matches_defaults():
    from pathlib import Path

    from drone_deblur.config import DEFAULT_CONFIG

    path = Path(__file__).parents[1] / "configs/default.json"
    assert json.loads(path.read_text()) == DEFAULT_CONFIG
