import hashlib
import json

from drone_deblur.artifacts import sha256, write_json


def test_hash_and_standard_json(tmp_path):
    source = tmp_path / "weights"
    source.write_bytes(b"checkpoint")
    assert sha256(source) == hashlib.sha256(b"checkpoint").hexdigest()
    output = tmp_path / "nested/result.json"
    write_json(output, {"path": source, "psnr": float("inf")})
    assert json.loads(output.read_text()) == {"path": str(source), "psnr": "inf"}
    assert not output.with_suffix(".json.tmp").exists()


def test_gpu_metadata_timeout_does_not_abort_artifact_recording(monkeypatch):
    from drone_deblur import artifacts

    monkeypatch.setattr(artifacts.shutil, "which", lambda name: "/mock/nvidia-smi")

    real_run = artifacts.subprocess.run

    def timeout(command, *args, **kwargs):
        if command[0] == "nvidia-smi":
            raise artifacts.subprocess.TimeoutExpired("nvidia-smi", 5)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(artifacts.subprocess, "run", timeout)
    result = artifacts.environment_manifest()
    assert result["nvidia_smi"] is None
    assert result["python"] and result["platform"]
