# ruff: noqa: E402
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("modelopt")

from modern_nafnet.data.images import write_rgb
from modern_nafnet.models.trt_r2r_light import TrtR2RLight
from modern_nafnet.training.quantize import (
    calibrate_int8,
    calibration_inputs,
    check_int8_convs,
    modelopt_state,
    restore_quantized_model,
)


def small_model():
    return TrtR2RLight(
        tasks=["derain", "deblur"],
        channels=[32, 32, 32, 32],
        encoder_blocks=[1, 0, 0, 0],
        decoder_blocks=[1, 0, 0],
        slots_per_task=2,
        query_dim=8,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stage_a_conv_policy_frozen_bank_backward_and_strict_restore(tmp_path, device):
    from modern_nafnet.models.factory import checkpoint_metadata

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    torch.manual_seed(71)
    model = small_model().to(device)
    image = torch.rand(1, 3, 32, 32, device=device)
    model = calibrate_int8(model, [image])
    report = {item["name"]: item for item in check_int8_convs(model)}
    for name in ("stem", "tail", "query_head", "bank.task_gate"):
        assert not report[f"{name}.input_quantizer"]["enabled"]
        assert not report[f"{name}.weight_quantizer"]["enabled"]
    assert report["downs.0.input_quantizer"]["axis"] is None
    assert report["downs.0.input_quantizer"]["amax_shape"] == []
    assert report["downs.0.weight_quantizer"]["axis"] == 0
    assert report["downs.0.weight_quantizer"]["amax_shape"] == [32, 1, 1, 1]
    assert report["downs.0.weight_quantizer"]["amax_finite"]
    key, value = model.bank.key.detach().clone(), model.bank.value.detach().clone()
    scales = {name: value.clone() for name, value in model.state_dict().items() if "_amax" in name}
    model.train()(image).square().mean().backward()
    assert model.downs[0].weight.grad is not None
    assert torch.isfinite(model.downs[0].weight.grad).all()
    assert model.bank.key.grad is None and model.bank.value.grad is None
    torch.testing.assert_close(model.bank.key, key, atol=0, rtol=0)
    torch.testing.assert_close(model.bank.value, value, atol=0, rtol=0)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in scales.items())
    config = {"architecture": model.architecture, "trt_r2r_light": model.model_spec}
    checkpoint = {
        **checkpoint_metadata(model, config),
        "state_dict": model.state_dict(),
        "modelopt_state": modelopt_state(model),
        "config": config,
    }
    path = tmp_path / "qat.pt"
    torch.save(checkpoint, path)
    restored = restore_quantized_model(
        small_model(), torch.load(path, weights_only=True, map_location="cpu"), config
    ).to(device)
    torch.testing.assert_close(restored.eval()(image), model.eval()(image), atol=1e-7, rtol=1e-7)
    restored.train()(image).mean().backward()
    assert restored.downs[0].weight.grad is not None
    tampered = deepcopy(checkpoint)
    tampered["task_order"] = list(reversed(tampered["task_order"]))
    with pytest.raises(ValueError, match="task|Task"):
        restore_quantized_model(small_model(), tampered, config)
    tampered = deepcopy(checkpoint)
    tampered["state_dict"]["bank.key"][0, 0] += 1
    with pytest.raises(ValueError, match="fingerprint|Bank"):
        restore_quantized_model(small_model(), tampered, config)


def test_stage_a_rejects_enabled_bank_quantizers_and_wrong_amax_shape():
    model = calibrate_int8(small_model(), [torch.rand(1, 3, 16, 16)])
    model.bank.task_gate.input_quantizer.enable()
    with pytest.raises(RuntimeError, match="Unexpected Stage A quantizer"):
        check_int8_convs(model)
    model.bank.task_gate.input_quantizer.disable()
    model.downs[0].input_quantizer._buffers["_amax"] = torch.ones(2)
    with pytest.raises(RuntimeError, match="amax shape"):
        check_int8_convs(model)


def test_restoration_calibration_preserves_rgb_and_external_padding(tmp_path):
    image = np.full((17, 19, 3), [20, 90, 160], dtype=np.uint8)
    path = tmp_path / "derain/input/first.png"
    write_rgb(path, image)
    actual = next(
        calibration_inputs(
            tmp_path,
            [1, 3, 32, 32],
            "cpu",
            1,
            samples=[{"input": path, "task": "derain"}],
            architecture="trt-r2r-light",
        )
    )
    assert actual.shape == (1, 3, 32, 32)
    torch.testing.assert_close(actual[0, :, 0, 0], torch.tensor([20, 90, 160]) / 255)
    assert torch.count_nonzero(actual[:, :, 17:, :]) == 0
    assert torch.count_nonzero(actual[:, :, :, 19:]) == 0
    with pytest.raises(ValueError, match="prepare deployment-sized images explicitly"):
        list(
            calibration_inputs(
                tmp_path,
                [1, 3, 48, 48],
                "cpu",
                1,
                samples=[{"input": path}],
                architecture="trt-r2r-light",
            )
        )


def test_nonfinite_bank_is_rejected_before_calibration():
    model = small_model()
    with torch.no_grad():
        model.bank.key[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN|Inf"):
        calibrate_int8(model, [torch.rand(1, 3, 16, 16)])
