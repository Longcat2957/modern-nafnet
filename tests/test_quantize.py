# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("modelopt")

from modern_nafnet.data.images import write_rgb
from modern_nafnet.training.quantize import (
    calibrate_int8,
    calibration_inputs,
    check_int8_convs,
    modelopt_state,
    restore_quantized_model,
)


class AttentionConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.sca = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Conv2d(3, 3, 1))

    def forward(self, image):
        value = self.conv(image)
        return value * self.sca(value)


def test_calibration_gradient_frozen_scales_and_safe_restore(tmp_path):
    torch.manual_seed(3)
    image = torch.rand(1, 3, 8, 8)
    model = calibrate_int8(AttentionConv(), [image])
    report = check_int8_convs(model)
    assert any(item["name"] == "conv.input_quantizer" and item["axis"] is None for item in report)
    assert any(item["name"] == "conv.weight_quantizer" and item["axis"] == 0 for item in report)
    assert not model.sca[1].input_quantizer.is_enabled
    scales = {
        name: value.clone() for name, value in model.state_dict().items() if name.endswith("_amax")
    }
    model.train()(image).square().mean().backward()
    assert model.conv.weight.grad is not None
    assert torch.isfinite(model.conv.weight.grad).all()
    assert model.conv.weight.grad.abs().sum() > 0
    assert all(torch.equal(model.state_dict()[name], value) for name, value in scales.items())
    checkpoint = {"state_dict": model.state_dict(), "modelopt_state": modelopt_state(model)}
    path = tmp_path / "qat.pt"
    torch.save(checkpoint, path)
    restored = restore_quantized_model(AttentionConv(), torch.load(path, weights_only=True))
    torch.testing.assert_close(restored.eval()(image), model.eval()(image), atol=1e-7, rtol=1e-7)
    restored.train()(image).mean().backward()
    assert restored.conv.weight.grad is not None


def test_empty_calibration_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        calibrate_int8(torch.nn.Conv2d(3, 3, 1), [])


def test_calibration_folder_reads_only_real_blur_with_limit_and_rgb_range(tmp_path):
    image = np.empty((8, 8, 3), dtype=np.uint8)
    image[:] = [20, 90, 160]
    write_rgb(tmp_path / "blur/first.png", image)
    write_rgb(tmp_path / "blur/second.png", np.full_like(image, 200))
    write_rgb(tmp_path / "sharp/first.png", np.full_like(image, 255))
    inputs = list(calibration_inputs(tmp_path, [1, 3, 8, 8], "cpu", 1))
    assert len(inputs) == 1
    assert inputs[0].shape == (1, 3, 8, 8)
    assert inputs[0].dtype == torch.float32
    assert inputs[0].is_contiguous()
    expected = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255
    torch.testing.assert_close(inputs[0][0], expected)


def test_calibration_folder_rejects_implicit_resize(tmp_path):
    write_rgb(tmp_path / "image.png", np.full((9, 8, 3), 70, dtype=np.uint8))
    with pytest.raises(ValueError, match="prepare deployment-sized images explicitly"):
        list(calibration_inputs(tmp_path, [1, 3, 8, 8], "cpu", 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is unavailable")
def test_nafnet_cuda_calibration_and_qat_backward():
    from modern_nafnet.models.nafnet import NAFNetLocal

    torch.manual_seed(21)
    model = NAFNetLocal(
        width=2,
        enc_blk_nums=[1],
        middle_blk_num=1,
        dec_blk_nums=[1],
        train_size=[1, 3, 8, 8],
    ).cuda()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("beta", "gamma")):
                parameter.fill_(0.1)
    image = torch.rand(1, 3, 8, 8, device="cuda")
    model = calibrate_int8(model, [image])
    scales = {
        name: value.clone() for name, value in model.state_dict().items() if name.endswith("_amax")
    }
    model.train()(image).square().mean().backward()
    assert model.intro.weight.grad.device.type == "cuda"
    assert model.intro.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert all(torch.equal(model.state_dict()[name], value) for name, value in scales.items())
