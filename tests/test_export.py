# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from drone_deblur.deployment.export import export_model, run_export


@pytest.mark.parametrize("quantized", [False, True])
def test_fixed_onnx_output_matches_pytorch(tmp_path, quantized):
    torch.manual_seed(9)
    model = torch.nn.Conv2d(3, 3, 3, padding=1).eval()
    image = torch.rand(1, 3, 8, 8)
    if quantized:
        pytest.importorskip("modelopt")
        from drone_deblur.training.quantize import calibrate_int8

        model = calibrate_int8(model, [image])
    else:
        pytest.importorskip("onnxscript")
    result = export_model(model, tmp_path / "model.onnx", [1, 3, 8, 8], quantized=quantized)
    session = ort.InferenceSession(result["output"], providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": image.numpy()})[0]
    expected = model(image).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert session.get_inputs()[0].shape == [1, 3, 8, 8]
    assert bool(result["quantize_nodes"]) is quantized
    assert bool(result["dequantize_nodes"]) is quantized
    assert not result["tensorrt_verified"]


@pytest.mark.parametrize("as_manifest", [False, True])
def test_export_rejects_overwriting_source_checkpoint(tmp_path, as_manifest):
    path = tmp_path / ("source.onnx.json" if as_manifest else "source.pt")
    torch.save({}, path)
    output = tmp_path / "source.onnx" if as_manifest else path
    config = {"export": {"checkpoint": str(path), "output": str(output), "shape": [1, 3, 8, 8]}}
    with pytest.raises(ValueError, match="overwrite the source checkpoint"):
        run_export(config)
    assert torch.load(path, weights_only=True) == {}


def test_existing_export_manifest_is_preserved(tmp_path):
    path = tmp_path / "model.onnx.json"
    path.write_bytes(b"previous manifest")
    config = {
        "export": {
            "checkpoint": str(tmp_path / "source.pt"),
            "output": str(tmp_path / "model.onnx"),
            "shape": [1, 3, 8, 8],
        }
    }
    with pytest.raises(FileExistsError, match="manifest already exists"):
        run_export(config)
    assert path.read_bytes() == b"previous manifest"
    assert not (tmp_path / "model.onnx").exists()


def test_existing_export_is_preserved(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"previous artifact")
    with pytest.raises(FileExistsError, match="already exists"):
        export_model(torch.nn.Conv2d(3, 3, 1), path, [1, 3, 8, 8])
    assert path.read_bytes() == b"previous artifact"


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("height,width", [(8, 8), (17, 19)])
def test_nafnet_tlc_global_and_local_onnx_outputs(tmp_path, quantized, height, width):
    from drone_deblur.models.nafnet import NAFNetLocal

    torch.manual_seed(19)
    model = NAFNetLocal(
        width=2,
        enc_blk_nums=[1],
        middle_blk_num=1,
        dec_blk_nums=[1],
        train_size=[1, 3, 8, 8],
    ).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("beta", "gamma")):
                parameter.fill_(0.1)
    image = torch.rand(1, 3, height, width)
    if quantized:
        pytest.importorskip("modelopt")
        from drone_deblur.training.quantize import calibrate_int8

        model = calibrate_int8(model, [image])
    result = export_model(
        model,
        tmp_path / "nafnet.onnx",
        [1, 3, height, width],
        quantized=quantized,
    )
    session = ort.InferenceSession(result["output"], providers=["CPUExecutionProvider"])
    expected = model(image).detach().numpy()
    actual = session.run(None, {"input": image.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    assert actual.shape == (1, 3, height, width)
    assert bool(result["quantize_nodes"]) is quantized
