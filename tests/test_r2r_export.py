# ruff: noqa: E402
import builtins
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from modern_nafnet.deployment.export import _r2r_graph_report, export_model, run_export
from modern_nafnet.models.trt_r2r_light import TrtR2RLight


def model_config():
    return {
        "architecture": "trt-r2r-light",
        "trt_r2r_light": {
            "tasks": ["derain", "deblur"],
            "channels": [32, 32, 32, 32],
            "encoder_blocks": [1, 0, 0, 0],
            "decoder_blocks": [1, 0, 0],
            "slots_per_task": 2,
            "query_dim": 8,
            "bank_enabled": True,
            "prior_scale_init": 0.1,
        },
    }


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stage_a_single_rgb_onnx_has_fixed_fp32_bank_and_matches_pytorch(
    tmp_path, quantized, device, monkeypatch
):
    from modern_nafnet.models.factory import checkpoint_metadata

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    config = model_config()
    torch.manual_seed(72)
    model = TrtR2RLight(**config["trt_r2r_light"]).to(device).eval()
    image = torch.rand(1, 3, 32, 32, device=device)
    checkpoint = {"config": deepcopy(config)}
    if quantized:
        pytest.importorskip("modelopt")
        from modern_nafnet.training.quantize import calibrate_int8, modelopt_state

        model = calibrate_int8(model, [image])
        checkpoint["modelopt_state"] = modelopt_state(model)
        checkpoint["training_state"] = {"calibration_shape": [1, 3, 32, 32]}
    checkpoint.update(checkpoint_metadata(model, config))
    checkpoint["state_dict"] = model.state_dict()
    path = tmp_path / "source.pt"
    torch.save(checkpoint, path)
    config["export"] = {
        "checkpoint": str(path),
        "output": str(tmp_path / "r2r.onnx"),
        "shape": [1, 3, 32, 32],
        "device": device,
        "quantized": quantized,
    }
    if not quantized:
        original_import = builtins.__import__

        def without_modelopt(name, *args, **kwargs):
            if name == "modelopt" or name.startswith("modelopt."):
                raise AssertionError("FP32 export must not import ModelOpt")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_modelopt)
    result = run_export(config)
    options = ort.SessionOptions()
    # Conv/ReLU fusion can cross Q/DQ rounding boundaries; compare raw ONNX semantics.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        result["output"], sess_options=options, providers=["CPUExecutionProvider"]
    )
    assert len(session.get_outputs()) == 1
    actual = session.run(None, {"input": image.cpu().numpy()})[0]
    np.testing.assert_allclose(actual, model(image).detach().cpu().numpy(), atol=3e-5, rtol=3e-5)
    assert result["architecture"] == "trt-r2r-light"
    assert result["task_order"] == ["derain", "deblur"]
    assert result["bank"]["frozen"]
    assert set(result["bank_initializers"]) == {"key", "value"}
    assert result["preprocess"] == {"pad_multiple": 16, "mode": "constant", "pad_value": 0}
    assert result["postprocess"]["crop_to_original_size"]
    assert result["padding"].startswith("external")
    assert result["add_boundaries"]
    assert result["quantization"]["skipped_conv_count"] == 2
    assert result["precision_policy"] == ("w8a8-conv-fp32-branch" if quantized else "fp32")
    assert not result["tensorrt_verified"]
    graph = onnx.load(result["output"])
    operators = {node.op_type for node in graph.graph.node}
    assert not operators & {"TopK", "ArgMax", "Gather", "Einsum", "QLinearConv", "ConvInteger"}
    assert {"MatMul", "Softmax", "Sigmoid"} <= operators
    assert bool(result["quantize_nodes"]) is quantized
    assert bool(result["dequantize_nodes"]) is quantized
    if quantized:
        assert sum(item["precision"] == "fp32" for item in result["conv_policy"]) == 2
        broken = deepcopy(graph)
        producers = {value: node for node in broken.graph.node for value in node.output}
        internal = next(
            node
            for node in broken.graph.node
            if node.op_type == "Conv"
            and producers.get(node.input[0], onnx.NodeProto()).op_type == "DequantizeLinear"
        )
        dequantize = producers[internal.input[0]]
        internal.input[0] = producers[dequantize.input[0]].input[0]
        with pytest.raises(RuntimeError, match="Q/DQ boundary"):
            _r2r_graph_report(broken, model, quantized=True)
        config["export"].update(shape=[1, 3, 16, 16], output=str(tmp_path / "mismatch.onnx"))
        with pytest.raises(ValueError, match="calibration_shape"):
            run_export(config)
        assert not (tmp_path / "mismatch.onnx").exists()


def test_export_requires_external_padding_and_stage_a(tmp_path):
    model = TrtR2RLight(**model_config()["trt_r2r_light"])
    with pytest.raises(ValueError, match="externally padded"):
        export_model(model, tmp_path / "bad.onnx", [1, 3, 17, 19])
    with pytest.raises(ValueError, match="Stage A"):
        export_model(model, tmp_path / "half.onnx", [1, 3, 32, 32], precision_policy="fp16")
