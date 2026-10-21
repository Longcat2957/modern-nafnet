"""Fixed batch-one FP and ModelOpt Q/DQ ONNX export on the training host."""

import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from modern_nafnet.artifacts import environment_manifest, sha256, write_json


def _r2r_graph_report(graph, model: torch.nn.Module, *, quantized: bool = False) -> dict:
    """Reject dynamic retrieval and verify the exported, immutable FP32 prototypes."""
    import numpy as np
    from onnx import TensorProto, numpy_helper

    forbidden = {
        "TopK",
        "ArgMax",
        "ArgMin",
        "Gather",
        "GatherElements",
        "GatherND",
        "Einsum",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "QLinearConv",
        "ConvInteger",
    }
    found = sorted({node.op_type for node in graph.graph.node} & forbidden)
    if found:
        raise RuntimeError(f"TRT-R2R-Light export contains forbidden operators: {found}")
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise RuntimeError("TRT-R2R-Light export must have one RGB input and one RGB output")
    for value in (*graph.graph.input, *graph.graph.output):
        if value.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise RuntimeError("Stage A RGB input and output must be FP32")
    if any(value.data_type == TensorProto.FLOAT16 for value in graph.graph.initializer):
        raise RuntimeError("Stage A must not contain FP16 initializers")
    producers = {output: node.op_type for node in graph.graph.node for output in node.output}
    nodes_by_output = {output: node for node in graph.graph.node for output in node.output}
    conv_policy = []
    if quantized:
        expected = {
            name for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)
        }
        seen = set()
        for node in graph.graph.node:
            if node.op_type != "Conv":
                continue
            weight = node.input[1]
            for _ in range(2):
                producer = nodes_by_output.get(weight)
                if producer is not None and producer.op_type in {
                    "QuantizeLinear",
                    "DequantizeLinear",
                }:
                    weight = producer.input[0]
            module_name = weight.removesuffix(".weight")
            if module_name not in expected or module_name in seen:
                raise RuntimeError(f"Unrecognized or duplicate exported Conv: {node.name}")
            seen.add(module_name)
            excluded = module_name in {"stem", "tail"}
            for role, value in zip(("input", "weight"), node.input[:2], strict=True):
                dequantize = nodes_by_output.get(value)
                is_dq = dequantize is not None and dequantize.op_type == "DequantizeLinear"
                if is_dq == excluded:
                    raise RuntimeError(f"Invalid Q/DQ boundary on Conv {module_name}.{role}")
                if is_dq:
                    quantize = nodes_by_output.get(dequantize.input[0])
                    if quantize is None or quantize.op_type != "QuantizeLinear":
                        raise RuntimeError(f"Missing QuantizeLinear on Conv {module_name}.{role}")
            conv_policy.append(
                {
                    "module": module_name,
                    "node": node.name,
                    "precision": "fp32" if excluded else "w8a8",
                }
            )
        if seen != expected:
            raise RuntimeError(f"Missing exported Conv modules: {sorted(expected - seen)}")
    consumers = {}
    for node in graph.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node.op_type)
    add_boundaries = [
        {
            "name": node.name,
            "input_operators": [producers.get(value, "initializer/input") for value in node.input],
            "output_operators": [
                kind for value in node.output for kind in consumers.get(value, [])
            ],
        }
        for node in graph.graph.node
        if node.op_type == "Add"
    ]
    bank_initializers = {}
    if model.bank is not None:
        for role in ("key", "value"):
            expected = getattr(model.bank, role).detach().cpu().numpy()
            for initializer in graph.graph.initializer:
                array = numpy_helper.to_array(initializer)
                if (
                    array.shape == expected.shape
                    and np.array_equal(array, expected)
                    or array.shape == expected.T.shape
                    and np.array_equal(array, expected.T)
                ):
                    if initializer.data_type != TensorProto.FLOAT:
                        raise RuntimeError(f"Stage A Bank {role} initializer must be FP32")
                    bank_initializers[role] = initializer.name
                    break
            else:
                raise RuntimeError(f"The exported graph is missing the fixed Bank {role}")
    return {
        "bank_initializers": bank_initializers,
        "forbidden_operators": found,
        "add_boundaries": add_boundaries,
        "conv_policy": conv_policy,
    }


def export_model(
    model: torch.nn.Module,
    output: str | Path,
    shape: list[int],
    *,
    quantized: bool = False,
    opset: int = 18,
    device: str = "cpu",
    precision_policy: str | None = None,
) -> dict:
    import onnx

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Export output already exists; choose a new path: {output}")
    if (
        len(shape) != 4
        or shape[:2] != [1, 3]
        or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in shape)
    ):
        raise ValueError("Export shape must be fixed [1, 3, height, width]")
    if opset < 13:
        raise ValueError("ONNX export needs opset >= 13")
    architecture = getattr(model, "architecture", "nafnet")
    r2r = architecture == "trt-r2r-light"
    if r2r:
        from modern_nafnet.training.quantize import R2R_PRECISION_POLICY

        if precision_policy not in {None, R2R_PRECISION_POLICY, "fp32"}:
            raise ValueError("Only TRT-R2R-Light precision Stage A is implemented")
        if shape[2] % 16 or shape[3] % 16:
            raise ValueError("TRT-R2R-Light export shape must be externally padded to multiple 16")
        model.validate_bank()
        model.freeze_bank()
    if quantized:
        from modern_nafnet.training.quantize import check_int8_convs

        check_int8_convs(model)
    # Stage A deliberately keeps the non-quantized branches and Q/DQ scales FP32.
    model = model.to(device).float().eval()
    dummy = torch.rand(shape, generator=torch.Generator().manual_seed(0)).to(device)
    with torch.inference_mode():
        expected = model(dummy)
    if not isinstance(expected, torch.Tensor) or list(expected.shape) != shape:
        raise ValueError("Export model must return one RGB tensor matching its input shape")
    if not bool(torch.isfinite(expected).all()):
        raise FloatingPointError("Export warm-up output contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".export-") as directory:
        staged = Path(directory) / "model.onnx"
        with torch.no_grad(), redirect_stdout(sys.stderr):
            torch.onnx.export(
                model,
                dummy,
                str(staged),
                input_names=["input"],
                output_names=["output"],
                opset_version=opset,
                dynamo=not quantized,
                external_data=False,
            )
        onnx.checker.check_model(str(staged))
        graph = onnx.load(str(staged))
        quantize = sum(node.op_type == "QuantizeLinear" for node in graph.graph.node)
        dequantize = sum(node.op_type == "DequantizeLinear" for node in graph.graph.node)
        if quantized and (not quantize or not dequantize):
            raise RuntimeError("The exported INT8 graph is missing QuantizeLinear/DequantizeLinear")
        if not quantized and (quantize or dequantize):
            raise RuntimeError("Floating-point export unexpectedly contains Q/DQ nodes")
        graph_report = _r2r_graph_report(graph, model, quantized=quantized) if r2r else {}
        staged.replace(output)
    return {
        "output": str(output),
        "sha256": sha256(output),
        "shape": shape,
        "dtype": "float32",
        "quantized": quantized,
        "exporter": "torchscript (ModelOpt INT8 symbolic)" if quantized else "dynamo",
        "opsets": {op.domain: op.version for op in graph.opset_import},
        "quantize_nodes": quantize,
        "dequantize_nodes": dequantize,
        "tensorrt_verified": False,
        **graph_report,
    }


def run_export(config: dict) -> dict:
    from modern_nafnet.models.factory import (
        bank_fingerprint,
        build_model_from_config,
        check_checkpoint_metadata,
        load_checkpoint,
    )
    from modern_nafnet.models.nafnet import tlc_kernels

    settings = config["export"]
    if not settings.get("checkpoint"):
        raise ValueError("export.checkpoint is required")
    if not settings.get("shape"):
        raise ValueError("export.shape is required and must be a fixed batch-one shape")
    output = Path(settings["output"])
    manifest = output.with_suffix(".onnx.json")
    source = Path(settings["checkpoint"]).resolve()
    if any(path.resolve() == source for path in (output, manifest)):
        raise ValueError("ONNX output and manifest must not overwrite the source checkpoint")
    if manifest.exists():
        raise FileExistsError(f"Export manifest already exists; choose a new path: {manifest}")
    model = build_model_from_config(config)
    calibration = None
    if settings.get("quantized", False):
        from modern_nafnet.training.quantize import restore_quantized_model

        checkpoint = torch.load(settings["checkpoint"], map_location="cpu", weights_only=True)
        check_checkpoint_metadata(model, checkpoint, config)
        if config.get("architecture", "nafnet") == "trt-r2r-light":
            state = checkpoint.get("training_state", {})
            if state.get("calibration_shape") != settings["shape"]:
                raise ValueError(
                    "TRT-R2R-Light Q/DQ export shape must match the saved calibration_shape; "
                    "calibrate separately for each deployment shape"
                )
            calibration = {
                "shape": state["calibration_shape"],
                "dataset": state.get("calibration_dataset"),
            }
        model = restore_quantized_model(model, checkpoint, config)
    else:
        load_checkpoint(model, settings["checkpoint"], config)
    result = export_model(
        model,
        settings["output"],
        settings["shape"],
        quantized=settings.get("quantized", False),
        opset=settings.get("opset", 18),
        device=settings.get("device", "cuda"),
        precision_policy=config.get("quantization", {}).get("precision_policy"),
    )
    if config.get("architecture", "nafnet") == "trt-r2r-light":
        from modern_nafnet.training.quantize import (
            R2R_PRECISION_POLICY,
            quantizer_report,
        )

        bank = model.bank
        quantized = settings.get("quantized", False)
        report = quantizer_report(model) if quantized else []
        skipped = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Conv2d) and name in {"stem", "tail"}
        ]
        result.update(
            {
                "architecture": "trt-r2r-light",
                "model_revision": "v0.1",
                "task_order": list(model.tasks),
                "model_spec": model.model_spec,
                "bank_fingerprint": bank_fingerprint(model),
                "bank": {
                    "enabled": model.bank_enabled,
                    "slots_per_task": model.model_spec["slots_per_task"],
                    "key_dim": model.model_spec["query_dim"],
                    "value_dim": model.model_spec["channels"][-1],
                    "frozen": bank is None
                    or not (bank.key.requires_grad or bank.value.requires_grad),
                },
                "precision_policy": R2R_PRECISION_POLICY if quantized else "fp32",
                "precision_stage": "A",
                "calibration": calibration,
                "input": {
                    "shape": settings["shape"],
                    "dtype": "float32",
                    "layout": "NCHW",
                    "color": "RGB",
                    "range": [0, 1],
                },
                "preprocess": {"pad_multiple": 16, "mode": "constant", "pad_value": 0},
                "postprocess": {"crop_to_original_size": True},
                "padding": "external right/bottom zero padding to a multiple of 16",
                "quantization": {
                    "format": "ONNX-QDQ" if quantized else "none",
                    "weight_axis": 0 if quantized else None,
                    "activation_axis": None,
                    "quantizers": report,
                    "skipped_conv_layers": skipped,
                    "skipped_conv_count": len(skipped),
                    "typing": "strongly-typed ONNX types; no builder precision flags",
                    "residual_add": "FP32; Conv input/weight Q/DQ boundaries only",
                },
            }
        )
    result.update(
        {
            "checkpoint_sha256": sha256(settings["checkpoint"]),
            "model": config.get(
                "trt_r2r_light" if config.get("architecture") == "trt-r2r-light" else "model", {}
            ),
            "tlc_kernels": tlc_kernels(model),
            "preprocessing": {"layout": "NCHW", "color": "RGB", "range": [0, 1]},
            "padding": (
                "external right/bottom zero padding to a multiple of 16"
                if config.get("architecture") == "trt-r2r-light"
                else f"zero padding to a multiple of {getattr(model, 'padder_size', 1)} inside the model"
            ),
            "crop": "crop the model output to the original input height and width",
            "environment": environment_manifest(),
            "torch_cuda": torch.version.cuda,
            "device": settings.get("device", "cuda"),
        }
    )
    write_json(manifest, result)
    result["manifest"] = str(manifest)
    return result
