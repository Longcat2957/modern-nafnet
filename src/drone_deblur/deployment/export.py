"""Fixed batch-one FP and ModelOpt Q/DQ ONNX export on the training host."""

import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from drone_deblur.artifacts import environment_manifest, sha256, write_json


def export_model(
    model: torch.nn.Module,
    output: str | Path,
    shape: list[int],
    *,
    quantized: bool = False,
    opset: int = 18,
    device: str = "cpu",
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
    if quantized:
        from drone_deblur.training.quantize import check_int8_convs

        check_int8_convs(model)
    model = model.to(device).float().eval()
    dummy = torch.rand(shape, generator=torch.Generator().manual_seed(0)).to(device)
    with torch.inference_mode():
        expected = model(dummy)
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
    }


def run_export(config: dict) -> dict:
    from drone_deblur.models.nafnet import (
        NAFNetLocal,
        check_checkpoint_metadata,
        load_checkpoint,
        tlc_kernels,
    )

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
    model = NAFNetLocal(**config.get("model", {}))
    if settings.get("quantized", False):
        from drone_deblur.training.quantize import restore_quantized_model

        checkpoint = torch.load(settings["checkpoint"], map_location="cpu", weights_only=True)
        check_checkpoint_metadata(model, checkpoint, config.get("model", {}))
        model = restore_quantized_model(model, checkpoint)
    else:
        metadata = load_checkpoint(model, settings["checkpoint"])
        check_checkpoint_metadata(model, metadata, config.get("model", {}))
    result = export_model(
        model,
        settings["output"],
        settings["shape"],
        quantized=settings.get("quantized", False),
        opset=settings.get("opset", 18),
        device=settings.get("device", "cuda"),
    )
    result.update(
        {
            "checkpoint_sha256": sha256(settings["checkpoint"]),
            "model": config.get("model", {}),
            "tlc_kernels": tlc_kernels(model),
            "preprocessing": {"layout": "NCHW", "color": "RGB", "range": [0, 1]},
            "padding": f"zero padding to a multiple of {getattr(model, 'padder_size', 1)} inside the model",
            "crop": "crop the model output to the original input height and width",
            "environment": environment_manifest(),
            "torch_cuda": torch.version.cuda,
            "device": settings.get("device", "cuda"),
        }
    )
    write_json(manifest, result)
    result["manifest"] = str(manifest)
    return result
