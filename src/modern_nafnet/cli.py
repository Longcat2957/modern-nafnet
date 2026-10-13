"""Thin Typer commands; optional ML dependencies load on execution only."""

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console

from modern_nafnet.artifacts import json_safe
from modern_nafnet.config import load_config

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console(stderr=True)
ConfigPath = Annotated[Path | None, typer.Option(exists=True, dir_okay=False)]
Checkpoint = Annotated[Path | None, typer.Option(exists=True, dir_okay=False)]
DataRoot = Annotated[Path | None, typer.Option(exists=True, file_okay=False)]


@app.callback()
def callback():
    """NAFNet training, evaluation and ONNX export on an x86 CUDA host."""


def _run(operation, extra: str):
    try:
        with redirect_stdout(sys.stderr):
            result = operation()
        typer.echo(json.dumps(json_safe(result), allow_nan=False))
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] not in {
            "torch",
            "albumentations",
            "modelopt",
            "onnx",
            "onnxscript",
            "onnxruntime",
        }:
            raise
        console.print(
            f"Missing {exc.name}. Run with uv run --locked {extra} modern-nafnet ...",
            style="red",
            markup=False,
        )
        raise typer.Exit(1) from exc
    except (ValueError, OSError, RuntimeError, ArithmeticError) as exc:
        console.print(str(exc), style="red", markup=False)
        raise typer.Exit(1) from exc


@app.command()
def train(
    config: ConfigPath = None,
    data: DataRoot = None,
    checkpoint: Checkpoint = None,
    resume: Checkpoint = None,
    steps: Annotated[int | None, typer.Option(min=1)] = None,
    device: Annotated[str | None, typer.Option()] = None,
    amp: Annotated[bool | None, typer.Option("--amp/--no-amp")] = None,
    synthetic_blur: Annotated[
        bool | None, typer.Option("--synthetic-blur/--no-synthetic-blur")
    ] = None,
    num_workers: Annotated[int | None, typer.Option(min=0)] = None,
):
    """Fine-tune a floating point model using paired images."""

    def execute():
        from modern_nafnet.training.loop import run_train

        settings = load_config(
            config,
            {
                "training": {
                    "checkpoint": checkpoint,
                    "resume": resume,
                    "steps": steps,
                    "device": device,
                    "amp": amp,
                    "num_workers": num_workers,
                },
                "data": {"path": data, "synthetic_blur": {"enabled": synthetic_blur}},
            },
        )
        return run_train(settings)

    _run(execute, "--extra train")


@app.command()
def qat(
    config: ConfigPath = None,
    data: DataRoot = None,
    checkpoint: Checkpoint = None,
    resume: Checkpoint = None,
    steps: Annotated[int | None, typer.Option(min=0)] = None,
    device: Annotated[str | None, typer.Option()] = None,
    synthetic_blur: Annotated[
        bool | None, typer.Option("--synthetic-blur/--no-synthetic-blur")
    ] = None,
    num_workers: Annotated[int | None, typer.Option(min=0)] = None,
):
    """Calibrate INT8 and fine-tune; --steps 0 saves a PTQ checkpoint."""

    def execute():
        from modern_nafnet.training.loop import run_train

        settings = load_config(
            config,
            {
                "training": {
                    "checkpoint": checkpoint,
                    "resume": resume,
                    "steps": steps,
                    "device": device,
                    "num_workers": num_workers,
                },
                "quantization": {"checkpoint": checkpoint},
                "data": {"path": data, "synthetic_blur": {"enabled": synthetic_blur}},
            },
        )
        return run_train(settings, qat=True)

    _run(execute, "--extra qat")


@app.command(name="export")
def export_model(
    config: ConfigPath = None,
    checkpoint: Checkpoint = None,
    output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
    height: Annotated[int | None, typer.Option(min=1)] = None,
    width: Annotated[int | None, typer.Option(min=1)] = None,
    device: Annotated[str | None, typer.Option()] = None,
    quantized: Annotated[bool | None, typer.Option("--quantized/--no-quantized")] = None,
):
    """Export a fixed batch-one FP or explicit INT8 Q/DQ ONNX model."""

    def execute():
        from modern_nafnet.deployment.export import run_export

        if (height is None) != (width is None):
            raise ValueError("Specify both --height and --width")
        settings = load_config(
            config,
            {
                "export": {
                    "checkpoint": checkpoint,
                    "output": output,
                    "device": device,
                    "quantized": quantized,
                    "shape": [1, 3, height, width] if height is not None else None,
                }
            },
        )
        return run_export(settings)

    _run(execute, "--extra export --extra qat")


@app.command()
def infer(
    config: ConfigPath = None,
    checkpoint: Checkpoint = None,
    input: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
    device: Annotated[str | None, typer.Option()] = None,
    quantized: Annotated[bool | None, typer.Option("--quantized/--no-quantized")] = None,
):
    """Restore one RGB image using a PyTorch checkpoint."""

    def execute():
        from modern_nafnet.inference import run_infer

        settings = load_config(
            config,
            {
                "inference": {
                    "checkpoint": checkpoint,
                    "input": input,
                    "output": output,
                    "device": device,
                    "quantized": quantized,
                }
            },
        )
        return run_infer(settings)

    _run(execute, "--extra train (or --extra qat for a quantized checkpoint)")


@app.command(name="eval")
def evaluate(
    config: ConfigPath = None,
    data: DataRoot = None,
    checkpoint: Checkpoint = None,
    split: Annotated[
        Literal["val", "test"] | None, typer.Option(help="Evaluation split (default: val)")
    ] = None,
    output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
    device: Annotated[str | None, typer.Option()] = None,
    quantized: Annotated[bool | None, typer.Option("--quantized/--no-quantized")] = None,
):
    """Evaluate paired images using the official uint8 GoPro protocol."""

    def execute():
        from modern_nafnet.evaluation.quality import run_evaluate

        settings = load_config(
            config,
            {
                "evaluation": {
                    "checkpoint": checkpoint,
                    "split": split,
                    "output": output,
                    "device": device,
                    "quantized": quantized,
                },
                "data": {"path": data},
            },
        )
        return run_evaluate(settings)

    _run(execute, "--extra train (or --extra qat for a quantized checkpoint)")


def main():
    app()
