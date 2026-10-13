"""Small model checks plus optional exact upstream snapshot comparisons."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="Install the train extra for model tests")
from torch.nn import functional as F  # noqa: E402

from modern_nafnet.artifacts import sha256  # noqa: E402
from modern_nafnet.config import DEFAULT_CONFIG  # noqa: E402
from modern_nafnet.models.layers import LayerNorm2d  # noqa: E402
from modern_nafnet.models.nafnet import (  # noqa: E402
    UPSTREAM_COMMIT,
    NAFNetLocal,
    build_model,
    check_checkpoint_metadata,
    load_checkpoint,
    tlc_kernels,
)
from modern_nafnet.models.tlc import LocalAvgPool2d  # noqa: E402

TINY = dict(width=2, enc_blk_nums=(1, 1), middle_blk_num=1, dec_blk_nums=(1, 1))
BUNDLED_CHECKPOINT = Path(__file__).resolve().parents[1] / "weights/nafnet-gopro-width32.pt"


@pytest.fixture(scope="module")
def upstream():
    """Load the pinned, unmodified development oracle without installing BasicSR."""
    root = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "reference"
        / "NAFNet"
        / UPSTREAM_COMMIT
        / "basicsr/models/archs"
    )
    paths = {name: root / f"{name}.py" for name in ("arch_util", "local_arch", "NAFNet_arch")}
    if not all(path.is_file() for path in paths.values()):
        pytest.skip(f"Pinned upstream oracle is not present at {root}")
    loaded = {}
    with pytest.MonkeyPatch.context() as patch:
        for name in ("basicsr", "basicsr.models", "basicsr.models.archs"):
            package = ModuleType(name)
            package.__path__ = []
            patch.setitem(sys.modules, name, package)
        utils = ModuleType("basicsr.utils")
        utils.get_root_logger = logging.getLogger
        patch.setitem(sys.modules, "basicsr.utils", utils)
        for name, path in paths.items():
            qualified = f"basicsr.models.archs.{name}"
            spec = importlib.util.spec_from_file_location(qualified, path)
            module = importlib.util.module_from_spec(spec)
            patch.setitem(sys.modules, qualified, module)
            spec.loader.exec_module(module)
            loaded[name] = module
        yield SimpleNamespace(**loaded)


@pytest.fixture(scope="module")
def official_models(upstream):
    checkpoint = BUNDLED_CHECKPOINT
    if not checkpoint.is_file():
        pytest.skip(f"Bundled GoPro width32 checkpoint is absent: {checkpoint}")
    if not torch.cuda.is_available():
        pytest.skip("Official width32 oracle comparison requires the CUDA training host")
    with torch.device("cuda"):
        original = upstream.NAFNet_arch.NAFNetLocal(
            img_channel=3,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=(1, 1, 1, 28),
            dec_blk_nums=(1, 1, 1, 1),
        )
    rebuilt = NAFNetLocal().cuda()
    load_checkpoint(original, checkpoint)
    load_checkpoint(rebuilt, checkpoint)
    return original.eval(), rebuilt.eval()


def test_layernorm_matches_channel_layer_norm():
    layer = LayerNorm2d(4).double()
    x = torch.randn(2, 4, 3, 5, dtype=torch.float64, requires_grad=True)
    expected = F.layer_norm(
        x.permute(0, 2, 3, 1), (4,), layer.weight, layer.bias, eps=1e-6
    ).permute(0, 3, 1, 2)
    actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    direction = torch.randn_like(actual)
    variables = (x, layer.weight, layer.bias)
    actual_grads = torch.autograd.grad((actual * direction).sum(), variables)
    expected_grads = torch.autograd.grad((expected * direction).sum(), variables)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-9)


@pytest.mark.reference
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("near_constant", [False, True])
def test_layernorm_upstream_output_and_all_gradients(upstream, dtype, near_constant):
    original = upstream.arch_util.LayerNorm2d(4).to(dtype=dtype)
    rebuilt = LayerNorm2d(4).to(dtype=dtype)
    with torch.no_grad():
        original.weight.copy_(torch.randn_like(original.weight))
        original.bias.copy_(torch.randn_like(original.bias))
    rebuilt.load_state_dict(original.state_dict(), strict=True)
    values = torch.randn(2, 4, 3, 5, dtype=dtype)
    if near_constant:
        values = 1.0 + values * 1e-5
    original_input = values.clone().requires_grad_()
    rebuilt_input = values.clone().requires_grad_()
    expected, actual = original(original_input), rebuilt(rebuilt_input)
    tolerance = dict(rtol=2e-5, atol=2e-5) if dtype == torch.float32 else dict(rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(actual, expected, **tolerance)
    direction = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(
        (expected * direction).sum(), (original_input, original.weight, original.bias)
    )
    actual_grads = torch.autograd.grad(
        (actual * direction).sum(), (rebuilt_input, rebuilt.weight, rebuilt.bias)
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, **tolerance)


@pytest.mark.parametrize("kernel", [(3, 4), (12, 4), (12, 14)])
def test_tlc_window_and_edge_gradients(kernel):
    x = torch.randn(1, 3, 9, 11, dtype=torch.float64, requires_grad=True)
    actual = LocalAvgPool2d(kernel)(x)
    if kernel[0] >= 9 and kernel[1] >= 11:
        expected = F.adaptive_avg_pool2d(x, 1)
    else:
        kh, kw = min(9, kernel[0]), min(11, kernel[1])
        expected = F.pad(
            F.avg_pool2d(x, (kh, kw), stride=1),
            ((kw - 1) // 2, kw // 2, (kh - 1) // 2, kh // 2),
            mode="replicate",
        )
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    direction = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * direction).sum(), x)[0]
    expected_grad = torch.autograd.grad((expected * direction).sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-9)


def test_tlc_fp16_uses_finite_fp32_accumulation():
    x = torch.full((1, 2, 64, 80), 1000.0, dtype=torch.float16)
    result = LocalAvgPool2d((17, 19))(x)
    assert result.dtype == x.dtype
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, x)


@pytest.mark.reference
@pytest.mark.parametrize("shape", [(1, 3, 13, 15), (1, 3, 31, 39)])
def test_model_upstream_forward_gradient_and_fixed_tlc(upstream, shape):
    original = upstream.NAFNet_arch.NAFNetLocal(**TINY, train_size=(1, 3, 16, 16))
    rebuilt = NAFNetLocal(**TINY, train_size=(1, 3, 16, 16))
    with torch.no_grad():
        for name, parameter in original.named_parameters():
            if name.endswith(("beta", "gamma")):
                parameter.fill_(0.2)
    rebuilt.load_state_dict(original.state_dict(), strict=True)
    original_pools = {
        name: tuple(module.kernel_size)
        for name, module in original.named_modules()
        if isinstance(module, upstream.local_arch.AvgPool2d)
    }
    rebuilt_pools = {
        name: module.kernel_size
        for name, module in rebuilt.named_modules()
        if isinstance(module, LocalAvgPool2d)
    }
    assert rebuilt_pools == original_pools
    values = torch.rand(shape)
    original_input = values.clone().requires_grad_()
    rebuilt_input = values.clone().requires_grad_()
    expected, actual = original(original_input), rebuilt(rebuilt_input)
    assert actual.shape == shape
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    direction = torch.randn_like(actual)
    (expected * direction).mean().backward()
    (actual * direction).mean().backward()
    torch.testing.assert_close(rebuilt_input.grad, original_input.grad, rtol=1e-3, atol=1e-5)
    original_parameters = dict(original.named_parameters())
    for name, parameter in rebuilt.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(
            parameter.grad, original_parameters[name].grad, rtol=1e-3, atol=1e-5, msg=name
        )
    assert rebuilt_pools == {
        name: module.kernel_size
        for name, module in rebuilt.named_modules()
        if isinstance(module, LocalAvgPool2d)
    }


def test_build_model_preserves_config_and_crops_padding():
    config = {**TINY, "train_size": [1, 3, 16, 16]}
    model = build_model(config)
    assert config["train_size"] == [1, 3, 16, 16]
    with torch.inference_mode():
        x = torch.rand(1, 3, 13, 15)
        assert model(x).shape == x.shape
    assert model.train_size == (1, 3, 16, 16)


def test_checkpoint_metadata_rejects_weight_compatible_tlc_mismatch():
    config = {**TINY, "train_size": [1, 3, 16, 16]}
    model = build_model(config)
    checkpoint = {
        "config": {"model": {**TINY, "train_size": (1, 3, 16, 16)}},
        "tlc_kernels": tlc_kernels(model),
    }
    check_checkpoint_metadata(model, checkpoint, config)
    checkpoint["config"]["model"]["train_size"] = (1, 3, 32, 32)
    with pytest.raises(ValueError, match="model config differs"):
        check_checkpoint_metadata(model, checkpoint, config)
    checkpoint.pop("config")
    different_tlc = build_model({**config, "train_size": [1, 3, 32, 32]})
    # The state dict shapes alone cannot detect a changed TLC initialization.
    model.load_state_dict(different_tlc.state_dict(), strict=True)
    checkpoint["tlc_kernels"] = tlc_kernels(different_tlc)
    with pytest.raises(ValueError, match="TLC kernels differ"):
        check_checkpoint_metadata(model, checkpoint, config)


@pytest.mark.parametrize("container", [None, "params", "state_dict"])
def test_strict_checkpoint_supported_formats(tmp_path, container):
    original = NAFNetLocal(**TINY)
    weights = original.state_dict()
    checkpoint = weights if container is None else {container: weights, "step": 7}
    path = tmp_path / "model.pth"
    torch.save(checkpoint, path)
    rebuilt = NAFNetLocal(**TINY)
    metadata = load_checkpoint(rebuilt, path)
    assert metadata == ({} if container is None else {"step": 7})
    for name, value in rebuilt.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    weights.pop("intro.weight")
    torch.save(weights, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(rebuilt, path)


def test_bundled_checkpoint_preserves_fp32_weights_and_source_metadata():
    assert BUNDLED_CHECKPOINT.is_file(), "The published FP32 checkpoint must be included"
    assert sha256(BUNDLED_CHECKPOINT) == (
        "f6abb1b16853670e8a03bc0842e4d948f0276a1110999c5886fa87f73cb760cb"
    )
    payload = torch.load(BUNDLED_CHECKPOINT, map_location="cpu", weights_only=True)
    weights = payload["state_dict"]
    assert len(weights) == 664
    assert "modelopt_state" not in payload
    assert all(tensor.dtype == torch.float32 for tensor in weights.values())
    model = build_model(DEFAULT_CONFIG["model"])
    metadata = load_checkpoint(model, BUNDLED_CHECKPOINT)
    check_checkpoint_metadata(model, metadata, DEFAULT_CONFIG["model"])
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.view(torch.int32), weights[name].view(torch.int32))
    provenance = metadata["provenance"]
    assert provenance["repository"] == "https://github.com/megvii-research/NAFNet"
    assert provenance["commit"] == UPSTREAM_COMMIT
    assert provenance["source_url"] == (
        "https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view"
    )
    assert provenance["source_sha256"] == (
        "19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c"
    )


@pytest.mark.reference
def test_width32_official_parameter_keys(upstream):
    original = upstream.NAFNet_arch.NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=(1, 1, 1, 28),
        dec_blk_nums=(1, 1, 1, 1),
    )
    rebuilt = NAFNetLocal()
    expected = {name: value.shape for name, value in original.state_dict().items()}
    assert {name: value.shape for name, value in rebuilt.state_dict().items()} == expected
    rebuilt.load_state_dict(original.state_dict(), strict=True)
    if torch.cuda.is_available():
        with torch.inference_mode():
            sample = torch.rand(1, 3, 33, 47, device="cuda")
            assert rebuilt.cuda()(sample).shape == sample.shape


@pytest.mark.reference
@pytest.mark.parametrize("shape", [(1, 3, 32, 35), (1, 3, 385, 401)])
def test_official_checkpoint_fp32_forward(official_models, shape):
    original, rebuilt = official_models
    with torch.inference_mode():
        sample = torch.rand(shape, device="cuda")
        expected, actual = original(sample), rebuilt(sample)
        assert actual.shape == shape
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
