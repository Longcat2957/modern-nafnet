import copy

import pytest

torch = pytest.importorskip("torch", reason="Install the train extra for model tests")

from modern_nafnet.config import load_config  # noqa: E402
from modern_nafnet.models.factory import (  # noqa: E402
    bank_fingerprint,
    build_model_from_config,
    check_checkpoint_metadata,
    checkpoint_metadata,
    load_checkpoint,
)
from modern_nafnet.models.trt_r2r_light import (  # noqa: E402
    TASKS,
    CompactDegradationBank,
    LiteResidual,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def r2r_config(**options):
    return load_config(
        overrides={
            "architecture": "trt-r2r-light",
            "trt_r2r_light": {"channels": [32, 32, 32, 32], **options},
        }
    )


@pytest.mark.parametrize("bank_enabled", [False, True])
@pytest.mark.parametrize("shape", [(1, 3, 256, 256), (1, 3, 128, 256)])
def test_restoration_tensor_auxiliary_contract_and_gradients(bank_enabled, shape):
    config = r2r_config(bank_enabled=bank_enabled)
    model = build_model_from_config(config)
    image = torch.rand(shape, requires_grad=True)
    restored, logits, attention = model.forward_with_aux(image)
    assert isinstance(model(image), torch.Tensor)
    torch.testing.assert_close(model(image), restored)
    assert restored.shape == shape and torch.isfinite(restored).all()
    if bank_enabled:
        assert logits.shape == (1, 5) and attention.shape == (1, 5, 8)
        torch.testing.assert_close(attention.sum(-1), torch.ones(1, 5))
        loss = restored.square().mean() + logits.square().mean()
    else:
        assert logits is None and attention is None
        assert model.query_head is None and model.bank is None
        loss = restored.square().mean()
    loss.backward()
    assert torch.isfinite(image.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_lite_residual_constant_inputs_and_backward(fill):
    block = LiteResidual(32)
    image = torch.full((1, 32, 16, 32), fill, requires_grad=True)
    output = block(image)
    assert output.shape == image.shape and torch.isfinite(output).all()
    output.mean().backward()
    assert torch.isfinite(image.grad).all()


@pytest.mark.parametrize("tasks", [("derain", "deblur"), TASKS])
def test_bank_task_local_retrieval_matches_equation_and_independent_gates(tasks):
    bank = CompactDegradationBank(tasks, 8, 64, 192)
    query = torch.randn(2, 64)
    with torch.no_grad():
        bank.task_gate.weight.zero_()
        bank.task_gate.bias.fill_(4)
    prior, logits, attention = bank(query)
    scores = (query @ bank.key.T) / 8
    expected_attention = scores.reshape(2, len(tasks), 8).softmax(-1)
    expected_prior = (expected_attention * logits.sigmoid().unsqueeze(-1)).flatten(1) @ bank.value
    assert bank.key.shape == (len(tasks) * 8, 64)
    assert bank.value.shape == (len(tasks) * 8, 192)
    assert (logits.sigmoid() > 0.98).all()
    torch.testing.assert_close(attention, expected_attention)
    torch.testing.assert_close(prior, expected_prior)


@pytest.mark.parametrize("shape", [(1, 3, 17, 32), (1, 3, 16, 15), (1, 4, 32, 32), (3, 32, 32)])
def test_model_rejects_inputs_outside_external_padding_contract(shape):
    with pytest.raises(ValueError, match="input"):
        build_model_from_config(r2r_config())(torch.rand(shape))


def test_bank_freeze_and_finite_guards():
    config = r2r_config()
    model = build_model_from_config(config)
    before = bank_fingerprint(model)
    model.freeze_bank()
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is (name not in ("bank.key", "bank.value")), name
    assert bank_fingerprint(model) == before
    with torch.no_grad():
        model.bank.key[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.validate_bank()
    with pytest.raises(ValueError, match="NaN or Inf"):
        checkpoint_metadata(model, config)


@pytest.mark.parametrize("bank_enabled", [False, True])
def test_checkpoint_round_trip_and_metadata_only_verification(tmp_path, bank_enabled):
    config = r2r_config(bank_enabled=bank_enabled)
    model = build_model_from_config(config)
    checkpoint = {
        **checkpoint_metadata(model, config),
        "config": config,
        "state_dict": model.state_dict(),
    }
    path = tmp_path / "r2r.pt"
    torch.save(checkpoint, path)
    restored = build_model_from_config(config)
    metadata = load_checkpoint(restored, path, config)
    check_checkpoint_metadata(restored, metadata, config)
    image = torch.rand(1, 3, 16, 32)
    torch.testing.assert_close(model(image), restored(image), rtol=0, atol=0)
    assert bank_fingerprint(restored) == bank_fingerprint(model)


@pytest.mark.parametrize("field", ["architecture", "task_order", "model_spec", "bank_fingerprint"])
def test_new_architecture_requires_identity_metadata(field):
    config = r2r_config()
    model = build_model_from_config(config)
    checkpoint = {**checkpoint_metadata(model, config), "state_dict": model.state_dict()}
    checkpoint.pop(field)
    with pytest.raises(ValueError, match="architecture|missing"):
        check_checkpoint_metadata(model, checkpoint, config)


@pytest.mark.parametrize("mutation", ["task_order", "model_spec", "bank_bytes", "nonfinite"])
def test_checkpoint_rejects_semantic_or_bank_content_mismatch_before_loading(tmp_path, mutation):
    config = r2r_config()
    original = build_model_from_config(config)
    checkpoint = copy.deepcopy(
        {
            **checkpoint_metadata(original, config),
            "config": config,
            "state_dict": original.state_dict(),
        }
    )
    if mutation == "task_order":
        checkpoint["task_order"].reverse()
    elif mutation == "model_spec":
        checkpoint["model_spec"]["prior_scale_init"] = 0.2
    elif mutation == "bank_bytes":
        checkpoint["state_dict"]["bank.key"][0, 0] += 1
    else:
        checkpoint["state_dict"]["bank.value"][0, 0] = float("inf")
    path = tmp_path / "invalid.pt"
    torch.save(checkpoint, path)
    target = build_model_from_config(config)
    before = bank_fingerprint(target)
    with pytest.raises(ValueError, match="task order|model spec|fingerprint|NaN or Inf"):
        load_checkpoint(target, path, config)
    assert bank_fingerprint(target) == before


def test_factory_preserves_legacy_nafnet_and_rejects_cross_architecture(tmp_path):
    config = load_config(
        overrides={
            "model": {
                "width": 2,
                "enc_blk_nums": [1],
                "dec_blk_nums": [1],
                "train_size": [1, 3, 16, 16],
            }
        }
    )
    model = build_model_from_config(config)
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": model.state_dict(), "config": {"model": config["model"]}}, path)
    load_checkpoint(build_model_from_config(config), path, config)
    r2r = r2r_config()
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(build_model_from_config(r2r), path, r2r)
    new_path = tmp_path / "r2r.pt"
    new_model = build_model_from_config(r2r)
    torch.save(
        {**checkpoint_metadata(new_model, r2r), "state_dict": new_model.state_dict()}, new_path
    )
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(model, new_path, config)


def test_bank_fingerprint_includes_task_order():
    first = build_model_from_config(r2r_config(tasks=["derain", "deblur"]))
    second = build_model_from_config(r2r_config(tasks=["deblur", "derain"]))
    second.load_state_dict(first.state_dict(), strict=True)
    assert bank_fingerprint(first) != bank_fingerprint(second)


@pytest.mark.parametrize("field,value", [("task_order", ["deblur"]), ("bank_fingerprint", "bad")])
def test_nafnet_rejects_new_metadata_claiming_a_degradation_bank(field, value):
    config = load_config(
        overrides={
            "model": {
                "width": 2,
                "enc_blk_nums": [0],
                "dec_blk_nums": [0],
                "train_size": [1, 3, 16, 16],
            }
        }
    )
    model = build_model_from_config(config)
    metadata = checkpoint_metadata(model, config)
    metadata[field] = value
    with pytest.raises(ValueError, match="task order|bank fingerprint"):
        check_checkpoint_metadata(model, metadata, config)
