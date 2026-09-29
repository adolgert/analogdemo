import pytest
import torch

from analogdemo.model import build_model, fold_batch_norm, load_checkpoint_model


def test_model_shape_names_and_size():
    model = build_model()
    assert model.layer_names == ("conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "classifier")
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert 1_000_000 < sum(p.numel() for p in model.parameters()) < 1_300_000


def test_batch_norm_fold_matches_eval_logits():
    torch.manual_seed(4)
    model = build_model().eval()
    for bn in model.bns:
        bn.running_mean.normal_()
        bn.running_var.uniform_(0.2, 2.0)
    inputs = torch.randn(3, 3, 32, 32)
    expected = model(inputs)
    folded = fold_batch_norm(model)
    torch.testing.assert_close(folded(inputs), expected, rtol=2e-5, atol=2e-5)
    assert all(isinstance(layer, torch.nn.Identity) for layer in folded.bns)
    assert all(layer.bias is not None for layer in folded.convs)


def test_checkpoint_loader(tmp_path):
    model = build_model()
    path = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "config": {"num_classes": 10}}, path)
    loaded = load_checkpoint_model(path)
    assert not loaded.training
    torch.testing.assert_close(loaded.classifier.weight, model.classifier.weight)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_folded_cpu_and_mps_forward_consistency():
    """Exercise the exact inference form and tensor shape used by sampling."""
    torch.manual_seed(20260929)
    model = fold_batch_norm(build_model().eval())
    inputs = torch.randn(8, 3, 32, 32)
    with torch.inference_mode():
        cpu_logits = model(inputs)
        mps_logits = model.to("mps")(inputs.to("mps")).cpu()
    assert torch.isfinite(mps_logits).all()
    torch.testing.assert_close(mps_logits, cpu_logits, rtol=2e-4, atol=2e-5)
