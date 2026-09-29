import math

import torch

from analogdemo.calibration import calibrate_model, load_calibration, save_calibration
from analogdemo.model import build_model, fold_batch_norm


def test_calibration_records_all_affine_layers_and_fixed_weight_rms():
    model = fold_batch_norm(build_model().eval())
    batches = [(torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))]
    result = calibrate_model(model, batches)
    assert result.num_examples == 2
    assert set(result.output_rms) == set(model.layer_names)
    assert all(value > 0 for value in result.output_rms.values())
    expected = model.convs[0].weight.detach().double().square().mean().sqrt().item()
    assert math.isclose(result.weight_rms["conv1"], expected, rel_tol=1e-12)


def test_calibration_restores_mode_and_removes_hooks():
    model = fold_batch_norm(build_model().eval())
    calibrate_model(model, [torch.randn(1, 3, 32, 32)])
    assert model.training is False
    assert all(not layer._forward_hooks for layer in [*model.convs, model.classifier])


def test_calibration_artifact_round_trip(tmp_path):
    model = fold_batch_norm(build_model().eval())
    result = calibrate_model(model, [torch.randn(1, 3, 32, 32)])
    path = save_calibration(result, tmp_path / "calibration.json")
    assert load_calibration(path) == result
