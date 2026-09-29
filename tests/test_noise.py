import pytest
import torch
from torch.nn import functional as F

from analogdemo.calibration import Calibration
from analogdemo.model import build_model, fold_batch_norm
from analogdemo.noise import NoiseConfig, _transient, _weight_for_chip, noisy_forward


LAYERS = tuple([f"conv{i}" for i in range(1, 7)] + ["classifier"])


def calibration() -> Calibration:
    return Calibration(1, {name: 1.0 for name in LAYERS}, {name: 1.0 for name in LAYERS}, 1)


def folded_model():
    return fold_batch_norm(build_model().eval())


def test_transient_variance_and_spatial_covariance():
    count = 1200
    output = torch.zeros(count, 1, 1, 2)
    ids = list(range(count))
    independent = _transient(
        output, "conv1", ids, NoiseConfig(transient={"conv1": 1.0}), calibration(), 9, 0, 0
    ).reshape(count, 2)
    shared = _transient(
        output,
        "conv1",
        ids,
        NoiseConfig(transient={"conv1": 1.0}, spatial_rho={"conv1": 0.7}),
        calibration(),
        9,
        0,
        0,
    ).reshape(count, 2)
    assert torch.allclose(independent.var(0), torch.ones(2), atol=0.12)
    assert torch.allclose(shared.var(0), torch.ones(2), atol=0.12)
    assert abs(torch.corrcoef(independent.T)[0, 1]) < 0.1
    assert abs(torch.corrcoef(shared.T)[0, 1] - 0.7) < 0.1


def test_weight_error_persists_across_reads_and_changes_across_chips():
    model = folded_model()
    images = torch.randn(2, 3, 32, 32)
    config = NoiseConfig(weight={"classifier": 0.2})
    first = noisy_forward(model, images, [10, 11], config, calibration(), seed=4, chip_id=0, read_id=0)
    reread = noisy_forward(model, images, [10, 11], config, calibration(), seed=4, chip_id=0, read_id=7)
    other_chip = noisy_forward(model, images, [10, 11], config, calibration(), seed=4, chip_id=1)
    assert torch.equal(first, reread)
    assert not torch.equal(first, other_chip)


def test_transient_changes_across_reads():
    model = folded_model()
    images = torch.randn(1, 3, 32, 32)
    config = NoiseConfig(transient={"classifier": 0.2})
    first = noisy_forward(model, images, [10], config, calibration(), seed=4, read_id=0)
    second = noisy_forward(model, images, [10], config, calibration(), seed=4, read_id=1)
    assert not torch.equal(first, second)


def test_linear_weight_error_has_predicted_gaussian_mean_and_covariance():
    layer = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.2, -0.4, 0.7], [-0.3, 0.1, 0.5]]))
    inputs = torch.tensor([0.5, -1.25, 0.8])
    beta, weight_rms = 0.3, 0.4
    scales = calibration()
    scales = Calibration(
        scales.version,
        scales.output_rms,
        {**scales.weight_rms, "classifier": weight_rms},
        scales.num_examples,
    )
    config = NoiseConfig(weight={"classifier": beta})
    outputs = torch.stack([
        F.linear(inputs, _weight_for_chip(
            layer, "classifier", config, scales, seed=91, chip_id=chip
        ))
        for chip in range(2000)
    ])

    expected_mean = F.linear(inputs, layer.weight)
    expected_variance = beta**2 * weight_rms**2 * inputs.square().sum()
    covariance = torch.cov(outputs.T)
    torch.testing.assert_close(outputs.mean(0), expected_mean, atol=0.015, rtol=0)
    torch.testing.assert_close(covariance.diag(), expected_variance.expand(2), atol=0.006, rtol=0.12)
    assert abs(float(covariance[0, 1].detach())) < 0.004


@pytest.mark.parametrize("field", ["transient", "weight", "threshold"])
@pytest.mark.parametrize("amplitude", [-0.1, float("nan"), float("inf")])
def test_invalid_scalar_noise_cannot_produce_invalid_samples(field, amplitude):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        NoiseConfig(**{field: amplitude})
