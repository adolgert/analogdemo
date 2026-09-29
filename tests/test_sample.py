import json

import numpy as np
import torch

from analogdemo.calibration import Calibration
from analogdemo.model import build_model, fold_batch_norm
from analogdemo.noise import NoiseConfig
from analogdemo.sample import load_sample_artifact, sample_logits, save_sample_artifact


LAYERS = tuple([f"conv{i}" for i in range(1, 7)] + ["classifier"])


def calibration() -> Calibration:
    return Calibration(1, {name: 1.0 for name in LAYERS}, {name: 1.0 for name in LAYERS}, 1)


def test_zero_noise_equals_clean_and_does_not_mutate_model():
    model = fold_batch_norm(build_model().eval())
    images = torch.randn(3, 3, 32, 32)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    sampled = sample_logits(model, images, [4, 8, 15], NoiseConfig(), calibration(), batch_size=2)
    with torch.inference_mode():
        clean = model(images)
    assert torch.allclose(sampled[0, 0], clean, atol=1e-6, rtol=1e-6)
    assert model.training is False
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_sampling_is_batch_invariant():
    model = fold_batch_norm(build_model().eval())
    images = torch.randn(3, 3, 32, 32)
    config = NoiseConfig(transient={"conv1": 0.03, "classifier": 0.1}, weight={"conv4": 0.02})
    one = sample_logits(model, images, [101, 102, 103], config, calibration(),
                        num_chips=2, num_reads=2, seed=7, batch_size=1)
    three = sample_logits(model, images, [101, 102, 103], config, calibration(),
                          num_chips=2, num_reads=2, seed=7, batch_size=3)
    # Convolution kernels may choose batch-size-dependent floating-point reduction order.
    assert torch.allclose(one, three, atol=1e-6, rtol=1e-6)


def test_artifact_round_trip_uses_float_logits_and_unicode_ids(tmp_path):
    logits = torch.randn(2, 3, 4, 10)
    npz_path, manifest_path = save_sample_artifact(
        tmp_path / "raw", logits, [0, 1, 2, 3], ["test:0", "test:1", "test:2", "test:3"],
        {"seed": 12}, config=NoiseConfig(), calibration=calibration()
    )
    arrays, manifest = load_sample_artifact(npz_path)
    assert arrays["logits"].dtype == np.float32
    assert arrays["image_ids"].dtype.kind == "U"
    assert arrays["logits"].shape == (2, 3, 4, 10)
    assert manifest["seed"] == 12
    assert json.loads(manifest_path.read_text())["arrays"]["logits"]["dtype"] == "float32"
