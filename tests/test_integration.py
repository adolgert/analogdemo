import copy

import numpy as np
import pytest
import torch

from analogdemo.calibration import calibrate_model
from analogdemo.metrics import summarize_logits
from analogdemo.model import build_model, fold_batch_norm
from analogdemo.noise import NoiseConfig
from analogdemo.sample import load_sample_artifact, sample_logits, save_sample_artifact


def test_folded_calibration_sampling_artifact_and_metrics_pipeline(tmp_path):
    torch.manual_seed(21)
    model = fold_batch_norm(build_model().eval())
    images = torch.randn(3, 3, 32, 32)
    labels = torch.tensor([2, 5, 1])
    image_ids = ["validation:00012", "validation:00104", "validation:04321"]

    calibration = calibrate_model(model, [(images, labels, image_ids)])
    config = NoiseConfig(
        transient={"classifier": 0.04},
        weight={"classifier": 0.02},
    )
    logits = sample_logits(
        model,
        images,
        image_ids,
        config,
        calibration,
        num_chips=2,
        num_reads=2,
        seed=812,
        batch_size=2,
    )
    clean = sample_logits(model, images, image_ids, NoiseConfig(), calibration)[0, 0]
    npz_path, _ = save_sample_artifact(
        tmp_path / "raw_logits",
        logits,
        labels,
        image_ids,
        {"checkpoint_sha256": "synthetic-test-checkpoint", "seed": 812},
        config=config,
        calibration=calibration,
    )

    arrays, manifest = load_sample_artifact(npz_path)
    summary, details = summarize_logits(
        arrays["logits"], arrays["labels"], clean_logits=clean.numpy()
    )
    assert arrays["image_ids"].tolist() == image_ids
    assert arrays["logits"].shape == (2, 2, 3, 10)
    assert manifest["noise_config"] == config.to_dict()
    assert manifest["calibration"]["num_examples"] == 3
    assert summary["sample_counts"] == {
        "chips": 2, "reads_per_chip": 2, "images": 3, "classes": 10
    }
    assert details["logit_covariance"].shape == (3, 10, 10)
    np.testing.assert_allclose(details["softmax_mean"].sum(axis=-1), 1.0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_calibration_and_clean_sampling_match_cpu():
    torch.manual_seed(31)
    folded = fold_batch_norm(build_model().eval())
    cpu_model = copy.deepcopy(folded)
    mps_model = copy.deepcopy(folded)
    images = torch.randn(2, 3, 32, 32)
    image_ids = ["validation:00001", "validation:00002"]

    cpu_calibration = calibrate_model(cpu_model, [images], device="cpu")
    mps_calibration = calibrate_model(mps_model, [images], device="mps")
    np.testing.assert_allclose(
        list(mps_calibration.output_rms.values()),
        list(cpu_calibration.output_rms.values()),
        rtol=2e-4,
        atol=2e-5,
    )
    cpu_logits = sample_logits(
        cpu_model, images, image_ids, NoiseConfig(), cpu_calibration, device="cpu"
    )
    mps_logits = sample_logits(
        mps_model, images, image_ids, NoiseConfig(), mps_calibration, device="mps"
    )
    torch.testing.assert_close(mps_logits, cpu_logits, rtol=2e-4, atol=2e-5)
