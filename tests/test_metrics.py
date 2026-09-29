import json

import numpy as np

from analogdemo.metrics import (
    paired_image_bootstrap,
    save_metric_artifacts,
    stable_softmax,
    summarize_logits,
    whole_chip_bootstrap,
)


def test_softmax_is_stable_and_normalized():
    probabilities = stable_softmax(np.array([[10_000.0, 9_999.0], [-10_000.0, -10_000.0]]))
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
    np.testing.assert_allclose(probabilities[1], [0.5, 0.5])


def test_summary_known_values_and_tie_rule():
    logits = np.array(
        [[[[2, 1, 0], [0, 1, 2]], [[0, 3, 1], [2, 2, 0]]]], dtype=float
    )
    labels = np.array([0, 1])
    clean = np.array([[3, 0, 0], [0, 3, 0]], dtype=float)
    summary, details = summarize_logits(logits, labels, clean_logits=clean, quantiles=(0, 0.5, 1))
    assert summary["noisy_accuracy"] == 0.25
    assert summary["clean_accuracy"] == 1.0
    assert summary["tie_rule"] == "numpy_argmax_lowest_class_index"
    np.testing.assert_allclose(details["accuracy"], [0.5, 0.0])
    np.testing.assert_allclose(details["class_win_frequency"], [[0.5, 0.5, 0], [0.5, 0, 0.5]])
    np.testing.assert_allclose(details["top_tie_frequency"], [0, 0.5])
    assert details["logit_covariance"].shape == (2, 3, 3)
    assert details["softmax_quantiles"].shape == (2, 3, 3)
    assert details["true_margin_quantiles"].shape == (2, 3)


def test_single_chip_read_has_zero_covariance_and_no_fake_standard_errors():
    logits = np.array([[[[1.0, 0.0], [0.0, 2.0]]]])
    summary, details = summarize_logits(logits, np.array([0, 1]))
    np.testing.assert_array_equal(details["logit_covariance"], 0)
    assert summary["uncertainty"]["read"]["per_chip_mean_accuracy_standard_error"] is None
    assert summary["uncertainty"]["chip"]["mean_accuracy_standard_error"] is None
    assert summary["uncertainty"]["image"]["mean_accuracy_standard_error"] == 0.0


def test_wrong_to_different_wrong_is_not_newly_wrong():
    # Label 0: clean predicts 1, noisy predicts 2. It stays wrong.
    logits = np.array([[[[0.0, 1.0, 2.0]]]])
    clean = np.array([[0.0, 2.0, 1.0]])
    summary, _ = summarize_logits(logits, np.array([0]), clean_logits=clean)
    assert summary["newly_wrong_frequency"] == 0.0
    assert summary["newly_correct_frequency"] == 0.0
    assert summary["wrong_to_different_wrong_frequency"] == 1.0


def test_bootstraps_use_declared_cluster_and_are_reproducible():
    labels = np.array([0, 1, 0, 1])
    a = np.zeros((3, 2, 4, 2))
    b = np.zeros_like(a)
    a[..., 0] = np.array([2, 0, 2, 0])
    a[..., 1] = np.array([0, 2, 0, 2])
    b[..., 0] = 1
    first = paired_image_bootstrap(a, b, labels, repetitions=100, seed=4)
    second = paired_image_bootstrap(a, b, labels, repetitions=100, seed=4)
    assert first == second
    assert first["resampling_unit"] == "paired_image"
    chip = whole_chip_bootstrap(a, b, labels, repetitions=100, seed=7)
    assert chip["available"] is True
    assert chip["resampling_unit"] == "paired_whole_chip"
    unavailable = whole_chip_bootstrap(a[:1], b[:1], labels)
    assert unavailable["available"] is False


def test_uncertainty_keeps_chip_read_and_image_units_separate():
    # Chip 0 is correct on both images and chip 1 is wrong. Within each chip,
    # the two complete reads are identical, so read SE is zero while chip SE is not.
    labels = np.array([0, 1])
    logits = np.array(
        [
            [[[2, 0], [0, 2]], [[2, 0], [0, 2]]],
            [[[0, 2], [2, 0]], [[0, 2], [2, 0]]],
        ],
        dtype=float,
    )
    summary, _ = summarize_logits(logits, labels)
    uncertainty = summary["uncertainty"]
    assert uncertainty["read"]["per_chip_mean_accuracy_standard_error"] == [0.0, 0.0]
    assert uncertainty["chip"]["mean_accuracy_standard_error"] == 0.5
    assert uncertainty["image"]["mean_accuracy_standard_error"] == 0.0
    assert summary["per_chip_accuracy"] == [1.0, 0.0]


def test_saved_summary_is_json_safe_and_details_round_trip(tmp_path):
    logits = np.array([[[[1.0, 0.0]]]])
    summary, details = summarize_logits(logits, np.array([0]))
    json_path, npz_path = save_metric_artifacts(tmp_path / "metrics", summary, details)
    assert json.loads(json_path.read_text())["sample_counts"]["images"] == 1
    with np.load(npz_path) as artifact:
        np.testing.assert_allclose(artifact["softmax_mean"].sum(axis=-1), 1)
