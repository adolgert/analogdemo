"""Statistics for stochastic logit samples.

The sampling axes have distinct meanings: chip, read, image, class.  Functions
in this module deliberately preserve those axes until the estimand is defined.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax without overflowing, in float64 for stable summaries."""
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def _inputs(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(logits)
    y = np.asarray(labels, dtype=np.int64)
    if z.ndim != 4:
        raise ValueError("logits must have shape [chip, read, image, class]")
    if z.shape[-1] < 2 or z.shape[2] != y.size or y.ndim != 1:
        raise ValueError("labels must be one-dimensional and match the image axis")
    if z.shape[0] < 1 or z.shape[1] < 1 or y.size < 1:
        raise ValueError("chip, read, and image axes must be nonempty")
    if np.any((y < 0) | (y >= z.shape[-1])):
        raise ValueError("labels contain a class outside the logit class axis")
    if not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite")
    return z.astype(np.float64, copy=False), y


def _clean_logits(clean_logits: np.ndarray | None, n: int, k: int) -> np.ndarray | None:
    if clean_logits is None:
        return None
    clean = np.asarray(clean_logits, dtype=np.float64)
    if clean.shape == (1, 1, n, k):
        clean = clean[0, 0]
    if clean.shape != (n, k):
        raise ValueError("clean_logits must have shape [image, class] (or [1,1,image,class])")
    return clean


def per_image_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
) -> dict[str, np.ndarray]:
    """Return numeric per-image details, averaging executions over chips and reads.

    NumPy ``argmax`` resolves exact ties in favor of the lowest class index.  The
    separate ``top_tie_frequency`` field makes that otherwise hidden behavior
    visible. Covariance is zero when only one execution exists.
    """
    z, y = _inputs(logits, labels)
    c, r, n, k = z.shape
    samples = z.transpose(2, 0, 1, 3).reshape(n, c * r, k)
    probabilities = stable_softmax(samples)
    winners = np.argmax(samples, axis=-1)
    maxima = samples.max(axis=-1, keepdims=True)
    ties = np.count_nonzero(samples == maxima, axis=-1) > 1
    true_logits = np.take_along_axis(samples, y[:, None, None], axis=-1)[..., 0]
    other = samples.copy()
    other[np.arange(n)[:, None], np.arange(c * r)[None, :], y[:, None]] = -np.inf
    margins = true_logits - other.max(axis=-1)
    q = np.asarray(tuple(quantiles), dtype=np.float64)
    if q.ndim != 1 or np.any((q < 0) | (q > 1)):
        raise ValueError("quantiles must be a one-dimensional sequence in [0, 1]")
    centered = samples - samples.mean(axis=1, keepdims=True)
    covariance = (
        np.einsum("nsi,nsj->nij", centered, centered) / (c * r - 1)
        if c * r > 1
        else np.zeros((n, k, k), dtype=np.float64)
    )
    return {
        "logit_mean": samples.mean(axis=1),
        "logit_covariance": covariance,
        "softmax_mean": probabilities.mean(axis=1),
        "softmax_quantiles": np.quantile(probabilities, q, axis=1).transpose(1, 0, 2),
        "class_win_frequency": np.eye(k, dtype=np.float64)[winners].mean(axis=1),
        "true_margin_quantiles": np.quantile(margins, q, axis=1).T,
        "accuracy": (winners == y[:, None]).mean(axis=1),
        "top_tie_frequency": ties.mean(axis=1),
        "quantile_levels": q,
    }


def summarize_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    clean_logits: np.ndarray | None = None,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Create a JSON-safe aggregate summary and NPZ-ready per-image details."""
    z, y = _inputs(logits, labels)
    c, r, n, k = z.shape
    clean = _clean_logits(clean_logits, n, k)
    details = per_image_metrics(z, y, quantiles=quantiles)
    predictions = np.argmax(z, axis=-1)
    correct = predictions == y[None, None, :]
    per_chip = correct.mean(axis=(1, 2))
    per_read = correct.mean(axis=2)
    per_image = details["accuracy"]
    uncertainty: dict[str, Any] = {
        "image": {
            "unit": "image",
            "count": n,
            "mean_accuracy_standard_error": float(per_image.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        },
        "read": {
            "unit": "complete read of the shared image panel, within chip",
            "count_per_chip": r,
            "per_chip_mean_accuracy_standard_error": (
                (per_read.std(axis=1, ddof=1) / np.sqrt(r)).tolist() if r > 1 else None
            ),
        },
        "chip": {
            "unit": "whole simulated chip, each evaluated on the shared image panel",
            "count": c,
            "mean_accuracy_standard_error": float(per_chip.std(ddof=1) / np.sqrt(c)) if c > 1 else None,
        },
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "sample_counts": {"chips": c, "reads_per_chip": r, "images": n, "classes": k},
        "tie_rule": "numpy_argmax_lowest_class_index",
        "noisy_accuracy": float(correct.mean()),
        "per_chip_accuracy": per_chip.tolist(),
        "per_chip_read_accuracy": per_read.tolist(),
        "mean_top_tie_frequency": float(details["top_tie_frequency"].mean()),
        "uncertainty": uncertainty,
    }
    if clean is not None:
        clean_prediction = np.argmax(clean, axis=-1)
        clean_correct = clean_prediction == y
        changed = predictions != clean_prediction[None, None, :]
        summary.update(
            clean_accuracy=float(clean_correct.mean()),
            accuracy_change=float(correct.mean() - clean_correct.mean()),
            class_change_frequency=float(changed.mean()),
            newly_wrong_frequency=float((clean_correct[None, None, :] & ~correct).mean()),
            newly_correct_frequency=float((~clean_correct[None, None, :] & correct).mean()),
            wrong_to_different_wrong_frequency=float(
                (changed & ~clean_correct[None, None, :] & ~correct).mean()
            ),
        )
        details["clean_prediction"] = clean_prediction
        details["clean_correct"] = clean_correct.astype(np.uint8)
        details["class_change_frequency"] = changed.mean(axis=(0, 1))
    return summary, details


def paired_image_bootstrap(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    labels: np.ndarray,
    *,
    repetitions: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare expected single-execution accuracy by paired image resampling."""
    if repetitions < 1 or not 0 < confidence < 1:
        raise ValueError("repetitions must be positive and confidence must be in (0, 1)")
    a, y = _inputs(logits_a, labels)
    b, _ = _inputs(logits_b, labels)
    if a.shape[2:] != b.shape[2:]:
        raise ValueError("both inputs must use the same ordered image and class axes")
    score_a = (np.argmax(a, axis=-1) == y[None, None, :]).mean(axis=(0, 1))
    score_b = (np.argmax(b, axis=-1) == y[None, None, :]).mean(axis=(0, 1))
    difference = score_a - score_b
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, y.size, size=(repetitions, y.size))
    estimates = difference[draws].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "resampling_unit": "paired_image",
        "repetitions": repetitions,
        "seed": seed,
        "estimate_a_minus_b": float(difference.mean()),
        "confidence": confidence,
        "confidence_interval": np.quantile(estimates, [alpha, 1 - alpha]).tolist(),
    }


def whole_chip_bootstrap(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    labels: np.ndarray,
    *,
    repetitions: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare paired chip realizations, resampling complete chips as clusters."""
    a, y = _inputs(logits_a, labels)
    b, _ = _inputs(logits_b, labels)
    if a.shape[0] != b.shape[0] or a.shape[2:] != b.shape[2:]:
        raise ValueError("paired chip comparison requires matching chip and image axes")
    if a.shape[0] < 2:
        return {"available": False, "reason": "at least two chip realizations are required", "chip_count": a.shape[0]}
    if repetitions < 1 or not 0 < confidence < 1:
        raise ValueError("repetitions must be positive and confidence must be in (0, 1)")
    chip_a = (np.argmax(a, axis=-1) == y[None, None, :]).mean(axis=(1, 2))
    chip_b = (np.argmax(b, axis=-1) == y[None, None, :]).mean(axis=(1, 2))
    difference = chip_a - chip_b
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, a.shape[0], size=(repetitions, a.shape[0]))
    estimates = difference[draws].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "available": True,
        "resampling_unit": "paired_whole_chip",
        "chip_count": a.shape[0],
        "repetitions": repetitions,
        "seed": seed,
        "estimate_a_minus_b": float(difference.mean()),
        "confidence": confidence,
        "confidence_interval": np.quantile(estimates, [alpha, 1 - alpha]).tolist(),
    }


def save_metric_artifacts(
    path: str | Path, summary: dict[str, Any], details: dict[str, np.ndarray]
) -> tuple[Path, Path]:
    """Write ``<path>.json`` and ``<path>.npz`` and return both paths."""
    base = Path(path)
    json_path = base.with_suffix(".json")
    npz_path = base.with_suffix(".npz")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    np.savez_compressed(npz_path, **details)
    return json_path, npz_path
