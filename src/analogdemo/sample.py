"""Batched sampling and portable raw-logit artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .calibration import Calibration
from .noise import NoiseConfig, chip_weights, noisy_forward


def sample_logits(model: nn.Module, images: Tensor, image_ids: Sequence[object],
                  config: NoiseConfig, calibration: Calibration, *, num_chips: int = 1,
                  num_reads: int = 1, seed: int = 0, device: str | torch.device = "cpu",
                  batch_size: int = 128) -> Tensor:
    """Return raw logits as a CPU float32 tensor shaped ``[C,R,N,10]``."""

    if num_chips < 1 or num_reads < 1 or batch_size < 1:
        raise ValueError("num_chips, num_reads, and batch_size must be positive")
    if len(images) != len(image_ids):
        raise ValueError("images and image_ids must have equal length")
    if len(images) == 0:
        raise ValueError("sampling requires at least one image")
    target = torch.device(device)
    original_device = next(model.parameters()).device
    was_training = model.training
    result = torch.empty((num_chips, num_reads, len(images), 10), dtype=torch.float32)
    try:
        model.to(target).eval()
        with torch.inference_mode():
            for chip in range(num_chips):
                persistent_weights = chip_weights(
                    model, config, calibration, chip_id=chip, seed=seed
                )
                for read in range(num_reads):
                    for start in range(0, len(images), batch_size):
                        stop = min(start + batch_size, len(images))
                        # Reuse the chip's physical weight realization across every image batch.
                        logits = noisy_forward(
                            model, images[start:stop].to(target), image_ids[start:stop], config,
                            calibration, chip_id=chip, read_id=read, seed=seed,
                            persistent_weights=persistent_weights,
                        )
                        result[chip, read, start:stop] = logits.detach().cpu().float()
    finally:
        model.to(original_device).train(was_training)
    return result


def save_sample_artifact(path: str | Path, logits: Tensor | np.ndarray, labels: Tensor | np.ndarray,
                         image_ids: Tensor | np.ndarray, metadata: Mapping[str, Any] | None = None,
                         *, config: NoiseConfig | None = None,
                         calibration: Calibration | None = None) -> tuple[Path, Path]:
    """Write compressed arrays and an adjacent JSON provenance manifest."""

    npz_path = Path(path)
    if npz_path.suffix != ".npz":
        npz_path = npz_path.with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "logits": np.asarray(logits.detach().cpu() if isinstance(logits, Tensor) else logits,
                             dtype=np.float32),
        "labels": np.asarray(labels.detach().cpu() if isinstance(labels, Tensor) else labels,
                             dtype=np.int64),
        "image_ids": np.asarray(image_ids.detach().cpu() if isinstance(image_ids, Tensor) else image_ids,
                                dtype=np.str_),
    }
    if arrays["logits"].ndim != 4 or arrays["logits"].shape[-1] != 10:
        raise ValueError("logits must have shape [chip, read, image, 10]")
    if arrays["labels"].shape != (arrays["logits"].shape[2],) or arrays["image_ids"].shape != arrays["labels"].shape:
        raise ValueError("labels and image_ids must match the logits image dimension")
    started = time.perf_counter()
    temporary_npz = npz_path.with_name(npz_path.name + ".tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary_npz, npz_path)
    manifest = dict(metadata or {})
    manifest.update({
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "array_file": npz_path.name,
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for name, value in arrays.items()},
        "num_chips": arrays["logits"].shape[0],
        "num_reads": arrays["logits"].shape[1],
        "num_images": arrays["logits"].shape[2],
        "seed_scheme": "blake2b(base_seed, mechanism, layer, chip, read, image, component); v1",
        "write_seconds": time.perf_counter() - started,
    })
    if config is not None:
        manifest["noise_config"] = config.to_dict()
    if calibration is not None:
        manifest["calibration"] = calibration.to_dict()
    manifest_path = npz_path.with_suffix(".json")
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return npz_path, manifest_path


def load_sample_artifact(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = Path(path)
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    manifest = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
    return arrays, manifest
