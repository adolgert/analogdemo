"""Fixed clean reference scales used by the analog noise simulator."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn


LAYER_NAMES = tuple([f"conv{i}" for i in range(1, 7)] + ["classifier"])


@dataclass(frozen=True)
class Calibration:
    version: int
    output_rms: Mapping[str, float]
    weight_rms: Mapping[str, float]
    num_examples: int

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported calibration version {self.version}")
        for field_name, scales in (("output_rms", self.output_rms), ("weight_rms", self.weight_rms)):
            missing = set(LAYER_NAMES) - set(scales)
            if missing:
                raise ValueError(f"{field_name} missing layers: {sorted(missing)}")
            if any(not math.isfinite(float(scales[name])) or float(scales[name]) < 0
                   for name in LAYER_NAMES):
                raise ValueError(f"{field_name} scales must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "output_rms": dict(self.output_rms),
                "weight_rms": dict(self.weight_rms), "num_examples": self.num_examples}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Calibration":
        return cls(**dict(value))


def save_calibration(calibration: Calibration, path: str | Path) -> Path:
    """Save calibration scales as a human-readable, versioned JSON artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(calibration.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def load_calibration(path: str | Path) -> Calibration:
    return Calibration.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    convs = list(model.convs)
    if len(convs) != 6:
        raise ValueError("calibration requires model.convs with six convolutions")
    if hasattr(model, "bns") and any(not isinstance(bn, nn.Identity) for bn in model.bns):
        raise ValueError("calibration requires a batch-normalization-folded model")
    return [(f"conv{i + 1}", layer) for i, layer in enumerate(convs)] + [("classifier", model.classifier)]


def calibrate_model(model: nn.Module, batches: Iterable[Any], device: str | torch.device = "cpu") -> Calibration:
    """Measure clean preactivation and folded-weight RMS on supplied batches."""

    target = torch.device(device)
    sums = {name: 0.0 for name in LAYER_NAMES}
    counts = {name: 0 for name in LAYER_NAMES}
    hooks = []

    def hook_for(name: str):
        def record(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            value = output.detach().cpu().double()
            sums[name] += value.square().sum().item()
            counts[name] += value.numel()
        return record

    layers = _layers(model)
    for name, layer in layers:
        hooks.append(layer.register_forward_hook(hook_for(name)))
    was_training = model.training
    original_device = next(model.parameters()).device
    num_examples = 0
    try:
        model.to(target).eval()
        with torch.inference_mode():
            for batch in batches:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                images = images.to(target)
                num_examples += len(images)
                model(images)
    finally:
        for hook in hooks:
            hook.remove()
        model.to(original_device).train(was_training)
    if num_examples == 0:
        raise ValueError("calibration requires at least one example")
    output_rms = {name: (sums[name] / counts[name]) ** 0.5 for name in LAYER_NAMES}
    weight_rms = {name: layer.weight.detach().cpu().double().square().mean().sqrt().item()
                  for name, layer in layers}
    return Calibration(1, output_rms, weight_rms, num_examples)
