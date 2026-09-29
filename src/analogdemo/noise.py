"""Noise configuration and deterministic analog-inference simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .calibration import Calibration


LAYER_NAMES = tuple([f"conv{i}" for i in range(1, 7)] + ["classifier"])
CONV_NAMES = LAYER_NAMES[:-1]


def _amplitudes(value: Mapping[str, float] | float | None) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, (int, float)):
        result = {name: float(value) for name in LAYER_NAMES}
    else:
        result = {str(name): float(amplitude) for name, amplitude in value.items()}
    unknown = set(result) - set(LAYER_NAMES)
    if unknown:
        raise ValueError(f"unknown affine layers: {sorted(unknown)}")
    if any(not math.isfinite(x) or x < 0 for x in result.values()):
        raise ValueError("noise amplitudes must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class NoiseConfig:
    """Versioned relative amplitudes for the supported noise mechanisms."""

    version: int = 1
    transient: Mapping[str, float] | float = field(default_factory=dict)
    weight: Mapping[str, float] | float = field(default_factory=dict)
    spatial_rho: Mapping[str, float] | float = 0.0
    threshold: Mapping[str, float] | float = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported noise config version {self.version}")
        object.__setattr__(self, "transient", _amplitudes(self.transient))
        object.__setattr__(self, "weight", _amplitudes(self.weight))
        if isinstance(self.threshold, (int, float)):
            threshold = _amplitudes({name: float(self.threshold) for name in CONV_NAMES})
        else:
            threshold = _amplitudes(self.threshold)
        if "classifier" in threshold and threshold["classifier"] != 0:
            raise ValueError("threshold error applies only to convolutional ReLUs")
        object.__setattr__(self, "threshold", threshold)
        rho = self.spatial_rho
        rho_map = ({name: float(rho) for name in CONV_NAMES}
                   if isinstance(rho, (int, float)) else
                   {str(name): float(value) for name, value in rho.items()})
        if set(rho_map) - set(CONV_NAMES):
            raise ValueError("spatial_rho applies only to convolutional layers")
        if any(not 0 <= value <= 1 for value in rho_map.values()):
            raise ValueError("spatial correlations must be between zero and one")
        object.__setattr__(self, "spatial_rho", rho_map)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "transient": dict(self.transient),
            "weight": dict(self.weight),
            "spatial_rho": dict(self.spatial_rho),
            "threshold": dict(self.threshold),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NoiseConfig":
        return cls(**dict(value))


def _seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    # PyTorch accepts signed 64-bit seeds. Hashing makes streams independent of
    # loop and batch order, while omitting amplitudes pairs amplitude sweeps.
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63 - 1)


def _normal(shape: Sequence[int], *key: object, dtype: torch.dtype, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed(*key))
    draw = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    return draw.to(device=device, dtype=dtype)


def _normal_cpu(shape: Sequence[int], *key: object) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed(*key))
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32)


def _layer_sequence(model: nn.Module) -> list[tuple[str, nn.Module]]:
    convs = list(model.convs)
    if len(convs) != 6:
        raise ValueError("simulator requires model.convs with six convolutions")
    if hasattr(model, "bns") and any(not isinstance(bn, nn.Identity) for bn in model.bns):
        raise ValueError("noise simulation requires a batch-normalization-folded model")
    return [(f"conv{i + 1}", layer) for i, layer in enumerate(convs)] + [("classifier", model.classifier)]


def _weight_for_chip(layer: nn.Module, name: str, config: NoiseConfig,
                     calibration: Calibration, seed: int, chip_id: int) -> Tensor:
    amplitude = config.weight.get(name, 0.0)
    if amplitude == 0:
        return layer.weight
    scale = amplitude * calibration.weight_rms[name]
    return layer.weight + scale * _normal(layer.weight.shape, seed, "weight", name, chip_id,
                                           dtype=layer.weight.dtype, device=layer.weight.device)


def _transient(output: Tensor, name: str, image_ids: Sequence[object], config: NoiseConfig,
               calibration: Calibration, seed: int, chip_id: int, read_id: int) -> Tensor:
    amplitude = config.transient.get(name, 0.0)
    if amplitude == 0:
        return output
    scale = amplitude * calibration.output_rms[name]
    rho = config.spatial_rho.get(name, 0.0) if output.ndim == 4 else 0.0
    # Generate keyed per-image draws on CPU and transfer the completed batch once.
    # This retains batch-independent streams without issuing one small device copy
    # for every image and noise component on MPS.
    errors = []
    for image_id in image_ids:
        independent = _normal_cpu(
            output.shape[1:], seed, "transient", name, chip_id, read_id, image_id, "independent"
        )
        if rho:
            shared = _normal_cpu(
                (output.shape[1], 1, 1),
                seed, "transient", name, chip_id, read_id, image_id, "shared",
            )
            independent = math.sqrt(1 - rho) * independent + math.sqrt(rho) * shared
        errors.append(independent)
    error_batch = torch.stack(errors).to(device=output.device, dtype=output.dtype)
    return output + scale * error_batch


def _threshold(output: Tensor, name: str, config: NoiseConfig, calibration: Calibration,
               seed: int, chip_id: int) -> Tensor:
    amplitude = config.threshold.get(name, 0.0)
    if amplitude == 0:
        return output
    offsets = _normal((1, output.shape[1], 1, 1), seed, "threshold", name, chip_id,
                      dtype=output.dtype, device=output.device)
    return output - amplitude * calibration.output_rms[name] * offsets


def chip_weights(model: nn.Module, config: NoiseConfig, calibration: Calibration, *,
                 chip_id: int = 0, seed: int = 0) -> dict[str, Tensor]:
    """Materialize one persistent weight realization for reuse across all batches."""

    return {name: _weight_for_chip(layer, name, config, calibration, seed, chip_id)
            for name, layer in _layer_sequence(model)}


def noisy_forward(model: nn.Module, images: Tensor, image_ids: Sequence[object],
                  config: NoiseConfig, calibration: Calibration, *, chip_id: int = 0,
                  read_id: int = 0, seed: int = 0,
                  persistent_weights: Mapping[str, Tensor] | None = None) -> Tensor:
    """Execute one chip/read realization without modifying ``model``."""

    if len(images) != len(image_ids):
        raise ValueError("images and image_ids must have equal length")
    x = images
    conv_layers = _layer_sequence(model)[:-1]
    for index, (name, layer) in enumerate(conv_layers):
        weight = (persistent_weights[name] if persistent_weights is not None else
                  _weight_for_chip(layer, name, config, calibration, seed, chip_id))
        x = F.conv2d(x, weight, layer.bias, layer.stride, layer.padding,
                     layer.dilation, layer.groups)
        x = _transient(x, name, image_ids, config, calibration, seed, chip_id, read_id)
        x = F.relu(_threshold(x, name, config, calibration, seed, chip_id))
        if index in (1, 3, 5):
            x = F.max_pool2d(x, 2)
    x = F.adaptive_avg_pool2d(x, 1).flatten(1)
    name, layer = _layer_sequence(model)[-1]
    weight = (persistent_weights[name] if persistent_weights is not None else
              _weight_for_chip(layer, name, config, calibration, seed, chip_id))
    x = F.linear(x, weight, layer.bias)
    return _transient(x, name, image_ids, config, calibration, seed, chip_id, read_id)
