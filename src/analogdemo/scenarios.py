"""Deterministic, JSON-safe scenario specifications for the demonstration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


AFFINE_LAYERS = ("conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "classifier")
LOCATION_LAYERS = {
    "early": ("conv1", "conv2"),
    "middle": ("conv3", "conv4"),
    "late": ("conv5", "conv6"),
    "head": ("classifier",),
    "all": AFFINE_LAYERS,
}
DEFAULT_AMPLITUDES = (0.01, 0.03, 0.1, 0.3, 1.0)


def noise_config(
    *,
    mechanism: str,
    location: str,
    amplitude: float,
    spatial_rho: float = 0.0,
) -> dict[str, Any]:
    """Return fields accepted by ``noise.NoiseConfig`` without importing torch."""
    if location not in LOCATION_LAYERS:
        raise ValueError(f"unknown location {location!r}")
    if mechanism not in {"independent_transient", "spatially_shared_transient", "persistent_weight"}:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    if amplitude < 0 or not 0 <= spatial_rho <= 1:
        raise ValueError("amplitude must be nonnegative and spatial_rho must be in [0, 1]")
    layers = LOCATION_LAYERS[location]
    transient = {layer: 0.0 for layer in AFFINE_LAYERS}
    weight = {layer: 0.0 for layer in AFFINE_LAYERS}
    rho: float | dict[str, float] = 0.0
    if mechanism == "persistent_weight":
        weight.update({layer: float(amplitude) for layer in layers})
    else:
        transient.update({layer: float(amplitude) for layer in layers})
        if mechanism == "spatially_shared_transient":
            # The classifier has no spatial axis and therefore remains independent.
            rho = {layer: float(spatial_rho) for layer in layers if layer != "classifier"}
    return {"version": 1, "transient": transient, "weight": weight, "spatial_rho": rho, "threshold": {}}


def demonstration_scenarios(
    amplitudes: tuple[float, ...] = DEFAULT_AMPLITUDES,
    *,
    shared_rho: float = 0.75,
    seed: int = 20260929,
) -> list[dict[str, Any]]:
    """Build the frozen coarse sweep; order and IDs are deterministic."""
    scenarios: list[dict[str, Any]] = []
    mechanisms = ("independent_transient", "spatially_shared_transient", "persistent_weight")
    for mechanism in mechanisms:
        for location in ("early", "middle", "late", "head", "all"):
            for amplitude in amplitudes:
                rho = shared_rho if mechanism == "spatially_shared_transient" else 0.0
                ident = f"{mechanism}.{location}.a{amplitude:g}"
                scenarios.append(
                    {
                        "schema_version": 1,
                        "id": ident,
                        "seed": seed,
                        "mechanism": mechanism,
                        "location": location,
                        "amplitude": float(amplitude),
                        "noise_config": noise_config(
                            mechanism=mechanism, location=location, amplitude=amplitude, spatial_rho=rho
                        ),
                        "sampling": {
                            "chips": 32 if mechanism == "persistent_weight" else 1,
                            "reads_per_chip": 1 if mechanism == "persistent_weight" else 64,
                        },
                    }
                )
    return deepcopy(scenarios)
