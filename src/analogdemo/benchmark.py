"""Measure local training throughput before launching long jobs."""

from __future__ import annotations

import argparse
import time

import torch

from .calibration import calibrate_model
from .model import build_model, fold_batch_norm
from .noise import NoiseConfig
from .runtime import environment_manifest, synchronize, write_json
from .sample import sample_logits
from .scenarios import noise_config
from .train import DEFAULTS, build_optimizer, select_device, train_batch


def benchmark(device: str = "auto", steps: int = 8) -> dict:
    target = select_device(device)
    manifest = environment_manifest()
    manifest.update(device=str(target), benchmark_steps=steps, results=[])
    torch.set_num_threads(4)
    cpu_model = build_model().train()
    optimizer = build_optimizer(cpu_model, DEFAULTS)
    train_batch(cpu_model, torch.randn(2, 3, 32, 32), torch.tensor([0, 1]), optimizer)
    manifest["cpu_forward_backward_passed"] = True
    for batch_size in (64, 128):
        for method in ("sgd", "sam"):
            torch.manual_seed(1729)
            model = build_model().to(target).train()
            optimizer = build_optimizer(model, DEFAULTS)
            images = torch.randn(batch_size, 3, 32, 32).to(target)
            labels = torch.arange(batch_size, device=target) % 10
            for _ in range(2):
                train_batch(model, images, labels, optimizer, method=method)
            synchronize(target)
            start = time.perf_counter()
            for _ in range(steps):
                train_batch(model, images, labels, optimizer, method=method)
            synchronize(target)
            elapsed = time.perf_counter() - start
            row = {"method": method, "batch_size": batch_size,
                   "seconds_per_batch": elapsed / steps,
                   "images_per_second": steps * batch_size / elapsed,
                   "training_only_seconds_per_45000_image_epoch": elapsed / steps * 45000 / batch_size}
            manifest["results"].append(row)
            print(row, flush=True)
            del model, optimizer, images, labels
            if target.type == "mps":
                torch.mps.empty_cache()
    return manifest


def benchmark_sampling(device: str = "auto") -> dict:
    """Measure reference Monte Carlo throughput for representative mechanisms."""

    target = select_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    model = fold_batch_norm(build_model().eval()).to(target)
    images = torch.randn(16, 3, 32, 32)
    image_ids = [f"synthetic:{index:05d}" for index in range(len(images))]

    synchronize(target)
    calibration_started = time.perf_counter()
    calibration = calibrate_model(model, [images], target)
    synchronize(target)
    calibration_seconds = time.perf_counter() - calibration_started

    # Warm kernels and device transfers without consuming or changing benchmark state.
    sample_logits(
        model, images[:1], image_ids[:1], NoiseConfig(), calibration,
        num_chips=1, num_reads=1, seed=20260929, device=target, batch_size=1,
    )
    synchronize(target)

    specifications = (
        ("independent_transient_all", "independent_transient", 1, 4),
        ("spatially_shared_transient_all", "spatially_shared_transient", 1, 4),
        ("persistent_weight_all", "persistent_weight", 4, 1),
    )
    rows = []
    for name, mechanism, chips, reads in specifications:
        config = NoiseConfig.from_dict(noise_config(
            mechanism=mechanism, location="all", amplitude=0.1, spatial_rho=0.75,
        ))
        synchronize(target)
        started = time.perf_counter()
        logits = sample_logits(
            model, images, image_ids, config, calibration,
            num_chips=chips, num_reads=reads, seed=20260929,
            device=target, batch_size=16,
        )
        synchronize(target)
        elapsed = time.perf_counter() - started
        executions = chips * reads * len(images)
        assert logits.shape == (chips, reads, len(images), 10)
        rows.append({
            "name": name,
            "mechanism": mechanism,
            "amplitude": 0.1,
            "spatial_rho": 0.75 if mechanism == "spatially_shared_transient" else 0.0,
            "chips": chips,
            "reads_per_chip": reads,
            "images": len(images),
            "realizations": chips * reads,
            "network_executions": executions,
            "seconds": elapsed,
            "images_per_second": executions / elapsed,
        })
    realization_counts = {row["realizations"] for row in rows}
    execution_counts = {row["network_executions"] for row in rows}
    if realization_counts != {4} or execution_counts != {64}:
        raise AssertionError("main sampling workloads must use equal Monte Carlo budgets")

    selected_config = NoiseConfig.from_dict(noise_config(
        mechanism="independent_transient", location="all", amplitude=0.1,
    ))
    synchronize(target)
    started = time.perf_counter()
    selected_logits = sample_logits(
        model, images[:1], image_ids[:1], selected_config, calibration,
        num_chips=1, num_reads=64, seed=20260929,
        device=target, batch_size=1,
    )
    synchronize(target)
    selected_elapsed = time.perf_counter() - started
    assert selected_logits.shape == (1, 64, 1, 10)
    selected = {
        "name": "selected_image_independent_transient_all",
        "mechanism": "independent_transient",
        "amplitude": 0.1,
        "chips": 1,
        "reads_per_chip": 64,
        "images": 1,
        "realizations": 64,
        "network_executions": 64,
        "seconds": selected_elapsed,
        "images_per_second": 64 / selected_elapsed,
    }
    manifest = environment_manifest()
    manifest.update({
        "benchmark": "stochastic_sampling",
        "device": str(target),
        "seed": 1729,
        "noise_seed": 20260929,
        "model": "random_folded_vgg7",
        "calibration_images": 16,
        "calibration_seconds": calibration_seconds,
        "cpu_threads": torch.get_num_threads(),
        "main_workloads": rows,
        "selected_image_workload": selected,
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--sampling", action="store_true",
                        help="benchmark stochastic inference sampling instead of training")
    parser.add_argument("--output", default="artifacts/benchmark.json")
    args = parser.parse_args()
    result = benchmark_sampling(args.device) if args.sampling else benchmark(args.device, args.steps)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
