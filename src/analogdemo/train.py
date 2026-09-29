"""CLI and reusable training loop for paired SGD and SAM experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from .data import make_dataset, make_loader, set_dataset_epoch
from .model import build_model


DEFAULTS: dict[str, Any] = {
    "method": "sgd", "seed": 1729, "epochs": 150, "batch_size": 128,
    "learning_rate": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
    "warmup_epochs": 5, "sam_rho": 0.05, "num_workers": 0,
    "device": "auto", "num_classes": 10,
}


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.SGD(parameter_groups(model, float(config["weight_decay"])),
                           lr=float(config["learning_rate"]),
                           momentum=float(config["momentum"]))


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    warmup, epochs = int(config["warmup_epochs"]), int(config["epochs"])
    def factor(epoch: int) -> float:
        if warmup and epoch < warmup:
            return float(epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _bn_state(model: nn.Module):
    state = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            state.append((module, module.running_mean.clone(), module.running_var.clone(),
                          module.num_batches_tracked.clone()))
    return state


def _restore_bn(state) -> None:
    with torch.no_grad():
        for module, mean, variance, batches in state:
            module.running_mean.copy_(mean)
            module.running_var.copy_(variance)
            module.num_batches_tracked.copy_(batches)


@torch.no_grad()
def sam_perturb(model: nn.Module, rho: float) -> list[tuple[nn.Parameter, torch.Tensor]]:
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not parameters:
        return []
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in parameters]))
    scale = rho / (norm + 1e-12)
    changes = []
    for parameter in parameters:
        original = parameter.detach().clone()
        change = parameter.grad * scale
        parameter.add_(change)
        changes.append((parameter, original))
    return changes


@torch.no_grad()
def sam_restore(changes: Iterable[tuple[nn.Parameter, torch.Tensor]]) -> None:
    for parameter, original in changes:
        parameter.copy_(original)


def train_batch(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
                optimizer: torch.optim.Optimizer, method: str = "sgd", rho: float = 0.05):
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    first_loss = criterion(logits, labels)
    if not torch.isfinite(first_loss):
        raise FloatingPointError(f"non-finite training loss: {float(first_loss.detach())}")
    first_loss.backward()
    if method == "sam":
        changes = sam_perturb(model, rho)
        bn_before_second = _bn_state(model)
        try:
            optimizer.zero_grad(set_to_none=True)
            second_loss = criterion(model(inputs), labels)
            if not torch.isfinite(second_loss):
                raise FloatingPointError(f"non-finite SAM loss: {float(second_loss.detach())}")
            second_loss.backward()
        finally:
            _restore_bn(bn_before_second)
            sam_restore(changes)
    elif method != "sgd":
        raise ValueError(f"unknown training method: {method}")
    optimizer.step()
    return float(first_loss.detach()), logits.detach()


def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = correct = count = 0
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction="sum"))
            correct += int((logits.argmax(1) == labels).sum())
            count += labels.numel()
    if count == 0:
        raise ValueError("evaluation loader is empty")
    return {"loss": loss_sum / count, "accuracy": correct / count}


def capture_rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler,
                    epoch: int, best_accuracy: float, config: dict[str, Any]) -> None:
    payload = {"version": 1, "model_state": model.state_dict(),
               "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
               "epoch": epoch, "best_validation_accuracy": best_accuracy,
               "config": config, "rng_state": capture_rng_state()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _manifest(config: dict[str, Any], device: torch.device, model: nn.Module) -> dict[str, Any]:
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                                  text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except ImportError:
        torchvision_version = None
    decayed, undecayed = [], []
    for name, parameter in model.named_parameters():
        (undecayed if parameter.ndim == 1 or name.endswith(".bias") else decayed).append(name)
    return {"config": config, "device": str(device), "parameter_count": sum(p.numel() for p in model.parameters()),
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "started_unix": time.time(),
            "torchvision": torchvision_version, "source_revision": revision,
            "metadata_sha256": config["metadata_sha256"],
            "optimizer_parameter_groups": {"weight_decay": decayed, "no_weight_decay": undecayed},
            "sam_perturbed_parameters": [name for name, p in model.named_parameters() if p.requires_grad],
            "resume_events": []}


def _validate_resume_config(current: dict[str, Any], saved: dict[str, Any]) -> None:
    keys = ("method", "seed", "epochs", "batch_size", "learning_rate", "momentum",
            "weight_decay", "warmup_epochs", "sam_rho", "num_classes", "metadata_path", "data_root")
    keys = (*keys, "metadata_sha256")
    differences = [key for key in keys if current.get(key) != saved.get(key)]
    if differences:
        raise ValueError("resume configuration differs for: " + ", ".join(differences))


def _reconcile_metrics(path: Path, checkpoint_epoch: int) -> None:
    """Remove rows written after the last durable checkpoint and deduplicate epochs."""
    if not path.exists():
        return
    by_epoch: dict[int, dict[str, Any]] = {}
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if int(row["epoch"]) <= checkpoint_epoch:
            by_epoch[int(row["epoch"])] = row
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(by_epoch[key]) + "\n" for key in sorted(by_epoch)))
    os.replace(temporary, path)


def run_training(config: dict[str, Any], resume: str | Path | None = None,
                 stop_after_epoch: int | None = None) -> Path:
    config = {**DEFAULTS, **config}
    required = ("data_root", "metadata_path", "output_dir")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing configuration fields: {', '.join(missing)}")
    if config["method"] not in {"sgd", "sam"}:
        raise ValueError("method must be 'sgd' or 'sam'")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any((output / name).exists() for name in ("last.pt", "best.pt", "metrics.jsonl")):
        raise FileExistsError(f"training outputs already exist in {output}; resume or choose another directory")
    seed_everything(int(config["seed"]))
    device = select_device(str(config["device"]))
    metadata_bytes = Path(config["metadata_path"]).read_bytes()
    config["metadata_sha256"] = hashlib.sha256(metadata_bytes).hexdigest()
    metadata = json.loads(metadata_bytes)
    training = make_dataset(config["data_root"], "train", metadata, augment=True, seed=int(config["seed"]))
    validation = make_dataset(config["data_root"], "validation", metadata)
    if len(training) == 0 or len(validation) == 0:
        raise ValueError("training and validation splits must both be nonempty")
    train_loader = make_loader(training, batch_size=int(config["batch_size"]), seed=int(config["seed"]),
                               shuffle=True, num_workers=int(config["num_workers"]))
    validation_loader = make_loader(validation, batch_size=int(config["batch_size"]),
                                    num_workers=int(config["num_workers"]))
    model = build_model(int(config["num_classes"])).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    start_epoch, best_accuracy = 0, -1.0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        _validate_resume_config(config, checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_accuracy = float(checkpoint["best_validation_accuracy"])
    metrics_path = output / "metrics.jsonl"
    manifest_path = output / "manifest.json"
    if resume:
        _reconcile_metrics(metrics_path, start_epoch - 1)
        manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists()
                    else _manifest(config, device, model))
        manifest.setdefault("resume_events", []).append(
            {"resumed_unix": time.time(), "checkpoint": str(resume), "next_epoch": start_epoch}
        )
    else:
        manifest = _manifest(config, device, model)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    final_epoch = int(config["epochs"])
    if stop_after_epoch is not None:
        final_epoch = min(final_epoch, int(stop_after_epoch) + 1)
    for epoch in range(start_epoch, final_epoch):
        model.train()
        set_dataset_epoch(training, epoch)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        started = time.perf_counter()
        loss_sum = correct = count = 0
        epoch_started = time.perf_counter()
        for batch_index, (inputs, labels, _) in enumerate(train_loader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)
            loss, logits = train_batch(model, inputs, labels, optimizer,
                                       str(config["method"]), float(config["sam_rho"]))
            loss_sum += loss * labels.numel()
            correct += int((logits.argmax(1) == labels).sum())
            count += labels.numel()
            if batch_index % 100 == 0:
                print(json.dumps({"epoch": epoch, "batch": batch_index,
                                  "batches": len(train_loader),
                                  "elapsed_seconds": time.perf_counter() - epoch_started,
                                  "mean_loss": loss_sum / count}), flush=True)
        validation_metrics = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": loss_sum / count, "train_accuracy": correct / count,
               "validation_loss": validation_metrics["loss"],
               "validation_accuracy": validation_metrics["accuracy"],
               "learning_rate": optimizer.param_groups[0]["lr"],
               "elapsed_seconds": time.perf_counter() - started}
        print(json.dumps(row), flush=True)
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        is_best = validation_metrics["accuracy"] > best_accuracy
        best_accuracy = max(best_accuracy, validation_metrics["accuracy"])
        scheduler.step()
        if is_best:
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, epoch, best_accuracy, config)
        save_checkpoint(output / "last.pt", model, optimizer, scheduler, epoch, best_accuracy, config)
    return output / "best.pt"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="stop after this zero-based epoch; the full LR schedule is preserved")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    print(run_training(config, args.resume, args.stop_after_epoch))


if __name__ == "__main__":
    main()
