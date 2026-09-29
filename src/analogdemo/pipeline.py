"""Restartable orchestration for the complete laptop experiment."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

import torch

from .train import DEFAULTS, _validate_resume_config, run_training


STAGES = ("wait_for_data", "pilot_sgd", "pilot_sam", "preflight_experiment",
          "train_sgd", "train_sam", "experiment", "plots")


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / "src" / "analogdemo").glob("*.py")),
             *sorted((root / "configs").glob("*.json")), root / "pyproject.toml"]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.exists()}


def _checkpoint_epoch(path: Path) -> int | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint["epoch"])


def _load_config(path: Path, device: str) -> dict[str, Any]:
    config = json.loads(path.read_text())
    config["device"] = device
    return config


def _quality_gate(output: Path) -> None:
    metrics_path = output / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    candidates = [row for row in rows if int(row["epoch"]) == 1]
    if not candidates:
        raise RuntimeError(f"pilot did not produce epoch 1 metrics in {metrics_path}")
    row = candidates[-1]
    fields = ("train_loss", "train_accuracy", "validation_loss", "validation_accuracy")
    if any(not math.isfinite(float(row[field])) for field in fields):
        raise RuntimeError(f"pilot produced non-finite metrics: {row}")
    if float(row["validation_accuracy"]) < 0.15:
        raise RuntimeError(f"pilot validation accuracy below 0.15 quality gate: {row}")


def _pilot(config: dict[str, Any]) -> None:
    output = Path(config["output_dir"])
    last = output / "last.pt"
    epoch = _checkpoint_epoch(last)
    if epoch is None:
        run_training(config, stop_after_epoch=1)
    elif epoch < 1:
        run_training(config, resume=last, stop_after_epoch=1)
    _quality_gate(output)


def _validated_completed_training(config: dict[str, Any], last: Path) -> bool:
    epoch = _checkpoint_epoch(last)
    if epoch is None or epoch < int(config["epochs"]) - 1:
        return False
    best = last.with_name("best.pt")
    if not best.exists():
        raise RuntimeError(f"completed last checkpoint has no selected best checkpoint: {best}")
    last_checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    best_checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    current = {**DEFAULTS, **config}
    metadata = Path(current["metadata_path"]).read_bytes()
    current["metadata_sha256"] = hashlib.sha256(metadata).hexdigest()
    _validate_resume_config(current, last_checkpoint["config"])
    _validate_resume_config(current, best_checkpoint["config"])
    if int(best_checkpoint["epoch"]) > epoch:
        raise RuntimeError("best checkpoint is newer than the completed last checkpoint")
    if (float(best_checkpoint["best_validation_accuracy"]) !=
            float(last_checkpoint["best_validation_accuracy"])):
        raise RuntimeError("best and last checkpoints disagree on selected validation accuracy")
    return True


def _full_training(config: dict[str, Any]) -> None:
    last = Path(config["output_dir"]) / "last.pt"
    epoch = _checkpoint_epoch(last)
    if _validated_completed_training(config, last):
        print(f"checkpoint already complete: {last} (epoch {epoch})", flush=True)
        return
    if epoch is None:
        raise RuntimeError(f"pilot checkpoint is missing: {last}")
    run_training(config, resume=last)


def _set_stage(status_path: Path, status: dict[str, Any], name: str, state: str,
               detail: str | None = None) -> None:
    entry = status["stages"].setdefault(name, {})
    entry["state"] = state
    entry[f"{state}_at"] = _now()
    if detail:
        entry["detail"] = detail
    status["current_stage"] = None if state in {"completed", "failed"} else name
    status["updated_at"] = _now()
    _atomic_json(status_path, status)


def _run_stage(status_path: Path, status: dict[str, Any], name: str,
               operation: Callable[[], None], *, preserve_completed: bool = False) -> None:
    if preserve_completed and status["stages"].get(name, {}).get("state") == "completed":
        print(f"[{_now()}] preserving completed {name}", flush=True)
        return
    print(f"[{_now()}] starting {name}", flush=True)
    _set_stage(status_path, status, name, "running")
    status["stages"][name]["source_hashes"] = _source_hashes()
    _atomic_json(status_path, status)
    try:
        operation()
    except BaseException as error:
        detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        status["last_error"] = {"stage": name, "message": str(error), "traceback": detail}
        _set_stage(status_path, status, name, "failed", str(error))
        print(f"[{_now()}] failed {name}: {error}", file=sys.stderr, flush=True)
        raise
    status["last_error"] = None
    _set_stage(status_path, status, name, "completed")
    print(f"[{_now()}] completed {name}", flush=True)


def _wait_for_data(status_path: Path, status: dict[str, Any], timeout: float) -> None:
    metadata = Path("artifacts/data/data_metadata.json")
    deadline = time.monotonic() + timeout
    while True:
        try:
            content = json.loads(metadata.read_text())
            required = ("train_indices", "validation_indices", "calibration_indices", "mean", "std")
            if all(key in content for key in required):
                return
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"data metadata did not appear within {timeout:.0f} seconds: {metadata}")
        status["stages"]["wait_for_data"]["detail"] = (
            f"waiting for {metadata}; {remaining:.0f} seconds remain"
        )
        status["updated_at"] = _now()
        _atomic_json(status_path, status)
        print(f"[{_now()}] waiting for {metadata}", flush=True)
        time.sleep(min(10.0, remaining))


def run_pipeline(*, device: str = "mps", smoke: bool = False,
                 status_dir: str | Path = "artifacts/pipeline",
                 wait_for_data: bool = False, data_timeout: float = 7200.0) -> Path:
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
    torch.set_num_threads(4)
    status_dir = Path(status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    lock_path = status_dir / "pipeline.lock"
    status_path = status_dir / "status.json"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(f"another pipeline process holds {lock_path}") from error
    try:
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        status = {
            "schema_version": 1,
            "created_at": previous.get("created_at", _now()),
            "updated_at": _now(),
            "pid": os.getpid(),
            "worker_pid": os.getpid(),
            "device": device,
            "smoke": smoke,
            "source_hashes": _source_hashes(),
            "previous_source_hashes": previous.get("source_hashes"),
            "current_stage": None,
            "last_error": None,
            "stages": previous.get("stages", {name: {"state": "pending"} for name in STAGES}),
        }
        _atomic_json(status_path, status)
        metadata_path = Path("artifacts/data/data_metadata.json")
        if wait_for_data:
            _run_stage(status_path, status, "wait_for_data",
                       lambda: _wait_for_data(status_path, status, data_timeout))
        else:
            try:
                content = json.loads(metadata_path.read_text())
                required = ("train_indices", "validation_indices", "calibration_indices", "mean", "std")
                if not all(key in content for key in required):
                    raise ValueError("required fields are missing")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                raise FileNotFoundError(
                    f"valid {metadata_path} is unavailable; run analogdemo.prepare or pass --wait-for-data"
                ) from error
        sgd = _load_config(Path("configs/train-sgd.json"), device)
        sam = _load_config(Path("configs/train-sam.json"), device)
        _run_stage(status_path, status, "pilot_sgd", lambda: _pilot(sgd))
        _run_stage(status_path, status, "pilot_sam", lambda: _pilot(sam))

        def preflight() -> None:
            from .experiment import run_experiment
            run_experiment(output="artifacts/experiment-smoke", device=device, smoke=True)

        _run_stage(status_path, status, "preflight_experiment", preflight,
                   preserve_completed=True)
        _run_stage(status_path, status, "train_sgd", lambda: _full_training(sgd))
        _run_stage(status_path, status, "train_sam", lambda: _full_training(sam))

        def experiment() -> None:
            from .experiment import run_experiment
            output = "artifacts/experiment-final-smoke" if smoke else "artifacts/experiment"
            run_experiment(output=output, device=device, smoke=smoke)

        _run_stage(status_path, status, "experiment", experiment)

        def plots() -> None:
            from .plot import run_plots
            output = "artifacts/experiment-final-smoke" if smoke else "artifacts/experiment"
            run_plots(experiment_dir=output)

        _run_stage(status_path, status, "plots", plots)
        status["completed_at"] = _now()
        _atomic_json(status_path, status)
        return status_path
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


def launch_detached(*, device: str = "mps", smoke: bool = False,
                    status_dir: str | Path = "artifacts/pipeline",
                    wait_for_data: bool = False, data_timeout: float = 7200.0) -> int:
    """Launch a caffeinated worker detached from the invoking terminal."""
    status_dir = Path(status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    log_path = status_dir / "run.log"
    lock_path = status_dir / "pipeline.lock"
    with lock_path.open("a+") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another pipeline process holds {lock_path}") from error
        finally:
            try:
                fcntl.flock(probe, fcntl.LOCK_UN)
            except OSError:
                pass
    command = ["/usr/bin/caffeinate", "-i", sys.executable, "-u", "-m", "analogdemo.pipeline",
               "--device", device, "--status-dir", str(status_dir),
               "--data-timeout", str(data_timeout)]
    if smoke:
        command.append("--smoke")
    if wait_for_data:
        command.append("--wait-for-data")
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    launcher = {
        "schema_version": 1, "created_at": _now(), "updated_at": _now(),
        "launcher_pid": process.pid, "worker_pid": None, "device": device,
        "state": "launched", "log_path": str(log_path), "source_hashes": _source_hashes(),
    }
    _atomic_json(status_dir / "launcher.json", launcher)
    return process.pid


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--smoke", action="store_true",
                        help="pass smoke mode to artifact generation after full training")
    parser.add_argument("--status-dir", default="artifacts/pipeline")
    parser.add_argument("--wait-for-data", action="store_true",
                        help="wait up to --data-timeout seconds for frozen data metadata")
    parser.add_argument("--data-timeout", type=float, default=7200.0)
    parser.add_argument("--detach", action="store_true",
                        help="launch under caffeinate with output in the pipeline run log")
    args = parser.parse_args(argv)
    if args.detach:
        pid = launch_detached(device=args.device, smoke=args.smoke, status_dir=args.status_dir,
                              wait_for_data=args.wait_for_data, data_timeout=args.data_timeout)
        print(f"launched pipeline process {pid}; log: {Path(args.status_dir) / 'run.log'}")
    else:
        print(run_pipeline(device=args.device, smoke=args.smoke, status_dir=args.status_dir,
                           wait_for_data=args.wait_for_data, data_timeout=args.data_timeout))


if __name__ == "__main__":
    main()
