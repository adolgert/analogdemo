"""Small helpers shared by local experiment commands."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, target)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def environment_manifest() -> dict:
    import numpy
    import torchvision

    result = {
        "created_unix": time.time(),
        "python": platform.python_version(), "platform": platform.platform(),
        "architecture": platform.machine(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": numpy.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }
    try:
        hardware = json.loads(subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "-json"], text=True
        ))["SPHardwareDataType"][0]
        # Deliberately exclude device serial numbers and hardware UUIDs.
        result["hardware"] = {key: hardware[key] for key in
                              ("machine_name", "machine_model", "chip_type", "physical_memory")
                              if key in hardware}
    except (OSError, subprocess.SubprocessError, KeyError, ValueError):
        pass
    try:
        result["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.SubprocessError:
        result["git_revision"] = None
    result["source_hashes"] = {str(path): sha256(path)
                               for path in sorted(Path("src/analogdemo").glob("*.py"))}
    if Path("uv.lock").exists():
        result["dependency_lock_sha256"] = sha256("uv.lock")
    return result
