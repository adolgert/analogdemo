"""Reproducible CIFAR-10 splits and augmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def _tv():
    from torchvision import datasets
    from torchvision.transforms import functional as TF
    return datasets, TF


def _stable_seed(seed: int, epoch: int, image_id: str) -> int:
    text = f"{seed}:{epoch}:{image_id}".encode()
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little") % (2**63 - 1)


def _stratified_indices(targets: np.ndarray, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    classes = np.unique(targets)
    base, extra = divmod(count, len(classes))
    result: list[int] = []
    for rank, cls in enumerate(classes):
        candidates = np.flatnonzero(targets == cls).copy()
        rng.shuffle(candidates)
        result.extend(candidates[: base + (rank < extra)].tolist())
    rng.shuffle(result)
    return result


def prepare_data(
    root: str | Path, output_dir: str | Path, *, seed: int = 1729,
    val_size: int = 5000, calibration_size: int = 1024,
) -> dict[str, Any]:
    """Download CIFAR-10 and write the immutable split/calibration metadata."""
    datasets, _ = _tv()
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "data_metadata.json"
    if path.exists():
        existing = load_data_metadata(path)
        requested = (seed, val_size, calibration_size)
        recorded = (existing.get("seed"), len(existing.get("validation_indices", [])),
                    len(existing.get("calibration_indices", [])))
        if requested != recorded:
            raise FileExistsError(f"frozen data metadata at {path} uses {recorded}, requested {requested}")
        return existing
    train = datasets.CIFAR10(root=root, train=True, download=True)
    datasets.CIFAR10(root=root, train=False, download=True)
    targets = np.asarray(train.targets)
    val_indices = _stratified_indices(targets, val_size, seed)
    val_set = set(val_indices)
    train_indices = [i for i in range(len(targets)) if i not in val_set]
    calibration_local = _stratified_indices(targets[train_indices], calibration_size, seed + 1)
    calibration_indices = [train_indices[i] for i in calibration_local]
    # Accumulate in bounded chunks instead of materializing a ~1.1 GB float64 copy.
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for start in range(0, len(train_indices), 1000):
        pixels = np.asarray(train.data[train_indices[start:start + 1000]], dtype=np.float64) / 255.0
        channel_sum += pixels.sum(axis=(0, 1, 2))
        channel_square_sum += np.square(pixels).sum(axis=(0, 1, 2))
        pixel_count += pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
    mean_array = channel_sum / pixel_count
    variance = channel_square_sum / pixel_count - np.square(mean_array)
    mean = mean_array.tolist()
    std = np.sqrt(np.maximum(variance, 0.0)).tolist()
    metadata: dict[str, Any] = {
        "version": 1, "dataset": "CIFAR10", "seed": seed,
        "train_indices": train_indices, "validation_indices": val_indices,
        "calibration_indices": calibration_indices, "mean": mean, "std": std,
        "class_names": list(train.classes),
        "id_convention": "train:00000 and test:00000 are official CIFAR-10 indices",
    }
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary, path)
    return metadata


def load_data_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


class CIFARView(Dataset):
    def __init__(self, base: Dataset, indices: list[int], prefix: str,
                 mean: list[float], std: list[float], augment: bool, seed: int) -> None:
        self.base, self.indices, self.prefix = base, indices, prefix
        self.mean, self.std, self.augment, self.seed = mean, std, augment, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        _, TF = _tv()
        source_index = self.indices[position]
        image, label = self.base[source_index]
        image_id = f"{self.prefix}:{source_index:05d}"
        tensor = TF.to_tensor(image)
        if self.augment:
            generator = torch.Generator().manual_seed(_stable_seed(self.seed, self.epoch, image_id))
            tensor = TF.pad(tensor, [4, 4, 4, 4], padding_mode="reflect")
            top = int(torch.randint(0, 9, (), generator=generator))
            left = int(torch.randint(0, 9, (), generator=generator))
            tensor = TF.crop(tensor, top, left, 32, 32)
            if bool(torch.rand((), generator=generator) < 0.5):
                tensor = TF.hflip(tensor)
        return TF.normalize(tensor, self.mean, self.std), int(label), image_id


def make_dataset(root: str | Path, split: str, metadata: Mapping[str, Any],
                 augment: bool = False, seed: int = 1729) -> CIFARView:
    datasets, _ = _tv()
    if split not in {"train", "validation", "calibration", "test"}:
        raise ValueError(f"unknown split: {split}")
    official_train = split != "test"
    base = datasets.CIFAR10(root=root, train=official_train, download=False)
    key = f"{split}_indices"
    indices = list(metadata[key]) if key in metadata else list(range(len(base)))
    return CIFARView(base, indices, "train" if official_train else "test",
                     list(metadata["mean"]), list(metadata["std"]), augment, seed)


class EpochSeededSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, seed: int) -> None:
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


def set_dataset_epoch(dataset: Dataset, epoch: int) -> None:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)  # type: ignore[attr-defined]


def make_loader(dataset: Dataset, *, batch_size: int, seed: int = 1729,
                shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    sampler = EpochSeededSampler(dataset, seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=False, num_workers=num_workers, pin_memory=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download CIFAR-10 and create fixed split metadata")
    parser.add_argument("--root", default="data")
    parser.add_argument("--output-dir", default="artifacts/data")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--calibration-size", type=int, default=1024)
    args = parser.parse_args(argv)
    prepare_data(args.root, args.output_dir, seed=args.seed, val_size=args.val_size,
                 calibration_size=args.calibration_size)
    print(Path(args.output_dir) / "data_metadata.json")


if __name__ == "__main__":
    main()
