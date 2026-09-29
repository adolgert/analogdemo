import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

import analogdemo.train as train_module


class TinyDataset(Dataset):
    def __init__(self, training: bool) -> None:
        self.training = training
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4) / 48
        image = image + index * 0.03 + self.epoch * 0.01
        if self.training:
            # Use the global stream so the test detects missing RNG restoration.
            image = image + torch.rand(()) * 0.02
        return image, index % 2, f"train:{index:05d}"


class TinyBatchNormClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.classifier = nn.Linear(3, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.bn(self.conv(inputs)))
        return self.classifier(F.adaptive_avg_pool2d(hidden, 1).flatten(1))


def _assert_nested_equal(left, right) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def _config(method: str, output, metadata) -> dict:
    return {
        "method": method,
        "seed": 31415,
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 0.03,
        "momentum": 0.9,
        "weight_decay": 1e-3,
        "warmup_epochs": 1,
        "sam_rho": 0.04,
        "num_workers": 0,
        "device": "cpu",
        "num_classes": 10,
        "data_root": "synthetic",
        "metadata_path": str(metadata),
        "output_dir": str(output),
    }


@pytest.mark.parametrize("method", ["sgd", "sam"])
def test_uninterrupted_and_resumed_training_are_identical(tmp_path, monkeypatch, method):
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"synthetic": true}\n')
    monkeypatch.setattr(train_module, "build_model", lambda _num_classes=10: TinyBatchNormClassifier())
    monkeypatch.setattr(
        train_module,
        "make_dataset",
        lambda _root, split, _metadata, **_kwargs: TinyDataset(training=split == "train"),
    )

    full_dir = tmp_path / f"{method}-full"
    resumed_dir = tmp_path / f"{method}-resumed"
    train_module.run_training(_config(method, full_dir, metadata))
    train_module.run_training(
        _config(method, resumed_dir, metadata), stop_after_epoch=0
    )
    interrupted = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)
    assert interrupted["epoch"] == 0
    train_module.run_training(
        _config(method, resumed_dir, metadata), resume=resumed_dir / "last.pt"
    )

    full = torch.load(full_dir / "last.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)
    assert full["epoch"] == resumed["epoch"] == 2
    assert full["best_validation_accuracy"] == resumed["best_validation_accuracy"]
    _assert_nested_equal(full["model_state"], resumed["model_state"])
    _assert_nested_equal(full["optimizer_state"], resumed["optimizer_state"])
    _assert_nested_equal(full["scheduler_state"], resumed["scheduler_state"])

    # This includes BN running mean/variance and the exact once-per-batch counter.
    assert torch.equal(full["model_state"]["bn.num_batches_tracked"], torch.tensor(6))
    full_rows = [json.loads(line) for line in (full_dir / "metrics.jsonl").read_text().splitlines()]
    resumed_rows = [json.loads(line) for line in
                    (resumed_dir / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in resumed_rows] == [0, 1, 2]
    for full_row, resumed_row in zip(full_rows, resumed_rows):
        full_row.pop("elapsed_seconds")
        resumed_row.pop("elapsed_seconds")
        assert full_row == resumed_row

    full_best = torch.load(full_dir / "best.pt", map_location="cpu", weights_only=False)
    resumed_best = torch.load(resumed_dir / "best.pt", map_location="cpu", weights_only=False)
    _assert_nested_equal(full_best["model_state"], resumed_best["model_state"])
    manifest = json.loads((resumed_dir / "manifest.json").read_text())
    assert manifest["resume_events"][-1]["next_epoch"] == 1
