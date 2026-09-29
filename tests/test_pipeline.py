import fcntl
import hashlib
import json
from pathlib import Path

import pytest

import analogdemo.experiment
import analogdemo.pipeline as pipeline
import analogdemo.plot
from analogdemo.train import DEFAULTS


def _isolate_pipeline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    metadata = tmp_path / "artifacts" / "data" / "data_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({
        "train_indices": [0],
        "validation_indices": [1],
        "calibration_indices": [0],
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    }) + "\n")
    monkeypatch.setattr(pipeline.torch, "set_num_threads", lambda _count: None)
    monkeypatch.setattr(pipeline, "_source_hashes", lambda: {"source": "test-hash"})
    monkeypatch.setattr(
        pipeline,
        "_load_config",
        lambda path, device: {
            "method": "sam" if "sam" in path.name else "sgd",
            "device": device,
            "epochs": 150,
            "output_dir": f"artifacts/training/{'sam' if 'sam' in path.name else 'sgd'}",
        },
    )


def test_pipeline_orders_preflight_before_full_training_and_final_outputs(monkeypatch, tmp_path):
    _isolate_pipeline(monkeypatch, tmp_path)
    operations = []
    monkeypatch.setattr(pipeline, "_pilot", lambda config: operations.append(f"pilot_{config['method']}"))
    monkeypatch.setattr(
        pipeline,
        "_full_training",
        lambda config: operations.append(f"train_{config['method']}"),
    )

    def experiment(*, output, device, smoke):
        operations.append(f"experiment:{output}:{device}:smoke={smoke}")

    monkeypatch.setattr(analogdemo.experiment, "run_experiment", experiment)
    monkeypatch.setattr(
        analogdemo.plot,
        "run_plots",
        lambda experiment_dir: operations.append(f"plots:{experiment_dir}"),
    )

    status_path = pipeline.run_pipeline(device="cpu", status_dir=tmp_path / "status")

    assert operations == [
        "pilot_sgd",
        "pilot_sam",
        "experiment:artifacts/experiment-smoke:cpu:smoke=True",
        "train_sgd",
        "train_sam",
        "experiment:artifacts/experiment:cpu:smoke=False",
        "plots:artifacts/experiment",
    ]
    status = json.loads(status_path.read_text())
    assert status["last_error"] is None
    assert status["current_stage"] is None
    assert all(status["stages"][stage]["state"] == "completed" for stage in pipeline.STAGES[1:])


def test_pipeline_lock_rejects_second_worker(monkeypatch, tmp_path):
    _isolate_pipeline(monkeypatch, tmp_path)
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    with (status_dir / "pipeline.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another pipeline process holds"):
            pipeline.run_pipeline(device="cpu", status_dir=status_dir)


def test_stage_failure_is_persisted_and_stops_later_work(monkeypatch, tmp_path):
    _isolate_pipeline(monkeypatch, tmp_path)
    operations = []

    def pilot(config):
        operations.append(f"pilot_{config['method']}")
        if config["method"] == "sam":
            raise ValueError("pilot exploded")

    monkeypatch.setattr(pipeline, "_pilot", pilot)
    monkeypatch.setattr(
        pipeline,
        "_full_training",
        lambda config: operations.append(f"train_{config['method']}"),
    )
    with pytest.raises(ValueError, match="pilot exploded"):
        pipeline.run_pipeline(device="cpu", status_dir=tmp_path / "status")

    assert operations == ["pilot_sgd", "pilot_sam"]
    status = json.loads((tmp_path / "status" / "status.json").read_text())
    assert status["last_error"]["stage"] == "pilot_sam"
    assert status["last_error"]["message"] == "pilot exploded"
    assert "ValueError: pilot exploded" in status["last_error"]["traceback"]
    assert status["stages"]["pilot_sgd"]["state"] == "completed"
    assert status["stages"]["pilot_sam"]["state"] == "failed"
    assert status["stages"]["train_sgd"]["state"] == "pending"


def _completed_config(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"frozen": true}\n')
    output = tmp_path / "training"
    output.mkdir()
    (output / "last.pt").touch()
    (output / "best.pt").touch()
    config = {
        "method": "sgd",
        "seed": 1,
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "warmup_epochs": 1,
        "metadata_path": str(metadata),
        "data_root": "data",
        "output_dir": str(output),
    }
    # Real config JSON omits defaults such as sam_rho and num_classes, while a
    # serialized checkpoint contains the merged training configuration.
    saved = {**DEFAULTS, **config}
    saved["metadata_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    return config, saved, output


def test_completed_training_skip_validates_both_checkpoints(monkeypatch, tmp_path):
    config, saved, output = _completed_config(tmp_path)
    checkpoints = {
        "last.pt": {"epoch": 2, "config": saved, "best_validation_accuracy": 0.8},
        "best.pt": {"epoch": 1, "config": saved, "best_validation_accuracy": 0.8},
    }
    monkeypatch.setattr(pipeline.torch, "load", lambda path, **_kwargs: checkpoints[Path(path).name])
    training_calls = []
    monkeypatch.setattr(pipeline, "run_training", lambda *args, **kwargs: training_calls.append((args, kwargs)))

    pipeline._full_training(config)

    assert training_calls == []
    assert (output / "best.pt").exists()


@pytest.mark.parametrize("change", ["epochs", "metadata"])
def test_completed_training_refuses_changed_config(monkeypatch, tmp_path, change):
    config, saved, _output = _completed_config(tmp_path)
    changed = dict(saved)
    if change == "epochs":
        changed["epochs"] = 99
    else:
        Path(config["metadata_path"]).write_text('{"frozen": false}\n')
    checkpoints = {
        "last.pt": {"epoch": 2, "config": changed, "best_validation_accuracy": 0.8},
        "best.pt": {"epoch": 1, "config": saved, "best_validation_accuracy": 0.8},
    }
    monkeypatch.setattr(pipeline.torch, "load", lambda path, **_kwargs: checkpoints[Path(path).name])
    monkeypatch.setattr(
        pipeline,
        "run_training",
        lambda *_args, **_kwargs: pytest.fail("mismatched completed run must not resume or skip"),
    )

    with pytest.raises(ValueError, match="epochs|metadata_sha256"):
        pipeline._full_training(config)


def test_detached_launch_preserves_worker_status_file(monkeypatch, tmp_path):
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    status_path = status_dir / "status.json"
    status_path.write_text('{"current_stage": "train_sgd", "worker_pid": 41}\n')
    launched = []

    class Process:
        pid = 73

    def popen(command, **kwargs):
        launched.append((command, kwargs))
        return Process()

    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    monkeypatch.setattr(pipeline, "_source_hashes", lambda: {"source": "test-hash"})

    pid = pipeline.launch_detached(device="cpu", status_dir=status_dir, wait_for_data=True)

    assert pid == 73
    assert json.loads(status_path.read_text()) == {"current_stage": "train_sgd", "worker_pid": 41}
    launcher = json.loads((status_dir / "launcher.json").read_text())
    assert launcher["launcher_pid"] == 73
    assert launcher["state"] == "launched"
    assert "--wait-for-data" in launched[0][0]
