import json
import types

import numpy as np
import pytest
import torch

import analogdemo.experiment as experiment_module
from analogdemo.calibration import Calibration
from analogdemo.experiment import Experiment, Panel, clean_scenario, select_scenarios
from analogdemo.sample import save_sample_artifact


def test_selection_ignores_official_test_rows():
    rows = []
    for mechanism in ("independent_transient", "persistent_weight"):
        for amplitude, accuracy in ((0.03, 0.88), (0.1, 0.80)):
            setting = {"mechanism": mechanism, "location": "all", "amplitude": amplitude}
            rows.append({"panel": "validation", "scenario": setting,
                         "sgd": {"noisy_accuracy": accuracy}})
        rows.append({"panel": "test", "scenario": {
            "mechanism": mechanism, "location": "all", "amplitude": 0.3,
        }, "sgd": {"noisy_accuracy": 0.80001}})
    selected = select_scenarios(rows, clean_accuracy=0.9, smoke=False)
    chosen = {row["mechanism"]: row["amplitude"] for row in selected[1:]
              if row["location"] == "all"}
    assert chosen["independent_transient"] == 0.1
    assert chosen["persistent_weight"] == 0.1


def _panel(name: str) -> Panel:
    return Panel(
        name=name,
        images=torch.zeros(2, 3, 32, 32),
        labels=np.array([0, 1]),
        image_ids=np.array([f"{name}:00000", f"{name}:00001"]),
        display_images=np.zeros((2, 32, 32, 3), dtype=np.uint8),
        selection={"source_split": "test" if name == "test" else "validation",
                   "positions_in_split": [0, 1], "seed": 1},
    )


def test_execute_freezes_selection_before_test_and_completes_after_report(tmp_path, monkeypatch):
    run = Experiment.__new__(Experiment)
    run.root = tmp_path
    run.device = torch.device("cpu")
    run.smoke = True
    run.metadata = {"class_names": [str(i) for i in range(10)]}
    run.models, run.calibrations, run.checkpoints = {}, {}, {}
    run.index = {"schema_version": 1, "status": "running", "smoke": True,
                 "created_unix": 1.0, "panels": {}, "models": {}, "runs": []}
    run.prepare_models = types.MethodType(lambda self: None, run)

    def make_panel(name, _metadata, **_kwargs):
        if name == "test":
            assert (tmp_path / "scenarios-selected.json").exists()
        return _panel(name)

    def save_panel(self, panel):
        path = tmp_path / "panels" / f"{panel.name}.npz"
        experiment_module.save_npz(path, image_ids=panel.image_ids, labels=panel.labels)
        self.index["panels"][panel.name] = {"arrays": str(path.relative_to(tmp_path)),
                                                   "sha256": experiment_module.sha256(path)}
        self.flush()

    def run_pair(self, panel, setting, clean=None):
        logits = np.zeros((1, 1, 2, 10), dtype=np.float32)
        logits[..., 0] = 1
        accuracy = 0.8 if setting["mechanism"] == "clean" else 0.7
        row = {"panel": panel.name, "scenario": setting,
               "sgd": {"noisy_accuracy": accuracy}, "sam": {"noisy_accuracy": accuracy}}
        self.index["runs"].append(row)
        self.flush()
        return row, {"sgd": logits.copy(), "sam": logits.copy()}

    def write_report(root, index):
        assert index["status"] == "running"
        (root / "results.md").write_text("complete report\n")

    run.save_panel = types.MethodType(save_panel, run)
    run.run_pair = types.MethodType(run_pair, run)
    monkeypatch.setattr(experiment_module, "make_panel", make_panel)
    monkeypatch.setattr(experiment_module, "write_report", write_report)

    index_path = run.execute()
    saved = json.loads(index_path.read_text())
    assert saved["status"] == "complete"
    assert saved["report"] == "results.md"
    assert saved["report_sha256"] == experiment_module.sha256(tmp_path / "results.md")
    assert saved["panels"]["test"]["arrays"] == "panels/test.npz"


def test_cached_samples_reject_stale_inference_implementation(tmp_path):
    run = Experiment.__new__(Experiment)
    run.root = tmp_path
    run.device = torch.device("cpu")
    run.smoke = True
    run.environment = {}
    run.inference_implementation = {"sha256": "current", "sources": {}}
    run.inference_runtime = {"python": "test", "torch": "test",
                             "torchvision": "test", "numpy": "test"}
    run.metadata_path = tmp_path / "metadata.json"
    run.metadata_path.write_text("{}")
    run.checkpoints = {"sgd": "sgd-checkpoint", "sam": "sam-checkpoint"}
    run.calibrations = {method: Calibration(
        1,
        {**{f"conv{i}": 1.0 for i in range(1, 7)}, "classifier": 1.0},
        {**{f"conv{i}": 1.0 for i in range(1, 7)}, "classifier": 1.0},
        2,
    ) for method in ("sgd", "sam")}
    run.models = {}
    panel = _panel("validation")
    panel_path = tmp_path / "panel.npz"
    panel_path.write_bytes(b"panel")
    run.index = {"models": {
        "sgd": {"calibration_sha256": "sgd-calibration"},
        "sam": {"calibration_sha256": "sam-calibration"},
    }, "panels": {"validation": {"sha256": experiment_module.sha256(panel_path)}}, "runs": []}
    setting = clean_scenario()
    raw = tmp_path / "runs" / "validation" / "clean" / "sgd.npz"
    save_sample_artifact(
        raw,
        np.zeros((1, 1, 2, 10), dtype=np.float32),
        panel.labels,
        panel.image_ids,
        {"experiment": {"inference_implementation_sha256": "stale"},
         "array_sha256": "irrelevant"},
    )
    with pytest.raises(ValueError, match="provenance changed"):
        run.run_pair(panel, setting)
