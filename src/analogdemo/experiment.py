"""Produce reusable demonstration data from the two trained checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .calibration import Calibration, calibrate_model
from .data import load_data_metadata, make_dataset
from .metrics import paired_image_bootstrap, summarize_logits, whole_chip_bootstrap
from .model import fold_batch_norm, load_checkpoint_model
from .noise import NoiseConfig
from .runtime import environment_manifest, sha256, synchronize, write_json
from .sample import load_sample_artifact, sample_logits, save_sample_artifact
from .scenarios import DEFAULT_AMPLITUDES, noise_config
from .train import select_device


METHODS = ("sgd", "sam")
SEED = 20260929
INFERENCE_SOURCES = ("model.py", "calibration.py", "noise.py", "sample.py", "metrics.py")


def inference_implementation() -> dict:
    """Identify code that can change calibrated or sampled numerical results."""

    source_dir = Path(__file__).parent
    sources = {name: sha256(source_dir / name) for name in INFERENCE_SOURCES}
    encoded = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "sources": sources}


def read_json_if_valid(path: Path) -> dict | None:
    """Return None for missing/interrupted JSON writes; preserve semantic mismatches."""

    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


@dataclass
class Panel:
    name: str
    images: torch.Tensor
    labels: np.ndarray
    image_ids: np.ndarray
    display_images: np.ndarray
    selection: dict


def make_panel(name: str, metadata: dict, *, count: int | None = None,
               positions: list[int] | None = None) -> Panel:
    dataset = make_dataset("data", "test" if name == "test" else "validation", metadata)
    if positions is None:
        if count is None or count >= len(dataset):
            positions = list(range(len(dataset)))
        else:
            targets = np.asarray(dataset.base.targets)[dataset.indices]
            rng = np.random.default_rng(SEED)
            positions = []
            for label in range(10):
                candidates = np.flatnonzero(targets == label)
                positions.extend(rng.choice(candidates, count // 10, replace=False).tolist())
            rng.shuffle(positions)
    items = [dataset[position] for position in positions]
    return Panel(name, torch.stack([item[0] for item in items]),
                 np.asarray([item[1] for item in items], dtype=np.int64),
                 np.asarray([item[2] for item in items], dtype=np.str_),
                 dataset.base.data[[dataset.indices[position] for position in positions]],
                 {"source_split": "test" if name == "test" else "validation",
                  "positions_in_split": positions, "seed": SEED})


def scenario(mechanism: str, location: str, amplitude: float) -> dict:
    return {"id": f"{mechanism}.{location}.a{amplitude:g}", "mechanism": mechanism,
            "location": location, "amplitude": amplitude, "seed": SEED,
            "noise_config": noise_config(mechanism=mechanism, location=location,
                                         amplitude=amplitude, spatial_rho=0.75)}


def clean_scenario() -> dict:
    return {"id": "clean", "mechanism": "clean", "location": "all", "amplitude": 0.0,
            "seed": SEED, "noise_config": NoiseConfig().to_dict()}


def coarse_scenarios(smoke: bool) -> list[dict]:
    amplitudes = (0.1,) if smoke else DEFAULT_AMPLITUDES
    combinations = [("independent_transient", location)
                    for location in ("early", "middle", "late", "all")]
    combinations += [("spatially_shared_transient", "all"), ("persistent_weight", "all")]
    return [scenario(mechanism, location, amplitude)
            for mechanism, location in combinations for amplitude in amplitudes]


def select_scenarios(rows: list[dict], clean_accuracy: float, smoke: bool) -> list[dict]:
    """Select two amplitudes using only SGD validation degradation, then freeze."""
    chosen = {}
    for mechanism in ("independent_transient", "persistent_weight"):
        candidates = [row for row in rows if row["scenario"]["mechanism"] == mechanism
                      and row["scenario"]["location"] == "all"
                      and row.get("panel", "validation") == "validation"]
        if not candidates:
            raise ValueError(f"no validation candidates for {mechanism}")
        winner = min(candidates, key=lambda row: (
            abs(clean_accuracy - row["sgd"]["noisy_accuracy"] - 0.10),
            row["scenario"]["amplitude"]))
        chosen[mechanism] = winner["scenario"]["amplitude"]
    return [clean_scenario()] + [item for item in coarse_scenarios(smoke)
        if item["amplitude"] == chosen[
            "persistent_weight" if item["mechanism"] == "persistent_weight"
            else "independent_transient"]]


class Experiment:
    def __init__(self, output: str, device: str, smoke: bool):
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = select_device(device)
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable in this process")
        self.smoke = smoke
        self.metadata_path = Path("artifacts/data/data_metadata.json")
        self.metadata = load_data_metadata(self.metadata_path)
        self.environment = environment_manifest()
        self.inference_implementation = inference_implementation()
        self.inference_runtime = {key: self.environment.get(key)
                                  for key in ("python", "torch", "torchvision", "numpy")}
        self.models = {}
        self.calibrations = {}
        self.checkpoints = {}
        self.index = {"schema_version": 1, "status": "running", "smoke": smoke,
                      "created_unix": time.time(), "class_names": self.metadata["class_names"],
                      "data_metadata_sha256": sha256(self.metadata_path),
                      "calibration_convention": "per_checkpoint_clean_layer_rms",
                      "inference_implementation": self.inference_implementation,
                      "inference_runtime": self.inference_runtime,
                      "environment": self.environment, "panels": {}, "models": {}, "runs": []}
        existing = self.root / "index.json"
        if existing.exists() and (previous := read_json_if_valid(existing)) is not None:
            if previous["smoke"] != smoke:
                raise ValueError("Use separate output directories for smoke and final experiments")
            self.index["created_unix"] = previous["created_unix"]
        self.flush()

    def flush(self) -> None:
        self.index["updated_unix"] = time.time()
        write_json(self.root / "index.json", self.index)

    def prepare_models(self) -> None:
        dataset = make_dataset("data", "calibration", self.metadata)
        count = 16 if self.smoke else len(dataset)
        calibration_items = [dataset[i] for i in range(count)]
        images = torch.stack([item[0] for item in calibration_items])
        ids = [item[2] for item in calibration_items]
        for method in METHODS:
            path = Path(f"artifacts/training/{method}/best.pt")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            original = load_checkpoint_model(path).to(self.device).eval()
            folded = fold_batch_norm(original)
            with torch.inference_mode():
                batch = images[:8].to(self.device)
                original_logits, folded_logits = original(batch), folded(batch)
                torch.testing.assert_close(original_logits, folded_logits, rtol=2e-4, atol=2e-5)
                fold_error = float((original_logits - folded_logits).abs().max())
            del original
            self.models[method] = folded
            self.checkpoints[method] = sha256(path)
            expected = {"checkpoint_sha256": self.checkpoints[method],
                        "data_metadata_sha256": sha256(self.metadata_path), "image_ids": ids,
                        "convention": "per_checkpoint_clean_layer_rms",
                        "inference_implementation_sha256": self.inference_implementation["sha256"],
                        "inference_runtime": self.inference_runtime}
            calibration_path = self.root / "calibration" / f"{method}.json"
            content = read_json_if_valid(calibration_path)
            if content is not None and not {"metadata", "calibration"} <= content.keys():
                content = None
            if content is not None:
                if content["metadata"] != expected:
                    raise ValueError(f"Calibration provenance changed: {calibration_path}")
                calibration = Calibration.from_dict(content["calibration"])
            else:
                calibration = calibrate_model(folded, images.split(64), self.device)
                write_json(calibration_path, {"schema_version": 1,
                    "calibration": calibration.to_dict(), "metadata": expected})
            self.calibrations[method] = calibration
            self.index["models"][method] = {
                "checkpoint_path": str(path.resolve()), "checkpoint_sha256": self.checkpoints[method],
                "selected_epoch_zero_based": checkpoint["epoch"],
                "best_validation_accuracy": checkpoint["best_validation_accuracy"],
                "calibration": str(calibration_path.relative_to(self.root)),
                "calibration_sha256": sha256(calibration_path), "fold_max_absolute_error": fold_error,
                "training_manifest_path": str((path.parent / "manifest.json").resolve()),
                "training_metrics_path": str((path.parent / "metrics.jsonl").resolve())}
            self.flush()

    def save_panel(self, panel: Panel) -> None:
        path = self.root / "panels" / f"{panel.name}.npz"
        expected = {**panel.selection, "data_metadata_sha256": sha256(self.metadata_path),
                    "class_names": self.metadata["class_names"], "num_images": len(panel.labels),
                    "images_uint8_layout": "NHWC", "mean": self.metadata["mean"],
                    "std": self.metadata["std"]}
        panel_valid = False
        panel_loaded = False
        saved_metadata = read_json_if_valid(path.with_suffix(".json"))
        if path.exists() and saved_metadata is not None:
            try:
                with np.load(path, allow_pickle=False) as saved:
                    panel_loaded = True
                    panel_valid = np.array_equal(saved["image_ids"], panel.image_ids)
            except (OSError, ValueError, KeyError):
                panel_valid = False
            if panel_loaded and not panel_valid:
                raise ValueError(f"Panel changed: {path}")
            if panel_valid and saved_metadata != expected:
                raise ValueError(f"Panel metadata changed: {path}")
        if not panel_valid:
            save_npz(path, images_uint8=panel.display_images, labels=panel.labels,
                     image_ids=panel.image_ids)
            write_json(path.with_suffix(".json"), expected)
        self.index["panels"][panel.name] = {"arrays": str(path.relative_to(self.root)),
                                            "manifest": str(path.with_suffix(".json").relative_to(self.root)),
                                            "sha256": sha256(path)}
        self.flush()

    def run_pair(self, panel: Panel, setting: dict, clean: dict[str, np.ndarray] | None = None
                 ) -> tuple[dict, dict]:
        if setting["mechanism"] == "clean":
            chips, reads = 1, 1
        else:
            count = 2 if self.smoke else (4 if panel.name == "validation" else 8)
            if panel.name == "examples" and not self.smoke:
                count = 128 if setting["mechanism"] == "persistent_weight" else 512
            chips, reads = ((count, 1) if setting["mechanism"] == "persistent_weight" else (1, count))
        arrays, summaries, paths = {}, {}, {}
        config = NoiseConfig.from_dict(setting["noise_config"])
        for method in METHODS:
            folder = self.root / "runs" / panel.name / setting["id"]
            path = folder / f"{method}.npz"
            expected = {"checkpoint_sha256": self.checkpoints[method],
                "calibration_sha256": self.index["models"][method]["calibration_sha256"],
                "panel_sha256": self.index["panels"][panel.name]["sha256"],
                "scenario": setting, "chips": chips, "reads": reads, "master_seed": SEED,
                "device": str(self.device), "data_metadata_sha256": sha256(self.metadata_path),
                "inference_implementation_sha256": self.inference_implementation["sha256"],
                "inference_runtime": self.inference_runtime}
            logits = None
            if path.exists() and path.with_suffix(".json").exists():
                try:
                    saved, manifest = load_sample_artifact(path)
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    saved = manifest = None
                if manifest is not None:
                    if "experiment" not in manifest:
                        manifest = None
                    elif manifest["experiment"] != expected:
                        raise ValueError(f"Cached sample provenance changed: {path}")
                if manifest is not None:
                    actual_checksum = sha256(path)
                    recorded_checksum = manifest.get("array_sha256")
                    if recorded_checksum is None:
                        # The process may have stopped after the atomic array write.
                        manifest["array_sha256"] = actual_checksum
                        write_json(path.with_suffix(".json"), manifest)
                        recorded_checksum = actual_checksum
                    if actual_checksum == recorded_checksum:
                        logits = saved["logits"]
            if logits is None:
                print(f"Sampling {panel.name} {setting['id']} {method} C={chips} R={reads} N={len(panel.labels)}", flush=True)
                synchronize(self.device)
                started = time.perf_counter()
                logits = sample_logits(self.models[method], panel.images, panel.image_ids,
                    config, self.calibrations[method], num_chips=chips, num_reads=reads,
                    seed=SEED, device=self.device, batch_size=128).numpy()
                elapsed = time.perf_counter() - started
                save_sample_artifact(path, logits, panel.labels, panel.image_ids,
                    {"experiment": expected, "environment": self.environment,
                     "sampling_seconds": elapsed,
                     "images_per_second": chips * reads * len(panel.labels) / elapsed},
                    config=config, calibration=self.calibrations[method])
                manifest = json.loads(path.with_suffix(".json").read_text())
                manifest["array_sha256"] = sha256(path)
                write_json(path.with_suffix(".json"), manifest)
            summary, details = summarize_logits(logits, panel.labels,
                clean_logits=clean[method] if clean is not None else logits[0, 0])
            summary["raw_sample_sha256"] = sha256(path)
            summary_path = folder / f"{method}.summary.json"
            details_path = folder / f"{method}.details.npz"
            write_json(summary_path, summary)
            save_npz(details_path, **details)
            arrays[method], summaries[method] = logits, summary
            paths[method] = {"raw": str(path.relative_to(self.root)),
                "manifest": str(path.with_suffix(".json").relative_to(self.root)),
                "summary": str(summary_path.relative_to(self.root)),
                "details": str(details_path.relative_to(self.root))}
        comparison = {"direction": "sam_minus_sgd", "training_seed_pairs": 1,
            "paired_images_conditional_on_sampled_noise": paired_image_bootstrap(
                arrays["sam"], arrays["sgd"], panel.labels, repetitions=1000, seed=SEED),
            "paired_chips_conditional_on_image_panel": whole_chip_bootstrap(
                arrays["sam"], arrays["sgd"], panel.labels, repetitions=1000, seed=SEED)}
        write_json(folder / "comparison.json", comparison)
        row = {"panel": panel.name, "scenario": setting, "sgd": summaries["sgd"],
               "sam": summaries["sam"], "files": paths,
               "comparison": str((folder / "comparison.json").relative_to(self.root))}
        self.index["runs"].append(row)
        self.flush()
        print(f"Completed {panel.name} {setting['id']}: SGD={summaries['sgd']['noisy_accuracy']:.4f}, SAM={summaries['sam']['noisy_accuracy']:.4f}", flush=True)
        return row, arrays

    def execute(self) -> Path:
        self.prepare_models()
        validation = make_panel("validation", self.metadata, count=20 if self.smoke else 500)
        self.save_panel(validation)
        clean_row, clean_arrays = self.run_pair(validation, clean_scenario())
        validation_clean = {method: clean_arrays[method][0, 0] for method in METHODS}
        coarse = coarse_scenarios(self.smoke)
        write_json(self.root / "scenarios-coarse.json", coarse)
        rows = [self.run_pair(validation, setting, validation_clean)[0] for setting in coarse]
        selected = select_scenarios(rows, clean_row["sgd"]["noisy_accuracy"], self.smoke)
        selected_record = {"schema_version": 1, "selection_split": "validation",
            "rule": "SGD all-layer accuracy drop nearest 0.10; lower amplitude breaks ties; independent amplitude shared across locations and correlations; persistent chosen separately",
            "scenarios": selected, "validation_panel_sha256": self.index["panels"]["validation"]["sha256"]}
        selected_path = self.root / "scenarios-selected.json"
        if selected_path.exists() and json.loads(selected_path.read_text()) != selected_record:
            raise ValueError("Frozen scenario selection changed; use a new experiment directory")
        write_json(selected_path, selected_record)
        self.index["selected_scenarios"] = str(selected_path.relative_to(self.root))
        # Freeze scenarios above before loading or evaluating the official test set.
        test = make_panel("test", self.metadata, count=20 if self.smoke else None)
        self.save_panel(test)
        _, test_clean_arrays = self.run_pair(test, clean_scenario())
        test_clean = {method: test_clean_arrays[method][0, 0] for method in METHODS}
        for setting in selected[1:]:
            self.run_pair(test, setting, test_clean)
        # Select examples across the baseline validation margin distribution.
        z = validation_clean["sgd"]
        other = z.copy()
        other[np.arange(len(z)), validation.labels] = -np.inf
        margins = z[np.arange(len(z)), validation.labels] - other.max(axis=1)
        ordering = np.argsort(margins, kind="stable")
        ranks = np.linspace(0, len(z) - 1, 2 if self.smoke else 6).round().astype(int)
        local_positions = ordering[ranks].tolist()
        positions = [validation.selection["positions_in_split"][i] for i in local_positions]
        examples = make_panel("examples", self.metadata, positions=positions)
        examples.selection.update(rule="six equally spaced ranks of SGD clean true-class margin on validation panel",
                                  validation_panel_positions=local_positions)
        self.save_panel(examples)
        _, example_clean_arrays = self.run_pair(examples, clean_scenario())
        for setting in selected[1:]:
            self.run_pair(examples, setting, {method: example_clean_arrays[method][0, 0] for method in METHODS})
        write_report(self.root, self.index)
        self.index["report"] = "results.md"
        self.index["report_sha256"] = sha256(self.root / "results.md")
        self.index["status"] = "complete"
        self.index.pop("error", None)
        self.flush()
        return self.root / "index.json"


def write_report(root: Path, index: dict) -> None:
    lines = ["# Stochastic experiment results", "",
             "This run compares one SGD/SAM training-seed pair. Amplitudes are relative to each checkpoint's own clean layer RMS.",
             "", "## Official test panel", "",
             "| Scenario | SGD accuracy | SAM accuracy | SAM minus SGD |",
             "|---|---:|---:|---:|"]
    for row in index["runs"]:
        if row["panel"] == "test":
            a, b = row["sgd"]["noisy_accuracy"], row["sam"]["noisy_accuracy"]
            lines.append(f"| {row['scenario']['id']} | {a:.2%} | {b:.2%} | {100*(b-a):+.2f} percentage points |")
    lines += ["", "Full per-image distributions, raw logits, sample counts, paired comparisons, and provenance are linked by `index.json`.",
              "The comparison intervals condition on either sampled noise or the chosen image panel; they do not capture training-seed variability.",
              "", "## Training", ""]
    for method in METHODS:
        metrics = Path(index["models"][method]["training_metrics_path"])
        if metrics.exists():
            rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
            elapsed = sum(row["elapsed_seconds"] for row in rows)
            lines.append(f"- {method.upper()}: {len(rows)} completed epochs; summed epoch time {elapsed/3600:.2f} hours; selected epoch {index['models'][method]['selected_epoch_zero_based']+1}.")
    if index["smoke"]:
        lines.insert(2, "SMOKE TEST ONLY: reduced panels and sample counts; not a final evaluation.\n")
    report = root / "results.md"
    temporary = report.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, report)


def run_experiment(output: str = "artifacts/experiment", device: str = "mps", smoke: bool = False) -> Path:
    torch.set_num_threads(4)
    experiment = Experiment(output, device, smoke)
    try:
        return experiment.execute()
    except BaseException as error:
        experiment.index.update(status="failed", error=f"{type(error).__name__}: {error}")
        experiment.flush()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/experiment")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = "artifacts/experiment-smoke" if args.smoke and args.output == "artifacts/experiment" else args.output
    print(run_experiment(output, args.device, args.smoke), flush=True)


if __name__ == "__main__":
    main()
