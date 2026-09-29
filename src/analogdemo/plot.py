"""Create offline diagnostic figures from a stochastic experiment index."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

# Headless and managed environments often make the user cache read-only.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "analogdemo-plot-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METHOD_STYLE = {
    "sgd": {"color": "#4477AA", "marker": "o", "label": "SGD"},
    "sam": {"color": "#CC6677", "marker": "s", "label": "SAM"},
}


def _completed_rows(index: dict[str, Any], panel: str) -> list[dict[str, Any]]:
    return [
        row
        for row in index.get("runs", [])
        if row.get("panel") == panel
        and all(method in row and "noisy_accuracy" in row[method] for method in METHOD_STYLE)
    ]


def plot_validation_sweep(index: dict[str, Any], output: Path) -> Path | None:
    """Plot observed validation accuracy curves for every completed coarse group."""
    rows = [row for row in _completed_rows(index, "validation") if row["scenario"]["amplitude"] > 0]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        scenario = row["scenario"]
        groups.setdefault((scenario["mechanism"], scenario["location"]), []).append(row)
    if not groups:
        return None
    columns = 2
    rows_count = (len(groups) + columns - 1) // columns
    figure, axes = plt.subplots(rows_count, columns, figsize=(10, 3.5 * rows_count), squeeze=False)
    for axis, ((mechanism, location), group) in zip(axes.flat, sorted(groups.items())):
        group.sort(key=lambda row: row["scenario"]["amplitude"])
        amplitudes = [row["scenario"]["amplitude"] for row in group]
        for method, style in METHOD_STYLE.items():
            axis.plot(amplitudes, [row[method]["noisy_accuracy"] for row in group],
                      color=style["color"], marker=style["marker"], label=style["label"])
        axis.set_xscale("log")
        axis.set_ylim(0, 1)
        axis.set_title(f"{mechanism.replace('_', ' ')} · {location}")
        axis.set_xlabel("Relative noise amplitude")
        axis.set_ylabel("Expected single-execution accuracy")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(groups):]:
        axis.set_visible(False)
    axes.flat[0].legend()
    figure.suptitle("Validation coarse sweep (observed Monte Carlo estimates)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_test_accuracy(index: dict[str, Any], output: Path) -> Path | None:
    """Plot paired SGD/SAM point estimates for completed official-test scenarios."""
    rows = _completed_rows(index, "test")
    if not rows:
        return None
    rows.sort(key=lambda row: (row["scenario"]["id"] != "clean", row["scenario"]["id"]))
    labels = [row["scenario"]["id"] for row in rows]
    positions = list(range(len(rows)))
    figure, axis = plt.subplots(figsize=(max(8, 0.75 * len(rows)), 5))
    offsets = {"sgd": -0.12, "sam": 0.12}
    for method, style in METHOD_STYLE.items():
        axis.scatter([position + offsets[method] for position in positions],
                     [row[method]["noisy_accuracy"] for row in rows],
                     color=style["color"], marker=style["marker"], s=55, label=style["label"])
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Expected single-execution accuracy")
    axis.set_title("Official test scenarios (observed Monte Carlo estimates)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def generate_plots(index_path: str | Path, output: str | Path | None = None) -> list[Path]:
    """Generate every plot supported by the completed rows in ``index_path``."""
    path = Path(index_path)
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("schema_version") != 1:
        raise ValueError(f"unsupported experiment index schema {index.get('schema_version')!r}")
    folder = Path(output) if output is not None else path.parent / "figures"
    candidates = (
        plot_validation_sweep(index, folder / "validation-accuracy-sweep.png"),
        plot_test_accuracy(index, folder / "test-accuracy.png"),
    )
    return [candidate for candidate in candidates if candidate is not None]


def run_plots(experiment_dir: str | Path = "artifacts/experiment") -> Path | None:
    """Create available figures for an experiment and return their directory."""
    root = Path(experiment_dir)
    paths = generate_plots(root / "index.json", root / "figures")
    return root / "figures" if paths else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="artifacts/experiment/index.json")
    parser.add_argument("--output", help="Output directory (default: INDEX_PARENT/figures)")
    arguments = parser.parse_args()
    for path in generate_plots(arguments.index, arguments.output):
        print(path)


if __name__ == "__main__":
    main()
