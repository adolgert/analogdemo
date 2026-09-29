"""Build the one-screen demonstration page from the completed experiment.

Reads only files listed in artifacts/experiment/index.json and writes a
self-contained demo/index.html with the numbers inlined.

    .venv/bin/python demo/build.py [--artifact OUT.html]

The optional --artifact path receives the same page without the document
wrapper, for publishing as a hosted artifact.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "artifacts" / "experiment"
MODELS = ("sgd", "sam")
AMPLITUDES = (0.01, 0.03, 0.1, 0.3, 1.0)
SETTINGS = (
    ("independent_transient", "early"),
    ("independent_transient", "middle"),
    ("independent_transient", "late"),
    ("independent_transient", "all"),
    ("spatially_shared_transient", "all"),
    ("persistent_weight", "all"),
)


def scenario_id(mechanism: str, location: str, amplitude: float) -> str:
    return f"{mechanism}.{location}.a{amplitude:g}"


def load_index() -> tuple[dict, dict]:
    index = json.loads((EXP / "index.json").read_text())
    if index["schema_version"] != 1:
        raise SystemExit(f"unknown schema version {index['schema_version']}")
    if index["smoke"] or index["status"] != "complete":
        raise SystemExit("index is a smoke run or incomplete; refusing to build")
    rows = {(row["panel"], row["scenario"]["id"]): row for row in index["runs"]}
    return index, rows


def run_dir(panel: str, sid: str) -> Path:
    return EXP / "runs" / panel / sid


def summary_entry(rows: dict, panel: str, sid: str) -> dict:
    row = rows[(panel, sid)]
    out = {}
    for model in MODELS:
        summary = row[model]
        counts = summary["sample_counts"]
        out[model] = {
            "acc": round(summary["noisy_accuracy"], 5),
            "chips": counts["chips"],
            "reads": counts["reads_per_chip"],
            "images": counts["images"],
        }
    comparison = json.loads((run_dir(panel, sid) / "comparison.json").read_text())
    if comparison["direction"] != "sam_minus_sgd":
        raise SystemExit(f"unexpected comparison direction in {panel}/{sid}")
    paired = comparison["paired_images_conditional_on_sampled_noise"]
    low, high = paired["confidence_interval"]
    out["diff"] = {
        "est": round(paired["estimate_a_minus_b"], 5),
        "lo": round(low, 5),
        "hi": round(high, 5),
        "conf": paired["confidence"],
    }
    return out


def details(panel: str, sid: str, model: str) -> dict:
    with np.load(run_dir(panel, sid) / f"{model}.details.npz") as data:
        return {key: data[key] for key in data.files}


def png_data_uri(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def build_data() -> dict:
    index, rows = load_index()
    class_names = index["class_names"]

    val_ids = ["clean"] + [scenario_id(m, l, a) for m, l in SETTINGS for a in AMPLITUDES]
    test_ids = [key[1] for key in rows if key[0] == "test"]
    for sid in val_ids:
        if ("validation", sid) not in rows:
            raise SystemExit(f"validation run missing from index: {sid}")

    data = {
        "classes": class_names,
        "amplitudes": list(AMPLITUDES),
        "settings": [{"mech": m, "loc": l} for m, l in SETTINGS],
        "val": {sid: summary_entry(rows, "validation", sid) for sid in val_ids},
        "test": {sid: summary_entry(rows, "test", sid) for sid in test_ids},
    }

    # 500-image field: rows are classes, columns sorted by SGD clean margin.
    with np.load(EXP / "panels" / "validation.npz") as panel:
        labels = panel["labels"]
    margin = details("validation", "clean", "sgd")["true_margin_quantiles"][:, 2]
    order = np.lexsort((-margin, labels))
    if not np.all(np.bincount(labels) == 50):
        raise SystemExit("field layout assumes 50 validation images per class")
    field = {}
    for sid in val_ids:
        field[sid] = {}
        for model in MODELS:
            entry = data["val"][sid][model]
            runs = entry["chips"] * entry["reads"]
            correct = np.rint(details("validation", sid, model)["accuracy"] * runs).astype(int)
            if runs > 9:
                raise SystemExit("field encoding assumes at most 9 runs per image")
            field[sid][model] = {"n": runs, "k": "".join(str(k) for k in correct[order])}
    data["field"] = {"runs": field}

    # Six illustrative images: refined 512-read runs where they exist, else the
    # validation sweep's runs for the same six images.
    examples_meta = json.loads((EXP / "panels" / "examples.json").read_text())
    positions = np.array(examples_meta["validation_panel_positions"])
    with np.load(EXP / "panels" / "examples.npz") as panel:
        images = panel["images_uint8"]
        ex_labels = panel["labels"]
        ex_ids = panel["image_ids"]
    card_order = np.argsort(-margin[positions])  # most confident first
    example_ids = {key[1] for key in rows if key[0] == "examples"}
    ex_runs = {}
    for sid in val_ids:
        ex_runs[sid] = {}
        for model in MODELS:
            if sid in example_ids:
                source, summary = "examples", rows[("examples", sid)][model]
                win = details("examples", sid, model)["class_win_frequency"][card_order]
            else:
                source, summary = "validation", rows[("validation", sid)][model]
                win = details("validation", sid, model)["class_win_frequency"][positions[card_order]]
            counts = summary["sample_counts"]
            ex_runs[sid][model] = {
                "src": source,
                "chips": counts["chips"],
                "reads": counts["reads_per_chip"],
                "win": np.round(win, 4).tolist(),
            }
    data["examples"] = {
        "images": [png_data_uri(images[i]) for i in card_order],
        "labels": ex_labels[card_order].tolist(),
        "ids": ex_ids[card_order].tolist(),
        "runs": ex_runs,
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, help="also write a wrapper-free copy here")
    args = parser.parse_args()

    payload = json.dumps(build_data(), separators=(",", ":"))
    template = (HERE / "template.html").read_text()
    page = template.replace("/*__DATA__*/null", payload)
    (HERE / "index.html").write_text(page)
    print(f"wrote {HERE / 'index.html'} ({len(page) / 1024:.0f} KiB)")

    if args.artifact:
        start = page.index("<!--BEGIN PAGE-->") + len("<!--BEGIN PAGE-->")
        end = page.index("<!--END PAGE-->")
        body = page[start:end].replace("</head>\n<body>\n", "")
        args.artifact.write_text(body)
        print(f"wrote {args.artifact}")


if __name__ == "__main__":
    main()
