"""Download and integrity-check CIFAR-10, then freeze its data split."""

import argparse
from pathlib import Path

from .data import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", default="artifacts/data")
    args = parser.parse_args()
    if (Path(args.output) / "data_metadata.json").exists():
        print("Data metadata already exists; preserving the frozen split.")
        return
    metadata = prepare_data(args.root, args.output)
    print({"train": len(metadata["train_indices"]),
           "validation": len(metadata["validation_indices"]),
           "calibration": len(metadata["calibration_indices"]),
           "mean": metadata["mean"], "std": metadata["std"]}, flush=True)


if __name__ == "__main__":
    main()
