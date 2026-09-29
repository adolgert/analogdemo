# Analog computation noise demonstration backend

This project trains a 1.15-million-parameter VGG-7 on CIFAR-10 and samples its logits under general analog computation error mechanisms. It compares paired SGD and SAM models. The repository builds the computational artifacts and a one-screen demonstration page in `demo/`.

**The first full experiment is complete.** Start with [the results overview](notes/results-overview.md) for what was run, what was learned, and where the data lives. The exact output schemas are in [the interface data contract](notes/interface-data.md).

## Demonstration page

`demo/index.html` is the presentation. It is a single self-contained file: every number and all six example images are inlined, so it needs no data files. Open it in a browser and press F for full screen. Keys: M changes the error mechanism, L the location, the left and right arrows the error size, and T the training method.

The page loads its fonts from Google Fonts. Without an internet connection it falls back to Georgia and Helvetica and otherwise works the same.

The page is generated from `demo/template.html` and the completed experiment. After changing the template or the results, rebuild it with:

```bash
.venv/bin/python demo/build.py
```

The build reads only rows listed in `artifacts/experiment/index.json` and refuses a smoke or incomplete index.

### Moving the data to another machine

The `artifacts/` and `data/` directories are not in git. Showing the demo needs neither of them, only a clone of the repository or a copy of `demo/index.html`.

To rebuild the page or continue the analysis elsewhere, zip `artifacts/` from the repository root. The archive is about 250 MB and includes the experiment results and the trained checkpoints:

```bash
zip -r -X analogdemo-artifacts.zip artifacts -x 'artifacts/experiment-smoke/*' '*.DS_Store'
```

On the other machine, clone the repository, unzip the archive at the repository root, and install the environment:

```bash
git clone https://github.com/adolgert/analogdemo.git
cd analogdemo
unzip /path/to/analogdemo-artifacts.zip
uv sync
.venv/bin/python demo/build.py
```

CIFAR-10 itself, the 354 MB `data/` directory, is only needed to retrain or resample. The preparation command below downloads it again and checks it against the frozen split in `artifacts/data`.

## Setup and data

Install the locked environment, then download CIFAR-10 and freeze the train, validation, and calibration split:

```bash
uv sync
.venv/bin/python -m analogdemo.prepare --root data --output artifacts/data
```

The preparation command preserves an existing split. It will not silently replace metadata created with different split settings.

## Benchmark

Run the short device benchmark before committing to the long training job:

```bash
.venv/bin/python -m analogdemo.benchmark --device mps
```

## Full restartable run

On a Mac with MPS support, launch the long process detached from the terminal. The worker runs under `caffeinate -i` and writes unbuffered output to `run.log`:

```bash
.venv/bin/python -m analogdemo.pipeline --device mps --detach
```

If data preparation is still running, add `--wait-for-data`. The worker will wait for up to two hours by default:

```bash
.venv/bin/python -m analogdemo.pipeline --device mps --detach --wait-for-data
```

The stages are:

1. Train the SGD pilot through zero-based epoch 1 and apply a generous quality gate.
2. Train the SAM pilot through epoch 1 with the same initialization, examples, and augmentation.
3. Run a small stochastic artifact preflight from the pilot checkpoints.
4. Resume SGD through epoch 149.
5. Resume SAM through epoch 149.
6. Calibrate the selected models and generate the final stochastic experiment artifacts.
7. Render offline validation curves and official-test accuracy figures.

The pilots retain the complete 150-epoch learning-rate schedule. They do not create shortened training runs. A second pipeline process exits if another process holds the pipeline lock.

Inspect progress from another terminal:

```bash
cat artifacts/pipeline/status.json
tail -f artifacts/pipeline/run.log
tail -n 5 artifacts/training/sgd/metrics.jsonl
tail -n 5 artifacts/training/sam/metrics.jsonl
```

Each stage has start or completion timestamps in `status.json`. Failures include the stage, exception, and traceback. The status also snapshots SHA-256 hashes of the source and configuration files used at launch.
Detached-launch information is stored separately in `launcher.json`; the worker records its own PID in `status.json`.

To resume after interruption, run the same pipeline command again. It reads each `last.pt`, reconciles metrics with the last durable checkpoint, skips a completed 150-epoch model, and continues at the next epoch. Do not delete `last.pt`; `best.pt` is the clean-validation checkpoint selected for the final experiment.

For a quick end-to-end artifact check after trained checkpoints already exist, add `--smoke`. This changes only the final stochastic artifact budget.

## Outputs for the future interface

Training outputs are under `artifacts/training/{sgd,sam}`. Each directory contains `best.pt`, `last.pt`, `metrics.jsonl`, and `manifest.json`. The final experiment writes raw logits, summary tables, calibration scales, scenario definitions, and provenance under `artifacts/experiment`.

The machine-readable artifact fields and their intended interface use are documented in [notes/interface-data.md](notes/interface-data.md). The experimental assumptions and sampling design are documented in [notes/backend-plan.md](notes/backend-plan.md).

Measured MacBook benchmarks and the local run handoff are in [notes/local-execution.md](notes/local-execution.md). Consult `artifacts/pipeline/status.json` for live progress; the final experiment index must report `status: complete` before treating its inventory as complete.

After an experiment has produced completed index rows, generate offline diagnostic figures with:

```bash
.venv/bin/python -m analogdemo.plot --index artifacts/experiment/index.json
```
