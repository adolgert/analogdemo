# Local execution and output handoff

**Completed:** the full pipeline finished on September 29, 2026 at 3:51 p.m.
Eastern, with both 150-epoch models, stochastic evaluations, and plots complete.
Start with [the results overview](results-overview.md) for the final findings and
data inventory. The benchmark and startup sections below are historical records.

## Machine and environment

The backend was implemented with three GPT-5.6 subagents covering training,
simulation, and metrics/documentation, followed by cross-review and integration.

The execution machine is an Apple M2 MacBook Air with 24 GB of unified memory.
The locked environment uses Python 3.12.11, PyTorch 2.14.0, and torchvision 0.29.0.
The training and simulation device is MPS. The managed command sandbox hides
the GPU, so GPU commands run with the approved host execution permissions.

The implementation includes exact uninterrupted-versus-resumed SGD/SAM tests,
CPU/MPS comparisons, analytical noise mean/variance checks, persistent-chip
semantics, and complete artifact serialization checks. The test suite can be
rerun with `.venv/bin/python -m pytest -q`; GPU checks skip if MPS is unavailable.

## Measured short benchmarks

These are warm, short measurements, not sustained training guarantees. Synthetic
images were used; data loading, validation, and laptop thermal changes can add
time to complete training epochs.

| Training method | Batch size | Seconds per batch | Images per second |
|---|---:|---:|---:|
| SGD | 64 | 0.0401 | 1,598 |
| SAM | 64 | 0.0851 | 752 |
| SGD | 128 | 0.0730 | 1,754 |
| SAM | 128 | 0.1504 | 851 |

At batch size 128, the short benchmark implies about 26 seconds of training
computation per SGD epoch and 53 seconds per SAM epoch over 45,000 images.
The sum over 150 epochs each is about 3.3 hours before the additional costs above.
Use measured `elapsed_seconds` from the real training logs for an updated estimate.

| Sampling mechanism | Workload | Elapsed seconds | Executions per second |
|---|---|---:|---:|
| Independent transient, all layers | 16 images × 4 reads | 0.195 | 327 |
| Spatially shared transient, all layers | 16 images × 4 reads | 0.196 | 327 |
| Persistent weights, all layers | 16 images × 4 chips | 0.067 | 952 |
| Independent transient, selected image | 1 image × 64 reads | 0.315 | 203 |

The sampling benchmark used a randomly initialized, folded network and relative
amplitude 0.1. It measures computational cost, not useful classification accuracy.
Direct Monte Carlo is the implemented inference engine; no moment approximation
is needed for this first backend.

Machine-readable measurements and source fingerprints are in:

- `artifacts/benchmark.json`
- `artifacts/sampling-benchmark.json`

## Execution process

The full workflow was launched on September 29, 2026 with:

```bash
.venv/bin/python -m analogdemo.pipeline --device mps --wait-for-data --detach
```

It runs under `caffeinate -i`, records a process lock, and writes:

- `artifacts/pipeline/launcher.json`: detached launch information.
- `artifacts/pipeline/status.json`: current stage, process ID, timestamps, source
  hashes, and errors.
- `artifacts/pipeline/run.log`: training progress and experiment output.

The status file records completion of every stage with no error. For future
runs, the same launch command resumes from the last completed epoch and reuses
completed stochastic artifacts whose inputs and provenance still match.

### Verified startup, September 29 at 11:09 a.m. Eastern

CIFAR-10 finished downloading and passed integrity verification. The archive is
170,498,071 bytes with MD5 `c58f30108f718f92721af3b95e74349a`. The frozen metadata
contains 45,000 training, 5,000 validation, and 1,024 calibration image indices.
Training and validation do not overlap, and every calibration image belongs to
the training split.

Both real-data pilots passed:

| Method | Epoch 1 validation accuracy | Epoch 2 validation accuracy | Seconds per epoch |
|---|---:|---:|---|
| SGD | 63.50% | 64.52% | 39.66, 38.00 |
| SAM | 61.38% | 65.06% | 68.68, 67.11 |

Both training losses decreased. The small stochastic preflight completed in
about 3.6 seconds and wrote raw logits, summaries, and the full artifact index
under `artifacts/experiment-smoke/`. Full SGD training resumed at the third epoch
at 11:09 a.m. Eastern. Full SAM training and final evaluation subsequently
completed; their results are in the overview linked above.

The pilot epoch times project to about 4.45 hours for 150 epochs of each method,
before final stochastic evaluation and any change in sustained performance.
The pilot accuracies are startup checks, not a conclusion about SAM robustness.

The sequence is data readiness, two SGD pilot epochs, two SAM pilot epochs, a
small artifact preflight, completion of 150 SGD epochs, completion of 150 SAM
epochs, final calibration/evaluation, then diagnostic plots. The pilots retain
the full learning-rate schedule and count toward the 150 epochs.

## Artifacts to use for the demonstration

Read [interface-data.md](interface-data.md) for exact array shapes, fields, units,
and loading examples. The entry point for the final data is
`artifacts/experiment/index.json`. It links:

- The selected checkpoints and fixed per-model calibration scales.
- Original display images, labels, canonical image IDs, and class names.
- Frozen noise scenarios and their selection rule.
- Raw float32 logits with axes `[chip, read, image, class]`.
- Per-image logit covariance, class-win frequencies, probability and margin
  quantiles, and dataset accuracy summaries.
- Paired model comparisons with explicitly labeled resampling units.
- A human-readable `results.md` report. Diagnostic plots are in `figures/`.

`artifacts/experiment-smoke/` is an implementation preflight based on pilot
checkpoints. It must not be presented as the final trained-model result.

Raw logits and summaries are empirical results of the stated Gaussian mechanism
model. Their sampling uncertainty does not include uncertainty about a physical
chip or variation across training seeds. The initial experiment uses one paired
training seed and reports SAM's measured behavior without assuming an advantage.
