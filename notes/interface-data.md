# Data contract for the demonstration interface

This document describes the version 1 artifacts produced by the backend. The
interface should read these files rather than rerun training or large stochastic
evaluations during a presentation. The first full experiment completed on
September 29, 2026; see [the results overview](results-overview.md) for its findings
and inventory. For any run, a file is available only when it appears in
`index.json` and exists at the recorded path. The interface must not infer
completion from the proposed scenario list.

## Experiment directory

The default root is `artifacts/experiment/`:

```text
index.json
panels/{validation,test,examples}.npz
panels/{validation,test,examples}.json
calibration/{sgd,sam}.json
runs/{validation,test,examples}/{scenario_id}/{sgd,sam}.npz
runs/{validation,test,examples}/{scenario_id}/{sgd,sam}.json
runs/{validation,test,examples}/{scenario_id}/{sgd,sam}.summary.json
runs/{validation,test,examples}/{scenario_id}/{sgd,sam}.details.npz
runs/{validation,test,examples}/{scenario_id}/comparison.json
scenarios-selected.json
scenarios-coarse.json
results.md
figures/validation-accuracy-sweep.png
figures/test-accuracy.png
```

`index.json` is the top-level inventory and provenance entry point. Its root
fields are `schema_version`, `status`, `smoke`, creation/update Unix timestamps,
`class_names`, `data_metadata_sha256`, `calibration_convention`, `environment`,
`inference_implementation`, `inference_runtime`, `panels`, `models`, and `runs`;
`selected_scenarios` is added after selection; `report` and `report_sha256` are
added when `results.md` is complete. `inference_implementation` gives
one combined SHA-256 plus individual hashes for `model.py`, `calibration.py`,
`noise.py`, `sample.py`, and `metrics.py`. `inference_runtime` records the Python,
PyTorch, torchvision, and NumPy versions that can affect numerical results.
Each completed run row embeds `panel`, the full `scenario`, the SGD and SAM
summaries, paths in `files`, and the `comparison` path. The runner updates the
index after each complete pair. The command
`python -m analogdemo.experiment --device mps` resumes work whose inputs and
metadata still match; `cpu` is a supported device choice.

`status: complete` means that particular experiment budget finished. The
interface must also inspect `smoke`: when it is `true`, panels, calibration, and
sample counts are deliberately reduced for an integration test and must be
labeled non-final. The full presentation should use an index with `smoke: false`.

## Artifact groups

### Training checkpoints

Each selected SGD or SAM checkpoint contains the VGG-7 model state, optimizer and
scheduler state for resumability, epoch, random-number states, merged training
configuration, frozen-data hash, and best validation accuracy. The epoch-by-epoch
training history is the separate `metrics.jsonl`; environment and optimizer-group
provenance are in `manifest.json`. For each method, `index.models` records
`checkpoint_path`, `checkpoint_sha256`, `selected_epoch_zero_based`,
`best_validation_accuracy`, calibration path and hash, maximum fold-check error,
and absolute paths to the training manifest and metrics. The interface should use
the checkpoint hash to join the selected model to stochastic results.

Checkpoint, manifest, and metric paths point outside the experiment directory and
can change when training resumes. This is especially visible after a smoke
experiment made from pilot checkpoints. The recorded `checkpoint_sha256` is the
model identity used for calibration and sampling; consumers must not assume the
current file at `checkpoint_path` still has that hash. A final experiment is
generated only after full training has completed.

### Images and class information

Each panel NPZ contains `images_uint8` with shape `[N,32,32,3]`, `labels`, and
`image_ids`; its adjacent JSON contains the split, selection rule, preprocessing,
hashes, and class map. The data artifacts also record fixed
train/validation/test split indices, split seed, augmentation seed, and the
CIFAR-10 class map. Image IDs refer to the original CIFAR-10 index within their
declared source split. `images_uint8` is already display-ready RGB data in NHWC
layout; the interface should use it directly. The saved mean and standard
deviation describe the normalized model-input tensors and are needed only if a
consumer reconstructs or reverses that model preprocessing. Illustrative images
are selected on validation data by a recorded clean-margin rule; aggregate test
results never use only that illustrative subset.

The index records a SHA-256 fingerprint of each panel NPZ. On reuse, the runner
also compares the saved image IDs and the complete panel JSON, including
`source_split`, `positions_in_split`, selection seed or rule,
`data_metadata_sha256`, class names, normalization, layout, and image count. A
changed source split or selection therefore invalidates the cached panel instead
of silently attaching old results to new images.

### Calibration

`calibration/sgd.json` and `calibration/sam.json` use this wrapper:

```text
schema_version: 1
calibration: {version, output_rms, weight_rms, num_examples}
metadata: {checkpoint_sha256, data_metadata_sha256, image_ids, convention,
           inference_implementation_sha256, inference_runtime}
```

Within it:

- `calibration.version` is `1`, and `calibration.num_examples` records the count;
- `output_rms`: clean preactivation RMS `s_l` for `conv1` through `conv6` and
  `classifier`;
- `weight_rms`: folded-weight RMS `r_l` for the same affine layers.

The ordered calibration IDs and hashes live under `metadata`, rather than inside
the reusable `Calibration` object. `metadata.convention` is
`per_checkpoint_clean_layer_rms`.

Transient amplitude `alpha` is dimensionless and gives standard deviation
`alpha * s_l`. Persistent weight amplitude `beta` is dimensionless and gives
standard deviation `beta * r_l`. Scales belong to one checkpoint and remain fixed
through a sweep. Equal amplitudes therefore mean equal relative perturbation,
not equal physical voltage or current error.

### Raw stochastic logits

Each `{method}.npz` under `runs/` is written by `sample.save_sample_artifact` and
has an adjacent JSON manifest.
The NPZ fields are:

| Field | dtype and shape | Meaning |
|---|---|---|
| `logits` | float32 `[C,R,N,10]` | Raw logits by chip, read, image, class |
| `labels` | int64 `[N]` | True class for each ordered image |
| `image_ids` | Unicode `[N]` | Stable source-split image identifiers |

The manifest contains `schema_version`, `created_utc`, `array_file`, an `arrays`
shape/dtype map, `num_chips`, `num_reads`, `num_images`, `seed_scheme`,
`write_seconds`, `noise_config`, `calibration`, `environment`,
`sampling_seconds`, `images_per_second`, and `array_sha256`. Its `experiment`
object contains `checkpoint_sha256`, `calibration_sha256`, `panel_sha256`, the
complete `scenario`, `chips`, `reads`, `master_seed`, `device`, and
`data_metadata_sha256`, plus `inference_implementation_sha256` and
`inference_runtime`. The noise configuration has `version`, per-layer
`transient`, per-layer `weight`, `spatial_rho`, and per-convolution `threshold`
fields. The embedded `calibration` is the reusable numeric object; its standalone
wrapper carries the checkpoint and implementation provenance.

Before accepting a cached raw result, the runner requires that entire
`experiment` fingerprint to match and verifies `array_sha256`. This binds the
result to its checkpoint, calibration, panel contents, frozen data metadata,
scenario, sampling budget, seed, and device. The pipeline status separately
records source and configuration file hashes for every stage; those hashes are
provenance for the run and are not a claim that artifacts from different
software versions are numerically interchangeable.

Random draws are keyed by mechanism, layer, chip ID, read ID, and image ID. A chip
draw persists for all images and reads on that chip. A transient draw changes by
read and image. Scenario ordering or batch size does not redefine those IDs.

### Derived metrics

For each raw `{method}.npz`, metrics are `{method}.summary.json` and
`{method}.details.npz`. The JSON is presentation-ready and contains:

- sample counts and the exact tie rule;
- expected single-execution noisy accuracy;
- optional clean accuracy, accuracy change, class-change frequency, newly-wrong
  frequency, newly-correct frequency, and wrong-to-different-wrong frequency;
- accuracy for each whole chip and for each read within each chip;
- uncertainty fields labeled by image, read, and whole-chip sampling units.

The NPZ contains numeric per-image arrays:

| Field | shape | Meaning |
|---|---|---|
| `logit_mean` | `[N,10]` | Mean across chip/read executions |
| `logit_covariance` | `[N,10,10]` | Empirical pooled chip/read covariance; zeros for one execution |
| `softmax_mean` | `[N,10]` | Mean of softmax for each execution |
| `softmax_quantiles` | `[N,Q,10]` | Probability distribution quantiles |
| `class_win_frequency` | `[N,10]` | Fraction of executions won by each class |
| `true_margin_quantiles` | `[N,Q]` | Quantiles of true logit minus best other logit |
| `accuracy` | `[N]` | Correct-execution fraction per image |
| `top_tie_frequency` | `[N]` | Fraction with two or more equal maximum logits |
| `quantile_levels` | `[Q]` | Quantiles used by both quantile arrays |

When clean logits are supplied, the NPZ also includes `clean_prediction`,
`clean_correct`, and per-image `class_change_frequency`. Softmax is calculated per
execution before averaging. NumPy `argmax` resolves an exact top tie to the lowest
class index; tie frequency remains visible. Accuracy means a randomly selected
single execution. It is not accuracy after averaging logits.

The covariance and quantiles describe the finite sampled execution distribution.
For a combined chip-by-read run, pooled executions are correlated within chip, so
the usual `n-1` covariance normalization is descriptive rather than an iid
uncertainty estimate. The separately labeled chip/read/image fields carry the
sampling-precision information.

Paired model comparisons resample whole image IDs. Chip intervals resample whole
chip realizations and are unavailable with fewer than two chips. No interval
treats flattened chip/read/image decisions as independent observations.

The JSON uncertainty fields have three distinct conditional meanings:

- `uncertainty.image.mean_accuracy_standard_error` describes variation over the
  panel's per-image expected accuracies, conditional on the sampled chips and
  reads. It is `null` for one image. Its usual standard-error interpretation
  treats sampled images as the replication unit; it does not include training-set
  or training-seed uncertainty.
- `uncertainty.read.per_chip_mean_accuracy_standard_error` uses complete-panel
  read accuracies within each chip. It is `null` when there is only one read and
  does not describe variation between chips.
- `uncertainty.chip.mean_accuracy_standard_error` uses complete-chip accuracies
  on the shared image panel, averaging that chip's reads. It is `null` for one
  chip and is conditional on the chosen image panel.

These are standard errors of estimated means, not quantiles of the underlying
noise distribution and not confidence intervals. Distribution quantiles live in
the per-image details. The interface should keep those two concepts labeled
separately.

Each `comparison.json` declares `direction: sam_minus_sgd` and
`training_seed_pairs: 1`. `paired_images_conditional_on_sampled_noise` contains a
paired-image bootstrap estimate, confidence level and interval, repetition count,
and seed. It conditions on the finite noise realizations already sampled.
`paired_chips_conditional_on_image_panel` contains the analogous whole-chip
bootstrap when at least two paired chips exist; otherwise it explicitly returns
`available: false` and a reason. Neither interval measures variability across
training seeds, because this demonstration trains one SGD/SAM seed pair.

## Demonstration scenarios

The scenario helper can generate the full deterministic mechanism/location grid,
but the experiment runner evaluates a smaller declared coarse design. At each
dimensionless amplitude `0.01, 0.03, 0.1, 0.3, 1.0`, it runs independent
transient noise at early (`conv1,conv2`), middle (`conv3,conv4`), late
(`conv5,conv6`), and all affine layers; spatially shared transient noise at all
layers with convolutional `rho=0.75`; and persistent weight error at all layers.
The classifier has no spatial correlation. `scenarios-coarse.json` records these
30 configurations. Keeping two convolutions in each named location avoids
confounding the early/middle/late comparison with the number of injected
sources. The clean baseline is a separate zero-noise run.

The experiment workflow starts with a balanced 500-image validation panel and
four reads or four chips at nonzero amplitudes `0.01, 0.03, 0.1, 0.3, 1.0`.
`scenarios-selected.json` freezes six nonzero scenarios plus clean before the final test
run, which starts with eight reads or chips. Six validation examples spanning
SGD clean-margin quantiles receive refined budgets of 512 transient reads or 128
persistent chips. Actual counts in each manifest take precedence over these
starting budgets. `comparison.json` holds the paired model comparison for a
scenario, and `results.md` is the human-readable report.

The regular-versus-SAM comparison uses paired draws and each checkpoint's own
calibration. SAM improvement is not assumed. Scenario selection uses validation
results and is frozen before final test evaluation.

## Loading example

```python
import json
from pathlib import Path

import numpy as np

base = Path("artifacts/experiment/runs/test/example-scenario/sgd")
manifest = json.loads(base.with_suffix(".json").read_text())
with np.load(base.with_suffix(".npz")) as raw:
    logits = raw["logits"]       # [chip, read, image, class]
    labels = raw["labels"]
    image_ids = raw["image_ids"]

summary = json.loads(base.with_suffix(".summary.json").read_text())
with np.load(base.with_suffix(".details.npz")) as derived:
    probability_mean = derived["softmax_mean"]
    margin_quantiles = derived["true_margin_quantiles"]
```

The interface should reject unknown major schema versions, show sample counts,
mechanism, location, amplitude, correlation, checkpoint, panel, and units near
the displayed result, and label the simulator as a mechanism study rather than a
prediction for a physical analog chip.

`python -m analogdemo.plot --index artifacts/experiment/index.json` reads only
completed rows currently present in the index and writes static PNG diagnostics
under `figures/`. The plots show observed sampled accuracies without confidence
bars; uncertainty remains in the explicitly labeled summary and comparison data.
