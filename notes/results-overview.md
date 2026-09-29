# Completed backend: what was run and what was produced

## Status

**The full workflow finished successfully on September 29, 2026 at 3:51 p.m.
Eastern.** Both models completed 150 epochs, followed by calibration, stochastic
evaluation, and diagnostic plots. The pipeline recorded no failure.

This is the starting point for reading the results. For exact array fields and
loading examples, continue with [the interface data contract](interface-data.md).
The original design is in [the backend plan](backend-plan.md), and local setup,
benchmarks, and startup history are in [the execution notes](local-execution.md).

## What was done

Everything ran on the M2 MacBook Air with 24 GB of memory, using PyTorch's MPS
backend. The network is a VGG-style classifier with six convolutions and a linear
head, totaling 1,148,874 trainable parameters.

CIFAR-10 was downloaded and integrity-checked. We froze 45,000 training images,
5,000 validation images, and the official 10,000-image test set. A separate
calibration selection contains 1,024 images from the training split.

Two models used the same initialization, data split, batch order, and augmented
inputs. One used SGD and the other used SAM with SGD as its underlying optimizer.
Each trained for 150 epochs. The checkpoint with highest clean validation
accuracy was selected independently for each method. Batch normalization was
then folded into the convolutions for inference.

| Model | Training time, summed epochs | Selected epoch | Best validation accuracy | Clean test accuracy |
|---|---:|---:|---:|---:|
| SGD | 1.40 hours | 144 | 93.94% | 93.07% |
| SAM | 2.91 hours | 148 | 93.86% | 93.67% |

Epoch numbers in this table start at one; checkpoint metadata starts at zero.
Final calibration and stochastic evaluation took about 27 minutes. The entire
workflow after data preparation took about 4 hours 46 minutes.

## What the noise experiments mean

The simulator executes the actual nonlinear network for every sampled
realization. It models three general error mechanisms:

1. **Independent transient noise:** independent Gaussian errors added to affine
   outputs before ReLU, or to the classifier's logits. Each execution gets a new
   draw.
2. **Spatially shared transient noise:** the same marginal error variance, but
   with correlation 0.75 across spatial positions within each output channel.
   Classifier noise remains independent across its outputs.
3. **Persistent weight error:** independent Gaussian perturbations of stored
   weights, held fixed across all images evaluated on a simulated chip.

An amplitude of 0.3 means noise standard deviation equal to 30% of the clean
layer-output RMS for transient errors, or 30% of folded-weight RMS for weight
errors. Calibration scales are fixed for each checkpoint. These are normalized
sensitivity comparisons, not physical voltage/current specifications.

The validation sweep covered amplitudes 0.01, 0.03, 0.1, 0.3, and 1.0. Final
amplitudes were selected using validation results and frozen before test-set
evaluation. The rule selected 0.3 for both transient and persistent mechanisms.
Early, middle, and late comparisons affect two convolutions each; the all-layer
case affects all six convolutions and the classifier.

## Main test results

These are expected accuracies of a single noisy execution, averaged over all
10,000 test images and eight reads or eight simulated chips. Clean inference
uses one deterministic execution per image.

| Condition | SGD | SAM |
|---|---:|---:|
| Clean | 93.07% | 93.67% |
| Independent noise, early layers | 87.83% | 88.38% |
| Independent noise, middle layers | 89.53% | 90.93% |
| Independent noise, late layers | 91.51% | 92.48% |
| Independent noise, all layers | 80.61% | 80.84% |
| Spatially shared noise, all layers | 56.19% | 61.21% |
| Persistent weight error, all layers | 88.88% | 89.94% |

Three observations are useful for the demonstration:

- **Location matters:** equal relative transient noise in the early pair of
  convolutions hurt accuracy more than noise in the late pair in this experiment.
- **Correlation matters:** shared transient noise was substantially more damaging
  than independent noise at the same marginal variance.
- **Training matters, but the effect depends on the mechanism:** SAM's advantage
  was about five percentage points under shared noise. Under independent noise
  in all layers, its advantage was only 0.24 percentage points; the paired-image
  95% interval includes zero (approximately -0.10 to +0.55 percentage points).
  SAM also started with a 0.60-point clean advantage, so higher noisy accuracy
  alone does not establish a smaller degradation from its own clean baseline.

This is **one pair of training seeds** under a simplified Gaussian mechanism
model. Stored uncertainty estimates describe specified image/read/chip sampling
units; they do not establish general superiority of SAM across training seeds
or predict a particular analog chip.

## What data was produced

The final experiment is about 224 MiB, excluding trained checkpoints and the
source CIFAR-10 files. Its entry point is
[artifacts/experiment/index.json](../artifacts/experiment/index.json).

| Panel | Images | Conditions | Sampling per nonzero condition and model |
|---|---:|---:|---|
| Validation sweep | 500, balanced across classes | 30 noisy + clean | 4 reads or 4 chips |
| Official test | 10,000 | 6 noisy + clean | 8 reads or 8 chips |
| Illustrative examples | 6 validation images | 6 noisy + clean | 512 reads or 128 chips |

The example images span ranks of the SGD model's clean classification margin.
They are illustrative; the aggregate test results do not use this small subset.

There are **90 raw-logit files**: 45 panel/condition combinations evaluated with
both models. Each raw NPZ contains:

- `logits`: float32 array with axes `[chip, read, image, class]` and 10 classes.
- `labels`: the true class for each image.
- `image_ids`: canonical identifiers that join directly to the image panels.

Each raw file has a JSON provenance manifest, a JSON accuracy/uncertainty summary,
and a derived NPZ with per-image logit means and covariance, mean softmax
probabilities, probability quantiles, class-win frequencies, and margin quantiles.
There are also 45 paired SGD/SAM comparison files.

Across both models, the recorded runs contain 1,133,268 image-forward executions.
A final audit of all 90 result sets found no broken links, checksum mismatches,
label/image-ID mismatches, nonfinite logits, or inconsistent array shapes.
Probability normalization, quantile ordering, covariance symmetry, and the two
current checkpoint identities also passed verification.

The index links to the original display images, class names, calibration scales,
noise configurations, checkpoint identities, and all result files. Training
checkpoints and epoch histories are under `artifacts/training/{sgd,sam}/`.
No retraining is needed to build the interface from these outputs.

### Recommended reading and viewing order

1. This overview: experimental design and findings.
2. [Validation accuracy curves](../artifacts/experiment/figures/validation-accuracy-sweep.png)
   and [test accuracy plot](../artifacts/experiment/figures/test-accuracy.png).
3. [Interface data contract](interface-data.md): exact fields, units, and loading
   examples for interface development.
4. [Generated results table](../artifacts/experiment/results.md) and
   [machine-readable inventory](../artifacts/experiment/index.json).

Use `artifacts/experiment/` for the demonstration. The separate
`artifacts/experiment-smoke/` directory contains the earlier pilot integration
check and is not the final dataset. The interface itself has not been built.
