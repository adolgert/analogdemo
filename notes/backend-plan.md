# Backend plan: analog inference noise on a MacBook

## Purpose and scope

Build the computational backend for a five-minute demonstration to AI scientists who do not specialize in hardware. Use a small VGG-style classifier trained on CIFAR-10 to show that the consequences of analog computation errors depend on their location, correlation, persistence, and the training procedure.

The demonstration will illustrate general error mechanisms. It will not assign noise laws to charge, voltage, or current technologies or predict the performance of a particular chip. All training, simulation, and artifact generation will run locally on this MacBook. The user interface is a separate next step.

The core comparison is ordinary training versus sharpness-aware minimization (SAM). Whether SAM improves robustness is an experimental result; the backend must preserve an unchanged or worse result as readily as an improvement.

## Main decisions

- Use PyTorch and torchvision, with the MPS device when supported and CPU as a reference and fallback.
- Train a VGG-style network with six convolutions and one linear classifier, approximately 1.15 million parameters.
- Train two checkpoints with the same architecture, data split, and epoch budget: SGD and SAM wrapping SGD.
- Simulate Gaussian errors at operation boundaries and in stored weights. Execute the nonlinear network for every sampled realization.
- Save raw logits, labels, example IDs, configurations, and random seeds. Derive presentation statistics from these artifacts.
- Start with direct Monte Carlo sampling. Add a moment approximation only if measured performance makes it necessary.
- Precompute the main demonstration scenarios so the eventual presentation does not depend on completing large evaluations live.

## 1. Establish the local execution environment

Create a small Python project with a locked dependency environment containing PyTorch, torchvision, NumPy, a plotting library, and a test runner. Use ordinary PyTorch tensor operations for the simulator; no circuit simulator is required.

Record the Mac model, chip, memory, macOS version, Python and package versions, selected device, and source revision in an experiment manifest. Check MPS availability at runtime instead of assuming every MacBook supports the same operations.

Before full training:

1. Download CIFAR-10 and verify its supplied integrity checks.
2. Run a clean forward/backward smoke test on CPU and the selected training device.
3. Benchmark batch sizes 64 and 128, including the two passes used by SAM.
4. Measure a complete short training run after warm-up, with device synchronization around timing measurements.
5. Estimate total training time from measured epochs. Include validation and checkpoint overhead, and allow for sustained laptop performance changing with temperature.

Start data loading with zero worker processes to simplify the first run, then benchmark additional workers if loading limits throughput. Train the two models sequentially. Save resumable checkpoints so a long run can continue after interruption.

Deliverable: an environment manifest and a short timing report. Do not promise a training duration or live inference latency before these measurements.

## 2. Data and model construction

### Data protocol

Make a fixed, stratified split of the 50,000 CIFAR-10 training images into 45,000 training and 5,000 validation examples. Keep the official 10,000-image test set separate.

- Training: random crop with four pixels of padding, horizontal flip, and fixed channel normalization.
- Validation and test: normalization only.
- Calculate normalization statistics from the training split and save them.
- Use validation data to choose training settings, noise ranges, and illustrative examples.
- Use the test set for final evaluation after those choices are fixed.

Save split indices and seeds so both training methods use identical examples. Augmentation randomness must have its own seed stream so SAM's extra forward pass does not change subsequent data augmentation choices.

### Architecture

Use 3×3 convolutions with stride 1 and padding 1. Each convolution is followed by batch normalization and ReLU. Convolutions have no bias during training.

| Stage | Operations | Output shape |
|---|---|---|
| Input | Normalized RGB image | 3 × 32 × 32 |
| Block 1 | Conv64, BN, ReLU; Conv64, BN, ReLU; max pool 2×2 | 64 × 16 × 16 |
| Block 2 | Conv128, BN, ReLU; Conv128, BN, ReLU; max pool 2×2 | 128 × 8 × 8 |
| Block 3 | Conv256, BN, ReLU; Conv256, BN, ReLU; max pool 2×2 | 256 × 4 × 4 |
| Head | Global average pool; linear 256→10 with bias | 10 logits |

This defines what “VGG-7” means in this project. Count and record the actual parameters when constructing it. Do not add dropout; inference variability should come from the specified error model.

After training, copy the model into evaluation mode and fold batch normalization into the adjacent convolutions. Verify that the folded model reproduces the original evaluation logits within a documented floating-point tolerance. Apply hardware errors to the folded weights and operations, never to a training-mode batch-normalized model.

Pooling, routing, bias addition, and the final software softmax are ideal in the initial simulator. Noise affects the six convolution outputs and the linear output; optional activation-threshold errors cover the nonlinearities. These are explicit abstraction choices.

Deliverable: a model constructor and a verified conversion from training checkpoint to inference model.

## 3. Train the regular and SAM checkpoints

### Starting recipe

Use the following as initial settings, subject to the short pilot and validation results:

| Setting | Initial value |
|---|---|
| Optimizer | SGD with momentum 0.9 |
| Batch size | 128 if the benchmark supports it |
| Initial learning rate | 0.1 at batch size 128 |
| Weight decay | 5e-4 on convolution and linear weights |
| Schedule | 5-epoch warm-up followed by cosine decay |
| Training duration | 150 epochs per method |
| SAM perturbation radius | 0.05, standard L2 SAM |
| Precision | Float32 initially |

Explicitly record optimizer parameter groups; keep batch-normalization parameters and biases out of weight decay. Record which trainable parameters receive the SAM perturbation. Use the same convention in every SAM run.

For each paired run, start both methods from the same initialization and use the same batch order and augmented inputs. Both methods see the same number of epochs; SAM performs approximately twice as many forward/backward passes, so this is a comparison at equal data exposure, not equal compute cost. Record wall-clock time for both.

### SAM implementation details

For each batch, compute the gradient at the current weights, perturb the weights in the normalized gradient direction, compute the gradient again on the same augmented batch, restore the original weights, and apply the SGD update using the second gradient.

Batch normalization needs explicit handling: update running statistics only on the first pass. The second pass should still use batch statistics without leaving another update in the running buffers. Verify this behavior, including the batch counter. Apply weight decay through the base optimizer update, not as a substitute for the SAM perturbation.

First run a short pilot for each method to catch learning, optimizer, and checkpoint problems. Then run the full pair. Select checkpoints by clean validation accuracy using a fixed tie-breaking rule; do not choose them for favorable noise robustness.

If validation indicates a poor training configuration, make a small, documented adjustment before final evaluation. Keep tuning effort comparable. A single paired seed is sufficient to build the demo. Repeat with two more paired seeds if a broad claim about SAM is part of the presentation; otherwise identify the displayed models as one trained pair.

### Saved training artifacts

- Best validation checkpoint and last resumable checkpoint.
- Model, optimizer, scheduler, epoch, and random-number states needed to resume.
- Training configuration, seed, split IDs, and environment manifest.
- Loss, accuracy, learning rate, and elapsed time by epoch.
- Clean validation and final test metrics for each selected checkpoint.

Deliverable: two trained models and their clean baselines. A SAM advantage is not an acceptance criterion.

## 4. Define and calibrate the noise mechanisms

### Amplitude convention

Use a fixed calibration subset of the training split, with deterministic preprocessing. For each folded model and affine layer, save the RMS of its clean preactivation output, denoted s_l, and the RMS of its folded weights, denoted r_l.

An output-noise amplitude alpha_l means a standard deviation of alpha_l × s_l. A weight-error amplitude beta_l means beta_l × r_l. These reference scales remain fixed when noise changes; never recalculate them from noisy activations or individual evaluation batches.

Each checkpoint uses its own clean calibration scales. Thus equal amplitudes compare equal relative perturbation levels in the two networks. This is a normalized sensitivity experiment, not identical physical noise voltages imposed on both models. Save and expose this convention in the result metadata.

### A. Independent transient computation noise

For an affine operation, including convolution:

    u_l = W_l h_(l-1) + b_l
    u_noisy_l = u_l + alpha_l s_l epsilon_l
    epsilon_l ~ N(0, I)

Inject the error before ReLU, or directly into the final logits for the classifier. Sample independently across output elements, images, layers, and repeated executions.

This is the simplest model of independently fluctuating computation/readout errors. It should be the first working mechanism.

### B. Persistent weight error

For each simulated chip, draw:

    W_chip_l = W_l + beta_l r_l E_l
    E_l ~ N(0, I)

Keep this realization fixed across the entire evaluation panel and all repeated reads for that chip. A convolution uses the same perturbed kernel at every spatial position. Different chip IDs receive independent realizations. Biases remain ideal in this mechanism.

The perturbations are applied to a copy or functional view of the clean weights. A new scenario must not accumulate changes on a previously perturbed checkpoint.

This mechanism demonstrates why repeated execution does not eliminate a fixed implementation error.

### C. Spatially correlated transient computation noise

For convolutional outputs, extend mechanism A to:

    error = alpha_l s_l [sqrt(1-rho) epsilon + sqrt(rho) c]

Here epsilon is independent at every output location and c is sampled once per image, output channel, layer, and execution, then broadcast over spatial positions. Set rho between 0 and 1.

This keeps each output element's marginal noise variance unchanged while changing spatial correlation. It permits a clear comparison of independent and shared errors at the same local RMS amplitude. Channels remain independent. The final classifier uses mechanism A, since it has no spatial dimension.

Do not use one common additive offset for every final logit as the primary correlation example: that leaves both softmax and argmax unchanged.

### Optional activation-threshold error

Represent a nonlinear element's threshold offset as:

    h_l = ReLU(u_noisy_l - theta_l)

Sample theta_l once per channel and simulated chip, scaled by the clean preactivation RMS. This is a persistent threshold error, not another name for transient affine-output noise. Implement after the three main mechanisms if nonlinear-element controls are needed.

Support per-layer amplitudes internally, but define the demonstration scenarios around early, middle, late, and all-layer changes. Keep clipping, quantization, drift, and circuit-specific effects outside the first version.

Deliverable: a versioned noise configuration, calibration artifact, and simulator with explicit chip and read lifetimes.

## 5. Sample and save logits

### Sampling contract

The central operation accepts a checkpoint, ordered image IDs, a noise configuration, a seed, a number of chip realizations C, and a number of transient reads R per chip.

Return logits with conceptual shape:

    [chip C, read R, image N, class 10]

Store labels, image IDs, checkpoint hash, calibration hash, complete noise configuration, seed scheme, sample counts, and timing alongside the logits. Large evaluations can stream summaries and save logits in chunks.

For transient-only scenarios, use C=1. For weight-only scenarios, use R=1: repeating an identical deterministic chip execution provides no additional evidence. Use both dimensions for combined mechanisms.

### Reproducibility and batching

Separate random streams for training, chip errors, transient errors, and scenario selection. Key noise realizations by seed, mechanism, layer, chip ID, read ID, and image ID so changing batch size does not silently change the experiment.

Reuse the same standardized draws across amplitude sweeps, scaling them by the requested amplitudes. Use paired draws across the two trained models as well. This reduces distracting Monte Carlo variation in comparisons; it does not guarantee monotone finite-sample curves.

Start with a CPU reference random-number path and check the selected device's reproducibility. Optimize noise generation only after profiling, preserving the sampling contract. Record device and version because cross-device bitwise equality is not required.

For transient noise, batch multiple images and reads together, within measured memory limits. For persistent errors, first implement a loop over chips, batching images and reads inside each chip. Avoid treating each image as a new chip. Cache clean activations before the earliest affected operation when this is useful and correct.

### Initial sampling budgets

These are starting budgets to benchmark, not latency promises:

| Use | Initial budget |
|---|---|
| Selected validation image, transient variability | 64 reads; refine to 512 |
| Selected image, persistent variability | 32 chips; refine to 128 |
| Quick aggregate preview | Fixed 500-image balanced validation panel, 4 reads or 4 chips |
| Offline final scenarios | Full test set; start with 8 reads or 8 chips and refine where uncertainty matters |

Combined scenarios need a separate C×R budget because their cost multiplies. Choose final sample counts from the precision of the displayed comparison, not from an arbitrary requirement to run the model thousands of times.

Deliverable: a reproducible command that creates raw logit artifacts without a user interface.

## 6. Derive scientifically interpretable statistics

For every sampled logit vector z, calculate probabilities with softmax and the predicted class with argmax. Keep these summaries distinct:

- Mean softmax probability: average of softmax(z) across executions.
- Class-win frequency: fraction of executions in which each class has the largest logit.
- Expected single-execution accuracy: average correctness over the declared images, chips, and reads.
- Ensemble accuracy, only if explicitly requested: accuracy after combining multiple executions. This is a different inference procedure.

Do not replace mean softmax with softmax of the mean logits, or single-execution accuracy with the accuracy of averaged logits.

For individual examples, save mean logits, the full 10×10 logit covariance, probability quantiles, class-win frequencies, and the true-class margin:

    margin = z_true - max(z_other classes)

The fraction of positive margins directly estimates correctness, apart from ties. A fixed pairwise margin between two named classes is also useful for showing how a decision changes.

For dataset summaries, report clean accuracy, noisy accuracy, change from each model's clean accuracy, and class changes relative to the clean prediction. Class changes include both newly wrong and newly corrected examples.

Distinguish three sources of uncertainty:

1. Repeated-read variation for a fixed chip and image set.
2. Variation between simulated chips.
3. Variation from evaluating a finite set of images.

For chip variation, evaluate each chip on the same image set and retain per-chip metrics. Do not treat all chip/read/image observations as independent Bernoulli trials. Use paired resampling of images for model comparisons and whole-chip resampling when estimating chip uncertainty. Label intervals by what was resampled; label distribution quantiles separately from uncertainty in an estimated mean.

Deliverable: summary tables and diagnostic plots generated from saved logits.

## 7. Prepare a small demonstration scenario set

Use validation data to choose amplitudes that span negligible, visible, and substantial degradation. Begin with a coarse relative-noise sweep such as 0, 0.01, 0.03, 0.1, and 0.3, then refine where behavior changes. Different mechanisms may require different useful ranges.

Prepare three main comparisons:

1. **Location:** equal relative noise in early versus late layers. Measure which is more consequential rather than assuming an ordering.
2. **Structure:** independent versus spatially shared noise at equal marginal variance; separately compare transient noise with persistent weight error.
3. **Training:** regular versus SAM under the same declared normalized-noise settings, showing clean baselines as well as degradation.

Choose a few validation images using a documented rule, such as covering large, medium, and small clean margins. Mark them as illustrative examples. Aggregate metrics must use the fixed evaluation panels, not a handpicked subset of dramatic failures.

Freeze these scenarios before final test evaluation. Precompute their refined logit samples and metrics. Retain results that complicate the story: the useful claim is that error structure and training matter, not that a particular layer or optimizer always wins.

## 8. Validate correctness and statistical behavior

Required checks focus on errors that could change the scientific interpretation:

- All amplitudes at zero reproduce clean inference.
- Folded and original evaluation models agree within the chosen tolerance.
- The SAM perturbation is restored before updating; batch-normalization running buffers advance only once per batch.
- A small linear example has the predicted Gaussian mean and covariance under independent additive or weight errors.
- Noise has the configured empirical variance, and mechanism C changes spatial covariance without changing marginal variance.
- Persistent weights stay identical across images and reads for one chip; transient draws change between reads.
- A common offset to all logits leaves class probabilities and predictions unchanged.
- Batching and chunking preserve the declared random realizations.
- Increasing sample counts stabilizes the chosen summaries within estimated sampling uncertainty.
- CPU and MPS produce compatible clean results and compatible sampled statistics; exact random trajectories across devices are unnecessary.

For the final artifact, report sampling precision and, where applicable, variation across training seeds. Statistical agreement with this simulator is not validation against a physical chip; the stated mechanism assumptions define the model's scope.

## 9. Performance decision: whether moments are needed

Benchmark the completed sampler before adding another inference model. Precomputed scenarios, batched reads, and progressive sample refinement may be sufficient for the later interface.

If immediate arbitrary slider feedback still requires acceleration, add a local approximation to the ten logits:

    covariance_z(x, amplitudes) ≈ sum_l amplitude_l² C_l(x)

Estimate each 10×10 contribution C_l from local derivatives or small-noise perturbation experiments for each checkpoint, image, and noise mechanism. This assumes independent sources and locally fixed sensitivities. Preserve the full class covariance, then sample logits from the resulting Gaussian approximation.

Validate it against direct network sampling at held-out amplitude combinations, comparing margin distributions and class-win probabilities as well as covariance. Changes in ReLU gates, max-pool winners, and noise-induced mean shifts can invalidate the approximation. Restrict its amplitude range accordingly and label approximate results. Never add independent single-layer accuracy losses to predict a combined-noise accuracy.

This is an optional acceleration phase, not a prerequisite for the first backend.

## 10. Implementation order and completion criteria

Suggested project organization:

```text
src/analogdemo/
    data.py          # Splits, preprocessing, example IDs
    model.py         # VGG-7 and batch-normalization folding
    train.py         # SGD/SAM, validation, resume
    calibration.py   # Frozen clean reference scales
    noise.py         # Mechanisms and chip/read state
    sample.py        # Batched logit sampling
    metrics.py       # Statistics and uncertainty estimates
    scenarios.py     # Reproducible experiment configurations
configs/
tests/
artifacts/           # Ignored by git; manifests accompany large outputs
notes/
```

Implement in this order:

1. Environment, data split, model, and short training benchmarks.
2. Correct regular and SAM training; full checkpoint pair.
3. Folding, calibration, and independent transient noise.
4. Persistent weight errors, spatial correlation, and sampler validation.
5. Logit artifacts, metrics, validation sweeps, and scenario selection.
6. Final evaluations and performance report.
7. Optional extra training seeds, activation errors, or logit approximation as justified by the results.

The backend is ready for interface work when both checkpoints and clean baselines exist, the noise and sampling checks pass, the three scenario comparisons are reproducible from saved configurations, and a local callable API returns logits and summaries with explicit sample counts and provenance. The final report should state measured training time, sampling throughput, useful noise ranges, observed SAM behavior, and unresolved limitations.

## References

- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html): local GPU execution on supported Macs.
- [Foret et al., Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412): the training method and its objective.
- [IBM AIHWKit hardware-aware training and inference](https://aihwkit.readthedocs.io/en/v0.9.0/hwa_training.html): persistent weight effects versus transient computation noise.
- [IBM AIHWKit operation parameters](https://aihwkit.readthedocs.io/en/latest/api/aihwkit.simulator.parameters.io.html): examples of separate input, output, and weight-error mechanisms.

These references inform the abstractions. The project uses its own small PyTorch simulator and makes no claim of reproducing a specific analog device.
