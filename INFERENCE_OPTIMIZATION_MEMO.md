# Optimizing native stem inference for an inexpensive Raspberry Pi

Design exploration, 21 October 2026. Companion to
[NATIVE_STREAMING_DESIGN.md](NATIVE_STREAMING_DESIGN.md).

Our goal is one useful audio stem from a continuous iPhone Bluetooth stream on an
inexpensive new Raspberry Pi 5 with 2GB RAM and analog audio output. Train and
experiment in Python; ship a native application with one exported model, bounded
buffers, and no Python, browser, or swap. Quality may be traded for throughput,
but each trade must be measured on representative music.

Testing takes place in a sandbox deliberately restricted to the target device's
resources. Direct testing on a Raspberry Pi is not part of this plan. Numerical
estimates justify candidates and establish budgets; they do not certify device
performance.

## Starting budget

Use the first memo's candidate: mono 16kHz, 250ms windows advancing 125ms, 512-point
STFT with 384-sample hop, 64 encoded frequency positions, feature and hidden widths
64, and two dual-path bidirectional LSTM blocks. Each block has two recurrent paths.

Its recurrent matrices and projections cost about 3.146 billion floating-point
operations per audio-second, including twofold overlap. Five times playback speed
requires about 15.73 GFLOP/second for this subtotal. Encoder, decoder, nonlinearities,
FFT, resampling, and data movement add work. There are about 299,264 parameters in
the recurrent core; allow 1–5 million for the whole model until it is specified.

Targets: average at most 25ms of local processing for each 125ms advance, p99 at
most 50ms after warmup, peak whole-pipeline occupied memory at most 1GiB including
loading, and one hour without underruns, accumulating backlog, or swap. The 2GB
sandbox limit includes supporting services, not just neural network tensors.

## Quantization and numeric precision

Quantization maps values onto a smaller integer range using a scale and possibly
a zero point; casting floats directly to uint8 does not implement it. INT8 and
UINT8 both use one byte. Select signedness and quantization scheme to match the
native backend's fast kernels. Integer dot products generally accumulate in INT32;
not every tensor or operation becomes eight-bit.

| Representation | Weight payload for 1–5 million parameters | Main opportunity |
| --- | ---: | --- |
| FP32 | 4–20MB | Reference behavior and broad operator support |
| FP16 | 2–10MB | Half-sized weights and eligible activations |
| INT8 or UINT8 | 1–5MB plus scales and metadata | Smaller weights and integer matrix kernels |
| INT4, exploratory only | 0.5–2.5MB plus metadata | Smaller payload if efficient kernels exist |

These are decimal MB and count stored weight values only. Packed copies, bias,
scales, accumulators, and runtime workspace remain. A fourfold reduction in weight
storage does not imply fourfold reduction in process RAM or execution time. INT4
may introduce unpacking overhead on this CPU and is a low priority for this small
model. FP16 storage savings do not establish FP16 CPU acceleration either.

Compare weight-only quantization with quantized weights and activations. Weight-only
schemes can still perform floating-point computation. Dynamic activation
quantization computes scales during inference; static calibration fixes scales
using representative inputs. Quantization-aware training lets the model adapt to
rounding and clipping. Start with representative calibration and evaluate whether
QAT or mixed precision is needed. Preserve floating-point FFT, reconstruction,
and sensitive recurrent state where appropriate; confirm that the chosen backend
actually supports the intended mixed path without excessive conversions.

Representative calibration/listening material should include quiet passages,
transients, strong bass, cymbals, dense mixes, varied levels, and long continuous
sequences. Listen for leakage, lost attacks, pumping, and recurrent-state drift.

For planning, consider quantized matrix-kernel speedups of 1.5–3 times. These are
scenarios to test, not benchmark predictions. If eligible matrix operations account
for 80% of runtime, Amdahl's law gives speedup = 1 / (0.2 + 0.8 / kernel_speedup):

| Kernel speedup | Whole-pipeline speedup | Hypothetical 25ms baseline becomes |
| --- | ---: | ---: |
| 1.5x | 1.36x | 18.3ms |
| 2x | 1.67x | 15.0ms |
| 3x | 2.14x | 11.7ms |

Profile the eligible fraction rather than assuming 80%. Quantization leaves the
number of logical multiply-adds largely unchanged; report integer operation counts
separately from floating-point counts. Operator fallback or conversion overhead
can remove the expected advantage.

Reference: [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## Tensor dimensions and layer count

For comparable recurrent paths, matrix arithmetic approximately scales as:

    blocks × encoded_frequency_positions × frames_per_second × overlap
        × hidden_width × (input_width + hidden_width)

When input and hidden widths shrink together, the width term is approximately
quadratic. This is the strongest simple architectural lever. Actual time also
depends on kernel utilization, packing, memory layout, and sequential dependencies.

| Change from the starting geometry | Recurrent/projection arithmetic relative to baseline | Subtotal GFLOP/audio-second |
| --- | ---: | ---: |
| Width 64 to 48, including proportional projections | 0.5625 | 1.77 |
| Width 64 to 32 | 0.25 | 0.79 |
| Two dual-path blocks to one | 0.5 | 1.57 |
| Encoded frequency positions 64 to 32 | 0.5 | 1.57 |
| Width 48 and one block | 0.28125 | 0.88 |

These exclude the surrounding model and small nonlinear/bias costs. Reducing
frequency positions needs an actual compression operation and can discard useful
frequency detail. Reducing depth or width needs retraining and quality assessment.
Structured pruning that physically removes channels/layers can produce these
smaller dense tensors. Arbitrary zero weights do not guarantee faster dense kernels.

Benchmark widths 32, 48, 64, and 96 with fixed shapes and batch size one. Multiples
of 8 or 16 are sensible candidates for vectorized kernels, not universal optimums.
Measure the complete path including transposes, contiguous copies, and padding;
a padded width 48 kernel might execute more work than its logical count suggests.

Use fixed-size windows, precomputed constants, persistent execution sessions,
reused allocations, and fused operators where supported. Avoid moving tensors
between incompatible layouts at each layer. Across-window batching can improve
utilization but adds waiting and buffering; keep it disabled in the initial stream.

A causal unidirectional recurrent design could remove the reverse direction and
reuse time state, but it changes projections, context, and training. It is a
separate model candidate, not a guaranteed twofold speedup for the whole system.
Also consider a small convolutional temporal model if its native kernels outperform
the LSTM at comparable quality. Choose by measured execution and listening results.

## Sample rate, channel count, and spectral resolution

Train and infer at the same selected internal sample rate and channel convention.
Bluetooth decoding and the output DAC may use another rate; resample explicitly.

| Internal rate | Samples in 250ms | Highest representable frequency | Frames/second at fixed hop of 384 samples |
| --- | ---: | ---: | ---: |
| 24kHz | 6,000 | Below 12kHz | 62.5 |
| 16kHz | 4,000 | Below 8kHz | 41.67 |
| 12kHz | 3,000 | Below 6kHz | 31.25 |
| 8kHz | 2,000 | Below 4kHz | 20.83 |

At fixed FFT size, sample hop, and encoded frequency positions, dropping 16kHz to
12kHz reduces recurrent frame work about 25%; 8kHz reduces it about 50%. But a
512-point FFT spans 32ms at 16kHz and 64ms at 8kHz. If we instead keep the hop fixed
in milliseconds, frames/second stay the same and this recurrent saving disappears
unless we also reduce frequency positions or other dimensions. Record all spectral
settings with each result. Lower sample rate is a fidelity/representation choice,
not an independent speed multiplier under every parameterization.

Mono halves raw input storage and channelwise DSP relative to stereo. It does not
halve a shared latent backbone whose width and shape are unchanged. Mix-down can
also cancel out-of-phase content and removes stereo spatial cues. Evaluate it on
real mixes rather than just duplicating one channel. A future stereo-preserving
mask predicted from a mono representation is another quality/complexity tradeoff,
not part of the initial one-mono-stem contract.

Sweep STFT sizes 256/512/1024 and appropriate hops jointly with model training.
Fewer frames can save more model work than shaving FFT implementation time, but
larger hops reduce time resolution and constrain reconstruction. Only use
window/hop combinations with verified overlap-add behavior. Use an established
native FFT library; focus on eliminating redundant transforms rather than writing
an FFT implementation.

## Windows, state reuse, and output scope

Keep window duration, advance, and overlap distinct. Shortening both duration and
advance proportionally does not inherently improve steady-state arithmetic per
audio-second, although it reduces first-window wait. It also increases call and
boundary overhead per second.

Changing twofold window overlap to no overlap could approximately halve repeated
model work, but may introduce boundary artifacts. A causal model with retained
state may avoid repeatedly processing old context without losing the same context
history. Test those as separate trained/reconstruction designs. Retain only bounded
history; never let state or queues grow with track length.

Predict one selected stem. Omitting other output channels saves their final head
and synthesis work, but does not divide a shared backbone's cost by four. A model
trained specifically for that stem may support a smaller backbone; measure that
quality tradeoff. Parallel independent stem models multiply total work and are
outside this one-stem cheap-device budget.

For first-output latency, the starting 250ms context is already a large floor.
Reducing inference from 25ms to 15ms changes approximately 275ms to 265ms before
transport/playback buffering. Improving throughput primarily buys steady-state
headroom, reduced duty cycle, or capacity for a better model. Shorter context or
a causal architecture addresses the larger algorithmic latency component.

## Native pipeline and resource control

Use one native inference session with bounded PCM queues and reusable workspace.
Remove development dependencies and unused operators from the deployed runtime.
Limit inference threading deliberately; compare one, two, and three inference
threads with four available CPU cores, leaving headroom for audio services. More
threads can lose on small operations and hurt p99 latency. Prevent nested thread
pools from multiplying workers. Keep allocation, logging, model loading, and
inference out of the audio callback.

Avoid repeated PCM conversions and resampling, but preserve the input/output
formats needed by the actual audio backend. Choose a fixed maximum queue depth and
an explicit overload policy; buffering must not hide a failure to keep up. Sustained
cooling and power are hardware assumptions, not benefits a sandbox can verify.

References: [ONNX Runtime custom builds](https://onnxruntime.ai/docs/build/custom.html)
and [thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

## Resource-limited sandbox validation

All planned performance experiments run in a sandbox constrained to the Pi target,
not directly on a Raspberry Pi. The test setup must record and enforce:

- A 2GB whole-sandbox memory limit, with swap disabled and no host swap activity
  attributable to the experiment. Include audio services and supporting processes.
- A maximum of four CPU cores and explicit inference thread counts. Record host
  CPU, architecture, affinity, quotas, and any contention.
- Fixed real-time-paced input, bounded queues, and a simulated output sink that
  consumes audio at playback rate. Separately run unpaced inference for throughput.
- One model session and representative Bluetooth decode/resampling workloads or
  recorded decoded PCM with separately budgeted transport costs.
- Peak resident/system memory, loading transients, processing p50/p95/p99/max,
  deadline misses, queue depth, and long-run output quality.

Four host cores or a matching clock rate do not reproduce four Cortex-A76 cores.
A CPU quota limits scheduled time rather than changing instruction throughput,
cache sizes, memory bandwidth, or latency. An ARM64 container on ARM hardware can
exercise native ARM code but still has a different microarchitecture. Emulation
can check compatibility, but its timings do not directly predict native speed.

Use a declared calibration workload representative of the selected kernels to
choose a conservative CPU budget when credible target performance data are
available. Report sensitivity to stricter quotas. Without that calibration,
sandbox results establish resource-constrained host feasibility, not a certified
Pi real-time result. CPU quotas can introduce artificial stalls, so report both
CPU time and wall-clock deadlines. A sandbox also does not reproduce the Pi radio,
DAC/driver behavior, thermals, or power draw; keep these outside claimed validation.

No direct Pi testing is required by this memo. The uncertainty is handled by
reporting assumptions, resource limits, and timing margin honestly.

## Experiment order and decision record

1. Build the native FP32 baseline and sandbox pipeline before full training.
   Verify export parity, fixed shapes, bounded memory, and actual operator kernels.
2. Profile recurrent paths, encoder/decoder, transforms, copies, and audio services.
   Benchmark representative kernels at widths 32/48/64/96 and one/two blocks.
3. Compare FP32, FP16, and supported INT8/UINT8 paths on identical geometry.
   Measure full-process speed and memory, including conversions and packed weights.
4. Train a small set of promising shapes. Separately vary mono/sample rate,
   compressed frequency positions, context, and overlap; avoid changing everything
   at once, so quality losses and savings remain attributable.
5. Apply static calibration or QAT and mixed precision where necessary. Consider
   distillation from the larger reference model to improve the smaller student's
   quality; distillation itself does not reduce an unchanged student's inference work.
6. Run a one-hour paced stream under the declared sandbox limits and stricter CPU
   budgets. Select the smallest configuration meeting quality and resource targets.

For each candidate, save graph/backend version, precision per operator, parameter
count, operation counts, tensor shapes, DSP settings, sandbox configuration,
latency/throughput, peak memory, and listening/quality results. Compare native FP32
against its PyTorch counterpart numerically. Evaluate quantized output with explicit
error and separation-quality measures; do not require FP32 allclose after intentional
lossy quantization. Compare the retrained small model to SCNet XL for quality rather
than numerical equality.

Savings must not be multiplied blindly. Width, depth, frequency compression, and
frame rate jointly change arithmetic; quantization changes kernel execution cost;
threading changes utilization. Profile and time the combined candidate. The goal is
a useful stem with measured margin under a declared cheap-device resource budget,
not the smallest checkpoint in isolation.
