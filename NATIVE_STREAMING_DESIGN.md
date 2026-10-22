# Native streaming stem separation on Raspberry Pi

Design concept, 21 October 2026. Proposed deployment target: Raspberry Pi 5,
2GB RAM, one selected output stem, Bluetooth input from an iPhone, and wired
analog output. Python and PyTorch are development and training tools only.

This design concerns a new, smaller, retrained model. The existing SCNet XL
checkpoint is a reference for quality and behavior, not the model to deploy on
this hardware. The estimates below justify building a prototype; they are not
Pi benchmarks or demonstrated separation quality.

## Product and runtime

The device behaves as a Bluetooth audio receiver. The user pairs an iPhone,
plays music, and listens to the selected stem through headphones or an amplifier.

Audio flows through these stages:

1. iPhone sends a Bluetooth A2DP stream to the Pi's built-in radio.
2. BlueZ and an audio backend such as BlueALSA decode the negotiated codec into PCM.
3. A native application resamples and mixes down to the model's input format.
4. Bounded buffers feed fixed-size windows to a native inference runtime.
5. Overlap-add reconstruction emits completed output samples into an ALSA buffer.
6. A USB audio DAC drives a 3.5mm headphone output. RCA line output can instead
   use a suitable DAC or cable and output-level configuration.

Run minimal 64-bit Linux without a desktop or browser. Load one model once.
Disable swap. Allocate and reuse audio buffers, recurrent state, feature tensors,
and scratch space wherever the backend allows. Keep inference work outside the
real-time audio callback; the callback only moves samples through bounded queues.

Export the model graph and weights from PyTorch to ONNX, then evaluate a reduced
native ONNX Runtime build against other native backends if necessary. Include
only required operators. Keep FFT, resampling, and overlap-add in native DSP code
if that gives a simpler and faster execution path than including them in the
exported graph. Exportability and operator performance must influence model design
before training. A browser/WebGPU application is another possible deployment,
but it is not part of this smallest dedicated-device implementation.

References: [ONNX Runtime C API](https://onnxruntime.ai/docs/get-started/with-c.html),
[reduced runtime builds](https://onnxruntime.ai/docs/build/custom.html), and
[BlueALSA A2DP receiver configuration](https://github.com/arkq/bluez-alsa/blob/master/doc/bluealsad.8.rst).
Bluetooth version alone does not configure an audio receiver; the Linux A2DP sink
service, codec support, pairing, and reconnect behavior are required software.

## Proposed model geometry

Use this concrete starting point for estimation, not as an already selected
architecture:

| Quantity | Starting assumption |
| --- | --- |
| Internal audio | 16,000 samples/second, mono |
| Output | One selected stem at the same internal sample rate |
| Window | 250ms, or 4,000 samples |
| Window advance | 125ms, or 2,000 samples; twofold overlap |
| STFT | 512-point FFT, 384-sample hop; about 41.67 frames/second |
| Encoded frequency positions | 64 |
| Feature width | 64 |
| Recurrent blocks | Two dual-path blocks, four recurrent paths total |
| Recurrent path | Single-layer bidirectional LSTM, input and hidden widths 64 |
| Projection | 128 bidirectional features projected back to 64 per path |
| Total model size allowance | 1–5 million parameters, including encoder/decoder |

The 64 encoded positions require an encoder or frequency reduction; they do not
follow automatically from the 257 bins of a 512-point real FFT. The encoder and
decoder must be explicitly designed and included in the eventual operation count.
Unlike current SCNet, this estimate uses fixed widths and does not include its
alternating feature-conversion FFTs. Reproducing those operations would require
revising the estimate.

Bidirectionality is confined to each available window. It requires waiting for
the window and recomputes overlapping context; it is not a causal recurrent stream.
A future causal model could retain state and reduce repeated work, but would need
separate training and quality evaluation. Do not assume that benefit in this budget.

16kHz limits the representable audio band to below 8kHz. This is an explicit fidelity
tradeoff that needs listening tests, especially for cymbals and other high-frequency
content. Lowering the model sample rate reduces samples and spectral frames; changing
compressed transport bitrate alone does not reduce the neural network's work on
an otherwise identical decoded signal. The sound device may run at a different
sample rate through explicit resampling.

## Compute estimate

Count one multiply-add as two floating-point operations. For one direction of an
LSTM with input width D and hidden width H, the four gates require approximately
4H(D + H) multiply-adds per sequence position. With D = H = 64 and two directions:

    2 directions × 4 × 64 × (64 + 64) × 2 FLOP = 131,072 FLOP/position

Each frequency or time path visits approximately F × T positions. Four paths,
64 encoded frequency positions, and 16,000 / 384 frames per second give:

    131,072 × 4 × 64 × (16,000 / 384) = 1.398 GFLOP/audio-second

The four 128-to-64 projections add:

    (2 × 128 × 64) × 4 × 64 × (16,000 / 384) = 0.175 GFLOP/audio-second

Thus recurrent matrices plus projections cost about 1.573 GFLOP/audio-second
before overlap, or 3.146 GFLOP/audio-second with twofold overlap. Biases, gate
nonlinearities, normalization, padding effects, encoder/decoder, FFT, resampling,
and memory movement are additional costs. Small windows also make exact padded
frame counts material; this continuous-frame calculation is approximate.

| Throughput | Required sustained arithmetic for this subtotal |
| --- | ---: |
| 1× playback speed | 3.15 GFLOP/second |
| 2× playback speed | 6.29 GFLOP/second |
| 5× playback speed | 15.73 GFLOP/second |

Budget roughly 20–50 effective GFLOP/second for the complete model at 5× speed
until a complete architecture provides an exact operation count. This is a required
throughput allowance, not a claim about measured Pi performance. Sequential LSTM
dependencies, small matrix shapes, layout conversions, and kernel overhead can
make nominal FLOP capacity misleading.

The Pi 5 has four Cortex-A76 CPU cores at 2.4GHz. Start with CPU inference and
active cooling. No GPU or accelerator speedup is assumed. The decisive experiment
is native execution of these tensor shapes, followed by the full model, in a
sandbox deliberately limited to the Pi resource budget. See the companion
[optimization memo](INFERENCE_OPTIMIZATION_MEMO.md) for test constraints and the
limits of mapping sandbox timing to the target CPU.
[Pi 5 specifications](https://www.raspberrypi.com/products/raspberry-pi-5/).

Keeping a 125ms advance while increasing context to 500ms doubles overlap to four
and approximately doubles window-based work. Alternatively, advancing 250ms keeps
twofold overlap but increases output block size and startup delay. Window duration,
advance, and retained state must be evaluated together.

## Memory estimate

The recurrent core itself is small. One bidirectional LSTM has approximately:

    2 × [4H(D + H) + 8H] = 66,560 parameters

Four LSTMs contain 266,240 parameters. Four projections add
4 × (128 × 64 + 64) = 33,024, bringing the subtotal to 299,264 parameters,
about 1.14MiB in FP32. The 1–5 million total-parameter allowance leaves room for
an encoder and decoder; their final size remains a design decision.

| Concrete allocation | Size |
| --- | ---: |
| 1–5 million FP32 weights | 3.8–19.1MiB |
| Same number of INT8 weight values, excluding scales/metadata | 0.95–4.77MiB |
| 250ms mono FP32 input window | 16,000 bytes |
| 500ms mono FP32 input window | 32,000 bytes |
| One 64 × 64 × 21 FP32 feature tensor | 336KiB |
| One second of 48kHz stereo FP32 audio | 384,000 bytes |

The feature example corresponds approximately to 500ms of spectral frames. Actual
framing/padding changes the count. Inference needs only live intermediates, rather
than a training graph retaining all activations for backpropagation. Nevertheless,
backend workspaces, packed weights, allocator pools, and load-time copies must be
counted; weights and raw audio alone are not the entire footprint.

Use the following provisional ceilings to make the 2GB target testable:

| Component | Planning ceiling |
| --- | ---: |
| FP32 model weights | 20MiB |
| Live model intermediates and recurrent state | 64MiB |
| Backend scratch space and packed-weight allowance | 128MiB |
| Native executable, libraries, and audio services | 96MiB |
| Minimal OS and background services | 320MiB |
| Bounded PCM queues and DSP buffers | 8MiB |
| Additional margin, including loading transients | 256MiB |
| Total planning allowance | 892MiB |

These are engineering budgets, not measured footprints. Target at most 1GiB of
whole-device occupied memory during startup and sustained playback, leaving
substantial room on the 2GB board. Measure system memory as well as process RSS;
shared mappings and reclaimable page cache need care when interpreting totals.
Avoid simultaneously retaining a loaded checkpoint, a second weight copy, and a
second inference session. Quantization is optional and requires backend and audio
quality validation; the FP32 memory budget does not depend on it.

Python is absent from production. The runtime still needs native libraries,
intermediates, and workspace, but there is no requirement to carry the Python
interpreter, PyTorch development package, training code, or a browser.

## Latency and acceptance targets

Throughput and first-output latency are related but distinct. For the proposed
windowed design, first output needs about 250ms of received PCM before inference
can begin. At the sustained 5× target below, each window invocation takes about
25ms, yielding about 275ms from the first available PCM sample to first model
output, before Bluetooth reception/decode and playback buffering are added. This
assumes the first window costs the same as a warmed steady-state invocation; cold
initialization and any extra reconstruction delay must be measured separately.

Steady state advances 125ms per inference. Because the 5× budget already includes
overlap, its average processing budget is 125 / 5 = 25ms per invocation. Do not
apply a second overlap penalty to this number. Proposed acceptance targets are:

- Average native processing time at most 25ms per 125ms advance, including local
  DSP and reconstruction; p99 at most 50ms after warmup.
- One hour of continuous Bluetooth input and analog playback without underruns,
  unbounded queues, thermal degradation, or swap use.
- Peak whole-device occupied memory at most 1GiB, including startup/model loading.
- Explicitly measured Bluetooth-to-analog latency and cold boot/model-load time,
  reported separately from warmed model execution.
- Listening evaluation on representative music, with a quality threshold selected
  before deciding that the speed/fidelity tradeoff is acceptable.

For export correctness, compare the same candidate model in PyTorch and the native
runtime on identical PCM windows and complete streamed reconstruction. Report
maximum/RMS error and investigate differences. For FP32, start with
np.allclose(atol=2e-6, rtol=1e-4), adjusting only with numerical evidence. A newly
trained smaller model will not numerically match SCNet XL; compare that reference
using separation quality metrics and listening, not samplewise equality.

## Prototype hardware and cost

New hardware only; assembly is acceptable. Prices checked 21 October 2026, excluding
tax and shipping. This is a combined shopping list, not a preconfigured audio kit.

| Item | Price |
| --- | ---: |
| [PiShop Pi 5 2GB starter kit](https://www.pishop.us/product/raspberry-pi-5-starter-kit-2gb/) | $137.45 |
| [Adafruit USB audio adapter, product 1475](https://www.adafruit.com/product/1475) | $4.95 |
| Total | $142.40 |

The starter kit includes the board with built-in Bluetooth, case, active cooler,
27W power supply, 64GB microSD card with 64-bit OS, card reader, and HDMI cable.
The USB adapter supplies the 3.5mm analog output; headphones/speakers and any RCA
cable are separate. The [4GB starter kit](https://www.pishop.us/product/raspberry-pi-5-starter-kit-4gb/)
would bring the total to $174.90, but additional RAM does not increase CPU throughput.

## First validation steps in a constrained sandbox

1. Construct the untrained candidate geometry and calculate exact parameters,
   operations, live tensors, and fixed-window frame counts.
2. Export and run native kernels in a sandbox limited to 2GB and at most four
   CPU cores before committing to full training. Disable swap and include layout
   changes and synchronization. Host CPU quotas do not establish Pi-equivalent
   instruction performance; record calibration assumptions and timing margin.
3. Build a bounded pipeline harness with paced PCM input and output consumption,
   including representative audio-service costs. Measure sandbox memory and
   processing delay; keep unmeasured physical Bluetooth/DAC latency separate.
4. Integrate one native inference session and verify export/reconstruction parity.
5. Train, assess quality, and repeat sustained latency/memory tests in the sandbox
   with the actual model. Direct Pi testing is outside this plan; physical thermal,
   radio, and power behavior remain assumptions rather than sandbox measurements.
   Change the model or target hardware if acceptance fails.

The existing SCNet XL validation remains useful as a reference. Its measured
9.132GiB process-tree peak and 255.910-second pass over an 18.125-second excerpt
in validation/RESULTS.md belong to a different architecture and validation harness;
they are not memory or speed estimates for this proposed native implementation.
