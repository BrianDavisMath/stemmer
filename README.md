# Stemmer

One complete stereo waveform in, four complete stems out. This is a fixed,
inference-only SCNet XL IHF implementation derived from
[ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training),
source snapshot `6fdad13`. The upstream MIT license is retained in LICENSE.

```python
from stemmer import Stemmer

# weights: an already-loaded XL IHF tensor state dictionary
# audio: NumPy float32 array or CPU float32 tensor, shape [2, samples]
stemmer = Stemmer(weights)
stems = stemmer(audio)  # CPU float32 tensor, shape [4, 2, samples]
drums, bass, other, vocals = stems.unbind(0)
# stems.numpy() gives a NumPy view when desired.
```

The caller supplies weights and decoded audio. There is no file I/O, checkpoint
download, sample-rate conversion, audio encoding, CLI, or service. An enclosing
checkpoint dictionary must be unpacked by the caller to its tensor state dictionary.
Inputs are assumed to be 44,100 Hz; arrays cannot establish their own sample rate.
Stereo channel-first float32 is required; mono, empty and nonfinite input is rejected.
Input samples are never modified. Output order is drums, bass, other, vocals;
output length exactly matches input length. Levels are not normalized or clipped.

The architecture is fixed to the config paired with the v1.0.15 release checkpoint
`model_scnet_ep_36_sdr_10.0891.ckpt` (SHA256
`ac25975f0f5704f3d1a3c3c251505b7a0f417a22eafe82773440ee4f7e14b74f`).
Weights are loaded strictly; parameter names and network computations are preserved.

Execution is CPU float32, one 485,100-sample window per forward, fourfold overlap
(step 121,275), original conditional reflected outer padding and reflected/zero
partial tails, and weight-normalized overlap-add. Edge tapers are assigned per
window position, correcting the old batching-dependent edge policy. Complete-track
buffers scale with track length; working inference buffers contain one window.
There is no claim of constant memory or arbitrary track-length capacity.

The model remains loaded across calls. Use one call at a time per instance.
The caller sets process/thread limits and protects actual inference with the
existing external memory watchdog. No automatic backend selection, multiprocessing,
compilation, mixed precision, tiling or alternate window settings are included.
This is a small correctness baseline, not a port of every karaoke optimization.

Only NumPy and PyTorch are runtime dependencies. Install with `pip install -e .`.
Run lightweight reconstruction tests with `python -m unittest discover -s tests -v`.
These tests use small deterministic models; they do not establish trained-model
speed, full-network numerical parity, or perceptual separation quality.

## Original-versus-refactor audio verification

`validation/original/` retains the original model and verbatim upstream demix
functions as an independent development reference, with provenance hashes. It is
excluded from the installed package. `validation/watchdog.py` retains the existing
macOS external memory/swap watchdog. Install the harness with
`pip install -e '.[validation]'`.

```sh
python validation/compare_audio.py --audio /path/to/stereo-44100.wav \
  --checkpoint /path/to/model_scnet_ep_36_sdr_10.0891.ckpt \
  --output .runtime/parity-full-track --guard-log .runtime/parity.guard.jsonl
```

Omit `--seconds` to compare the entire track, or explicitly set an excerpt length.
Both implementations use the same decoded samples, CPU FP32, checkpoint, thread
count, 11-second windows, overlap four and single-window forwards. Every output
sample is checked with `np.allclose(atol=2e-6, rtol=1e-4)`; each stem gets maximum
absolute and RMS difference metrics. Raw outputs and reports are saved for review.
A comparison passes only if both the report and final watchdog result pass. No
normalization, fitted gain, time shift, or tolerance adjustment conceals differences.
The watchdog requires a zero-swap starting baseline and aborts on swap growth,
unsafe pressure, excess compression, RSS budget, unavailable telemetry or timeout.

The comparison controller never imports PyTorch or constructs a model. Original
and refactor each run in one isolated child process, sequentially; the original
process is reaped before the refactor launches. Process exit releases weights and
allocator caches. The external watchdog covers the controller and its child.
