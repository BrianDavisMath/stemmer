# Refactor verification

## Real audio

The guarded original-versus-refactor comparison passed on an explicitly selected
18.125-second excerpt (799,312 stereo frames at 44.1 kHz) from the cached Arctic
Monkeys test source. This is excerpt qualification, not a full-song timing study
or a listening-quality assessment.

Both runs used the identical decoded input and XL IHF checkpoint, CPU float32,
two intra-op threads, one inter-op thread, 11-second windows, fourfold overlap,
and one window per forward. The original model process exited and was reaped
before the refactor model process started. The controller loads no model.

All 6,394,496 output sample values across drums, bass, other and vocals matched
exactly: maximum absolute difference 0, RMS difference 0 for every stem.
`np.allclose(atol=2e-6, rtol=1e-4)` passed for every stem. Output shape was
[4, 2, 799312]. No gain adjustment, normalization, time shifting or clipping.

Original pass: 258.560 seconds; refactor pass: 255.910 seconds. These single-run
observations do not establish a performance improvement. No optimization of
model execution is claimed by this refactor.

Final watchdog valid: zero swap before/during/after; no swap-in, swap-out, pageout
or compression growth. Peak process-tree RSS 9.132 GiB; minimum available memory
28.366 GiB; maximum telemetry gap 0.272 seconds. No limits were relaxed.

Input SHA256: 068f7be12778b34273eb80a744c8e97452a0f32877e850461cfffcd3a2efaef2

Checkpoint SHA256: ac25975f0f5704f3d1a3c3c251505b7a0f417a22eafe82773440ee4f7e14b74f

Raw arrays and comparison.json: .runtime/parity-after-restart/

Before/during/after telemetry: .runtime/parity-after-restart.guard.jsonl

## Lightweight checks

Six unittest cases pass, including upstream demixer parity at 16 lengths covering
short tracks, exact boundaries, partial tails and reflected outer padding;
input immutability; NumPy/tensor equivalence; source mapping; strict weight loading;
and invalid input/output rejection. These tests use tiny context-sensitive models.

The cached real checkpoint strictly matches all 510 parameter tensors (53,466,800
elements) in both architectures. Network source inspection confirms all 19 retained
methods match upstream syntax except documentation; only SCNet construction was
specialized to the fixed XL IHF geometry. Git whitespace checks pass.
