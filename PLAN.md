# Lean complete-track refactor

## Contract and decisions

Complete-track stereo float32 array/tensor at 44.1 kHz to four CPU float32 stems
with identical length. Caller supplies an already-loaded state dictionary.
Fix SCNet XL IHF architecture, CPU FP32, 11-second windows, overlap four and one
window per forward. No audio or checkpoint file I/O. No training or alternate models.

## Implementation

1. Preserve upstream SCNet/recurrent computation and parameter names; hardcode the
   XL IHF architecture from the cached checkpoint's config.
2. Add one public Stemmer class for strict weight loading and array validation.
3. Implement complete-track reflected/zero tail padding and weighted overlap-add,
   with correct first/last window tapers and exact original-length output.
4. Remove training, evaluation, GUI, model registries, configuration files, folder
   processing and other models. Retain LICENSE and reduce dependencies to torch/numpy.
5. Test short, boundary-length and long tracks, tail policies, source mapping,
   input immutability, invalid inputs/outputs and context-sensitive reconstruction
   against an independent full-buffer oracle. Check real checkpoint keys/shapes
   without a real forward. Inspect and verify Git changes.

## Deliberately deferred

CPU tiling/vectorization, parallel windows, GPUs, services, audio adapters, training
and performance experiments. This baseline must not imply their measured speedups.
Full trained-model forward/output parity needs a separately guarded inference run;
this refactor does not launch expensive model inference.

## User correction: retain real audio verification

Keep a frozen runnable upstream model/demix reference, provenance and the existing
external watchdog under validation/. Keep I/O exclusively in a development audio
harness. Compare original/refactor outputs samplewise using np.allclose, storing
raw tensors, maximum/RMS errors and explicit tolerance. A real inference comparison
must pass the watchdog; synthetic tests alone do not satisfy the audio acceptance.

## Restart and execution isolation

The user requires swap to remain zero. The watchdog remains unchanged. The first
pre-restart comparison was refused before launching a model because the system
already had swap in use. After restart, zero swap was confirmed. The comparison
now runs original and refactor in isolated sequential OS processes; it waits for
and reaps the original before launching the refactor, so model/allocator memory
cannot overlap across implementations. Initial real-audio qualification uses an
explicit 18.125-second excerpt, exercising outer padding and partial tail windows;
it is not a full-song speed or perceptual-quality claim.

## Completion

Implementation and retained verification infrastructure are complete. Six tests
pass; the real cached checkpoint matches strictly. The post-restart audio
comparison completed under the unchanged zero-swap watchdog, with exact output
agreement for all four stems. See validation/RESULTS.md for scope and evidence.
Changes remain uncommitted for review. A full-song equivalence run and perceptual
review were not performed; the real-audio result covers an 18.125-second excerpt.
