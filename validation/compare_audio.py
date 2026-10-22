#!/usr/bin/env python3
"""Compare real audio under the watchdog, with exactly one model process at a time."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker():
    """Internal child. Process exit releases all model and allocator memory."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--implementation', choices=('original', 'refactor'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, required=True)
    args = parser.parse_args()
    if os.environ.get('STEMMER_PARITY_PARENT') != str(os.getppid()):
        raise SystemExit('Internal worker must be launched by the guarded comparison')
    import numpy as np
    import torch
    from stemmer import Stemmer
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    audio = np.load(args.output/'input.npy')
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    for key in ('state', 'state_dict', 'model_state_dict'):
        if key in weights:
            weights = weights[key]
    if args.implementation == 'original':
        from validation.original.model import SCNet
        from validation.original.demix import demix
        model = SCNet(sources=list(Stemmer.sources), dims=[4, 64, 128, 256],
                      band_SR=[0.230, 0.370, 0.400], band_stride=[1, 4, 4],
                      band_kernel=[3, 4, 4], conv_depths=[3, 3, 3], num_dplayer=8)
        model.load_state_dict(weights, strict=True)
        model.eval().requires_grad_(False)
        del weights
        class Config(dict):
            __getattr__ = dict.__getitem__
        config = Config(audio=Config(chunk_size=485100),
                        inference=Config(batch_size=1, num_overlap=4),
                        training=Config(instruments=list(Stemmer.sources), target_instrument=None, use_amp=False))
        started = time.perf_counter()
        stems = demix(config, model, audio, 'cpu', 'scnet')
        stems = np.stack([stems[name] for name in Stemmer.sources])
    else:
        model = Stemmer(weights)
        del weights
        started = time.perf_counter()
        stems = model(audio).numpy()
    elapsed = time.perf_counter() - started
    np.save(args.output/f'{args.implementation}.npy', stems)
    record = dict(implementation=args.implementation, seconds=elapsed,
                  pid=os.getpid(), shape=list(stems.shape))
    (args.output/f'{args.implementation}.json').write_text(json.dumps(record)+'\n')
    print(json.dumps(dict(event='implementation_complete', **record)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=None,
                        help='Use an initial excerpt; omitted means the entire track')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or (args.seconds is not None and (not 0 < args.seconds < float('inf'))):
        parser.error('threads and seconds must be positive and finite')
    if args.output.exists():
        parser.error('output directory must be fresh')
    args.output.mkdir(parents=True)
    import hashlib
    import numpy as np
    import soundfile as sf
    with sf.SoundFile(args.audio) as source:
        if source.samplerate != 44100 or source.channels != 2:
            raise ValueError('fixture must already be stereo 44.1 kHz')
        frames = -1 if args.seconds is None else round(args.seconds * 44100)
        audio = source.read(frames=frames, dtype='float32', always_2d=True).T.copy()
    np.save(args.output/'input.npy', audio)
    source_hash = hashlib.sha256(audio.tobytes()).hexdigest()
    frame_count = audio.shape[-1]
    del audio
    environment = dict(os.environ, STEMMER_PARITY_PARENT=str(os.getpid()))
    runs = {}
    for implementation in ('original', 'refactor'):
        print(json.dumps(dict(event='implementation_start', implementation=implementation)), flush=True)
        # Blocking wait: the original OS process is reaped before the next starts.
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        '--implementation', implementation, '--checkpoint', str(args.checkpoint),
                        '--output', str(args.output), '--threads', str(args.threads)],
                       env=environment, check=True)
        runs[implementation] = json.loads((args.output/f'{implementation}.json').read_text())
        print(json.dumps(dict(event='implementation_reaped', implementation=implementation)), flush=True)
    reference = np.load(args.output/'original.npy', mmap_mode='r')
    actual = np.load(args.output/'refactor.npy', mmap_mode='r')
    if reference.shape != actual.shape or actual.shape != (4, 2, frame_count):
        raise ValueError('original/refactor output shape mismatch')
    sources = ('drums', 'bass', 'other', 'vocals')
    metrics = {}
    for name, expected, observed in zip(sources, reference, actual):
        delta = observed.astype(np.float64) - expected.astype(np.float64)
        metrics[name] = dict(allclose=bool(np.allclose(observed, expected, atol=2e-6, rtol=1e-4)),
                             max_absolute_difference=float(np.abs(delta).max()),
                             rms_difference=float(np.sqrt(np.mean(delta ** 2))))
    with args.checkpoint.open('rb') as checkpoint:
        checkpoint_hash = hashlib.file_digest(checkpoint, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(checkpoint.read()).hexdigest()
    report = dict(audio=str(args.audio), frames=frame_count, sample_rate=44100,
                  input_sha256=source_hash, checkpoint_sha256=checkpoint_hash,
                  sources=sources, shape=list(actual.shape), atol=2e-6, rtol=1e-4,
                  execution='sequential isolated processes; original reaped before refactor launch',
                  runs=runs, stems=metrics, allclose=all(m['allclose'] for m in metrics.values()),
                  note='Timing is diagnostic. Final watchdog validity is also required.')
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not report['allclose']:
        raise SystemExit('Real-audio parity failed')


if __name__ == '__main__':
    if '--implementation' in sys.argv:
        worker()
    else:
        from validation.watchdog import guarded_entry
        guarded_entry(main)
