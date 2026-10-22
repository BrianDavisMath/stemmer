"""Fixed CPU/float32 inference and sample-aligned complete-track reconstruction."""
from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional as F

from .model import SCNet

_WINDOW = 485_100  # 11 seconds at 44.1 kHz


class Stemmer:
    """One loaded XL IHF model. Calls accept stereo float32 at 44.1 kHz.

    Pass the checkpoint's tensor state dictionary, already loaded by the caller.
    Output is a CPU float32 tensor [4, 2, samples], ordered as ``sources``.
    No decoding, resampling, normalization, clipping, or filesystem access occurs.
    Instances process one track at a time; callers own concurrency and CPU limits.
    """

    sample_rate = 44_100
    sources = ('drums', 'bass', 'other', 'vocals')

    def __init__(self, weights: Mapping[str, torch.Tensor]):
        self._model = SCNet().cpu().float()
        self._model.load_state_dict(weights, strict=True)
        self._model.requires_grad_(False)
        self._model.eval()

    @torch.inference_mode()
    def __call__(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            if audio.dtype != np.float32:
                raise ValueError('audio must be float32')
            mix = torch.from_numpy(np.array(audio, copy=True, order='C'))
        elif isinstance(audio, torch.Tensor):
            if audio.device.type != 'cpu' or audio.dtype != torch.float32:
                raise ValueError('audio must be a CPU float32 tensor')
            mix = audio.detach()
        else:
            raise TypeError('audio must be a NumPy array or PyTorch tensor')
        if mix.ndim != 2 or mix.shape[0] != 2 or mix.shape[1] == 0:
            raise ValueError('audio must have shape [2, nonzero samples]')
        if not torch.isfinite(mix).all():
            raise ValueError('audio must contain only finite samples')
        return _separate(self._model, mix)


def _separate(model, mix):
    """Original fourfold overlap and padding, with position-specific edge tapers."""
    samples = mix.shape[-1]
    step = _WINDOW // 4
    border = _WINDOW - step
    padded = samples > 2 * border
    if padded:
        mix = F.pad(mix, (border, border), mode='reflect')
    length = mix.shape[-1]
    crop_start = border if padded else 0
    fade = _WINDOW // 10
    taper = torch.ones(_WINDOW, dtype=torch.float32, device='cpu')
    taper[:fade] = torch.linspace(0, 1, fade, device='cpu')
    taper[-fade:] = torch.linspace(1, 0, fade, device='cpu')
    result = torch.zeros((4, 2, samples), dtype=torch.float32, device='cpu')
    counter = torch.zeros(samples, dtype=torch.float32, device='cpu')

    for start in range(0, length, step):
        part = mix[:, start:start + _WINDOW]
        valid = part.shape[-1]
        mode = 'reflect' if valid > _WINDOW // 2 else 'constant'
        window = F.pad(part, (0, _WINDOW - valid), mode=mode)
        prediction = model(window.unsqueeze(0))
        if (prediction.shape != (1, 4, 2, _WINDOW)
                or prediction.dtype != torch.float32 or prediction.device.type != 'cpu'
                or not torch.isfinite(prediction).all()):
            raise ValueError('model must return finite CPU float32 [1, 4, 2, window]')
        weights = taper.clone()
        if start == 0:
            weights[:fade] = 1
        if start + step >= length:
            weights[-fade:] = 1
        # Accumulate only the returned interval, avoiding four-stem padding buffers.
        left = max(start, crop_start)
        right = min(start + valid, crop_start + samples)
        if left < right:
            source = slice(left - start, right - start)
            target = slice(left - crop_start, right - crop_start)
            result[..., target] += prediction[0, ..., source] * weights[source]
            counter[target] += weights[source]
        del prediction, window

    if not (counter > 0).all():
        raise RuntimeError('window plan left uncovered samples')
    result /= counter
    if not torch.isfinite(result).all():
        raise ValueError('reconstructed stems contain nonfinite samples')
    return result
