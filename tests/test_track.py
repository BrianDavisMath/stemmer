import unittest
from unittest.mock import patch

import numpy as np
import torch

import stemmer.track as track
from stemmer import Stemmer
from validation.original.demix import demix


class Config(dict):
    __getattr__ = dict.__getitem__


def config(window):
    return Config(audio=Config(chunk_size=window),
                  inference=Config(batch_size=1, num_overlap=4),
                  training=Config(instruments=list(Stemmer.sources), use_amp=False,
                                  target_instrument=None))


class ContextModel(torch.nn.Module):
    """Distinct stems sensitive to context, padding and local window position."""
    def __init__(self):
        super().__init__()
        self.register_buffer('gain', torch.tensor(0.75))

    def forward(self, x):
        n = x.shape[-1]
        position = torch.arange(n, dtype=x.dtype) / n
        context = x.mean(-1, keepdim=True)
        return torch.stack([x * (k + 1) * self.gain + context * position * (k + 2)
                            for k in range(4)], dim=1)


class TrackTests(unittest.TestCase):
    def setUp(self):
        self.window = patch.object(track, '_WINDOW', 40)
        self.factory = patch.object(track, 'SCNet', ContextModel)
        self.window.start()
        self.factory.start()
        self.addCleanup(self.window.stop)
        self.addCleanup(self.factory.stop)
        self.stemmer = Stemmer({'gain': torch.tensor(0.75)})

    def test_upstream_parity_all_boundaries_and_partial_tails(self):
        for n in (1, 9, 10, 19, 20, 21, 39, 40, 41, 59, 60, 61, 79, 80, 81, 137):
            with self.subTest(samples=n):
                audio = np.random.default_rng(n).normal(size=(2, n)).astype(np.float32)
                original = demix(config(40), ContextModel().eval(), audio, 'cpu', 'scnet')
                expected = np.stack([original[s] for s in Stemmer.sources])
                actual = self.stemmer(audio).numpy()
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-4)
                self.assertEqual(actual.shape, (4, 2, n))

    def test_tensor_numpy_and_noncontiguous_inputs_are_unchanged(self):
        a = np.random.default_rng(22).normal(size=(2, 137)).astype(np.float32)
        for audio in (a, a[:, ::-1], torch.from_numpy(a), torch.from_numpy(a).requires_grad_()):
            before = audio.detach().numpy().copy() if torch.is_tensor(audio) else audio.copy()
            result = self.stemmer(audio)
            np.testing.assert_array_equal(audio.detach().numpy() if torch.is_tensor(audio) else audio, before)
            self.assertFalse(result.requires_grad)
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(result.device.type, 'cpu')
        np.testing.assert_array_equal(self.stemmer(a), self.stemmer(torch.from_numpy(a)))

    def test_identity_stems_cover_endpoints_and_preserve_mapping(self):
        class Identity(torch.nn.Module):
            def forward(self, x):
                return torch.stack([x * (k + 1) for k in range(4)], 1)
        self.stemmer._model = Identity()
        a = np.random.default_rng(33).normal(size=(2, 137)).astype(np.float32)
        result = self.stemmer(a).numpy()
        np.testing.assert_allclose(result, np.stack([a * (k + 1) for k in range(4)]), atol=2e-6, rtol=1e-6)

    def test_reject_invalid_inputs(self):
        for audio in (np.zeros((1, 10), np.float32), np.zeros((2, 0), np.float32),
                      np.zeros(10, np.float32), np.zeros((2, 10)),
                      torch.zeros(2, 10, dtype=torch.int16),
                      np.full((2, 10), np.nan, np.float32),
                      np.full((2, 10), np.inf, np.float32)):
            with self.subTest(shape=audio.shape), self.assertRaises(ValueError):
                self.stemmer(audio)
        with self.assertRaises(TypeError):
            self.stemmer([[1.0]])

    def test_reject_bad_model_outputs(self):
        class Bad(torch.nn.Module):
            def __init__(self, kind):
                super().__init__()
                self.kind = kind
            def forward(self, x):
                if self.kind == 'shape': return torch.zeros(1, 3, 2, 40)
                if self.kind == 'dtype': return torch.zeros(1, 4, 2, 40, dtype=torch.float64)
                return torch.full((1, 4, 2, 40), float('nan'))
        for kind in ('shape', 'dtype', 'nan'):
            self.stemmer._model = Bad(kind)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.stemmer(np.zeros((2, 10), np.float32))

    def test_strict_weights_and_evaluation_mode(self):
        with self.assertRaises(RuntimeError):
            Stemmer({})
        with self.assertRaises(RuntimeError):
            Stemmer({'gain': torch.tensor(1.0), 'unexpected': torch.tensor(1.0)})
        self.assertFalse(self.stemmer._model.training)


if __name__ == '__main__':
    unittest.main()
