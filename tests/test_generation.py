"""Tests for greedy, temperature, top-k, and top-p generation."""

from pathlib import Path
import unittest

import torch

from odiagpt.config import ModelConfig
from odiagpt.generation import generate_tokens
from odiagpt.model import OdiaGPT


ROOT = Path("/content/drive/MyDrive/OdiaGPT")


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(91)
        config = ModelConfig.from_yaml(ROOT / "configs/smoke.yaml")
        cls.model = OdiaGPT(config).eval()

    def _generate(self, **kwargs):
        return generate_tokens(
            self.model, [2, 17, 24], max_new_tokens=8,
            eos_id=3, pad_id=0, unk_id=1, bos_id=2,
            **kwargs,
        )

    def test_greedy_executes(self):
        ids = self._generate(temperature=0)
        self.assertGreater(len(ids), 3)
        self.assertLessEqual(len(ids), 11)
        self.assertTrue(all(0 <= i < self.model.config.vocab_size for i in ids))

    def test_sample_reproducible(self):
        first = self._generate(temperature=0.8, top_k=50, top_p=0.9, seed=77)
        second = self._generate(temperature=0.8, top_k=50, top_p=0.9, seed=77)
        self.assertEqual(first, second)

    def test_modes_and_limits(self):
        for kwargs in (
            {"temperature": 1.0, "seed": 1},
            {"temperature": 0.7, "top_k": 10, "seed": 1},
            {"temperature": 0.9, "top_p": 0.8, "seed": 1},
        ):
            with self.subTest(kwargs=kwargs):
                ids = self._generate(**kwargs)
                self.assertTrue(all(i not in (0, 1, 2) for i in ids[3:]))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            self._generate(temperature=-1)
        with self.assertRaises(ValueError):
            self._generate(top_k=0)
        with self.assertRaises(ValueError):
            self._generate(top_p=0)
