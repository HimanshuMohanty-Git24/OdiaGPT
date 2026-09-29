"""Correctness tests for the randomly initialized OdiaGPT decoder."""

from pathlib import Path
import tempfile
import unittest

import torch

from odiagpt.config import ModelConfig
from odiagpt.model import OdiaGPT
from odiagpt.rope import RotaryEmbedding, apply_rope


ROOT = Path("/content/drive/MyDrive/OdiaGPT")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1234)
        cls.config = ModelConfig.from_yaml(ROOT / "configs/smoke.yaml")

    def test_config_and_parameter_count(self):
        for name in ("smoke", "nano", "tiny", "odia30m"):
            config = ModelConfig.from_yaml(ROOT / "configs" / f"{name}.yaml")
            model = OdiaGPT(config)
            d = config.d_model
            expected = (
                config.vocab_size * d
                + config.n_layers * (4 * d * d + 3 * d * config.d_ff + 2 * d)
                + d
            )
            self.assertEqual(model.parameter_count(), expected)
            self.assertIs(model.lm_head.weight, model.token_embedding.weight)
            print(f"{name} parameters: {expected:,}")
            del model

    def test_forward_loss_and_backward(self):
        model = OdiaGPT(self.config)
        tokens = torch.randint(0, self.config.vocab_size, (2, 16), dtype=torch.long)
        logits, loss = model(tokens[:, :-1], tokens[:, 1:])
        self.assertEqual(logits.shape, (2, 15, self.config.vocab_size))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all().item() for g in gradients))

    def test_causality(self):
        model = OdiaGPT(self.config).eval()
        a = torch.randint(0, self.config.vocab_size, (2, 12))
        b = a.clone()
        b[:, 6:] = (b[:, 6:] + 13) % self.config.vocab_size
        with torch.no_grad():
            first, _ = model(a)
            changed, _ = model(b)
        self.assertTrue(torch.allclose(first[:, :6], changed[:, :6], atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(first[:, 6:], changed[:, 6:]))

    def test_rope_shape_device_dtype(self):
        embedding = RotaryEmbedding(32)
        x = torch.randn(2, 4, 7, 32)
        cos, sin = embedding(7, x.device, x.dtype)
        result = apply_rope(x, cos, sin)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device, x.device)
        self.assertTrue(torch.allclose(result[:, :, 0], x[:, :, 0], atol=1e-6))
        self.assertTrue(torch.isfinite(result).all().item())

    def test_save_load_logits_match(self):
        model = OdiaGPT(self.config).eval()
        tokens = torch.randint(0, self.config.vocab_size, (1, 9))
        with torch.no_grad():
            original, _ = model(tokens)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save(model.state_dict(), path)
            restored = OdiaGPT(self.config).eval()
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            with torch.no_grad():
                result, _ = restored(tokens)
        self.assertTrue(torch.equal(original, result))

    def test_short_greedy_generation(self):
        model = OdiaGPT(self.config).eval()
        ids = torch.tensor([[2, 42, 73]], dtype=torch.long)
        with torch.no_grad():
            for _ in range(3):
                logits, _ = model(ids[:, -self.config.context_length:])
                next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
                ids = torch.cat((ids, next_id), dim=1)
        self.assertEqual(ids.shape, (1, 6))
        self.assertTrue(((ids >= 0) & (ids < self.config.vocab_size)).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_backward(self):
        device = torch.device("cuda")
        model = OdiaGPT(self.config).to(device)
        tokens = torch.randint(0, self.config.vocab_size, (2, 9), device=device)
        logits, loss = model(tokens[:, :-1], tokens[:, 1:])
        self.assertEqual(logits.device.type, "cuda")
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(model.token_embedding.weight.grad).all().item())
        print("CUDA device:", torch.cuda.get_device_name(device))
        print("BF16 supported:", torch.cuda.is_bf16_supported())
