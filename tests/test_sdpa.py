"""Check PyTorch SDPA against our manual causal attention."""

from dataclasses import replace
from pathlib import Path
import unittest

import torch

from odiagpt.config import ModelConfig
from odiagpt.model import OdiaGPT


ROOT = Path("/content/drive/MyDrive/OdiaGPT")


class SDPAAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manual_config = ModelConfig.from_yaml(ROOT / "configs/smoke.yaml")
        cls.sdpa_config = replace(cls.manual_config, attention_backend="sdpa")

    def _pair(self, device):
        torch.manual_seed(556)
        manual = OdiaGPT(self.manual_config).to(device).eval()
        sdpa = OdiaGPT(self.sdpa_config).to(device).eval()
        sdpa.load_state_dict(manual.state_dict())
        return manual, sdpa

    def test_cpu_outputs_and_causality(self):
        manual, sdpa = self._pair(torch.device("cpu"))
        tokens = torch.randint(0, self.manual_config.vocab_size, (2, 12))
        with torch.no_grad():
            a, _ = manual(tokens)
            b, _ = sdpa(tokens)
        difference = (a - b).abs().max().item()
        print("CPU manual/SDPA max logit difference:", difference)
        self.assertLess(difference, 1e-4)
        changed = tokens.clone()
        changed[:, 6:] = (changed[:, 6:] + 17) % self.manual_config.vocab_size
        with torch.no_grad():
            future, _ = sdpa(changed)
        self.assertTrue(torch.allclose(b[:, :6], future[:, :6], atol=1e-6, rtol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_fp16_forward_backward(self):
        device = torch.device("cuda")
        manual, sdpa = self._pair(device)
        tokens = torch.randint(0, self.manual_config.vocab_size, (2, 24), device=device)
        targets = torch.randint(0, self.manual_config.vocab_size, (2, 24), device=device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            a, _ = manual(tokens)
            b, loss = sdpa(tokens, targets)
        difference = (a.float() - b.float()).abs().max().item()
        print("CUDA FP16 manual/SDPA max logit difference:", difference)
        self.assertLess(difference, 0.02)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(sdpa.token_embedding.weight.grad).all().item())
        changed = tokens.clone()
        changed[:, 12:] = (changed[:, 12:] + 17) % self.manual_config.vocab_size
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            future, _ = sdpa(changed)
        self.assertTrue(torch.allclose(b[:, :12], future[:, :12], atol=0.02, rtol=0.02))
