"""Tests for the from-scratch nano 8K SentencePiece tokenizer."""

from pathlib import Path
import unittest

import sentencepiece as spm


ROOT = Path("/content/drive/MyDrive/OdiaGPT")


class NanoTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp = spm.SentencePieceProcessor(
            model_file=str(ROOT / "tokenizer/nano_8k.model")
        )

    def test_vocabulary_and_special_ids(self):
        self.assertEqual(self.sp.get_piece_size(), 8192)
        self.assertEqual(
            (self.sp.pad_id(), self.sp.unk_id(), self.sp.bos_id(), self.sp.eos_id()),
            (0, 1, 2, 3),
        )

    def test_exact_roundtrips(self):
        samples = [
            "ଓଡ଼ିଆ ଭାଷାରେ ଏହା ଏକ ବାକ୍ୟ।",
            "କି କୁ କେ କୌ ଙ୍କ ଣ୍ଡ",
            "୧୨୩, 2026! ମୂଲ୍ୟ ₹୫୦।",
            "ଓଡ଼ିଆ and English code-mixed text.",
            "https://example.org/ଓଡ଼ିଆ?q=୧",
            "ଦୁର୍ଲଭ ♞ 字 ଚିହ୍ନ",
            "ପ୍ରଥମ ଧାଡ଼ି\nଦ୍ୱିତୀୟ ଧାଡ଼ି",
            "",
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                ids = self.sp.encode(sample, out_type=int)
                self.assertEqual(self.sp.decode(ids), sample)
                self.assertNotIn(self.sp.unk_id(), ids)

    def test_empty_input(self):
        self.assertEqual(self.sp.encode("", out_type=int), [])

    def test_token_ids_in_range(self):
        ids = self.sp.encode("ଓଡ଼ିଆ ଭାଷା ୧୨୩ ABC", out_type=int)
        self.assertTrue(all(0 <= i < 8192 for i in ids))
