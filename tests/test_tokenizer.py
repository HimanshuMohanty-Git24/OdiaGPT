"""Tests for the from-scratch OdiaGPT smoke tokenizer."""

from pathlib import Path
import unittest

from odiagpt.tokenizer import load_tokenizer


ROOT = Path("/content/drive/MyDrive/OdiaGPT")


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp = load_tokenizer(ROOT)

    def test_vocabulary_and_special_ids(self):
        self.assertEqual(self.sp.get_piece_size(), 4096)
        self.assertEqual(
            (self.sp.pad_id(), self.sp.unk_id(), self.sp.bos_id(), self.sp.eos_id()),
            (0, 1, 2, 3),
        )

    def test_empty_input(self):
        self.assertEqual(self.sp.encode("", out_type=int), [])
        self.assertEqual(self.sp.decode([]), "")

    def test_round_trips(self):
        samples = [
            "ଓଡ଼ିଆ ଭାଷାରେ ଏହା ଏକ ବାକ୍ୟ।",
            "କି କୁ କେ କୌ ଙ୍କ ଣ୍ଡ",
            "୧୨୩, 2026! ମୂଲ୍ୟ ₹୫୦।",
            "ଓଡ଼ିଆ and English code-mixed text.",
            "https://example.com/odia?q=123",
            "ଓଡ଼ିଆ 🙂 rare 漢字",
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                ids = self.sp.encode(sample, out_type=int)
                self.assertTrue(ids)
                self.assertTrue(all(0 <= index < 4096 for index in ids))
                self.assertEqual(self.sp.decode(ids), sample)

    def test_no_unknowns_for_samples(self):
        for sample in ("ଓଡ଼ିଆ ଭାଷା", "rare 🙂 漢字"):
            ids = self.sp.encode(sample, out_type=int)
            self.assertNotIn(self.sp.unk_id(), ids)


if __name__ == "__main__":
    unittest.main()
