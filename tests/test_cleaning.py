"""Focused tests for conservative Odia document cleaning."""

import unittest
import unicodedata

from odiagpt.cleaning import CleaningConfig, clean_document, exact_fingerprint
from odiagpt.unicode_utils import (
    is_odia_letter_or_mark,
    normalize_text,
    text_stats,
)


class CleaningTests(unittest.TestCase):
    def test_newlines_controls_and_whitespace(self):
        raw = " \r\nକି\x00\tପଦ । ୧୨୩\r\n\r\n\r\n"
        self.assertEqual(normalize_text(raw), "କି ପଦ । ୧୨୩")

    def test_combining_mark_is_preserved(self):
        text = "କି"
        self.assertEqual(normalize_text(text), text)
        self.assertTrue(is_odia_letter_or_mark(text[0]))
        self.assertTrue(is_odia_letter_or_mark(text[1]))
        self.assertEqual(text_stats(text).odia_letter_mark_count, 2)

    def test_nfc_normalization(self):
        self.assertEqual(normalize_text("e\u0301"), unicodedata.normalize("NFC", "e\u0301"))

    def test_punctuation_digits_english_and_url_survive(self):
        text = "ଓଡ଼ିଆ ୧୨୩, English 2026! https://example.com/path"
        cleaned = normalize_text(text)
        for fragment in ("୧୨୩,", "English 2026!", "https://example.com/path"):
            self.assertIn(fragment, cleaned)

    def test_script_ratio_ignores_digits_and_punctuation(self):
        stats = text_stats("କି 123!")
        self.assertEqual(stats.odia_letter_mark_count, 2)
        self.assertEqual(stats.letter_mark_count, 2)
        self.assertEqual(stats.odia_script_ratio, 1.0)
        self.assertEqual(text_stats("English 123").odia_script_ratio, 0.0)

    def test_code_mixed_document_can_pass(self):
        raw = "ଓଡ଼ିଆ ଭାଷା ଏବଂ English words 123. " * 4
        result = clean_document(raw)
        self.assertTrue(result.accepted, result.rejection_reason)
        self.assertIn("English words 123.", result.text)

    def test_empty_document_rejected(self):
        self.assertEqual(clean_document("\x00  \r").rejection_reason, "empty")

    def test_short_document_rejected(self):
        self.assertEqual(clean_document("ଓଡ଼ିଆ").rejection_reason, "too_short")

    def test_too_few_odia_characters_rejected(self):
        raw = "କି " + "English text " * 10
        self.assertEqual(
            clean_document(raw).rejection_reason,
            "few_odia_characters",
        )

    def test_low_odia_ratio_rejected(self):
        raw = "ଓଡ଼ିଆ " * 5 + "English " * 40
        self.assertEqual(clean_document(raw).rejection_reason, "low_odia_ratio")

    def test_repeated_lines_rejected(self):
        line = "ଓଡ଼ିଆ ଭାଷାର ଏହା ଏକ ପୂର୍ଣ୍ଣ ବାକ୍ୟ ।"
        raw = "\n".join([line] * 6)
        self.assertEqual(clean_document(raw).rejection_reason, "repeated_lines")

    def test_exact_fingerprint_uses_cleaned_text(self):
        left = exact_fingerprint(normalize_text("ଓଡ଼ିଆ  ଭାଷା"))
        right = exact_fingerprint(normalize_text("ଓଡ଼ିଆ ଭାଷା"))
        self.assertEqual(left, right)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            CleaningConfig(min_odia_script_ratio=1.5)
        with self.assertRaises(ValueError):
            CleaningConfig(min_chars=-1)


if __name__ == "__main__":
    unittest.main()
