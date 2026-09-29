"""Unicode-aware text statistics and conservative normalization for Odia."""

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata


ODIA_START = 0x0B00
ODIA_END = 0x0B7F


@dataclass(frozen=True)
class TextStats:
    char_count: int
    odia_letter_mark_count: int
    letter_mark_count: int
    odia_script_ratio: float
    line_count: int
    repeated_line_fraction: float


def is_odia_letter_or_mark(char: str) -> bool:
    """Include Odia letters and their combining marks."""
    if len(char) != 1:
        raise ValueError("Expected exactly one Unicode character")
    codepoint = ord(char)
    return (
        ODIA_START <= codepoint <= ODIA_END
        and unicodedata.category(char)[0] in {"L", "M"}
    )


def normalize_text(text: str) -> str:
    """Normalize conservatively while preserving Odia marks and punctuation."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    text = unicodedata.normalize("NFC", text)
    text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u0085", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
    )

    # Remove nulls and other Cc controls; retain tabs and newlines until
    # whitespace normalization. Format characters such as ZWJ are preserved.
    text = "".join(
        char
        for char in text
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )

    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_stats(text: str) -> TextStats:
    """Measure script use among Unicode letters and combining marks."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    letter_mark_count = 0
    odia_count = 0
    for char in text:
        if unicodedata.category(char)[0] in {"L", "M"}:
            letter_mark_count += 1
            if is_odia_letter_or_mark(char):
                odia_count += 1

    ratio = odia_count / letter_mark_count if letter_mark_count else 0.0

    # Repeated long lines can indicate boilerplate or extraction errors.
    # Count only extra copies, so a line appearing twice contributes one.
    eligible_lines = [
        line.strip() for line in text.splitlines() if len(line.strip()) >= 20
    ]
    if len(eligible_lines) >= 4:
        counts = Counter(eligible_lines)
        repeated = sum(count - 1 for count in counts.values())
        repeated_fraction = repeated / len(eligible_lines)
    else:
        repeated_fraction = 0.0

    return TextStats(
        char_count=len(text),
        odia_letter_mark_count=odia_count,
        letter_mark_count=letter_mark_count,
        odia_script_ratio=ratio,
        line_count=len(text.splitlines()) if text else 0,
        repeated_line_fraction=repeated_fraction,
    )
