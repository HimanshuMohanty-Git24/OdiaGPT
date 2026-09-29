"""Configurable, conservative document cleaning for OdiaGPT."""

from dataclasses import dataclass
from hashlib import sha256

from .unicode_utils import TextStats, normalize_text, text_stats


@dataclass(frozen=True)
class CleaningConfig:
    min_chars: int = 80
    min_odia_letter_marks: int = 20
    min_odia_script_ratio: float = 0.35
    max_repeated_line_fraction: float = 0.50

    def __post_init__(self) -> None:
        if self.min_chars < 0 or self.min_odia_letter_marks < 0:
            raise ValueError("Minimum counts must be nonnegative")
        if not 0.0 <= self.min_odia_script_ratio <= 1.0:
            raise ValueError("min_odia_script_ratio must be between 0 and 1")
        if not 0.0 <= self.max_repeated_line_fraction <= 1.0:
            raise ValueError("max_repeated_line_fraction must be between 0 and 1")


@dataclass(frozen=True)
class CleaningResult:
    text: str | None
    rejection_reason: str | None
    stats: TextStats

    @property
    def accepted(self) -> bool:
        return self.text is not None


def clean_document(raw_text: str, config: CleaningConfig | None = None) -> CleaningResult:
    """Normalize and either accept a document or give one primary reason."""
    if config is None:
        config = CleaningConfig()

    cleaned = normalize_text(raw_text)
    stats = text_stats(cleaned)

    reason: str | None = None
    if not cleaned:
        reason = "empty"
    elif stats.char_count < config.min_chars:
        reason = "too_short"
    elif stats.odia_letter_mark_count < config.min_odia_letter_marks:
        reason = "few_odia_characters"
    elif stats.odia_script_ratio < config.min_odia_script_ratio:
        reason = "low_odia_ratio"
    elif stats.repeated_line_fraction > config.max_repeated_line_fraction:
        reason = "repeated_lines"

    return CleaningResult(
        text=cleaned if reason is None else None,
        rejection_reason=reason,
        stats=stats,
    )


def exact_fingerprint(cleaned_text: str) -> str:
    """Hash normalized text for exact document deduplication."""
    return sha256(cleaned_text.encode("utf-8")).hexdigest()
