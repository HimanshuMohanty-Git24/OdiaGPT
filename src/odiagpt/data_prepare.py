"""Build a small, document-split, cleaned Sangraha Odia corpus."""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean, median
import json
import os
import random
import tempfile
import unicodedata

from datasets import load_dataset
from huggingface_hub import HfApi

from .cleaning import CleaningConfig, clean_document, exact_fingerprint


REPO_ID = "ai4bharat/sangraha"


@dataclass(frozen=True)
class CorpusBuildConfig:
    target_clean_chars: int = 1_500_000
    max_stream_rows: int = 3_000
    max_document_chars: int = 20_000
    validation_fraction: float = 0.01
    split_seed: int = 42

    def __post_init__(self) -> None:
        if self.target_clean_chars <= 0 or self.max_stream_rows <= 0:
            raise ValueError("Corpus size limits must be positive")
        if self.max_document_chars <= 0:
            raise ValueError("max_document_chars must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    """Write beside the destination, then rename after a successful flush."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Will not overwrite existing artifact: {path}")

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        file.write(content)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_path, path)


def _jsonl(documents: list[dict]) -> str:
    return "".join(
        json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n"
        for doc in documents
    )


def _preview(text: str) -> str:
    """Keep report excerpts short and safe for a Markdown table."""
    return text[:70].replace("\n", "⏎").replace("|", "\\|")


def build_smoke_corpus(
    project_root: Path,
    build_config: CorpusBuildConfig | None = None,
    cleaning_config: CleaningConfig | None = None,
) -> dict:
    if build_config is None:
        build_config = CorpusBuildConfig()
    if cleaning_config is None:
        cleaning_config = CleaningConfig()

    train_path = project_root / "data/processed/smoke_train.jsonl"
    val_path = project_root / "data/processed/smoke_val.jsonl"
    manifest_path = project_root / "data/manifests/smoke_corpus.json"
    report_path = project_root / "reports/data_quality_report.md"
    outputs = (train_path, val_path, manifest_path, report_path)

    if manifest_path.exists():
        if not all(path.is_file() for path in outputs):
            raise RuntimeError("Corpus manifest exists, but an output file is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["build_config"] != asdict(build_config):
            raise RuntimeError("Existing corpus uses a different build config")
        if manifest["cleaning_config"] != asdict(cleaning_config):
            raise RuntimeError("Existing corpus uses a different cleaning config")
        if manifest["train_sha256"] != _sha256_file(train_path):
            raise RuntimeError("Existing training corpus hash does not match manifest")
        if manifest["val_sha256"] != _sha256_file(val_path):
            raise RuntimeError("Existing validation corpus hash does not match manifest")
        print("Existing corpus verified; no files overwritten.", flush=True)
        return manifest

    if any(path.exists() for path in outputs):
        raise RuntimeError(
            "Partial corpus outputs exist without a manifest. "
            "Stop and inspect them before retrying."
        )

    revision = HfApi().dataset_info(REPO_ID).sha
    if not revision:
        raise RuntimeError("Could not resolve a fixed Sangraha revision")
    print(f"Dataset revision: {revision}", flush=True)
    print("Streaming verified/ori; no full dataset download requested.", flush=True)

    dataset = load_dataset(
        REPO_ID,
        "verified",
        split="ori",
        revision=revision,
        streaming=True,
    )

    documents: list[dict] = []
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    rejections: Counter[str] = Counter()
    source_types: Counter[str] = Counter()
    raw_clean_examples: list[tuple[str, str]] = []
    inspected = 0
    total_chars = 0
    nfc_raw_count = 0

    for row in dataset:
        if inspected >= build_config.max_stream_rows:
            break
        inspected += 1

        raw_text = row.get("text")
        doc_id = row.get("doc_id")
        source_type = row.get("type", "unknown")
        source_types[str(source_type)] += 1

        if not isinstance(raw_text, str) or not isinstance(doc_id, str):
            rejections["missing_text_or_doc_id"] += 1
            continue
        if unicodedata.is_normalized("NFC", raw_text):
            nfc_raw_count += 1

        result = clean_document(raw_text, cleaning_config)
        if not result.accepted:
            rejections[result.rejection_reason or "unknown"] += 1
            continue

        cleaned = result.text
        assert cleaned is not None

        if len(cleaned) > build_config.max_document_chars:
            rejections["too_long_for_smoke"] += 1
            continue
        if doc_id in seen_ids:
            rejections["duplicate_doc_id"] += 1
            continue

        fingerprint = exact_fingerprint(cleaned)
        if fingerprint in seen_fingerprints:
            rejections["exact_duplicate_text"] += 1
            continue

        seen_ids.add(doc_id)
        seen_fingerprints.add(fingerprint)
        documents.append(
            {"doc_id": doc_id, "type": str(source_type), "text": cleaned}
        )
        total_chars += len(cleaned)

        if len(raw_clean_examples) < 3:
            raw_clean_examples.append((raw_text, cleaned))

        if inspected % 250 == 0:
            print(
                f"inspected={inspected} accepted={len(documents)} "
                f"clean_chars={total_chars:,}",
                flush=True,
            )
        if total_chars >= build_config.target_clean_chars:
            break

    if len(documents) < 2 or total_chars < 500_000:
        raise RuntimeError(
            f"Too little usable data: {len(documents)} documents, "
            f"{total_chars} cleaned characters"
        )

    validation_count = max(
        1, round(len(documents) * build_config.validation_fraction)
    )
    validation_count = min(validation_count, len(documents) - 1)
    rng = random.Random(build_config.split_seed)
    validation_indices = set(
        rng.sample(range(len(documents)), validation_count)
    )
    train_documents = [
        doc for index, doc in enumerate(documents)
        if index not in validation_indices
    ]
    val_documents = [
        doc for index, doc in enumerate(documents)
        if index in validation_indices
    ]

    train_ids = {doc["doc_id"] for doc in train_documents}
    val_ids = {doc["doc_id"] for doc in val_documents}
    assert train_ids.isdisjoint(val_ids)

    train_text = _jsonl(train_documents)
    val_text = _jsonl(val_documents)
    _atomic_write_text(train_path, train_text)
    _atomic_write_text(val_path, val_text)

    lengths = [len(doc["text"]) for doc in documents]
    ratios = [
        clean_document(doc["text"], cleaning_config).stats.odia_script_ratio
        for doc in documents
    ]
    openings = Counter(
        doc["text"].splitlines()[0][:80]
        for doc in documents
        if doc["text"].splitlines()
    )
    repeated_openings = sum(count >= 3 for count in openings.values())

    report_lines = [
        "# Initial Odia data quality report",
        "",
        "This report covers a small prefix of the verified Odia stream. "
        "It is not a representative random sample.",
        "",
        f"- Dataset: `{REPO_ID}`, configuration `verified`, split `ori`",
        f"- Fixed dataset revision: `{revision}`",
        f"- Documents inspected: {inspected:,}",
        f"- Documents retained: {len(documents):,}",
        f"- Documents rejected: {sum(rejections.values()):,}",
        f"- Exact duplicate texts rejected: "
        f"{rejections['exact_duplicate_text']:,}",
        f"- Duplicate document IDs rejected: "
        f"{rejections['duplicate_doc_id']:,}",
        f"- Raw documents already NFC: {nfc_raw_count:,}",
        f"- Cleaned characters: {total_chars:,}",
        f"- Cleaned length mean / median: "
        f"{mean(lengths):.1f} / {median(lengths):.1f}",
        f"- Odia letter/mark ratio mean / median: "
        f"{mean(ratios):.4f} / {median(ratios):.4f}",
        f"- Opening-line prefixes seen in at least three documents: "
        f"{repeated_openings:,} (detection only)",
        f"- Train / validation documents: "
        f"{len(train_documents):,} / {len(val_documents):,}",
        "",
        "## Source types among inspected documents",
        "",
    ]
    report_lines.extend(
        f"- `{kind}`: {count:,}" for kind, count in sorted(source_types.items())
    )
    report_lines.extend(["", "## Rejection reasons", ""])
    report_lines.extend(
        f"- `{reason}`: {count:,}"
        for reason, count in sorted(rejections.items())
    )
    if not rejections:
        report_lines.append("- None in this sample")
    report_lines.extend(
        [
            "",
            "## Short raw / clean comparisons",
            "",
            "Only the first 70 characters of each version are shown.",
            "",
            "| Example | Raw prefix | Clean prefix |",
            "|---|---|---|",
        ]
    )
    report_lines.extend(
        f"| {index} | {_preview(raw)} | {_preview(clean)} |"
        for index, (raw, clean) in enumerate(raw_clean_examples, start=1)
    )
    report_lines.extend(
        [
            "",
            "## Unicode observations",
            "",
            "NFC normalization was applied. Odia combining marks, punctuation, "
            "digits, and code-mixed text were preserved by the cleaning rules. "
            "The initial 200-row inspection found no unwanted controls, but "
            "that does not establish their absence from the full dataset.",
            "",
            "Document splitting occurred before token packing. Exact normalized "
            "text duplicates were removed before the split.",
            "",
        ]
    )
    _atomic_write_text(report_path, "\n".join(report_lines))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": REPO_ID,
        "source_revision": revision,
        "configuration": "verified",
        "split": "ori",
        "build_config": asdict(build_config),
        "cleaning_config": asdict(cleaning_config),
        "documents_inspected": inspected,
        "documents_retained": len(documents),
        "rejections": dict(rejections),
        "cleaned_characters": total_chars,
        "train_documents": len(train_documents),
        "validation_documents": len(val_documents),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_sha256": _sha256_file(train_path),
        "val_sha256": _sha256_file(val_path),
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )

    print(
        f"Corpus saved: inspected={inspected}, "
        f"retained={len(documents)}, clean_chars={total_chars:,}",
        flush=True,
    )
    print(
        f"Train docs={len(train_documents)}, "
        f"validation docs={len(val_documents)}",
        flush=True,
    )
    return manifest
