"""Build a larger, resumable Odia corpus for the nano experiment."""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean, median
import json
import os
import tempfile
import unicodedata

from datasets import load_dataset

from .cleaning import CleaningConfig, clean_document, exact_fingerprint
from .prepare_tokens import hash_file


@dataclass(frozen=True)
class NanoCorpusConfig:
    target_clean_chars: int = 20_000_000
    max_stream_rows: int = 40_000
    max_document_chars: int = 20_000
    validation_fraction: float = 0.01
    split_seed: int = 42


def atomic_text(path: Path, value: str, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=".nano_", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _split_for_id(doc_id: str, seed: int, fraction: float) -> str:
    digest = sha256(f"{seed}:{doc_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if value < fraction else "train"


def build_nano_corpus(root: Path, config: NanoCorpusConfig | None = None) -> dict:
    config = config or NanoCorpusConfig()
    if config.target_clean_chars <= 0 or config.max_stream_rows <= 0:
        raise ValueError("Invalid nano corpus limits")
    if not 0 < config.validation_fraction < 1:
        raise ValueError("Invalid validation fraction")
    smoke = json.loads((root / "data/manifests/smoke_corpus.json").read_text(encoding="utf-8"))
    cleaning = CleaningConfig(**smoke["cleaning_config"])
    revision = smoke["source_revision"]
    train_path = root / "data/processed/nano_train.jsonl"
    val_path = root / "data/processed/nano_val.jsonl"
    manifest_path = root / "data/manifests/nano_corpus.json"
    report_path = root / "reports/nano_data_quality_report.md"
    journal_path = root / "data/raw/nano_accepted.jsonl"
    progress_path = root / "data/metadata/nano_build_progress.json"
    if manifest_path.is_file():
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        if result["build_config"] != asdict(config):
            raise RuntimeError("Existing nano corpus has a different config")
        if result["source_revision"] != revision:
            raise RuntimeError("Existing nano corpus has a different source revision")
        if hash_file(train_path) != result["train_sha256"]:
            raise RuntimeError("Existing nano train hash mismatch")
        if hash_file(val_path) != result["val_sha256"]:
            raise RuntimeError("Existing nano validation hash mismatch")
        print("Existing nano corpus verified; nothing overwritten.", flush=True)
        return result
    if train_path.exists() or val_path.exists() or report_path.exists():
        raise RuntimeError("Partial final nano outputs exist without manifest; inspect before retry")
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["config"] != asdict(config) or progress["revision"] != revision:
            raise RuntimeError("Saved nano progress uses a different config or revision")
        committed_rows = int(progress["committed_rows"])
        source_types = Counter(progress["source_types"])
        rejections = Counter(progress["rejections"])
        nfc_count = int(progress["nfc_count"])
    else:
        committed_rows = 0
        source_types = Counter()
        rejections = Counter()
        nfc_count = 0
    documents = []
    seen_ids = {}
    seen_fingerprints = {}
    maximum_journal_row = 0
    if journal_path.exists():
        with journal_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                doc = json.loads(line)
                index = int(doc["stream_index"])
                fingerprint = exact_fingerprint(doc["text"])
                if doc["doc_id"] in seen_ids or fingerprint in seen_fingerprints:
                    raise RuntimeError("Duplicate in nano journal")
                seen_ids[doc["doc_id"]] = (index, fingerprint)
                seen_fingerprints[fingerprint] = index
                documents.append(doc)
                maximum_journal_row = max(maximum_journal_row, index)
    total_chars = sum(len(doc["text"]) for doc in documents)
    print(
        f"Nano resume: committed_rows={committed_rows}, "
        f"journal_docs={len(documents)}, cleaned_chars={total_chars:,}",
        flush=True,
    )
    dataset = load_dataset(
        smoke["source"], smoke["configuration"], split=smoke["split"],
        revision=revision, streaming=True,
    )
    iterator = iter(dataset.skip(committed_rows))
    inspected = committed_rows
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as journal:
        while inspected < config.max_stream_rows:
            try:
                row = next(iterator)
            except StopIteration:
                break
            inspected += 1
            raw = row.get("text")
            doc_id = row.get("doc_id")
            source_types[str(row.get("type", "unknown"))] += 1
            if not isinstance(raw, str) or not isinstance(doc_id, str):
                rejections["missing_text_or_doc_id"] += 1
            else:
                if unicodedata.is_normalized("NFC", raw):
                    nfc_count += 1
                cleaned = clean_document(raw, cleaning)
                if not cleaned.accepted:
                    rejections[cleaned.rejection_reason or "unknown"] += 1
                elif len(cleaned.text) > config.max_document_chars:
                    rejections["too_long_for_nano"] += 1
                else:
                    fingerprint = exact_fingerprint(cleaned.text)
                    if doc_id in seen_ids:
                        old_index, old_fingerprint = seen_ids[doc_id]
                        if old_index != inspected or old_fingerprint != fingerprint:
                            rejections["duplicate_doc_id"] += 1
                    elif fingerprint in seen_fingerprints:
                        rejections["exact_duplicate_text"] += 1
                    else:
                        doc = {
                            "stream_index": inspected, "doc_id": doc_id,
                            "type": str(row.get("type", "unknown")),
                            "text": cleaned.text,
                        }
                        journal.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        documents.append(doc)
                        seen_ids[doc_id] = (inspected, fingerprint)
                        seen_fingerprints[fingerprint] = inspected
                        total_chars += len(cleaned.text)
            if inspected % 250 == 0:
                journal.flush()
                os.fsync(journal.fileno())
                atomic_text(progress_path, json.dumps({
                    "config": asdict(config), "revision": revision,
                    "committed_rows": inspected,
                    "source_types": dict(source_types),
                    "rejections": dict(rejections),
                    "nfc_count": nfc_count,
                    "documents": len(documents),
                    "cleaned_chars": total_chars,
                }, indent=2) + "\n", overwrite=True)
                print(
                    f"inspected={inspected:,} retained={len(documents):,} "
                    f"clean_chars={total_chars:,}/{config.target_clean_chars:,}",
                    flush=True,
                )
            if total_chars >= config.target_clean_chars and inspected >= maximum_journal_row:
                break
        journal.flush()
        os.fsync(journal.fileno())
    if total_chars < config.target_clean_chars:
        raise RuntimeError(
            f"Nano target not reached: {total_chars:,} cleaned chars after "
            f"{inspected:,} streamed rows. Journal and progress remain for retry."
        )
    if not documents:
        raise RuntimeError("No nano documents retained")

    smoke_train_ids = {
        json.loads(line)["doc_id"]
        for line in Path(smoke["train_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    smoke_val_ids = {
        json.loads(line)["doc_id"]
        for line in Path(smoke["val_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if smoke_train_ids & smoke_val_ids:
        raise RuntimeError("Smoke split IDs overlap")
    train_docs = []
    val_docs = []
    for doc in documents:
        doc_id = doc["doc_id"]
        if doc_id in smoke_train_ids:
            destination = train_docs
        elif doc_id in smoke_val_ids:
            destination = val_docs
        else:
            destination = (
                val_docs if _split_for_id(
                    doc_id, config.split_seed, config.validation_fraction
                ) == "validation" else train_docs
            )
        destination.append({
            "doc_id": doc_id, "type": doc["type"], "text": doc["text"]
        })
    if not train_docs or not val_docs:
        raise RuntimeError("Nano train or validation split is empty")
    if {d["doc_id"] for d in train_docs} & {d["doc_id"] for d in val_docs}:
        raise RuntimeError("Nano split IDs overlap")
    for path, rows in ((train_path, train_docs), (val_path, val_docs)):
        atomic_text(
            path,
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        )
    lengths = [len(doc["text"]) for doc in documents]
    ratios = [clean_document(doc["text"], cleaning).stats.odia_script_ratio for doc in documents]
    report = [
        "# Nano Odia data quality report", "",
        "This is a prefix of the fixed verified/ori stream; it is not a random sample.",
        f"- Dataset: {smoke['source']}, {smoke['configuration']}/{smoke['split']}",
        f"- Revision: {revision}",
        f"- Documents inspected: {inspected:,}",
        f"- Retained: {len(documents):,}",
        f"- Rejected: {sum(rejections.values()):,}",
        f"- Cleaned characters: {total_chars:,}",
        f"- Train / validation documents: {len(train_docs):,} / {len(val_docs):,}",
        f"- Length mean / median: {mean(lengths):.1f} / {median(lengths):.1f}",
        f"- Odia ratio mean / median: {mean(ratios):.4f} / {median(ratios):.4f}",
        f"- Raw NFC documents: {nfc_count:,}", "",
        "## Rejection reasons", "",
    ]
    report.extend(f"- {reason}: {count:,}" for reason, count in sorted(rejections.items()))
    report.extend(["", "## Source types", ""])
    report.extend(f"- {kind}: {count:,}" for kind, count in sorted(source_types.items()))
    report.extend(["", "Exact duplicate text removal and document-level splitting precede token packing.",
                   "Existing smoke validation IDs remain in validation for this corpus.", ""])
    atomic_text(report_path, "\n".join(report))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": smoke["source"],
        "source_revision": revision,
        "configuration": smoke["configuration"],
        "split": smoke["split"],
        "build_config": asdict(config),
        "cleaning_config": asdict(cleaning),
        "documents_inspected": inspected,
        "documents_retained": len(documents),
        "rejections": dict(rejections),
        "cleaned_characters": total_chars,
        "train_documents": len(train_docs),
        "validation_documents": len(val_docs),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_sha256": hash_file(train_path),
        "val_sha256": hash_file(val_path),
        "journal_path": str(journal_path),
        "journal_sha256": hash_file(journal_path),
    }
    atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Nano corpus saved: {len(train_docs):,} train, {len(val_docs):,} val, "
        f"{total_chars:,} cleaned chars",
        flush=True,
    )
    return manifest
