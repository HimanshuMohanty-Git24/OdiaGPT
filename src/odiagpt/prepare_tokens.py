"""Convert verified document splits to packed uint16 token streams."""

from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import tempfile

import numpy as np
import sentencepiece as spm


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_split(
    input_path: Path,
    output_path: Path,
    processor: spm.SentencePieceProcessor,
    expected_documents: int,
    expected_input_hash: str,
) -> dict:
    if hash_file(input_path) != expected_input_hash:
        raise RuntimeError(f"Input hash mismatch: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    document_count = 0
    maximum_id = -1
    # Stage on Drive, next to the target, so os.replace is atomic.
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_path.parent, prefix=".tokens_",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            with input_path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    row = json.loads(line)
                    text = row["text"]
                    if not isinstance(text, str) or not text:
                        raise RuntimeError(f"Invalid text at line {line_number}: {input_path}")
                    ids = processor.encode(text, out_type=int)
                    if not ids:
                        raise RuntimeError(f"Empty encoding at line {line_number}: {input_path}")
                    if any(i in (processor.pad_id(), processor.unk_id()) for i in ids):
                        raise RuntimeError(f"Pad/unknown token at line {line_number}: {input_path}")
                    ids = [processor.bos_id(), *ids, processor.eos_id()]
                    if min(ids) < 0 or max(ids) >= processor.get_piece_size():
                        raise RuntimeError(f"Token ID out of range at line {line_number}")
                    block = np.asarray(ids, dtype="<u2")
                    handle.write(block.tobytes())
                    count += len(ids)
                    document_count += 1
                    maximum_id = max(maximum_id, max(ids))
            handle.flush()
            os.fsync(handle.fileno())
            if document_count != expected_documents:
                raise RuntimeError(
                    f"Document count mismatch: {document_count} != {expected_documents}"
                )
            if count <= 1:
                raise RuntimeError("Token stream is too short")
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "path": str(output_path),
        "documents": document_count,
        "tokens": count,
        "bytes": output_path.stat().st_size,
        "sha256": hash_file(output_path),
        "maximum_token_id": maximum_id,
        "source_sha256": expected_input_hash,
    }


def prepare_smoke_tokens(project_root: Path) -> dict:
    corpus_path = project_root / "data/manifests/smoke_corpus.json"
    tokenizer_path = project_root / "tokenizer/smoke_4k.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model_path = Path(tokenizer["model_path"])
    if hash_file(model_path) != tokenizer["model_sha256"]:
        raise RuntimeError("Tokenizer model hash mismatch")
    if corpus["train_sha256"] != tokenizer["corpus_sha256"]:
        raise RuntimeError("Tokenizer trained on a different corpus")
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = processor.get_piece_size()
    if vocab_size > 65536:
        raise ValueError("Vocabulary does not fit uint16")
    if vocab_size != tokenizer["actual_vocab_size"]:
        raise RuntimeError("Tokenizer vocabulary size mismatch")
    expected_special = tokenizer["special_token_ids"]
    actual_special = {
        "pad": processor.pad_id(), "unk": processor.unk_id(),
        "bos": processor.bos_id(), "eos": processor.eos_id(),
    }
    if actual_special != expected_special:
        raise RuntimeError("Special-token ID mismatch")

    directory = project_root / "data/tokenized"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "smoke_4k.json"
    outputs = {
        "train": directory / "smoke_4k_train.bin",
        "validation": directory / "smoke_4k_val.bin",
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["tokenizer_sha256"] != tokenizer["model_sha256"]:
            raise RuntimeError("Existing token data uses a different tokenizer")
        if manifest["corpus_manifest_sha256"] != hash_file(corpus_path):
            raise RuntimeError("Existing token data uses a different corpus manifest")
        for name, path in outputs.items():
            if not path.is_file():
                raise RuntimeError(f"Missing binary file: {path}")
            entry = manifest["splits"][name]
            if hash_file(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                raise RuntimeError(f"Binary file differs from manifest: {path}")
        print("Existing tokenized binaries verified; nothing overwritten.", flush=True)
        return manifest
    if any(path.exists() for path in outputs.values()):
        raise RuntimeError("Partial tokenized output exists without manifest; inspect before retry")

    train = _write_split(
        Path(corpus["train_path"]), outputs["train"], processor,
        corpus["train_documents"], corpus["train_sha256"],
    )
    validation = _write_split(
        Path(corpus["val_path"]), outputs["validation"], processor,
        corpus["validation_documents"], corpus["val_sha256"],
    )
    manifest = {
        "name": "smoke_4k",
        "dtype": "uint16_le",
        "vocab_size": vocab_size,
        "special_token_ids": actual_special,
        "document_boundaries": "BOS before and EOS after each document",
        "tokenizer_sha256": tokenizer["model_sha256"],
        "corpus_manifest_sha256": hash_file(corpus_path),
        "cleaning_config": corpus["cleaning_config"],
        "split_seed": corpus["build_config"]["split_seed"],
        "dataset_source": corpus["source"],
        "dataset_configuration": corpus["configuration"],
        "dataset_split": corpus["split"],
        "dataset_revision": corpus["source_revision"],
        "splits": {"train": train, "validation": validation},
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory,
        prefix=".manifest_", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest
