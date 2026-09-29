"""Prepare verified nano train and validation token binaries."""

from pathlib import Path
import json
import os
import tempfile

import sentencepiece as spm

from .prepare_tokens import _write_split, hash_file


def prepare_nano_tokens(root: Path) -> dict:
    corpus_path = root / "data/manifests/nano_corpus.json"
    tokenizer_path = root / "tokenizer/nano_8k.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if tokenizer["corpus_sha256"] != corpus["train_sha256"]:
        raise RuntimeError("Nano tokenizer trained on a different corpus")
    model_path = Path(tokenizer["model_path"])
    if hash_file(model_path) != tokenizer["model_sha256"]:
        raise RuntimeError("Nano tokenizer model hash mismatch")
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    if processor.get_piece_size() != 8192:
        raise RuntimeError("Nano tokenizer vocabulary mismatch")
    directory = root / "data/tokenized"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "nano_8k.json"
    paths = {
        "train": directory / "nano_8k_train.bin",
        "validation": directory / "nano_8k_val.bin",
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["tokenizer_sha256"] != tokenizer["model_sha256"]:
            raise RuntimeError("Existing nano token data uses another tokenizer")
        if manifest["corpus_manifest_sha256"] != hash_file(corpus_path):
            raise RuntimeError("Existing nano token data uses another corpus")
        for name, path in paths.items():
            entry = manifest["splits"][name]
            if not path.is_file() or hash_file(path) != entry["sha256"]:
                raise RuntimeError(f"Existing nano binary mismatch: {path}")
        print("Existing nano token binaries verified; nothing overwritten.", flush=True)
        return manifest
    if any(path.exists() for path in paths.values()):
        raise RuntimeError("Partial nano binaries exist without manifest")
    train = _write_split(
        Path(corpus["train_path"]), paths["train"], processor,
        corpus["train_documents"], corpus["train_sha256"],
    )
    validation = _write_split(
        Path(corpus["val_path"]), paths["validation"], processor,
        corpus["validation_documents"], corpus["val_sha256"],
    )
    manifest = {
        "name": "nano_8k",
        "dtype": "uint16_le",
        "vocab_size": processor.get_piece_size(),
        "special_token_ids": tokenizer["special_token_ids"],
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
        prefix=".nano_manifest_", delete=False,
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
