"""Train an 8K SentencePiece BPE tokenizer on nano training documents only."""

from dataclasses import asdict
from pathlib import Path
import json
import os
import shutil
import tempfile

import sentencepiece as spm

from .prepare_tokens import hash_file
from .tokenizer import TokenizerConfig


def _atomic_copy(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=target.parent, prefix=".nano_spm_",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, handle)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def train_nano_tokenizer(root: Path) -> dict:
    corpus_manifest_path = root / "data/manifests/nano_corpus.json"
    corpus = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    train_path = Path(corpus["train_path"])
    if hash_file(train_path) != corpus["train_sha256"]:
        raise RuntimeError("Nano training corpus hash mismatch")
    config = TokenizerConfig(vocab_size=8192)
    model_path = root / "tokenizer/nano_8k.model"
    vocab_path = root / "tokenizer/nano_8k.vocab"
    manifest_path = root / "tokenizer/nano_8k.json"
    outputs = (model_path, vocab_path, manifest_path)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not all(path.is_file() for path in outputs):
            raise RuntimeError("Nano tokenizer artifacts incomplete")
        if manifest["config"] != asdict(config):
            raise RuntimeError("Nano tokenizer config mismatch")
        if manifest["corpus_sha256"] != corpus["train_sha256"]:
            raise RuntimeError("Nano tokenizer corpus mismatch")
        if hash_file(model_path) != manifest["model_sha256"]:
            raise RuntimeError("Nano tokenizer model hash mismatch")
        print("Existing nano tokenizer verified; nothing overwritten.", flush=True)
        return manifest
    if any(path.exists() for path in outputs):
        raise RuntimeError("Partial nano tokenizer artifacts without manifest")

    runtime = Path("/content/odiagpt_runtime")
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nano_spm_", dir=runtime) as folder:
        work = Path(folder)
        text_path = work / "train.txt"
        count = 0
        with train_path.open("r", encoding="utf-8") as source, text_path.open(
            "w", encoding="utf-8"
        ) as destination:
            for line in source:
                row = json.loads(line)
                text = row["text"].replace("\n", " ").replace("\r", " ").strip()
                if not text:
                    raise RuntimeError(f"Empty nano training document at row {count+1}")
                destination.write(text + "\n")
                count += 1
        if count != corpus["train_documents"]:
            raise RuntimeError("Nano tokenizer document count mismatch")
        prefix = work / "nano_8k"
        print(
            f"Training SentencePiece BPE: {count:,} documents, "
            f"{text_path.stat().st_size:,} bytes, vocab=8192",
            flush=True,
        )
        spm.SentencePieceTrainer.train(
            input=str(text_path),
            model_prefix=str(prefix),
            vocab_size=config.vocab_size,
            model_type=config.model_type,
            character_coverage=config.character_coverage,
            pad_id=config.pad_id,
            unk_id=config.unk_id,
            bos_id=config.bos_id,
            eos_id=config.eos_id,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            byte_fallback=True,
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            max_sentence_length=100000,
            input_sentence_size=0,
            shuffle_input_sentence=False,
            num_threads=4,
            hard_vocab_limit=True,
            minloglevel=2,
        )
        source_model = Path(str(prefix) + ".model")
        source_vocab = Path(str(prefix) + ".vocab")
        processor = spm.SentencePieceProcessor(model_file=str(source_model))
        ids = {
            "pad": processor.pad_id(), "unk": processor.unk_id(),
            "bos": processor.bos_id(), "eos": processor.eos_id(),
        }
        expected = {
            "pad": config.pad_id, "unk": config.unk_id,
            "bos": config.bos_id, "eos": config.eos_id,
        }
        if processor.get_piece_size() != 8192 or ids != expected:
            raise RuntimeError("Nano tokenizer vocabulary or special IDs mismatch")
        _atomic_copy(source_model, model_path)
        _atomic_copy(source_vocab, vocab_path)
    manifest = {
        "name": "nano_8k",
        "config": asdict(config),
        "actual_vocab_size": processor.get_piece_size(),
        "special_token_ids": ids,
        "training_documents": count,
        "corpus_sha256": corpus["train_sha256"],
        "source_revision": corpus["source_revision"],
        "sentencepiece_version": spm.__version__,
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "model_sha256": hash_file(model_path),
        "vocab_sha256": hash_file(vocab_path),
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=manifest_path.parent,
        prefix=".nano_spm_", delete=False,
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
    print(
        f"Nano tokenizer saved: vocab=8192, "
        f"SHA256={manifest['model_sha256']}",
        flush=True,
    )
    return manifest
