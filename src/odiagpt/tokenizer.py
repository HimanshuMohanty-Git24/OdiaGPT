"""Train and load a SentencePiece tokenizer from OdiaGPT training text."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import tempfile

import sentencepiece as spm


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 4096
    model_type: str = "bpe"
    character_coverage: float = 1.0
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    def __post_init__(self) -> None:
        if self.vocab_size < 512:
            raise ValueError("vocab_size must be at least 512")
        if self.model_type not in {"bpe", "unigram"}:
            raise ValueError("model_type must be bpe or unigram")


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".spm_", delete=False) as handle:
        temporary = Path(handle.name)
        with source.open("rb") as input_file:
            shutil.copyfileobj(input_file, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _atomic_text(target: Path, content: str) -> None:
    if target.exists():
        raise FileExistsError(target)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent,
        prefix=".spm_", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def train_smoke_tokenizer(
    project_root: Path,
    config: TokenizerConfig | None = None,
) -> dict:
    config = config or TokenizerConfig()
    corpus_manifest_path = project_root / "data/manifests/smoke_corpus.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    train_path = Path(corpus_manifest["train_path"])
    if _hash_file(train_path) != corpus_manifest["train_sha256"]:
        raise RuntimeError("Training corpus hash differs from its manifest")

    model_path = project_root / "tokenizer/smoke_4k.model"
    vocab_path = project_root / "tokenizer/smoke_4k.vocab"
    manifest_path = project_root / "tokenizer/smoke_4k.json"
    outputs = (model_path, vocab_path, manifest_path)

    if manifest_path.exists():
        if not all(path.is_file() for path in outputs):
            raise RuntimeError("Tokenizer manifest exists but an artifact is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"] != asdict(config):
            raise RuntimeError("Existing tokenizer uses a different config")
        if manifest["corpus_sha256"] != corpus_manifest["train_sha256"]:
            raise RuntimeError("Existing tokenizer uses a different corpus")
        if manifest["model_sha256"] != _hash_file(model_path):
            raise RuntimeError("Existing tokenizer model hash mismatch")
        print("Existing tokenizer verified; no artifacts overwritten.", flush=True)
        return manifest

    if any(path.exists() for path in outputs):
        raise RuntimeError("Partial tokenizer artifacts exist without a manifest")

    runtime_root = Path("/content/odiagpt_runtime")
    runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spm_", dir=runtime_root) as work_dir:
        work = Path(work_dir)
        training_text = work / "train.txt"
        document_count = 0
        with train_path.open("r", encoding="utf-8") as source, training_text.open(
            "w", encoding="utf-8"
        ) as destination:
            for line in source:
                document = json.loads(line)
                text = document["text"].replace("\n", " ").replace("\r", " ").strip()
                if not text:
                    continue
                destination.write(text + "\n")
                document_count += 1
        if document_count != corpus_manifest["train_documents"]:
            raise RuntimeError("Tokenizer input document count mismatch")

        prefix = work / "smoke_4k"
        print(
            f"Training SentencePiece {config.model_type}: "
            f"{document_count} train documents, vocab={config.vocab_size}",
            flush=True,
        )
        spm.SentencePieceTrainer.train(
            input=str(training_text),
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
            num_threads=1,
            hard_vocab_limit=True,
        )
        processor = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
        actual_size = processor.get_piece_size()
        if actual_size != config.vocab_size:
            raise RuntimeError(f"Expected {config.vocab_size} pieces; got {actual_size}")
        ids = {
            "pad": processor.pad_id(),
            "unk": processor.unk_id(),
            "bos": processor.bos_id(),
            "eos": processor.eos_id(),
        }
        expected = {
            "pad": config.pad_id,
            "unk": config.unk_id,
            "bos": config.bos_id,
            "eos": config.eos_id,
        }
        if ids != expected:
            raise RuntimeError(f"Special token IDs mismatch: {ids} != {expected}")
        _atomic_copy(Path(str(prefix) + ".model"), model_path)
        _atomic_copy(Path(str(prefix) + ".vocab"), vocab_path)

    manifest = {
        "name": "smoke_4k",
        "config": asdict(config),
        "actual_vocab_size": actual_size,
        "special_token_ids": ids,
        "training_documents": document_count,
        "corpus_sha256": corpus_manifest["train_sha256"],
        "source_revision": corpus_manifest["source_revision"],
        "sentencepiece_version": spm.__version__,
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "model_sha256": _hash_file(model_path),
        "vocab_sha256": _hash_file(vocab_path),
    }
    _atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Tokenizer saved: vocab={actual_size}, model_sha256={manifest['model_sha256']}",
        flush=True,
    )
    return manifest


def load_tokenizer(project_root: Path, name: str = "smoke_4k") -> spm.SentencePieceProcessor:
    model_path = project_root / "tokenizer" / f"{name}.model"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    return spm.SentencePieceProcessor(model_file=str(model_path))
