"""Atomic training checkpoints with hash-verified latest pointers."""

from pathlib import Path
import json
import os
import tempfile

import torch

from .prepare_tokens import hash_file


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=".checkpoint_pointer_", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def save_training_checkpoint(root: Path, experiment_id: str, payload: dict) -> dict:
    if not experiment_id or "/" in experiment_id or "\\" in experiment_id:
        raise ValueError("Invalid experiment ID")
    if payload.get("experiment_id") != experiment_id:
        raise ValueError("Checkpoint experiment ID mismatch")
    step = int(payload["step"])
    if step < 0:
        raise ValueError("Checkpoint step must be nonnegative")
    directory = root / "checkpoints" / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{step:07d}.pt"
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=directory, prefix=".checkpoint_",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    pointer = {
        "experiment_id": experiment_id,
        "step": step,
        "path": str(path),
        "sha256": hash_file(path),
    }
    _atomic_json(directory / "latest.json", pointer)
    _atomic_json(root / "checkpoints/latest.json", pointer)
    return pointer


def load_latest_checkpoint(
    root: Path, experiment_id: str, map_location: str | torch.device = "cpu",
) -> tuple[dict, dict] | None:
    pointer_path = root / "checkpoints" / experiment_id / "latest.json"
    if not pointer_path.is_file():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer["experiment_id"] != experiment_id:
        raise RuntimeError("Latest pointer experiment mismatch")
    path = Path(pointer["path"])
    expected_directory = (root / "checkpoints" / experiment_id).resolve()
    if path.resolve().parent != expected_directory:
        raise RuntimeError("Latest pointer path outside experiment directory")
    if not path.is_file() or hash_file(path) != pointer["sha256"]:
        raise RuntimeError("Latest checkpoint missing or hash mismatch")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("experiment_id") != experiment_id or int(payload["step"]) != pointer["step"]:
        raise RuntimeError("Checkpoint content differs from latest pointer")
    return payload, pointer
