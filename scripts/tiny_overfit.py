"""Memorize one short Odia token sequence as a model/training sanity check."""

from datetime import datetime
from hashlib import sha256
from pathlib import Path
from time import monotonic
from zoneinfo import ZoneInfo
import json
import os
import random
import sys
import tempfile

import numpy as np
import torch

ROOT = Path("/content/drive/MyDrive/OdiaGPT")
sys.path.insert(0, str(ROOT / "src"))

from odiagpt.config import ModelConfig
from odiagpt.model import OdiaGPT
from odiagpt.prepare_tokens import hash_file


EXPERIMENT_ID = "tiny-overfit-smoke-v1"
MAX_STEPS = 300
LEARNING_RATE = 0.002
SEQUENCE_LENGTH = 32
SEED = 173


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=".write_", delete=False,
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


def update_status(step: int, loss: float | None, checkpoint: Path | None, note: str) -> None:
    stamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S %Z")
    (ROOT / "status.md").write_text(
        f"# Current OdiaGPT status\n\n"
        f"- Timestamp: {stamp}\n"
        "- Current phase: Tiny-overfit test\n"
        "- Last successful cell: CELL 21 (CPU runtime inspected)\n"
        f"- Latest runtime output: {note}\n"
        "- Current GPU: None in MCP scratch runtime; CPU used for tiny overfit\n"
        "- Dataset status: Smoke corpus and token binaries verified\n"
        "- Tokenizer status: Smoke 4K model and tests verified\n"
        "- Model status: Implemented; six CPU unit tests passed; CUDA test pending\n"
        f"- Active experiment: {EXPERIMENT_ID}\n"
        f"- Current training step: {step}\n"
        f"- Latest train loss: {loss if loss is not None else 'None'}\n"
        "- Latest validation loss: None (same-sequence overfit diagnostic)\n"
        f"- Latest checkpoint: {checkpoint if checkpoint else 'None'}\n"
        "- Blockers: GPU runtime control needs OAuth credentials for later experiments\n"
        "- Exact next action: Continue tiny-overfit until strong loss reduction.\n",
        encoding="utf-8",
    )


def save_checkpoint(
    model: OdiaGPT, optimizer: torch.optim.Optimizer, step: int,
    initial_loss: float, current_loss: float, config: ModelConfig, data_sha256: str,
) -> Path:
    path = ROOT / "checkpoints" / f"{EXPERIMENT_ID}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=".checkpoint_",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save({
            "experiment_id": EXPERIMENT_ID, "step": step,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "model_config": config.to_dict(), "data_sha256": data_sha256,
            "initial_loss": initial_loss, "current_loss": current_loss,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hash_file(path)
    atomic_json(ROOT / "checkpoints/latest.json", {
        "experiment_id": EXPERIMENT_ID, "step": step,
        "path": str(path), "sha256": digest,
    })
    return path


def main() -> None:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    config = ModelConfig.from_yaml(ROOT / "configs/smoke.yaml")
    token_manifest = json.loads(
        (ROOT / "data/tokenized/smoke_4k.json").read_text(encoding="utf-8")
    )
    if config.vocab_size != token_manifest["vocab_size"]:
        raise RuntimeError("Model/tokenizer vocabulary mismatch")
    train_entry = token_manifest["splits"]["train"]
    train_path = Path(train_entry["path"])
    data_hash = hash_file(train_path)
    if data_hash != train_entry["sha256"]:
        raise RuntimeError("Training binary hash mismatch")
    data = np.memmap(train_path, dtype="<u2", mode="r")
    if len(data) < SEQUENCE_LENGTH + 1:
        raise RuntimeError("Not enough tokens for tiny overfit")
    window = torch.tensor(np.asarray(data[:SEQUENCE_LENGTH + 1], dtype=np.int64))
    inputs = window[:-1].unsqueeze(0)
    targets = window[1:].unsqueeze(0)
    model = OdiaGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    checkpoint_path = ROOT / "checkpoints" / f"{EXPERIMENT_ID}.pt"
    start_step = 0
    initial_loss = None
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["experiment_id"] != EXPERIMENT_ID:
            raise RuntimeError("Checkpoint experiment mismatch")
        if saved["model_config"] != config.to_dict() or saved["data_sha256"] != data_hash:
            raise RuntimeError("Checkpoint config/data mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
        random.setstate(saved["python_rng_state"])
        start_step = int(saved["step"])
        initial_loss = float(saved["initial_loss"])
        print(f"Resumed saved step {start_step} from {checkpoint_path}", flush=True)
    log_path = ROOT / "logs" / f"{EXPERIMENT_ID}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Device: CPU; parameters: {model.parameter_count():,}; "
        f"sequence: {SEQUENCE_LENGTH}; max steps: {MAX_STEPS}",
        flush=True,
    )
    update_status(start_step, None, checkpoint_path if start_step else None, "Tiny overfit started")
    elapsed_start = monotonic()
    final_step = start_step
    final_loss = None
    passed = False
    for step in range(start_step + 1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        value = float(loss.item())
        if initial_loss is None:
            initial_loss = value
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not np.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}")
        optimizer.step()
        final_step, final_loss = step, value
        elapsed = monotonic() - elapsed_start
        if step == 1 or step % 10 == 0:
            event = {
                "experiment_id": EXPERIMENT_ID, "step": step,
                "train_loss": value, "initial_loss": initial_loss,
                "gradient_norm": gradient_norm, "learning_rate": LEARNING_RATE,
                "tokens_seen": step * SEQUENCE_LENGTH,
                "tokens_per_second": (step - start_step) * SEQUENCE_LENGTH / max(elapsed, 1e-9),
                "elapsed_seconds_this_session": elapsed,
                "device": "cpu",
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(
                f"step={step:3d} loss={value:.4f} "
                f"ratio={value/initial_loss:.3f} grad_norm={gradient_norm:.3f} "
                f"tokens/s={event['tokens_per_second']:.1f}",
                flush=True,
            )
        passed = value <= 0.5 and value <= initial_loss * 0.2
        if step % 25 == 0 or passed or step == MAX_STEPS:
            saved_path = save_checkpoint(
                model, optimizer, step, initial_loss, value, config, data_hash
            )
            update_status(step, value, saved_path, "Tiny overfit progress saved")
        if passed:
            break
    if final_loss is None:
        raise RuntimeError("No training steps executed")
    result = "passed" if passed else "failed_to_overfit"
    record = {
        "experiment_id": EXPERIMENT_ID,
        "date": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "model_config": config.to_dict(),
        "parameter_count": model.parameter_count(),
        "dataset": "smoke_4k_train.bin first 33 tokens",
        "token_count": SEQUENCE_LENGTH + 1,
        "tokenizer": "smoke_4k",
        "seed": SEED,
        "GPU": None,
        "precision": "float32",
        "context_length": SEQUENCE_LENGTH,
        "microbatch": 1,
        "gradient_accumulation": 1,
        "effective_batch_tokens": SEQUENCE_LENGTH,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "training_steps": final_step,
        "total_tokens_seen": final_step * SEQUENCE_LENGTH,
        "training_loss": final_loss,
        "initial_loss": initial_loss,
        "validation_loss": None,
        "perplexity": None,
        "tokens_per_second": final_step * SEQUENCE_LENGTH / max(monotonic()-elapsed_start,1e-9),
        "peak_VRAM": None,
        "start_checkpoint": None if start_step == 0 else str(checkpoint_path),
        "final_checkpoint": str(checkpoint_path),
        "result": result,
        "notes": "Single fixed Odia token window; diagnostic memorization only.",
    }
    history = ROOT / "experiments.jsonl"
    existing_ids = {
        json.loads(line)["experiment_id"]
        for line in history.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    if EXPERIMENT_ID not in existing_ids:
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = (
        "# Tiny overfit report\n\n"
        f"- Result: {result}\n"
        f"- Initial loss: {initial_loss:.6f}\n"
        f"- Final loss: {final_loss:.6f}\n"
        f"- Final / initial: {final_loss/initial_loss:.4f}\n"
        f"- Steps: {final_step}\n"
        f"- Parameters: {model.parameter_count():,}\n"
        f"- Device: CPU\n"
        f"- Fixed input length: {SEQUENCE_LENGTH} tokens\n"
        "- This verifies memorization of one sequence; it is not language-quality evidence.\n"
    )
    (ROOT / "reports/tiny_overfit_report.md").write_text(report, encoding="utf-8")
    print(f"RESULT: {result}; initial={initial_loss:.4f}; final={final_loss:.4f}", flush=True)
    if not passed:
        raise RuntimeError("Tiny-overfit criterion not met; stop before nano experiment")


if __name__ == "__main__":
    main()
