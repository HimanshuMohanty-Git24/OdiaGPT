"""Train OdiaGPT nano with mixed precision, evaluation, and resume."""

from datetime import datetime
from pathlib import Path
from time import monotonic
from zoneinfo import ZoneInfo
import json
import math
import os
import random
import tempfile

import numpy as np
import torch

from .checkpoint import load_latest_checkpoint, save_training_checkpoint
from .config import ModelConfig
from .dataset import PackedTokenData
from .model import OdiaGPT
from .prepare_tokens import hash_file
from .train_config import TrainConfig


EXPERIMENT_ID = "nano-8k-v1"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=".training_", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _status(
    root: Path, step: int, train_loss: float | None,
    val_loss: float | None, checkpoint: str | None, note: str,
) -> None:
    stamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S %Z")
    text = (
        "# Current OdiaGPT status\n\n"
        f"- Timestamp: {stamp}\n"
        "- Current phase: Nano experiment training\n"
        "- Last successful cell: CELL 46 (T4 FP16 and checkpoint preflight)\n"
        f"- Latest runtime output: {note}\n"
        "- Current GPU: Tesla T4 in MCP runtime (recheck after reconnect)\n"
        "- Dataset status: Nano 5,152,387 train / 56,727 validation tokens verified\n"
        "- Tokenizer status: Nano 8K trained from scratch; tests passed\n"
        "- Model status: Nano 8K model, 5,507,328 parameters\n"
        f"- Active experiment: {EXPERIMENT_ID}\n"
        f"- Current training step: {step}\n"
        f"- Latest train loss: {train_loss if train_loss is not None else 'None'}\n"
        f"- Latest validation loss: {val_loss if val_loss is not None else 'None'}\n"
        f"- Latest checkpoint: {checkpoint if checkpoint else 'None'}\n"
        "- Blockers: None known\n"
        "- Exact next action: Resume or continue nano training to the configured stopping point.\n"
    )
    _atomic_text(root / "status.md", text)


def _make_optimizer(model: OdiaGPT, config: TrainConfig) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    for _, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )


def _schedule_factor(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    span = max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / span)
    floor = config.min_learning_rate / config.learning_rate
    return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * progress))


@torch.inference_mode()
def _evaluate(
    model: OdiaGPT, data: PackedTokenData, config: TrainConfig,
    autocast_dtype: torch.dtype | None,
) -> float:
    model.eval()
    losses = []
    for index in range(config.eval_batches):
        x, y = data.validation_batch(index)
        with torch.amp.autocast(
            "cuda", dtype=autocast_dtype or torch.float16,
            enabled=autocast_dtype is not None,
        ):
            _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def train_nano(root: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Nano experiment requires a CUDA GPU")
    if not os.path.ismount("/content/drive"):
        raise RuntimeError("Google Drive must be mounted before training")
    device = torch.device("cuda")
    model_config = ModelConfig.from_yaml(root / "configs/nano.yaml")
    train_config = TrainConfig.from_yaml(root / "configs/nano.yaml")
    if train_config.precision == "fp16":
        autocast_dtype = torch.float16
    elif train_config.precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 not supported by runtime")
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None
    token_manifest_path = root / "data/tokenized/nano_8k.json"
    token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    token_manifest_hash = hash_file(token_manifest_path)
    if model_config.vocab_size != token_manifest["vocab_size"]:
        raise RuntimeError("Model and tokenizer vocabulary mismatch")
    if token_manifest["splits"]["train"]["tokens"] < 5_000_000:
        raise RuntimeError("Nano corpus below the 5M-token target")
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed_all(train_config.seed)
    np.random.seed(train_config.seed)
    random.seed(train_config.seed)
    data = PackedTokenData(
        Path(token_manifest["splits"]["train"]["path"]),
        token_manifest["splits"]["train"]["sha256"],
        Path(token_manifest["splits"]["validation"]["path"]),
        token_manifest["splits"]["validation"]["sha256"],
        device, model_config.context_length, train_config.seed,
        train_config.eval_batches, train_config.microbatch,
    )
    model = OdiaGPT(model_config).to(device)
    optimizer = _make_optimizer(model, train_config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: _schedule_factor(step, train_config)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=train_config.precision == "fp16")
    tokens_per_update = (
        train_config.microbatch * train_config.gradient_accumulation
        * model_config.context_length
    )
    run_dir = root / "runs" / EXPERIMENT_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    run_meta_path = run_dir / "config.json"
    run_meta = {
        "experiment_id": EXPERIMENT_ID,
        "model_config": model_config.to_dict(),
        "training_config": train_config.to_dict(),
        "token_manifest_sha256": token_manifest_hash,
        "tokenizer_sha256": token_manifest["tokenizer_sha256"],
        "train_binary_sha256": token_manifest["splits"]["train"]["sha256"],
        "validation_binary_sha256": token_manifest["splits"]["validation"]["sha256"],
        "parameter_count": model.parameter_count(),
        "tokens_per_update": tokens_per_update,
        "gpu": torch.cuda.get_device_name(0),
    }
    if run_meta_path.exists():
        existing = json.loads(run_meta_path.read_text(encoding="utf-8"))
        if existing != run_meta:
            raise RuntimeError("Saved run configuration differs; use another experiment ID")
    else:
        _atomic_text(run_meta_path, json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n")
    checkpoint = load_latest_checkpoint(root, EXPERIMENT_ID)
    step = 0
    tokens_seen = 0
    best_val = float("inf")
    latest_val = None
    latest_train = None
    elapsed_before = 0.0
    start_checkpoint = None
    checkpoint_path = None
    if checkpoint is not None:
        saved, pointer = checkpoint
        if saved["model_config"] != model_config.to_dict():
            raise RuntimeError("Checkpoint model config mismatch")
        if saved["training_config"] != train_config.to_dict():
            raise RuntimeError("Checkpoint training config mismatch")
        if saved["token_manifest_sha256"] != token_manifest_hash:
            raise RuntimeError("Checkpoint token data mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        np.random.set_state(saved["numpy_rng_state"])
        random.setstate(saved["python_rng_state"])
        data.train_generator.set_state(saved["sampler_rng_state"])
        step = int(saved["step"])
        tokens_seen = int(saved["tokens_seen"])
        best_val = float(saved["best_validation_loss"])
        latest_val = saved["latest_validation_loss"]
        latest_train = saved["latest_train_loss"]
        elapsed_before = float(saved["elapsed_seconds"])
        start_checkpoint = pointer["path"]
        checkpoint_path = pointer["path"]
        print(f"Resumed step {step} from {checkpoint_path}", flush=True)
    log_path = root / "logs" / f"{EXPERIMENT_ID}.jsonl"
    if step == 0:
        if log_path.exists() and log_path.stat().st_size:
            raise RuntimeError(
                "Log exists without a checkpoint; inspect before restarting experiment"
            )
        latest_val = _evaluate(model, data, train_config, autocast_dtype)
        best_val = latest_val
        _append_jsonl(log_path, {
            "experiment_id": EXPERIMENT_ID, "step": 0,
            "validation_loss": latest_val,
            "validation_perplexity": math.exp(min(latest_val, 20)),
            "tokens_seen": 0, "event": "baseline",
        })
        print(f"step=0 validation_loss={latest_val:.4f}", flush=True)
    print(
        f"Training {EXPERIMENT_ID}: {model.parameter_count():,} parameters, "
        f"{tokens_per_update:,} tokens/update, max_steps={train_config.max_steps}, "
        f"precision={train_config.precision}, gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    _status(root, step, latest_train, latest_val, checkpoint_path, "Nano training active")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    session_start = monotonic()
    last_log_time = session_start
    last_log_step = step
    smoothed = latest_train
    skipped_attempts = 0
    while step < train_config.max_steps:
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(train_config.gradient_accumulation):
            x, y = data.train_batch(train_config.microbatch)
            with torch.amp.autocast(
                "cuda", dtype=autocast_dtype or torch.float16,
                enabled=autocast_dtype is not None,
            ):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite train loss before step {step+1}")
            micro_losses.append(float(loss.detach().item()))
            scaler.scale(loss / train_config.gradient_accumulation).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.gradient_clip
        )
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if not torch.isfinite(norm) or scaler.get_scale() < old_scale:
            skipped_attempts += 1
            print(f"FP16 overflow; update skipped ({skipped_attempts})", flush=True)
            if skipped_attempts > 20:
                raise RuntimeError("Repeated FP16 overflow; stop for diagnosis")
            continue
        skipped_attempts = 0
        used_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        step += 1
        tokens_seen += tokens_per_update
        latest_train = sum(micro_losses) / len(micro_losses)
        smoothed = (
            latest_train if smoothed is None
            else 0.95 * smoothed + 0.05 * latest_train
        )
        if step % train_config.eval_interval == 0 or step == train_config.max_steps:
            latest_val = _evaluate(model, data, train_config, autocast_dtype)
            best_val = min(best_val, latest_val)
        if step % train_config.log_interval == 0 or step == 1:
            torch.cuda.synchronize()
            now = monotonic()
            interval_tokens = (step - last_log_step) * tokens_per_update
            speed = interval_tokens / max(now - last_log_time, 1e-9)
            event = {
                "experiment_id": EXPERIMENT_ID,
                "step": step,
                "train_loss": latest_train,
                "smoothed_loss": smoothed,
                "validation_loss": latest_val,
                "validation_perplexity": (
                    math.exp(min(latest_val, 20)) if latest_val is not None else None
                ),
                "learning_rate": used_lr,
                "gradient_norm": float(norm),
                "tokens_per_second": speed,
                "tokens_seen": tokens_seen,
                "elapsed_seconds": elapsed_before + now - session_start,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            _append_jsonl(log_path, event)
            print(
                f"step={step:4d} train={latest_train:.4f} "
                f"val={latest_val:.4f} lr={used_lr:.2e} "
                f"tokens/s={speed:.0f} seen={tokens_seen:,}",
                flush=True,
            )
            last_log_time, last_log_step = now, step
            _status(
                root, step, latest_train, latest_val, checkpoint_path,
                f"step {step}; train {latest_train:.4f}; validation {latest_val:.4f}",
            )
        stop_for_tokens = (
            train_config.max_tokens is not None
            and tokens_seen >= train_config.max_tokens
        )
        checkpoint_due = (
            step % train_config.checkpoint_interval == 0
            or step == train_config.max_steps or stop_for_tokens
        )
        if checkpoint_due:
            payload = {
                "experiment_id": EXPERIMENT_ID,
                "step": step,
                "tokens_seen": tokens_seen,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "model_config": model_config.to_dict(),
                "training_config": train_config.to_dict(),
                "token_manifest_sha256": token_manifest_hash,
                "best_validation_loss": best_val,
                "latest_validation_loss": latest_val,
                "latest_train_loss": latest_train,
                "elapsed_seconds": elapsed_before + monotonic() - session_start,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "sampler_rng_state": data.train_generator.get_state(),
            }
            pointer = save_training_checkpoint(root, EXPERIMENT_ID, payload)
            checkpoint_path = pointer["path"]
            print(f"checkpoint step={step}: {checkpoint_path}", flush=True)
            _status(
                root, step, latest_train, latest_val, checkpoint_path,
                f"checkpoint saved at step {step}",
            )
        if stop_for_tokens:
            break
    elapsed_total = elapsed_before + monotonic() - session_start
    result = {
        "experiment_id": EXPERIMENT_ID,
        "date": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "model_config": model_config.to_dict(),
        "parameter_count": model.parameter_count(),
        "dataset": token_manifest["dataset_source"] + " verified/ori nano",
        "token_count": token_manifest["splits"]["train"]["tokens"],
        "tokenizer": "nano_8k",
        "seed": train_config.seed,
        "GPU": torch.cuda.get_device_name(0),
        "precision": train_config.precision,
        "context_length": model_config.context_length,
        "microbatch": train_config.microbatch,
        "gradient_accumulation": train_config.gradient_accumulation,
        "effective_batch_tokens": tokens_per_update,
        "optimizer": "AdamW",
        "learning_rate": train_config.learning_rate,
        "training_steps": step,
        "total_tokens_seen": tokens_seen,
        "training_loss": latest_train,
        "validation_loss": latest_val,
        "perplexity": math.exp(min(latest_val, 20)),
        "tokens_per_second": tokens_seen / max(elapsed_total, 1e-9),
        "peak_VRAM": torch.cuda.max_memory_allocated(),
        "start_checkpoint": start_checkpoint,
        "final_checkpoint": checkpoint_path,
        "result": "completed",
        "notes": "Initial nano experiment on 8K tokenizer and 5.15M-token corpus.",
    }
    history = root / "experiments.jsonl"
    existing_ids = {
        json.loads(line)["experiment_id"]
        for line in history.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if EXPERIMENT_ID not in existing_ids:
        _append_jsonl(history, result)
    report = (
        "# Nano experiment\n\n"
        f"- GPU: {result['GPU']}\n"
        f"- Parameters: {result['parameter_count']:,}\n"
        f"- Precision: {result['precision']}\n"
        f"- Steps: {step:,}\n"
        f"- Tokens seen: {tokens_seen:,}\n"
        f"- Final train loss: {latest_train:.4f}\n"
        f"- Final validation loss: {latest_val:.4f}\n"
        f"- Validation perplexity: {result['perplexity']:.2f}\n"
        f"- Peak allocated VRAM: {result['peak_VRAM']/2**30:.3f} GiB\n"
        f"- Final checkpoint: {checkpoint_path}\n"
    )
    _atomic_text(root / "reports/nano_experiment.md", report)
    _status(
        root, step, latest_train, latest_val, checkpoint_path,
        f"Nano training completed at step {step}",
    )
    print(
        f"COMPLETED {EXPERIMENT_ID}: train={latest_train:.4f}, "
        f"val={latest_val:.4f}, tokens={tokens_seen:,}",
        flush=True,
    )
    return result
