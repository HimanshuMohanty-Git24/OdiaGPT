"""Validated settings for a pretraining experiment."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    gradient_clip: float
    microbatch: int
    gradient_accumulation: int
    max_steps: int
    warmup_steps: int
    eval_interval: int
    checkpoint_interval: int
    log_interval: int
    eval_batches: int
    precision: str
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not 0 < self.min_learning_rate <= self.learning_rate:
            raise ValueError("Invalid learning rate range")
        if self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid weight decay or gradient clip")
        if not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1:
            raise ValueError("Invalid Adam betas")
        for name in (
            "microbatch", "gradient_accumulation", "max_steps", "warmup_steps",
            "eval_interval", "checkpoint_interval", "log_interval", "eval_batches",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps exceeds max_steps")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        import yaml
        import yaml as _yaml
        values = _yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict) or not isinstance(values.get("training"), dict):
            raise ValueError("Config YAML must contain a training mapping")
        return cls(**values["training"])
