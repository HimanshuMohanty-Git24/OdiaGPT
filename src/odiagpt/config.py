"""Validated model configuration for OdiaGPT."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.0
    rope_base: float = 10000.0
    attention_backend: str = "manual"

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context_length", "n_layers", "d_model", "n_heads", "d_ff"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.vocab_size < 4:
            raise ValueError("vocab_size must include the four special tokens")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_base <= 1:
            raise ValueError("rope_base must be greater than 1")
        if self.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("attention_backend must be manual or sdpa")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        values = asdict(self)
        if self.attention_backend == "manual":
            values.pop("attention_backend")  # Preserve existing checkpoint config format.
        return values

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        import yaml
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict) or not isinstance(values.get("model"), dict):
            raise ValueError("Config YAML must contain a model mapping")
        return cls.from_dict(values["model"])
