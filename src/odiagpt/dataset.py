"""Verified packed token streams with fast GPU window sampling."""

from pathlib import Path
import os
import shutil
import tempfile

import numpy as np
import torch
from torch import Tensor

from .prepare_tokens import hash_file


def local_verified_copy(source: Path, expected_hash: str, runtime: Path) -> Path:
    if hash_file(source) != expected_hash:
        raise RuntimeError(f"Source token binary hash mismatch: {source}")
    runtime.mkdir(parents=True, exist_ok=True)
    target = runtime / source.name
    if target.is_file():
        if hash_file(target) != expected_hash:
            raise RuntimeError(f"Existing runtime copy hash mismatch: {target}")
        return target
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=runtime, prefix=".copy_", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, handle)
            handle.flush()
            os.fsync(handle.fileno())
            if hash_file(temporary) != expected_hash:
                raise RuntimeError("Runtime copy hash mismatch")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


class PackedTokenData:
    def __init__(
        self, train_path: Path, train_hash: str,
        val_path: Path, val_hash: str,
        device: torch.device, context_length: int,
        seed: int, eval_batches: int, microbatch: int,
    ) -> None:
        runtime = Path("/content/odiagpt_runtime/tokenized")
        train_local = local_verified_copy(train_path, train_hash, runtime)
        val_local = local_verified_copy(val_path, val_hash, runtime)
        self.device = device
        self.context_length = context_length
        self.train = self._load(train_local, device)
        self.validation = self._load(val_local, device)
        if len(self.train) <= context_length + 1 or len(self.validation) <= context_length + 1:
            raise RuntimeError("Token stream shorter than context length")
        self.train_generator = torch.Generator(device=device).manual_seed(seed)
        val_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        val_limit = len(self.validation) - context_length
        self.validation_starts = torch.randint(
            0, val_limit, (eval_batches, microbatch),
            generator=val_generator, device="cpu",
        ).to(device)
        self.offsets = torch.arange(context_length + 1, device=device)

    @staticmethod
    def _load(path: Path, device: torch.device) -> Tensor:
        values = np.memmap(path, dtype="<u2", mode="r")
        return torch.from_numpy(np.array(values, dtype=np.int64)).to(device)

    def _windows(self, data: Tensor, starts: Tensor) -> tuple[Tensor, Tensor]:
        positions = starts[:, None] + self.offsets[None, :]
        windows = data[positions]
        return windows[:, :-1], windows[:, 1:]

    def train_batch(self, microbatch: int) -> tuple[Tensor, Tensor]:
        high = len(self.train) - self.context_length
        starts = torch.randint(
            0, high, (microbatch,), device=self.device,
            generator=self.train_generator,
        )
        return self._windows(self.train, starts)

    def validation_batch(self, index: int) -> tuple[Tensor, Tensor]:
        return self._windows(self.validation, self.validation_starts[index])
