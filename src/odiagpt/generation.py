"""Autoregressive OdiaGPT generation without library generate APIs."""

import torch
from torch import Tensor

from .model import OdiaGPT


@torch.inference_mode()
def generate_tokens(
    model: OdiaGPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
    unk_id: int,
    bos_id: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int = 0,
) -> list[int]:
    if not prompt_ids or max_new_tokens <= 0:
        raise ValueError("Prompt and max_new_tokens must be nonempty/positive")
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if any(i < 0 or i >= model.config.vocab_size for i in prompt_ids):
        raise ValueError("Prompt token ID out of vocabulary range")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    generator = torch.Generator(device=device).manual_seed(seed)
    try:
        for _ in range(max_new_tokens):
            context = ids[:, -model.config.context_length:]
            logits, _ = model(context)
            next_logits = logits[:, -1, :].float().clone()
            next_logits[:, [pad_id, unk_id, bos_id]] = -torch.inf
            if temperature == 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits /= temperature
                if top_k is not None:
                    k = min(top_k, next_logits.size(-1))
                    boundary = torch.topk(next_logits, k=k, dim=-1).values[:, -1:]
                    next_logits = next_logits.masked_fill(next_logits < boundary, -torch.inf)
                if top_p is not None and top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True, dim=-1
                    )
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative = sorted_probs.cumsum(dim=-1)
                    remove = cumulative - sorted_probs > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
                    next_logits = torch.full_like(next_logits, -torch.inf).scatter(
                        dim=-1, index=sorted_indices, src=sorted_logits
                    )
                probs = torch.softmax(next_logits, dim=-1)
                if not torch.isfinite(probs).all():
                    raise RuntimeError("Invalid sampling probabilities")
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            ids = torch.cat((ids, next_id), dim=1)
            if int(next_id.item()) == eos_id:
                break
    finally:
        model.train(was_training)
    return ids.squeeze(0).tolist()
