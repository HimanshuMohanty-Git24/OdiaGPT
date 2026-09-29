"""Generate fixed Odia prompts from a verified nano checkpoint."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import sys

import sentencepiece as spm
import torch

ROOT = Path("/content/drive/MyDrive/OdiaGPT")
sys.path.insert(0, str(ROOT / "src"))

from odiagpt.checkpoint import load_latest_checkpoint
from odiagpt.config import ModelConfig
from odiagpt.generation import generate_tokens
from odiagpt.model import OdiaGPT
from odiagpt.prepare_tokens import hash_file


def main() -> None:
    checkpoint = load_latest_checkpoint(ROOT, "nano-8k-v1", map_location="cpu")
    if checkpoint is None:
        raise RuntimeError("Nano checkpoint not found")
    saved, pointer = checkpoint
    config = ModelConfig.from_dict(saved["model_config"])
    manifest = json.loads(
        (ROOT / "data/tokenized/nano_8k.json").read_text(encoding="utf-8")
    )
    tokenizer_info = json.loads(
        (ROOT / "tokenizer/nano_8k.json").read_text(encoding="utf-8")
    )
    if config.vocab_size != manifest["vocab_size"]:
        raise RuntimeError("Checkpoint and tokenizer vocabulary mismatch")
    if tokenizer_info["model_sha256"] != manifest["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer manifest mismatch")
    if hash_file(Path(tokenizer_info["model_path"])) != tokenizer_info["model_sha256"]:
        raise RuntimeError("Tokenizer model hash mismatch")
    sp = spm.SentencePieceProcessor(model_file=tokenizer_info["model_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OdiaGPT(config).to(device).eval()
    model.load_state_dict(saved["model"])
    prompts = [
        "ଓଡ଼ିଶାର ଲୋକମାନେ",
        "ଆଜି ସକାଳେ",
        "ଓଡ଼ିଆ ଭାଷାର",
    ]
    modes = [
        {"name": "greedy", "temperature": 0.0, "top_k": None, "top_p": None},
        {"name": "sample", "temperature": 0.8, "top_k": 50, "top_p": 0.9},
    ]
    output = ROOT / "generations/nano-8k-v1-step1000.jsonl"
    if output.exists():
        raise FileExistsError(f"Generation artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_ids = [sp.bos_id(), *sp.encode(prompt, out_type=int)]
        for mode in modes:
            all_ids = generate_tokens(
                model, prompt_ids, max_new_tokens=64,
                eos_id=sp.eos_id(), pad_id=sp.pad_id(),
                unk_id=sp.unk_id(), bos_id=sp.bos_id(),
                temperature=mode["temperature"],
                top_k=mode["top_k"], top_p=mode["top_p"],
                seed=2026 + prompt_index,
            )
            new_ids = all_ids[len(prompt_ids):]
            decode_ids = [i for i in new_ids if i != sp.eos_id()]
            completion = sp.decode(decode_ids)
            row = {
                "experiment_id": "nano-8k-v1",
                "checkpoint_step": saved["step"],
                "checkpoint_sha256": pointer["sha256"],
                "prompt": prompt,
                "mode": mode["name"],
                "temperature": mode["temperature"],
                "top_k": mode["top_k"],
                "top_p": mode["top_p"],
                "seed": 2026 + prompt_index,
                "prompt_token_count": len(prompt_ids),
                "generated_token_count": len(new_ids),
                "ended_on_eos": bool(new_ids and new_ids[-1] == sp.eos_id()),
                "completion": completion,
            }
            rows.append(row)
            print(
                f"{mode['name']} | {prompt} | tokens={len(new_ids)} | "
                f"completion={completion[:250]!r}",
                flush=True,
            )
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Saved generations:", output)
    print("Generation rows:", len(rows))


if __name__ == "__main__":
    main()
