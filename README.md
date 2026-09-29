# OdiaGPT

An educational Odia autoregressive decoder-only Transformer built and trained **completely from scratch**.

## Highlights
- **100% From Scratch**: No pretrained weights (no GPT-2, no Llama), no pretrained tokenizers, and no ready-made model classes. Every layer is built directly with PyTorch primitives.
- **Modern Transformer Architecture**:
  - RMSNorm (pre-normalization)
  - Rotary Position Embeddings (RoPE)
  - SwiGLU feed-forward networks
  - Causal Multi-Head Self-Attention with PyTorch Scaled Dot-Product Attention (SDPA)
  - Tied input embeddings and output LM head projection
- **Scratch Odia Tokenizer**: 8,192-vocabulary SentencePiece BPE tokenizer with byte fallback, trained strictly on our cleaned Odia corpus.
- **Cleaned Indic Corpus**: Filtered and NFC-normalized subset of AI4Bharat Sangraha (`verified/ori`) with boilerplate line scrubbing.
- **Hardware-Efficient**: Trained on Google Colab (Tesla T4) utilizing PyTorch AMP `fp16` at over 35,000 tokens/sec.

---

## Model Configurations

| Model | Layers | Hidden ($d_{model}$) | Heads | Context | Vocab | Parameters | Final Val Loss | Val PPL |
|---|---|---|---|---|---|---|---|---|
| **Smoke** | 2 | 128 | 4 | 128 | 4,096 | 950,912 | - | - |
| **Nano** | 4 | 256 | 4 | 256 | 8,192 | 5,507,328 | 5.922 | 373.30 |
| **OdiaGPT-30M** | 8 | 512 | 8 | 512 | 8,192 | 31,465,984 | 4.664 | 106.09 |

---

## Project Structure
```text
OdiaGPT/
├── configs/               # smoke.yaml, nano.yaml, odia30m.yaml
├── src/odiagpt/           # Core library
│   ├── model.py           # Transformer architecture (RMSNorm, SwiGLU, OdiaGPT)
│   ├── attention.py       # Causal self-attention with SDPA
│   ├── rope.py            # Rotary Position Embeddings
│   ├── tokenizer.py       # SentencePiece training and wrapper
│   ├── dataset.py         # Memory-mapped packed uint16 token dataset
│   ├── generation.py      # Autoregressive decoding (greedy, temp, top-k, top-p)
│   ├── checkpoint.py      # Resumption and state persistence
│   ├── cleaning.py        # Odia script ratio and normalization filters
│   └── unicode_utils.py   # NFC normalization and diacritic preservation
├── scripts/               # Training, generation, and data prep runners
├── tests/                 # Unit tests (cleaning, model causality, tokenizer, SDPA)
├── reports/               # Quality, benchmark, and evaluation reports
├── tokenizer/             # Trained .model and .vocab files
├── checkpoints/           # Atomic checkpoints with latest.json pointer
├── logs/                  # Machine-readable JSONL training logs
└── generations/           # Output sample logs from benchmark prompts
```

---

## Running Generation
To sample completions using the trained 30M checkpoint:
```python
from odiagpt.checkpoint import load_latest_checkpoint
from odiagpt.config import ModelConfig
from odiagpt.model import OdiaGPT
from odiagpt.generation import generate_tokens
import sentencepiece as spm
import torch

# Load model and tokenizer
checkpoint, _ = load_latest_checkpoint("checkpoints", "pilot-30m-v1")
config = ModelConfig.from_dict(checkpoint["model_config"])
model = OdiaGPT(config).eval().cuda()
model.load_state_dict(checkpoint["model"])
sp = spm.SentencePieceProcessor(model_file="tokenizer/nano_8k.model")

# Generate
prompt = "ଓଡ଼ିଶାର ଲୋକମାନେ"
prompt_ids = [sp.bos_id(), *sp.encode(prompt, out_type=int)]
out_ids = generate_tokens(model, prompt_ids, max_new_tokens=64, temperature=0.8, top_k=50, top_p=0.9)
print(sp.decode(out_ids))
```

---

## Sample Outputs (OdiaGPT-30M)
> **Prompt**: `ଆଜି ସକାଳେ`  
> **Completion**: *ଆଜି ସକାଳେ ସକାଳେ ଦୁଇ ଝିଅ ସହ ଆସି ଶୋଇପଡ଼ିଲା । ସେହି ସମୟରେ ବାପା ଘର ଲୋକ କୁ ଖବର ଦେଇ କହିଥିଲେ ଯେ, "ତେବେ ମାଆ, ତମେ ଆଉ କିଛି କହିଦେବୁ । ମୁଁ ଏ କଥା କିଛି ଶୁଣିନାହିଁ । କିନ୍ତୁ ବାପା, ମା'ଙ୍କୁ ନେଇ ମା'ର ମା'ର ପୁଅ ଝିଅ । ଏ କଥା କହି ସେ ଆଉ କାହା କଥା ଜାଣିନଥିଲା ।*

---

## License & Attribution
- Corpus: [AI4Bharat Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)
- Developed under the OdiaGPT scratch training curriculum.
