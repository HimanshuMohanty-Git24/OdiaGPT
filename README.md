# OdiaGPT (ଓଡ଼ିଆଜିପିଟି)

<div align="center">

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-CodeHima%2FOdiaGPT--30M-FFD21E?style=for-the-badge)](https://huggingface.co/CodeHima/OdiaGPT-30M)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Demo-OdiaGPT--Web-blue?style=for-the-badge)](https://huggingface.co/spaces/CodeHima/OdiaGPT-Web)
[![Gradio Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Gradio-orange?style=for-the-badge)](https://7165f80b8a0ec54d28.gradio.live)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)

**A Small, Modern Odia Decoder-Only Language Model Built & Trained 100% From Scratch**

[Explore on Hugging Face](https://huggingface.co/CodeHima/OdiaGPT-30M) • [Interactive Web Demo](https://huggingface.co/spaces/CodeHima/OdiaGPT-Web) • [Live Gradio Session](https://7165f80b8a0ec54d28.gradio.live) • [Architecture](#architecture) • [Getting Started](#getting-started) • [Evaluation](#evaluation--results)

</div>

---

## 🌟 Overview

**OdiaGPT** is an educational, production-clean research project that implements and trains an autoregressive decoder-only Transformer specifically for the **Odia (ଓଡ଼ିଆ)** language completely from scratch.

Unlike common transfer-learning workflows, OdiaGPT adheres strictly to a zero-pretrained-dependency policy:
- ❌ **No pretrained weights**: Zero GPT-2, Llama, Mistral, or Indic foundation model checkpoints.
- ❌ **No off-the-shelf tokenizers**: No Llama, IndicBERT, or multilingual tokenizers.
- ❌ **No ready-made model wrappers**: No `GPT2LMHeadModel` or `AutoModelForCausalLM` during model construction.
- ✅ **100% PyTorch primitives**: Every tensor operation, rotary embedding, normalization layer, and attention mechanism is written from the ground up.
- ✅ **From-scratch Indic Tokenizer**: An 8,192-vocabulary SentencePiece BPE tokenizer trained exclusively on cleaned Odia text with byte-fallback and 100% character coverage.

The flagship model, **OdiaGPT-30M** (31,465,984 parameters), was trained on Google Colab (NVIDIA Tesla T4) over 29.5 million tokens from a filtered, NFC-normalized subset of the [AI4Bharat Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) verified Odia partition.

---

## 🏗️ Architecture

OdiaGPT utilizes state-of-the-art decoder-only Transformer design principles:

| Component | Specification | Description |
|---|---|---|
| **Parameters** | **31,465,984** (~31.5M) | Compact, fast, and highly sample-efficient |
| **Layers ($N$)** | **8** Transformer Blocks | Deep residual representation |
| **Hidden Dimension ($d_{model}$)** | **512** | Embedding width |
| **Attention Heads ($H$)** | **8** Heads (Head Dim: 64) | Multi-head self-attention |
| **Feed-Forward ($d_{ff}$)** | **1,536** | SwiGLU activation ($\text{Swish}(x W_{gate}) \otimes x W_{up}$) |
| **Position Embeddings** | **RoPE** | Rotary Position Embeddings (base $\theta = 10,000.0$) |
| **Normalization** | **RMSNorm** | Root Mean Square Pre-Normalization ($\epsilon = 10^{-6}$) |
| **Attention Backend** | **SDPA** | PyTorch Scaled Dot-Product Attention (FlashAttention) |
| **Weight Tying** | **Yes** | Input token embeddings are tied to output LM projection head |
| **Context Length** | **512 Tokens** | Max attention context window |
| **Vocabulary** | **8,192 Tokens** | SentencePiece BPE with byte fallback & diacritic preservation |

---

## 🚀 Quickstart: Inference with Hugging Face Transformers

The model is natively exported to Hugging Face format and published to the Hugging Face Hub under [`CodeHima/OdiaGPT-30M`](https://huggingface.co/CodeHima/OdiaGPT-30M).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "CodeHima/OdiaGPT-30M"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Prompt in Odia
prompt = "ଓଡ଼ିଶାର ଲୋକମାନେ"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate with nucleus sampling and repetition penalty
outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.15,
    do_sample=True,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🎨 Interactive Gradio Web UI

An interactive web interface with dual-model selection (30M vs Nano), generation parameters, and one-click Odia presets is included:

- **Live Hugging Face Space**: [huggingface.co/spaces/CodeHima/OdiaGPT-Web](https://huggingface.co/spaces/CodeHima/OdiaGPT-Web)
- **Live Colab Session**: [7165f80b8a0ec54d28.gradio.live](https://7165f80b8a0ec54d28.gradio.live)

To run the Gradio interface locally:
```bash
git clone https://github.com/HimanshuMohanty-Git24/OdiaGPT.git
cd OdiaGPT
pip install -r requirements.txt
python app.py
```

---

## 📊 Training Progression & Results

The project followed a rigorous 22-stage curriculum:

```text
Smoke Memorization (33 tokens) ──> Nano Experiment (5.5M params) ──> OdiaGPT-30M (31.5M params)
     Loss: 8.33 -> 0.43                   Val PPL: 373.30                      Val PPL: 106.09
```

### Quantitative Metrics Across Tiers

| Tier | Parameters | Training Tokens Seen | Final Train Loss | Final Val Loss | Val Perplexity | Status |
|---|---|---|---|---|---|---|
| **Random Init** | 31.5M | 0 | ~9.07 | ~9.09 | 8,865.0 | Random Gaussian Baseline |
| **Tiny Overfit** | 0.95M | 768 (32 tok window) | 0.4317 | N/A | N/A | Diagnostic Memorization Passed |
| **Nano Model** | 5.51M | 8,192,000 | 5.7545 | 5.9224 | 373.30 | Architecture Validated |
| **OdiaGPT-30M** | **31.47M** | **29,491,200** | **4.5156** | **4.6643** | **106.09** | **Fully Trained Base Model** |

### Training Dynamics (OdiaGPT-30M)
- **Throughput**: ~35,202 tokens/sec on a single NVIDIA Tesla T4 GPU.
- **VRAM Utilization**: Peak allocated 3.53 GiB (only 24.2% of 14.56 GiB T4 memory; zero OOMs).
- **Optimization**: AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay 0.1, grad clip 1.0) with linear warmup (100 steps) and cosine learning rate decay ($3 \times 10^{-4} \to 3 \times 10^{-5}$).

---

## 📝 Qualitative Generation Samples

Samples generated from the final OdiaGPT-30M checkpoint (`step_0001800.pt`) under Nucleus Sampling ($T=0.8, \text{top-k}=50, \text{top-p}=0.9, \text{rep-penalty}=1.15$):

### 1. Prompt: `ଓଡ଼ିଶାର ଲୋକମାନେ` (People of Odisha)
> *ଓଡ଼ିଶାର ଲୋକମାନେ ବିରିୟାରେ ସାନ୍ଦ୍ରାର ଅଭାବକୁ ନେଇ ନିଜକୁ ନିଜେ ଅନେକ କଥା କହୁଥିବା କଥା । ଆମେ ସମସ୍ତେ ଜାଣନ୍ତି ଯେ ଏହି ମହାଭାରତ ତଥା ଦେଶର ବିଭିନ୍ନ ରାଜ୍ୟରୁ ଆସୁଥିବା ଓଡ଼ିଆଙ୍କ ପାଇଁ ଏକ ପ୍ରକାର ପ୍ରୟାସ । ଏହି ସମୟରେ ଆମେ ସମସ୍ତେ ସମସ୍ତେ ଏହି ଅବସରରେ ରହି ଏହିସବୁ କାର୍ଯ୍ୟକ୍ରମ ବିଷୟରେ କହିବାକୁ ଯାଉଛୁ । ଆମେ ମଧ୍ୟ ଏହି ସଭାରେ ଅଂଶଗ୍ରହଣ କରିବା ପାଇଁ ଏକ ନୂଆ ପ୍ରସ୍ତୁତି*  
> **Linguistic Analysis**: Learned accurate compound clause formation, correct postpositions (`ପାଇଁ`, `ବିଷୟରେ`), plural noun markers (`ଙ୍କ`), and topical Odia vocabulary (`ଓଡ଼ିଆଙ୍କ`, `ରାଜ୍ୟରୁ`, `କାର୍ଯ୍ୟକ୍ରମ`).

### 2. Prompt: `ଆଜି ସକାଳେ` (Today morning)
> *ଆଜି ସକାଳେ ସକାଳେ ଦୁଇ ଝିଅ ସହ ଆସି ଶୋଇପଡ଼ିଲା । ସେହି ସମୟରେ ବାପା ଘର ଲୋକ କୁ ଖବର ଦେଇ କହିଥିଲେ ଯେ, "ତେବେ ମାଆ, ତମେ ଆଉ କିଛି କହିଦେବୁ । ମୁଁ ଏ କଥା କିଛି ଶୁଣିନାହିଁ । କିନ୍ତୁ ବାପା, ମା'ଙ୍କୁ ନେଇ ମା'ର ମା'ର ପୁଅ ଝିଅ । ଏ କଥା କହି ସେ ଆଉ କାହା କଥା ଜାଣିନଥିଲା ।*  
> **Linguistic Analysis**: Demonstrates natural narrative pacing, correct past tense verb agreement (`ଆସି ଶୋଇପଡ଼ିଲା`, `କହିଥିଲେ`, `ଜାଣିନଥିଲା`), dialogue quotation marks, and familial nouns.

### 3. Prompt: `ଓଡ଼ିଆ ଭାଷାର` (Of the Odia language)
> *ଓଡ଼ିଆ ଭାଷାର ଅନେକ ଦୃଶ୍ୟ, ଚିତ୍ର-ଅର୍ଣ୍ଣିମ, ନୂଆବୋହୂକ ଓ ଲୋକନାଥଙ୍କ ଭଳି କିଛି ନୂଆ କାହାଣୀକୁ ନେଇ ଅନେକ ପ୍ରଶ୍ନବାଚୀ ସୃଷ୍ଟି ହୋଇଛି । ତେଣୁ ଏହି ଦୁଇ ଚରିତ୍ରର କଥା ଜାଣିଲେ ଆଶ୍ଚର୍ଯ୍ୟ ହୋଇଯିବେ । ପ୍ରଧାନମନ୍ତ୍ରୀଙ୍କ ଚିଠିରେ ପ୍ରକାଶ ପାଇବା ପରେ ଅନେକ ଘଟଣା ସାମ୍ନାକୁ ଆସିଛି ।*  
> **Linguistic Analysis**: Formal, journalistic Odia tone with complex nominal morphology and cohesive discourse markers (`ତେଣୁ`, `ଯେଉଁଥିରେ`).

---

## 📁 Repository Structure

```text
OdiaGPT/
├── configs/               # Reproducible experiment configurations
│   ├── smoke.yaml         # 2-layer smoke config
│   ├── nano.yaml          # 4-layer 5.5M config
│   └── odia30m.yaml       # 8-layer 31.5M flagship config
├── src/odiagpt/           # Core library
│   ├── model.py           # Transformer architecture (RMSNorm, SwiGLU, OdiaGPT)
│   ├── attention.py       # Causal multi-head self-attention with SDPA
│   ├── rope.py            # Rotary Position Embeddings
│   ├── tokenizer.py       # SentencePiece training and inference wrapper
│   ├── dataset.py         # Memory-mapped packed uint16 binary dataset loader
│   ├── generation.py      # Autoregressive generation (greedy, top-k, top-p)
│   ├── checkpoint.py      # Stateful optimizer, scheduler, and RNG checkpointing
│   ├── cleaning.py        # Conservative Odia script & boilerplate filters
│   └── unicode_utils.py   # NFC normalization and diacritic preservation
├── scripts/               # Training, benchmarking, and export scripts
│   ├── train_pilot.py     # OdiaGPT-30M training execution
│   ├── generate.py        # Benchmark generation runner
│   ├── upload_to_hf.py    # Hugging Face Hub uploader
│   └── tiny_overfit.py    # Memorization verification test
├── tests/                 # Unit test suite
│   ├── test_cleaning.py   # Unicode normalization & filtering tests (13 tests)
│   ├── test_model.py      # Causality, shapes, gradients, and RoPE tests
│   └── test_sdpa.py       # FlashAttention numerical equivalency tests
├── reports/               # Technical documentation
│   ├── main_training_report.md
│   ├── evaluation_report.md
│   ├── final_report.md
│   └── data_quality_report.md
├── tokenizer/             # Trained 8K and 4K SentencePiece .model/.vocab files
├── app.py                 # Interactive Gradio application
├── requirements.txt       # Project dependencies
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

---

## 🛠️ Testing & Verification

Run the test suite to verify model causality, normalization, and attention implementation:

```bash
python -m unittest discover -s tests -v
```

---

## 📜 Limitations & Ethical Considerations

- **Base Foundation Model**: OdiaGPT is a pure autoregressive next-token predictor. It is not instruction-tuned or RLHF aligned, and will not reliably answer questions in a conversational assistant format without fine-tuning.
- **Factual Knowledge**: Generations reflect statistical language patterns from web data; they should not be relied upon for factual or historical queries.
- **Community Use**: Provided freely under the MIT License for educational and linguistic research into low-resource Indic language models.

---

## 📄 License & Attribution

- **Source Code**: [MIT License](LICENSE) © 2026 Himanshu Mohanty
- **Corpus Attribution**: Trained on subsets of [AI4Bharat Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) (`verified/ori`).
- **Creator**: [Himanshu Mohanty (CodeHima)](https://github.com/HimanshuMohanty-Git24)
