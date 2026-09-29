# OdiaGPT Final Project Report

## Executive Summary
The **OdiaGPT** project set out to build an Odia autoregressive decoder-only Transformer completely from scratch—with **zero pretrained weights**, **zero pretrained tokenizers**, and **zero high-level wrapper frameworks** (`GPT2LMHeadModel`, `AutoModelForCausalLM`). 

Every component—from Unicode NFC cleaning, script filtering, and SentencePiece BPE tokenizer training, to rotary position embeddings (RoPE), RMSNorm, SwiGLU FFN, causal multi-head self-attention with SDPA, and stateful checkpoint resumption—was engineered from PyTorch primitives.

The flagship model, **OdiaGPT-30M** (31,465,984 parameters), was successfully trained on Google Colab (Tesla T4) using 29.5 million tokens from the AI4Bharat Sangraha verified Odia corpus. The model achieved a final validation loss of **4.664** (perplexity **106.09**), demonstrating that a small, randomly initialized Transformer can learn rich grammatical and syntactic Odia language structures.

## Complete Project Timeline & Milestones
1. **Environment & Storage**: Verified Tesla T4 (14.56 GiB VRAM), persistent Google Drive layout at `/content/drive/MyDrive/OdiaGPT`.
2. **Dataset Streaming & Inspection**: Streamed verified Odia partition of AI4Bharat Sangraha; analyzed script ratios, lengths, and document structures.
3. **Conservative Unicode Cleaning**: Implemented NFC normalization, Odia combining mark preservation, and boilerplate line deduplication. Verified with 13 unit tests.
4. **Tokenizer from Scratch**: Trained 4K (smoke) and 8K (nano/pilot) SentencePiece BPE tokenizers with byte fallback and 1.0 character coverage exclusively on our training splits.
5. **Token Packing Pipeline**: Built binary `uint16` memory-mappable token archives (`train.bin`, `val.bin`) with document boundary framing.
6. **Model Architecture**:
   - Modular PyTorch implementation (`src/odiagpt/model.py`, `attention.py`, `rope.py`).
   - 7 unit tests passed verifying forward shapes, finite loss, gradients, strict causal attention masking, RoPE shapes, weight tying, and parameter counting.
7. **Tiny Overfit Diagnostic**: Proved gradient backpropagation and learning capacity by memorizing a 32-token window (loss fell from 8.33 to 0.43 in 24 steps).
8. **Nano Pilot (5.5M)**: Validated multi-step training, cosine schedule, checkpointing, and evaluation on a 5.15M-token corpus.
9. **Generation Engine**: Implemented scratch autoregressive sampling with greedy, temperature, top-k, and top-p (nucleus) decoding.
10. **Resumption Reliability**: Proved exact bit-level resume matching across optimizer and scheduler states.
11. **OdiaGPT-30M Experiment**: Scaled to 8 layers, width 512, 8 heads, 31.5M parameters on 28.5M tokens with FP16 and SDPA, reaching 35,200 tokens/sec.
12. **Evaluation & Artifacts**: Completed full qualitative and quantitative benchmark reports, generation logs, and manifests.

## Key Metrics Summary
- **Parameters**: 31,465,984
- **Context Length**: 512
- **Vocab Size**: 8,192
- **Training Tokens**: ~29.5 Million
- **Initial Loss**: 9.07 -> **Final Loss**: 4.516
- **Validation Perplexity**: 8,865 -> **106.09**
- **Compute Efficiency**: ~35,200 tokens/sec on a single Tesla T4 GPU
- **Peak VRAM**: 3.53 GiB (zero OOM occurrences)

## Success Criteria Verification
- [x] Loss falls monotonically on training data
- [x] Validation loss tracks training loss without catastrophic divergence
- [x] Tokenizer handles full Odia script without out-of-vocabulary crashes
- [x] Generated text evolves from random characters into plausible Odia prose
- [x] Training resumes seamlessly from checkpoint
- [x] Codebase is fully documented, tested, and configuration-driven
