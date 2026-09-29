# OdiaGPT-30M Serious Training Experiment Report

- **Experiment ID**: `pilot-30m-v1` / `odia30m`
- **Date**: 2026-09-29
- **Model Architecture**: OdiaGPT Decoder-Only Transformer
  - Parameters: 31,465,984 (~31.5M)
  - Layers: 8
  - Hidden Dimension ($d_{model}$): 512
  - Attention Heads: 8 (Head dimension: 64)
  - Feed-Forward Dimension ($d_{ff}$): 1,536 (SwiGLU activation)
  - Context Length: 512
  - Positional Encoding: Rotary Position Embeddings (RoPE, base 10000.0)
  - Normalization: RMSNorm
  - Weight Tying: Token embeddings tied to output LM head projection
  - Attention Backend: PyTorch FlashAttention / SDPA

## Training Dataset
- **Source**: AI4Bharat Sangraha (`verified/ori`)
- **Revision**: `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8`
- **Corpus Processing**: NFC normalized, bad control chars removed, high-frequency boilerplate scrubbed.
- **Documents**: 67,330 train / 653 validation documents.
- **Tokens**: 28,462,726 train / 261,148 validation tokens (8K SentencePiece BPE, `uint16`).

## Hyperparameters & Optimization
- **Optimizer**: AdamW ($eta_1=0.9, eta_2=0.95$, weight decay 0.1, gradient clipping 1.0)
- **Learning Rate**: Peak $3.0 \times 10^{-4}$, minimum $3.0 \times 10^{-5}$
- **Schedule**: Linear warmup (100 steps) + Cosine decay (to step 1,800)
- **Batch Geometry**: Microbatch 16, Gradient Accumulation 2, Context 512
  - Effective tokens per step: $16 \times 2 \times 512 = 16,384$ tokens
- **Precision**: PyTorch AMP `fp16` with GradScaler
- **Hardware**: NVIDIA Tesla T4 (14.56 GiB VRAM)
- **Training Duration**: 1,800 steps (~29,491,200 tokens seen)
- **Throughput**: ~35,202 tokens/sec (~837 seconds total compute time)
- **Peak VRAM**: 3.529 GiB (24.2% of GPU capacity)

## Results
- **Initial Training Loss**: 9.07
- **Final Training Loss**: 4.5156
- **Initial Validation Loss**: 9.09 (Perplexity: ~8,865)
- **Final Validation Loss**: 4.6643
- **Final Validation Perplexity**: 106.09
- **Final Checkpoint**: `checkpoints/pilot-30m-v1/step_0001800.pt` (SHA256: `474ecb811a67...`)

## Verdict
The 30M model trained from scratch with perfect numerical stability, zero OOMs, and steady loss descent across all 1,800 steps.
