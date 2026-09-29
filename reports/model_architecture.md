# OdiaGPT model architecture

Randomly initialized, decoder-only Transformer implemented with PyTorch primitives.
Token embedding and output projection are tied. Each block uses pre-RMSNorm,
causal multi-head self-attention with RoPE, and a SwiGLU feedforward network.
Attention uses explicit masked softmax for initial correctness testing.

| Config | Layers | Width | Heads | Context | Vocabulary | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 2 | 128 | 4 | 128 | 4096 | 950,912 |
| nano | 4 | 256 | 4 | 256 | 8192 | 5,507,328 |
| tiny | 6 | 384 | 6 | 384 | 4096 | 12,194,688 |
| odia30m | 8 | 512 | 8 | 512 | 8192 | 31,465,984 |

The smoke, nano, and tiny configurations use the provisional 4K
tokenizer. The odia30m configuration requires a separately trained 8K tokenizer
and a larger validated corpus before training.
