# Technical decisions

## 2026-09-29 11:23:46 IST — Storage layout
- Issue: Colab runtime storage is temporary.
- Options: Store everything only in Drive; use only runtime storage; use both.
- Chosen solution: Persist source, state, reports, and checkpoints in Drive.
  Use `/content/odiagpt_runtime` for temporary training copies.
- Reason: Preserve work across disconnects while allowing fast local data access.

## 2026-09-29 11:23:46 IST — Execution sequence
- Issue: Larger training runs could conceal data or model errors.
- Options: Start with the 30M model; validate each smaller stage first.
- Chosen solution: Follow the staged sequence in `TODO.md`.
- Reason: Verify dataset, tokenizer, model correctness, tiny overfit, and
  checkpoint resume before serious training.

## 2026-09-29 11:35:52 IST — Initial cleaning policy
- Issue: Remove unusable documents without damaging Odia text.
- Options: Aggressive character stripping; conservative Unicode normalization
  with configurable quality checks.
- Chosen solution: NFC, restrained whitespace/control cleanup, and provisional
  minimum length, Odia-script, and repeated-line thresholds.
- Reason: Preserve combining marks, punctuation, digits, and code-mixed text.
  Validate thresholds with tests and corpus observations before scaling.

## 2026-09-29 11:59:50 IST — Smoke tokenizer vocabulary
- Issue: The 1.5M-character smoke corpus is small for an 8K tokenizer.
- Options: Train 8K immediately; use a smaller provisional tokenizer.
- Chosen solution: Train a 4,096-piece SentencePiece BPE tokenizer on training documents only.
- Reason: Validate the pipeline now; train a new 8K tokenizer on a larger corpus for serious experiments.
- Use identity normalization, byte fallback, and fixed special-token IDs 0–3.

## 2026-09-29 12:12:00 IST — Nano corpus target
- Issue: Nano training needs roughly 5M–10M tokens.
- Options: Train on the 452K-token smoke corpus; build a larger verified Odia prefix.
- Chosen solution: Stream the fixed Sangraha revision until 20M cleaned characters, then train a separate 8K tokenizer and measure actual tokens.
- Reason: The smoke tokenizer gave about 0.31 tokens per character; 20M characters should provide a provisional 5M–10M-token corpus. The exact count will be checked after tokenization.
- Preserve smoke train/validation document assignments in the larger split.

## 2026-09-29 12:17:10 IST — Nano 8K model configuration
- Issue: The initial nano.yaml targeted the provisional 4K smoke tokenizer.
- Options: Keep 4K; match the newly trained nano 8K tokenizer.
- Chosen solution: Set nano vocabulary to 8,192; retain 4 layers, width 256, 4 heads, context 256.
- Reason: Train with the tokenizer fitted to the 5,152,387-token nano corpus. Exact tied-weight parameter count: 5,507,328.
- Initial training settings: T4 FP16, AdamW, warmup plus cosine, microbatch 8, accumulation 4, 1,000 steps. These will be checked in preflight.

## 2026-09-29 16:00:24 IST — Pilot boilerplate cleaning
- Issue: The nano corpus contains repeated publisher promotion and digitization-credit lines; one footer appears in 859 training documents and surfaced in generation.
- Options: Keep all text; broad rule-based filtering; remove exact high-frequency lines derived from training documents.
- Chosen solution: For pilot data, remove seven exact lines of at least 20 characters found in at least 30 nano training documents.
- Reason: This targets observed repeated boilerplate while preserving ordinary prose and punctuation. The list and source hash are saved in data/metadata/boilerplate_lines_v1.json.

## 2026-09-29 16:10:51 IST — Pilot attention backend
- Issue: Manual masked attention is clear for correctness but may limit 30M model throughput at context 512.
- Options: Continue manual attention; use PyTorch causal scaled dot product attention after correctness checks.
- Chosen solution: Keep manual as the default and select sdpa in configs/odia30m.yaml. QKV projections, RoPE, blocks, and model weights remain our own implementation.
- Reason: Manual attention passed model causality, backward, and CUDA tests. CELL 75 will compare SDPA output and causality before training.

## 2026-09-29 16:12:49 IST — Pilot batch and step count
- Issue: Choose 30M pilot settings on the verified Tesla T4.
- Options: Microbatch 8, 16, or 32; vary accumulation.
- Chosen solution: FP16 SDPA, microbatch 16, accumulation 2, context 512, 1,800 updates, AdamW 3e-4 peak LR with 100-step warmup and cosine decay.
- Reason: CELL 76 measured about 36.9K tokens/sec at microbatch 16 with 3.23 GiB peak allocated VRAM. The plan processes 29,491,200 training tokens, just over the 28,462,726-token pilot corpus size, while keeping memory headroom.
