# Nano 8K tokenizer report

- Trained from scratch with SentencePiece BPE on nano training documents only.
- Vocabulary: 8,192
- Special IDs: pad=0, unk=1, bos=2, eos=3
- Identity normalization, byte fallback, no dummy prefix.
- Test suite: 4 tests passed.
- Documents measured: 230 (first 100 train and all validation).
- Tokens measured: 102,407.
- Unknown token rate: 0.000000.
- Byte fallback rate: 0.010322.
- Tokens per character: 0.2596.
- Tokens per word: 1.6466.
- Document roundtrip failures: 0.
