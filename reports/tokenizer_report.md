# Smoke tokenizer report

Tokenizer: SentencePiece BPE trained from scratch on 877 smoke training documents.
Vocabulary: 4,096 pieces. Special IDs: pad=0, unk=1, bos=2, eos=3.
This is a provisional smoke tokenizer; the serious 8K tokenizer will use more data.

## Synthetic cases

| Case | Characters | Tokens | Exact round trip | Unknown tokens |
|---|---:|---:|---|---:|
| Odia | 26 | 9 | True | 0 |
| punctuation and digits | 21 | 15 | True | 0 |
| code-mixed | 34 | 26 | True | 0 |
| URL | 30 | 28 | True | 0 |
| rare characters | 15 | 19 | True | 0 |
| multiline | 25 | 12 | True | 0 |
| empty | 0 | 0 | True | 0 |

## Saved-corpus sample

- documents_measured: 109
- tokens_measured: 57794
- characters_measured: 185277
- words_measured: 29643
- unknown_token_rate: 0.0
- byte_fallback_rate: 0.008582205765304357
- tokens_per_character: 0.31193294364654006
- tokens_per_word: 1.9496677124447592
- mean_document_tokens_per_character: 0.282964792216537

The corpus metrics use the first 100 training documents and all available validation documents. They are diagnostics, not representative estimates.

SentencePiece identity normalization and byte fallback were selected to preserve uncommon Unicode characters. Multiline behavior is reported above.
