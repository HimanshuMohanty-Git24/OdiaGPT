# OdiaGPT project memory

## Objective
Build an educational Odia autoregressive decoder-only Transformer from random
initialization, with a tokenizer trained from scratch.

## Execution protocol
The user runs numbered cells manually in Google Colab and returns complete
output. Runtime results are recorded only after the user provides them.

## Confirmed environment
From user-provided CELL 00 output: Python 3.13.15, PyTorch 2.11.0+cu128,
CUDA 12.8, Tesla T4, 14.56 GiB GPU VRAM, 12.67 GiB total RAM.
Google Drive mounted and the project root is writable.

From user-provided CELL 01 output: persistent and temporary paths are writable;
the project root was empty before project initialization.

## Paths
Persistent project: `/content/drive/MyDrive/OdiaGPT`
Temporary runtime: `/content/odiagpt_runtime`

## Architecture target
Configuration-driven decoder-only Transformer using RMSNorm, RoPE, causal
self-attention, SwiGLU, and tied token embeddings/output weights where practical.
The 30M target starts at 8 layers, width 512, 8 heads, context 512, and
approximately 8,000 tokenizer tokens. Exact parameter count is pending.

## Dataset target
AI4Bharat Sangraha, Odia portion. Dataset access and schema are unverified.
Use a small streaming access test before downloading or processing data.

## Resume procedure
After reconnecting to Colab, inspect `memory.md`, `status.md`, and filesystem
state before continuing. A compact resume cell will be added later.

## Confirmed dataset access
CELL 03: `datasets` 4.8.5 streamed `ai4bharat/sangraha`, configuration `verified`, split `ori`. The iterable reported 11 shards. Three rows were read with string fields `doc_id`, `type`, and `text`; all three had type `web`.

## Initial dataset inspection
CELL 04 inspected the first 200 streamed verified Odia rows: 174 web, 26 pdf; median text length 1,041 characters; median Odia-block ratio 0.9753 using non-whitespace characters. One row had ratio 0.0. All 200 texts were NFC normalized in this sample. These first-row statistics are not representative estimates.

## Initial cleaning implementation
`unicode_utils.py` performs NFC, newline, control-character, and whitespace normalization while retaining Odia combining marks. `cleaning.py` uses configurable length, Odia script, and repeated-line checks. Exact text hashes support later deduplication. Thresholds are provisional until tests and corpus inspection.

## Cleaning test result
User-provided CELL 06 output showed all 13 cleaning unit tests passed.

## Smoke corpus
CELL 08 built or verified a cleaned corpus from Sangraha revision `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8`. Inspected 893 documents; retained 886; cleaned characters 1500127. Document-level split: 877 train and 9 validation documents. See `data/manifests/smoke_corpus.json` and `reports/data_quality_report.md` for details.

## Smoke tokenizer
CELL 11 trained or verified SentencePiece bpe with 4096 pieces on 877 training documents. Special IDs: {'pad': 0, 'unk': 1, 'bos': 2, 'eos': 3}. Model SHA256: c6bb0d75541659166554b45a437f15e8e44535fae19d99be1f46b542eb268551. See tokenizer/smoke_4k.json for its source revision and configuration.

## Smoke token binaries
CELL 16 verified data/tokenized/smoke_4k_train.bin and smoke_4k_val.bin against every source document. Streams use little-endian uint16 with BOS before and EOS after each document. Manifest: data/tokenized/smoke_4k.json. Train tokens: 452256; validation tokens: 3418.

## Runtime replacement lesson
Changing the Colab accelerator reset the kernel and unmounted Drive. A directory check for /content/drive/MyDrive is insufficient: use os.path.ismount('/content/drive') and verify status.md before writes. On 2026-09-29 two nano builder files were temporarily written into an unmounted placeholder; CELL 32 preserved and restored both files to the real Drive. CELL 28 then passed all seven model tests on Tesla T4.

## Nano corpus and tokenizer
Fixed Sangraha verified/ori revision 8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8. 20,004,188 cleaned chars, 12,069 train and 130 validation documents. 8K SentencePiece BPE tokenizer trained from scratch on train only; model SHA256 26b8026425d1e4f58f9acc6c7334968de2885988a824c005574e2baa079e6dfb. uint16 binaries verified against every document: 5,152,387 train tokens and 56,727 validation tokens. See data/manifests/nano_corpus.json and data/tokenized/nano_8k.json.

## Checkpoint restore order
Construct model, optimizer, and scheduler before loading their state dicts. Then load model, optimizer, scheduler, and GradScaler state in that order. Constructing LambdaLR after optimizer.load_state_dict resets the restored optimizer learning rate; this caused the initial CELL 60 diagnostic to diverge. CELL 60C used the production order and matched uninterrupted step-three weights exactly.

## Pilot corpus
Fixed Sangraha verified/ori revision 8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8. 110,001,302 cleaned characters after removal of 9,005 exact high-frequency boilerplate lines using data/metadata/boilerplate_lines_v1.json. 67,330 train and 653 validation documents, preserving nano assignments. Using the from-scratch nano_8k tokenizer, pilot_8k uint16 binaries contain 28,462,726 train tokens and 261,148 validation tokens; CELL 73 verified every document. Manifests: data/manifests/pilot_corpus.json and data/tokenized/pilot_8k.json.
