# OdiaGPT-30M T4 memory benchmark

- Date: 2026-09-29 15:59:01 IST
- GPU: Tesla T4
- Compute capability: (7, 5)
- Parameters: 31,465,984
- Precision: FP16 autocast with GradScaler.
- Workload: one random-token forward, backward, AdamW step at context 512.
- No real dataset or language-quality result was measured.

| Microbatch | Result | Peak allocated GiB | Peak reserved GiB | Step seconds |
|---:|---|---:|---:|---:|
| 1 | passed | 0.666 | 0.770 | 0.61 |
| 2 | passed | 0.985 | 1.033 | 0.64 |
| 4 | passed | 1.476 | 1.570 | 0.68 |
| 8 | passed | 2.461 | 2.613 | 0.74 |

## SDPA throughput

Validated against manual attention in CELL 75.
One warmup update and three timed synthetic updates per microbatch.

| Microbatch | Result | Peak allocated GiB | Tokens/sec | Seconds/update |
|---:|---|---:|---:|---:|
| 8 | passed | 1.944 | 34580 | 0.118 |
| 16 | passed | 3.234 | 36916 | 0.222 |
| 32 | passed | 5.959 | 37363 | 0.439 |
