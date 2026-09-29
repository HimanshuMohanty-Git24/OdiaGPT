# Nano generation test

- Checkpoint: nano-8k-v1 step 1000.
- Prompts: three fixed Odia prefixes, each with greedy and temperature 0.8 / top-k 50 / top-p 0.9 sampling.
- Generated continuations: six; random-init comparisons: six.
- Trained outputs contain recognizable Odia words and occasional sentence patterns.
- Greedy outputs show severe repetition on two prompts.
- Sampled outputs are often fragmented, with weak grammar and occasional malformed or code-mixed spans.
- One greedy output resembles a generic news/Telegram boilerplate call to action.
- These samples do not establish factual knowledge or reasoning.

## Artifacts

- generations/nano-8k-v1-step1000.jsonl
- generations/nano-8k-v1-random-baseline.jsonl
