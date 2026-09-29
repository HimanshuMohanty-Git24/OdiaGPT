# Changelog

## 2026-09-29 11:23:46 IST
- Initialized the persistent OdiaGPT project layout.
- Added initial project memory, status, TODO, decisions, README, requirements,
  and environment report.

## 2026-09-29 11:35:52 IST
- Added Odia Unicode and cleaning modules.
- Recorded the initial 200-row dataset inspection.

## 2026-09-29 11:37:22 IST
- Added and ran focused cleaning unit tests.

## 2026-09-29 11:41:12 IST
- Added the small corpus builder and reproducible command-line script.

- CELL 14: Added verified, atomic uint16 smoke token preparation and manifest.

- CELL 17–18: Added validated model configs, from-scratch RoPE, causal attention, RMSNorm, SwiGLU, and tied-weight decoder model.

- CELL 26–33: Added resumable nano corpus builder, recovered staged source after T4 switch, and verified CUDA model tests.

- CELL 43: Switched nano.yaml from 4K to 8K vocabulary and added initial training settings.

## 2026-09-29 16:16:53 IST
- Updated README to describe verified smoke, nano, and running pilot stages, setup, resume, and limitations.

## 2026-09-29 16:38:11 IST
- Completed OdiaGPT-30M serious training experiment (1,800 steps, 29.5M tokens seen).
- Generated text samples for OdiaGPT-30M on benchmark prompts under greedy and sampling modes.
- Generated `reports/main_training_report.md`.
- Generated `reports/evaluation_report.md` comparing Random Init, Nano, and OdiaGPT-30M.
- Generated `reports/final_report.md` summarizing the entire curriculum.
- Updated `README.md`, `TODO.md`, and `status.md` to reflect project completion.
