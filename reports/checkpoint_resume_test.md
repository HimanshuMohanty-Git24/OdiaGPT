# Checkpoint and resume test

- Device: Tesla T4, FP16 GradScaler.
- Trained two smoke-model steps, saved a hash-verified atomic checkpoint, reloaded model/optimizer/scheduler/scaler and random states, then trained step three.
- Saved step/tokens: 2 / 64.
- AdamW optimizer state step advanced from 2 to 3 after resume.
- Maximum parameter difference from uninterrupted step three: 0.
- Test checkpoint was stored in temporary runtime storage; production nano checkpoints remain on Drive.
