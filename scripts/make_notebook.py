"""Generate the OdiaGPT Colab orchestration notebook from persistent source."""

from pathlib import Path
import json

ROOT = Path("/content/drive/MyDrive/OdiaGPT")


def markdown(body: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": body.splitlines(keepends=True)}


def code(body: str) -> dict:
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": body.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        markdown("""# OdiaGPT from scratch

This notebook orchestrates the persistent Python project on Google Drive. Run
the reconnect and inspection cells first after every Colab runtime change.
Training cells are guarded to avoid starting a second long run accidentally.
Model weights and both SentencePiece tokenizers are trained from scratch.
The verified experiment history is in experiments.jsonl.
"""),
        markdown("## WORKFLOW 00 — Mount Drive and verify the saved project"),
        code("""from pathlib import Path
import os
from google.colab import drive

DRIVE = Path("/content/drive")
if not os.path.ismount(DRIVE):
    if DRIVE.is_dir() and any(DRIVE.iterdir()):
        raise RuntimeError("Unmounted /content/drive contains files; inspect before mounting")
    drive.mount(str(DRIVE), force_remount=False)
ROOT = DRIVE / "MyDrive/OdiaGPT"
if not os.path.ismount(DRIVE) or not (ROOT / "status.md").is_file():
    raise RuntimeError("Saved OdiaGPT project is unavailable")
print("Drive mounted:", os.path.ismount(DRIVE))
print("Project:", ROOT)
print((ROOT / "status.md").read_text(encoding="utf-8"))
"""),
        markdown("## WORKFLOW 01 — Environment inspection"),
        code("""import platform, shutil, sys, torch
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
try:
    import psutil
    print("RAM total GiB:", round(psutil.virtual_memory().total / 2**30, 2))
except ImportError:
    pass
disk = shutil.disk_usage("/content")
print("Disk /content free GiB:", round(disk.free / 2**30, 2))
sys.path.insert(0, str(ROOT / "src"))
"""),
        markdown("## WORKFLOW 02 — Project state and checkpoint pointers"),
        code("""import json
for relative in ("memory.md", "TODO.md", "decisions.md"):
    path = ROOT / relative
    print("\\n###", relative)
    print(path.read_text(encoding="utf-8")[:5000])
for relative in (
    "checkpoints/nano-8k-v1/latest.json",
    "checkpoints/pilot-30m-v1/latest.json",
    "checkpoints/latest.json",
):
    path = ROOT / relative
    print(relative, json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None)
"""),
        markdown("""## WORKFLOW 03 — Dependency and dataset access

The saved data manifests identify the pinned Sangraha revision. Installing
packages or streaming new data is a separate, intentional step. Do not rerun
corpus builders during training.
"""),
        code("""import importlib.metadata as metadata
for package in ("datasets", "sentencepiece", "PyYAML", "numpy", "torch"):
    try:
        print(package, metadata.version(package))
    except metadata.PackageNotFoundError:
        print(package, "MISSING")
for relative in (
    "data/manifests/smoke_corpus.json",
    "data/manifests/nano_corpus.json",
    "data/manifests/pilot_corpus.json",
    "data/tokenized/nano_8k.json",
    "data/tokenized/pilot_8k.json",
):
    path = ROOT / relative
    print(relative, "present" if path.is_file() else "absent")
"""),
        markdown("## WORKFLOW 04 — Cleaning, tokenizer, and model tests"),
        code("""import os, subprocess, sys
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
result = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
    cwd=ROOT, env=env, check=False,
)
print("Test process exit code:", result.returncode)
if result.returncode:
    raise RuntimeError("Project unit tests failed")
"""),
        markdown("""## WORKFLOW 05 — Optional data preparation

The builders are staged and reuse existing verified outputs. They may stream
substantial data when a stage has not been prepared. Set RUN_PREPARATION to True
only when intentionally preparing the selected stage.
"""),
        code("""RUN_PREPARATION = False
STAGE = "smoke"  # smoke, nano, or pilot
if RUN_PREPARATION:
    pipeline = {
        "smoke": ("build_corpus.py", "train_tokenizer.py", "prepare_tokens.py"),
        "nano": ("build_nano_corpus.py", "train_nano_tokenizer.py", "prepare_nano_tokens.py"),
        "pilot": ("build_pilot_corpus.py", "prepare_pilot_tokens.py"),
    }
    if STAGE not in pipeline:
        raise ValueError(f"Unknown preparation stage: {STAGE}")
    for script in pipeline[STAGE]:
        print("Running", script, flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script)],
                       cwd=ROOT, env=env, check=True)
else:
    print("Preparation skipped. Set RUN_PREPARATION=True for a selected stage.")
"""),
        markdown("## WORKFLOW 06 — Inspect live training without starting a duplicate"),
        code("""import json
EXPERIMENT_ID = "pilot-30m-v1"
process_file = ROOT / "runs" / EXPERIMENT_ID / "process.json"
if process_file.is_file():
    pid = int(json.loads(process_file.read_text(encoding="utf-8"))["pid"])
    proc = Path(f"/proc/{pid}")
    state = proc.joinpath("stat").read_text().split()[2] if proc.exists() else "exited"
    print("PID:", pid, "state:", state)
else:
    print("No process metadata for", EXPERIMENT_ID)
console = ROOT / "logs" / f"{EXPERIMENT_ID}.console.log"
if console.is_file():
    print("\\n".join(console.read_text(encoding="utf-8").splitlines()[-20:]))
pointer = ROOT / "checkpoints" / EXPERIMENT_ID / "latest.json"
print("Checkpoint:", json.loads(pointer.read_text(encoding="utf-8")) if pointer.is_file() else None)
print((ROOT / "status.md").read_text(encoding="utf-8"))
"""),
        markdown("""## WORKFLOW 07 — Explicitly resume training

Only enable when the relevant experiment is unfinished, CUDA is available,
and WORKFLOW 06 shows no active process. The training script resumes from its
hash-verified checkpoint and writes progress to Drive. This cell can run for
a long time.
"""),
        code("""RUN_TRAINING = False
EXPERIMENT_ID = "pilot-30m-v1"  # or nano-8k-v1
if RUN_TRAINING:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run")
    if EXPERIMENT_ID not in {"pilot-30m-v1", "nano-8k-v1"}:
        raise ValueError(EXPERIMENT_ID)
    metadata_path = ROOT / "runs" / EXPERIMENT_ID / "process.json"
    if metadata_path.is_file():
        pid = int(json.loads(metadata_path.read_text(encoding="utf-8"))["pid"])
        proc = Path(f"/proc/{pid}")
        if proc.exists() and proc.joinpath("stat").read_text().split()[2] != "Z":
            raise RuntimeError(f"Existing training process is active: PID {pid}")
    script = "train_pilot.py" if EXPERIMENT_ID == "pilot-30m-v1" else "train.py"
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script)],
                   cwd=ROOT, env=env, check=True)
else:
    print("Training skipped. Inspect state, then set RUN_TRAINING=True to resume.")
"""),
        markdown("## WORKFLOW 08 — Completed experiment records and reports"),
        code("""records = [json.loads(line) for line in
           (ROOT / "experiments.jsonl").read_text(encoding="utf-8").splitlines()
           if line.strip()]
print("Completed experiment records:", len(records))
for record in records:
    print(record.get("experiment_id"), record.get("result"),
          "steps", record.get("training_steps"),
          "validation loss", record.get("validation_loss"))
for name in ("nano_experiment.md", "pilot_report.md", "evaluation_report.md", "final_report.md"):
    path = ROOT / "reports" / name
    print(name, "present" if path.is_file() else "pending")
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    output = ROOT / "OdiaGPT_From_Scratch.ipynb"
    notebook = build_notebook()
    encoded = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"Existing notebook differs; refusing to overwrite: {output}")
    output.write_text(encoded, encoding="utf-8")
    print("Notebook:", output)
    print("Cells:", len(notebook["cells"]))
