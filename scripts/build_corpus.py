"""Build or verify the small OdiaGPT cleaned corpus."""

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path("/content/drive/MyDrive/OdiaGPT")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from odiagpt.data_prepare import build_smoke_corpus


if __name__ == "__main__":
    manifest = build_smoke_corpus(PROJECT_ROOT)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
