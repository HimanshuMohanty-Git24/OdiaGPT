"""Build or verify the resumable nano Odia corpus."""

from pathlib import Path
import json
import sys

ROOT = Path("/content/drive/MyDrive/OdiaGPT")
sys.path.insert(0, str(ROOT / "src"))
from odiagpt.nano_data import build_nano_corpus

if __name__ == "__main__":
    print(json.dumps(build_nano_corpus(ROOT), ensure_ascii=False, indent=2))
