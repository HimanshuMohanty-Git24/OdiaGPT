"""Build or verify the resumable pilot Odia corpus."""

from pathlib import Path
import json
import sys

ROOT = Path("/content/drive/MyDrive/OdiaGPT")
sys.path.insert(0, str(ROOT / "src"))
from odiagpt.pilot_data import build_pilot_corpus

if __name__ == "__main__":
    print(json.dumps(build_pilot_corpus(ROOT), ensure_ascii=False, indent=2))
