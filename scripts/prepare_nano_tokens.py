"""Prepare or verify nano token binaries."""

from pathlib import Path
import json
import sys

ROOT = Path("/content/drive/MyDrive/OdiaGPT")
sys.path.insert(0, str(ROOT / "src"))
from odiagpt.prepare_nano_tokens import prepare_nano_tokens

if __name__ == "__main__":
    print(json.dumps(prepare_nano_tokens(ROOT), ensure_ascii=False, indent=2))
