"""Prepare or verify smoke token binaries."""

from pathlib import Path
import json
import sys

ROOT = Path("/content/drive/MyDrive/OdiaGPT")
sys.path.insert(0, str(ROOT / "src"))

from odiagpt.prepare_tokens import prepare_smoke_tokens


if __name__ == "__main__":
    result = prepare_smoke_tokens(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))
