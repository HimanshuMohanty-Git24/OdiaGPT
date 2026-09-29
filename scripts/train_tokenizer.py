"""Train or verify the OdiaGPT smoke tokenizer."""

from pathlib import Path
import json
import sys

ROOT = Path("/content/drive/MyDrive/OdiaGPT")
sys.path.insert(0, str(ROOT / "src"))

from odiagpt.tokenizer import train_smoke_tokenizer


if __name__ == "__main__":
    result = train_smoke_tokenizer(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))
