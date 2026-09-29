"""Train or verify the OdiaGPT nano 8K tokenizer."""

from pathlib import Path
import json
import sys

ROOT = Path("/content/drive/MyDrive/OdiaGPT")
sys.path.insert(0, str(ROOT / "src"))
from odiagpt.nano_tokenizer import train_nano_tokenizer

if __name__ == "__main__":
    print(json.dumps(train_nano_tokenizer(ROOT), ensure_ascii=False, indent=2))
