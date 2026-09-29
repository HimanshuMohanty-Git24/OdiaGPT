"""Run or resume the OdiaGPT nano experiment."""

from pathlib import Path
import json
import sys

ROOT = Path("/content/drive/MyDrive/OdiaGPT")
sys.path.insert(0, str(ROOT / "src"))
from odiagpt.training import train_nano

if __name__ == "__main__":
    print(json.dumps(train_nano(ROOT), ensure_ascii=False, indent=2))
