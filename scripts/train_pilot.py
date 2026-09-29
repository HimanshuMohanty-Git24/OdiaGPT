"""Run or resume the OdiaGPT-30M pilot experiment."""

from pathlib import Path
import json
import sys

ROOT = Path("/content/drive/MyDrive/OdiaGPT")
sys.path.insert(0, str(ROOT / "src"))
from odiagpt.pilot_training import train_pilot

if __name__ == "__main__":
    print(json.dumps(train_pilot(ROOT), ensure_ascii=False, indent=2))
