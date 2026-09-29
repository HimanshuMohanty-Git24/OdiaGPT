"""
Upload exported OdiaGPT-30M to the Hugging Face Hub.
Usage:
    python scripts/upload_to_hf.py --repo-id <username>/OdiaGPT-30M --token <HF_TOKEN>
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "export/hf_odiagpt_30m"


def main():
    parser = argparse.ArgumentParser(description="Upload OdiaGPT to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="HF repo ID, e.g. username/OdiaGPT-30M")
    parser.add_argument("--token", type=str, required=False, default=None, help="Hugging Face User Access Token")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    api = HfApi(token=args.token)
    print(f"Creating repository {args.repo_id} (if not already exists)...")
    create_repo(args.repo_id, token=args.token, private=args.private, exist_ok=True)

    print(f"Uploading files from {EXPORT_DIR} to {args.repo_id}...")
    api.upload_folder(
        folder_path=str(EXPORT_DIR),
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"\nSuccessfully uploaded OdiaGPT-30M to Hugging Face!")
    print(f"Model URL: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
