#!/usr/bin/env python3
"""
Helper to push the model outputs and README to the Hugging Face Hub.

Usage:
  export HUGGINGFACE_HUB_TOKEN="hf_..."
  export HF_REPO_ID="username/repo-name"  # optional (defaults to 4nkh/theme_model)
  python push_model.py

This script will create the model repo (if it doesn't exist) and upload
the `theme_model_outputs/` folder and the `README.md` model card.
"""
import os
import sys
from huggingface_hub import HfApi


def main():
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("ERROR: HUGGINGFACE_HUB_TOKEN not set", file=sys.stderr)
        sys.exit(2)

    raw_repo = os.getenv("HF_REPO_ID", "4nkh/theme_model")
    # Accept full URL or namespace
    if raw_repo.startswith("https://"):
        raw_repo = raw_repo.rstrip("/").split("/")[-2] + "/" + raw_repo.rstrip("/").split("/")[-1]

    repo_id = raw_repo
    folder = os.getenv("MODEL_FOLDER", "theme-model/theme_model_outputs")
    readme_path = os.getenv("README_PATH", "theme-model/README.md")

    api = HfApi(token=token)
    print("Using repo:", repo_id)

    try:
        print("Creating repo if needed...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        # Not fatal — repo might already exist or we might not have permission to create it
        print("Warning: create_repo failed or not needed:", e)

    try:
        print(f"Uploading folder {folder} to {repo_id} ...")
        api.upload_folder(repo_id=repo_id, folder_path=folder, path_in_repo="", repo_type="model")
        if os.path.exists(readme_path):
            print(f"Uploading README {readme_path} ...")
            with open(readme_path, "rb") as f:
                api.upload_file(path_or_fileobj=f, path_in_repo="README.md", repo_id=repo_id)
        print("Upload finished: https://huggingface.co/" + repo_id)
    except Exception as e:
        print("UPLOAD_ERROR", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
