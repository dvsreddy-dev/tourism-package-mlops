"""
Data Registration Script
------------------------
Registers the raw tourism.csv dataset to the Hugging Face Datasets Hub.

Requires:
    - HF_TOKEN environment variable (write-scoped Hugging Face token)
    - The raw dataset at tourism_project/data/trourism.csv
"""
import os
from huggingface_hub import HfApi, create_repo

# Configuration
HF_USERNAME = os.environ.get("HF_USERNAME", "your-hf-username")
HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO_NAME = "tourism-package-dataset"
DATASET_REPO_ID = f"{HF_USERNAME}/{DATASET_REPO_NAME}"

# Create a dataset repository on Hugging Face Hub
api = HfApi(token=HF_TOKEN)
create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

# Upload the tourism.csv to HuggingFace dataset space
api.upload_file(
    path_or_fileobj="tourism_project/data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    token="HF_TOKEN",
    commit_message="Register raw tourism dataset"
)

print(f"Dataset registered successfully at: https://huggingface.co/datasets/{DATASET_REPO_ID}")
