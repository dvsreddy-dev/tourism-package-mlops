"""
Hosting Script: Pushes deployment files (app.py, model, encoders, requirements.txt, Dockerfile)
to a Hugging Face Space for Streamlit hosting.

NOTE: Requires HF_TOKEN environment variable set with a valid Hugging Face write token.
"""

import os
import shutil
from huggingface_hub import HfApi, create_repo

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
SPACE_NAME = "tourism-package-predictor"
HF_USERNAME = "your-hf-username"

REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"

# Create the HF Space (Streamlit SDK)
api = HfApi(token=HF_TOKEN)
create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)

# Prepare deployment folder
deploy_dir = "tourism_project/deployment"

# Copy model and encoders to deployment folder
shutil.copy("tourism_project/model_building/model/best_model.pkl", f"{deploy_dir}/best_model.pkl")
shutil.copy(
    "tourism_project/model_building/processed_data/label_encoders.pkl",
    f"{deploy_dir}/label_encoders.pkl",
)

# Upload all files in deployment folder to the Space
api.upload_folder(
    folder_path=deploy_dir,
    repo_id=REPO_ID,
    repo_type="space",
)

print(f"Successfully deployed to : https://huggingface.co/spaces/{REPO_ID}")
print("The Streamlit app should be live within a few minutes.")
