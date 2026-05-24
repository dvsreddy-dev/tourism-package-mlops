"""
Hosting Script: Pushes deployment files (app.py, model, encoders, requirements.txt, Dockerfile)
to a Hugging Face Space for Streamlit hosting.

Pulls best_model.pkl and label_encoders.pkl from the HF Model Hub rather than the lcoal filesystem,
so this script works standalone without needing the local files created by the model training step.

NOTE: Requires HF_TOKEN environment variable set with a valid Hugging Face write token.
"""

import os
import shutil
from huggingface_hub import HfApi, create_repo, hf_hub_download

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME", "your-hf-username")
SPACE_NAME = "tourism-package-predictor"
SPACE_REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-package-model"


# Create the HF Space (Streamlit SDK)
api = HfApi(token=HF_TOKEN)
create_repo(
    repo_id=SPACE_REPO_ID, 
    repo_type="space", 
    space_sdk="docker", 
    exist_ok=True,
    token=HF_TOKEN
)


# ---- Prepare deployment folder ----
deploy_dir = "tourism_project/deployment"
os.makedirs(deploy_dir, exist_ok=True)

# Download the model + encoders from HF Model Hub and save to deployment folder
# Using HF as the source-of-truth for these artifacts rather than local files created by the model training step,
# so this hosting script can be run independently without needing the local files.
model_src = hf_hub_download(
    repo_id=MODEL_REPO_ID, filename="best_model.pkl", token=HF_TOKEN
)
encoders_src = hf_hub_download(
    repo_id=MODEL_REPO_ID, filename="label_encoders.pkl", token=HF_TOKEN
)
shutil.copy(model_src, os.path.join(deploy_dir, "best_model.pkl"))
shutil.copy(encoders_src, os.path.join(deploy_dir, "label_encoders.pkl"))

# ---- Upload deployment files to HF Space repo ----
api.upload_folder(
    folder_path=deploy_dir,
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    token=HF_TOKEN,
    commit_message="Deploy Streamlit app with model and encoders"
)

print(f"Deployment files uploaded to HF Space: https://huggingface.co/spaces/{SPACE_REPO_ID}")
print("The Streamlit app should be live within a few minutes!")
