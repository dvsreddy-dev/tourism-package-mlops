"""
Data Preparation Script
-----------------------
1. Loads dataset directly from HuggingFace data space
2. Clean: drop unnecessary columns, handle missing values, encode categoricals.
3. Split into train / test data sets and save locally.
4. Upload processed train / test sets back to the Hugging Face dataset space.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from huggingface_hub import HfApi, hf_hub_download

# ---- Configuration ----
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME", "your-hf-username")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-package-dataset"

# ---- Step 1: Load the raw dataset from Hugging Face ----
raw_csv_path = hf_hub_download(
    repo_id=DATASET_REPO_ID,
    filename="tourism.csv",
    repo_type="dataset",
    token=HF_TOKEN
)
df = pd.read_csv(raw_csv_path)

# ---- Step 2: Data cleaning ----
# Drop CustomerID (non-predictive) and any unnamed index column
df = df.drop(columns=["CustomerID"], errors="ignore")
df = df.loc[:, ~df.columns.str.contains("^unnamed")]

# Numerical: fill missing with median
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical: fill missing with mode, then label-encode
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ---- Step 3: Train/Test split and save locally ----
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=442, stratify=y
)

os.make_dirs("tourism_project/model_building/processed_data", exist_ok=True)
X_train.to_csv("tourism_project/model_building/processed_data/X_train.csv", index=False)
X_test.to_csv("tourism_project/model_building/processed_data/X_test.csv", index=False)
y_train.to_csv("tourism_project/model_building/processed_data/y_train.csv", index=False)
y_test.to_csv("tourism_project/model_building/processed_data/y_test.csv", index=False)
joblib.dump(label_encoders, "tourism_project/model_building/processed_data/label_encoders.pkl")

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---- Step 4: Upload train/test datasets back to Hugging Face ----
api = HfApi(token=HF_TOKEN)
for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "label_encoders.pkl"]:
    api.upload_file(
        path_or_fileobj=f"tourism_project/model_building/processed_data/{fname}",
        path_in_repo=f"processed/{fname}",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_messsage=f"Upload processed {fname}"
    )

print(f"Processsed datasets uploaded to: https://huggingface.co/datasets/{DATASET_REPO_ID}")
