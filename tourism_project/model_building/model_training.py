"""
Model Training Script with Experimentation Tracking
---------------------------------------------------
1. Load processed train/test data from the Hugging Face dataset space.
2. Define candidate models and hyperparameter grids (RandomForest, GradientBoosting)
3. Tue hyperparameters via GridSearchCV with MLFlow tracking.
4. Log all tuned parameters and evaluation metrics to MLFlow.
5. Evaluate and pick the best model (by F1 Score)
6. Register the best model in the Hugging Face Model Hub.
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import(
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import mlflow
import mlflow.sklearn
import joblib
from huggingface_hub import HfApi, hf_hub_download, create_repo

# ---- Configuration ----
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME", "your-hf-username")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-package-dataset"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-package-model"

# ---- Setp 1: Load train/test data from Hugging Face ----
def _hf_load(filename):
    path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=f"processed/{filename}",
        repo_type="dataset",
        token=HF_TOKEN
    )
    return pd.read_csv(path)

X_train = _hf_load("X_train.csv")
X_test = _hf_load("X_test.csv")
y_train = _hf_load("y_train.csv")
y_test = _hf_load("y_test.csv")

# ---- Step 2: Configure MLFlow ----
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Package_Prediction")

# ---- Step 3: Define candidate models and hyperparameter grids ----
search_space = {
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [8, 12, None],
            "min_samples_split": [2, 5],
        },
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_extimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
    },
}

best_overall = {"name": None, "model": None, "f1": 0.0, "params": None}

# ---- Step 4: Tune, log, and evaluate each model ----
for name, cfg in search_space.items():
    with mlflow.start_run(run_name=f"GridSearchCV_{name}"):
        grid = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["params"],
            cv=3,
            scoring="f1",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        y_pred = best_estimator.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "best_cv_f1": grid.best_score_,
        }

        # Log all tuned parameters and metrics
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_estimator, name)

        print(f"\n--- {name} ---")
        print(f"Best Params: {grid.best_params_}")

        for k, v in metrics.items():
            print(f".   {k}: {v:.4f}")
        print(classification_report(y_test, y_pred))

        if metrics["f1_score"] > best_overall["f1"]:
            best_overall.update(
                {"name": name, "model": best_estimator, "f1": metrics["f1_score"], "params": grid.best_params_}
            )

# ---- Step 5: Save best model locally ----
os.makedirs("tourism_project/model_building/model", exist_ok=True)
model_path = "tourism_project/model_building/model/best_model.pkl"
joblib.dump(best_overall["model"], model_path)

with mlflow.start_run(run_name=f"Best_Model_{best_overall['name']}"):
    mlflow.log_params(best_overall["params"])
    mlflow.log_metrics("best_f1_score", best_overall["f1"])
    mlflow.sklearn.log_model(
        best_overall["model"], "best_model", registered_model_name="TourismPackagePredictor"
    )

print(f"\nBest Model: {best_overall['name']} | F1: {best_overall['f1']:.4f}")

# ---- Step 6: Register best model in Hugging Face Model Hub ----
api = HfApi(token=HF_TOKEN)
create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=f"best_model.pkl",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
    commit_message=f"Register best model: {best_overall['name']}, F1={best_overall['f1']:.4f}"
)

# Also push the label encoders alongside the model so the deployment can use them
api.upload_file(
    path_or_fileobj="tourism_project/model_building/processed_data/label_encoders.pkl",
    path_in_repo=f"label_encoders.pkl",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
    commit_message="Added label encoders"
)

print(f"Model registerd at: https://huggingface.co/{MODEL_REPO_ID}")
