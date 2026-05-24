"""
Local end-to-end pipeline for the tourism project.

Reads:
    tourism_project/data/tourism.csv
Writes:
    tourism_project/local/artifacts/
        X_train.csv
        X_test.csv
        y_train.csv
        y_test.csv
        label_encoder.pkl
        best_model.pkl
        metrics.json

Run from the root of the project with:
    python tourism_project/local/pipeline_local.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

# ---- Paths (resolved relative to this file so cwd doesn't matter) ----
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RAW_CSV = PROJECT / "data" / "tourism.csv"
OUTPUT_DIR = HERE / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    if not RAW_CSV.exists():
        raise SystemExit(f"Raw data file not found at {RAW_CSV}"
                         f"Place tourism.csv there before running this script.")
    df = pd.read_csv(RAW_CSV)
    df = df.drop(columns=["CustomerID"], errors="ignore")  # Drop if exists, ignore if not
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]  # Drop unnamed columns if exist

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())  # Fill missing numeric with median

    label_encoders: dict[str, LabelEncoder] = {}
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])  # Fill missing categorical with mode
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store the label encoder for this column
    
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)
    joblib.dump(label_encoders, OUTPUT_DIR / "label_encoders.pkl")

    print(f"Train/test split done. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

def train(X_train, X_test, y_train, y_test) -> dict:
    search_space = {
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [8, 12, None],
                "min_samples_split": [2, 5]
            }
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        },
    }

    best = {"name": None, "model": None, "f1": 0.0, "params": None, "metrics": None}

    for name, config in search_space.items():
        print(f"Training {name}...")
        grid = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["params"],
            scoring="f1",
            cv=3,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        est = grid.best_estimator_
        y_pred = est.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "classification_report": classification_report(y_test, y_pred, zero_division=0)
        }

        print(f"{name} - F1 Score: {metrics['f1_score']:.4f}")
        print(f"Best Params: {grid.best_params_}")
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v}")

        
        if metrics["f1_score"] > best["f1"]:
            best.update({
                "name": name,
                "model": est,
                "f1": metrics["f1_score"],
                "params": grid.best_params_,
                "metrics": metrics
            })
    
    joblib.dump(best["model"], OUTPUT_DIR / "best_model.pkl")
    summary = {
        "best_model": best["name"],
        "best_f1_score": best["f1"],
        "best_params": best["params"],
        "metrics": best["metrics"]
    }

    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=4))
    print(f"Best model: {best['name']} with F1 Score: {best['f1']:.4f}")
    print(f"Saved -> {OUTPUT_DIR / 'best_model.pkl'} and {OUTPUT_DIR / 'metrics.json'}")
    return summary

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare()
    train(X_train, X_test, y_train, y_test)
