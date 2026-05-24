# How to Run the project Locally (No Hugging Face, No GitHub Actions)

This guide runs the **entire MLOps pipeline on your laptop/desktop** - data prep, training, and the Streamlit UI - with no external services. Useful for development, debugging, or air-gapped demos.

The only things skipped are: HF Hub uploads, HF Spaces hosting, and GitHub Actions CI. The model logic and Streamlit UI are identical to the cloud version.

---

## 1. Pre-requisites

- Python **3.14** installed and on `PATH`
- The repo cloned locally
- The raw dataset `tourism.csv` placed at:
  ```
  tourism_project/data/tourism.csv
  ```

---

## 2. Create a virtual envionrment and install dependencies

From the **repo root**

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r tourism_project/requirements.txt
```

### Windows Powershell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r tourism_project\requirements.txt
```

This single `requirements.txt` covers both the pipeline and Streamlit (it's the supoerset of the deployment model).

---

## 3. Run the local pipeline (data prep + model training)

From root folder:

```bash
python tourism_project/local/pipeline_local.py
```

This Script:
1. Reads `tourism_project/data/tourism.csv`
2. Cleans, encodes, train/test-splits
3. Runs `GridSearchCV` over `RandomForest` and `GradientBoosting`
4. Saves the best model + encoders to `tourism_project/local/artifacts/`

Expected output (last lines):
```
Train/test split done. Train shape: (****, **), Test shape: (***, 18**
Training RandomForest...
RandomForest - F1 Score: 0.****
Best Params: {...}
Accuracy: 0.***************
Precision: 0.***************
Recall: 0.***************
F1_score: 0.***************
Classification_report:               precision    recall  f1-score   support


Training GradientBoosting...
GradientBoosting - F1 Score: 0.****
Best Params: {...}
Accuracy: 0.***************
Precision: 0.***************
Recall: 0.***************
F1_score: 0.***************
Classification_report:               precision    recall  f1-score   support
```

---

## 4. Launch the local streamlit app

Still from root foler:

```bash
streamlit run tourism_project/local/app_local.py
```

Strimlist will print a URL - opn it in a browser:

```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.29.108:8501
```

Adjust the sidebar inputs and click **🔍 Predict**.

---

## 5. (Optional) Use MLflow tracking locally

The local pipeline script intentionally **skips MLflow** so you don't need a traciing server. If you want it anyway:

```bash
# Terminal 1 - start the tracking UI
mlflow ui --host 127.0.0.1 --port 5000

# Terminal 2 - set the tracking URI then run the *original* notebook script
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000     # Powershell: $env:MLFLOW_TRACKING_URI=...
# (re-run the notebook's model_traning cell, which already logs to MLflow)
```

Browse to http://127.0.0.1:5000 to see runs, params, and metrics.
