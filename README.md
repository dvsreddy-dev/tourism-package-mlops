# Tourism Package Purchase - MLOps Project

End-to-End MLOps pipeline that trains a classifier to predict whether a customer
will purchase a tourism package, registers data and model artifacts on the Hugging Face Hub,
and serves predictions through a Streamlit app deployed to a Hugging Face Space.

## Project Layout
```
tourism_project/
  ├── data/
  │   ├── register_dataset.py        # Upload raw CSV to HF dataset repo
  │   └── tourism.csv                # Raw Dataset
  ├── model_building/
  │   ├── data_preparation.py        # Clean + Split, push processed splits to HF
  │   ├── hosting.py                 # Create/configure HF Space (vars + secrets)
  │   └── model_training.py          # Train RandomForest, log to MLflow, push model to HF
  ├── deployment/
  │   ├── app.py                     # Streamlit UI served on HF Space
  │   ├── Dockerfile                 # HF Space Container (port 7860)
  │   └── requirements.txt           # Runtime dependencies for the Space
  ├── local/
  |   ├── pipeline_local.py          # Standalone end-to-end pipeline (no HF)
  |   ├── app_local.py               # Local Streamlit UI (loads model from ./artifacts)
  |   └── artifacts/                 # Local model + processed splits
  └── .github/workflows/
  |   └── pipeline.yml               # CI/CD: data -> train -> deploy on push to main
  └── requirements.txt               # Pipeline + training dependencies
```

## Pipeline Stages
```text
┌───────────────────────┐          ┌───────────────────────┐         ┌───────────────────────┐         ┌───────────────────────┐
|                       |          |                       |         |                       |         |                       |
│  register_dataset.py  ├────────▶ │  data_preparation.py  ├───────▶ │   model_training.py   ├───────▶ │      hosting.py       |
|                       |          |                       |         |                       |         |                       |
└───────────┬───────────┘          └────────────┬──────────┘         └────────────┬──────────┘         └────────────┬──────────┘
            |                                   |                                 |                                 |
            ▼                                   ▼                                 ▼                                 ▼
        HF Dataset                         HF Dataset                         HF Model                          HF Space
        (raw CSV)                         (processed/)               Hub (Best_Model + encoders)         (Streamlit app, Docker)

```

## Running Locally

For a fully self-contained run that bypasses Hugging Face, see
[HOW_TO_RUN_LOCAL.md](./HOW_TO_RUN_LOCAL.md).

In Short:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r tourism_project/requirements.txt
python3 tourism_project/local/app_local.py
streamlit run tourism_project/local/app_local.py
```
