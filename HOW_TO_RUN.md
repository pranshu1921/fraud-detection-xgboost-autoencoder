# HOW TO RUN — Fraud Detection: XGBoost + Autoencoder Ensemble

This file walks through every step from a fresh clone to a fully running system.
Follow these steps in order. Every command is copy-paste ready.

---

## Prerequisites

| Tool | Version | Check Command | Install Link |
|---|---|---|---|
| Python | 3.11+ | `python --version` | https://python.org |
| Conda | any | `conda --version` | https://anaconda.com |
| Git | 2.0+ | `git --version` | https://git-scm.com |
| Kaggle account | free | — | https://kaggle.com |

**Terminal note:** Use **Anaconda Prompt** or **Command Prompt** for all Python commands.
Use **GitBash** for Git commands only. Do not run Python in GitBash on Windows — it causes segmentation faults with PyTorch and XGBoost.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/pranshu1921/fraud-detection-xgboost-autoencoder.git
cd fraud-detection-xgboost-autoencoder
```

---

## Step 2 — Create Conda Environment

```bash
conda create -n fraud-detection python=3.11 -y
conda activate fraud-detection
```

Verify:
```bash
python --version
# Expected: Python 3.11.x
```

---

## Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install setuptools
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Verify:

Create a file called `verify.py` with this content:

```python
import torch, xgboost, mlflow, evidently, fastapi, streamlit, sklearn, pandas, numpy, shap
print("numpy:      ", numpy.__version__)
print("pandas:     ", pandas.__version__)
print("torch:      ", torch.__version__)
print("xgboost:    ", xgboost.__version__)
print("sklearn:    ", sklearn.__version__)
print("mlflow:     ", mlflow.__version__)
print("shap:       ", shap.__version__)
print("fastapi:    ", fastapi.__version__)
print("streamlit:  ", streamlit.__version__)
print("All packages OK")
```

Run it:
```bash
python verify.py
```

Expected: `All packages OK`

---

## Step 4 — Set Up Kaggle API

**4a. Get your Kaggle API key**

1. Go to https://www.kaggle.com/settings/account
2. Scroll to the **API** section
3. Click **Create New API Token** (not "Create New Token")
4. A `kaggle.json` file downloads to your Downloads folder

**4b. Place the file in the right location**

```bash
mkdir -p ~/.kaggle
cp /c/Users/YOUR_USERNAME/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Replace `YOUR_USERNAME` with your actual Windows username.

**4c. Accept the competition data terms**

Go to https://www.kaggle.com/c/ieee-fraud-detection and click **Join Competition**. Accept the rules. No submission needed.

**4d. Verify authentication**

```bash
kaggle competitions files ieee-fraud-detection
```

Expected: a list of files including `train_transaction.csv` and `train_identity.csv`.

---

## Step 5 — Run the Full Training Pipeline

Run from the **project root** (not from inside src/):

```bash
python src/train.py
```

This automatically:
1. Downloads the IEEE-CIS dataset from Kaggle (~677MB, one time only)
2. Validates the data with Great Expectations
3. Engineers 439 features from 590K transactions
4. Trains the PyTorch Autoencoder on 570K legitimate transactions
5. Trains XGBoost with temporal split and SHAP explainability
6. Trains the ensemble meta-learner
7. Logs everything to MLflow
8. Saves all model artifacts to `models/`

**Expected runtime:** 25 to 40 minutes on CPU.

**Expected final output:**
```
TRAINING COMPLETE
Final ensemble PR-AUC: ~0.52
MLflow Run ID: xxxxxxxx
Models saved to: models/
```

**Note on PR-AUC:** The IEEE-CIS dataset is one of the most challenging fraud datasets publicly available. A PR-AUC of 0.50 to 0.55 is consistent with published benchmarks on this dataset. The ROC-AUC of ~0.91 confirms the model has strong discriminative power.

---

## Step 6 — View MLflow Experiment Results

```bash
mlflow ui --backend-store-uri mlruns --port 5001
```

Open http://localhost:5001 in your browser.

Click the `fraud_detection_ieee_cis` experiment to see all logged parameters, metrics, and model artifacts.

Stop with `Ctrl + C` when done.

---

## Step 7 — Generate Monitoring Reports

```bash
python src/monitor.py
```

Expected output:
```
Drift within acceptable range. No retraining needed.
Reports saved to reports/
```

Open the reports:
```bash
start reports/data_drift_report.html
start reports/model_performance_report.html
```

---

## Step 8 — Launch the FastAPI Inference Endpoint

```bash
uvicorn api.main:app --reload --port 8000
```

Open http://localhost:8000/docs in your browser.

Test the predict endpoint with this request body:

```json
{
  "TransactionAmt": 2500.0,
  "ProductCD": "C",
  "card1": 4932,
  "card4": "visa",
  "card6": "credit",
  "P_emaildomain": "protonmail.com",
  "DeviceType": "mobile"
}
```

Expected response includes fraud probability, anomaly score, ensemble decision, risk level, and top 3 SHAP features.

Stop with `Ctrl + C` when done.

---

## Step 9 — Launch the Streamlit Dashboard

In a new terminal (with conda environment active):

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

Click **Score 100 Transactions** to simulate a live transaction feed. Click any flagged transaction in the right panel to see the SHAP explanation detail.

---

## Running Order Summary

| Step | Command | Runtime |
|---|---|---|
| Install packages | `pip install -r requirements.txt ...` | 5-10 min |
| Train pipeline | `python src/train.py` | 25-40 min |
| MLflow UI | `mlflow ui --backend-store-uri mlruns --port 5001` | instant |
| Monitoring reports | `python src/monitor.py` | 2-3 min |
| FastAPI server | `uvicorn api.main:app --reload --port 8000` | instant |
| Streamlit dashboard | `streamlit run dashboard/app.py` | instant |

---

## File Locations Reference

| What | Where |
|---|---|
| Raw data (auto-downloaded) | `data/train_transaction.csv`, `data/train_identity.csv` |
| Processed features | `data/X_features.parquet`, `data/y_labels.parquet` |
| Encoders | `data/encoders.pkl` |
| Autoencoder model | `models/autoencoder.pt` |
| XGBoost model | `models/xgboost_model.pkl` |
| Meta-learner | `models/meta_learner.pkl` |
| Training summary | `models/training_summary.json` |
| MLflow runs | `mlruns/` |
| Evidently reports | `reports/data_drift_report.html`, `reports/model_performance_report.html` |
