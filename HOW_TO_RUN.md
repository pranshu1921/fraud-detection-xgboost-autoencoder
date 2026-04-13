# HOW TO RUN — Fraud Detection: XGBoost + Autoencoder Ensemble

This file walks through every step from a fresh clone to a fully running system with screenshots.
Follow these steps in order. Every command is copy-paste ready.

---

## Prerequisites

Before you start, make sure you have all of these installed:

| Tool | Version | Check Command | Install Link |
|---|---|---|---|
| Python | 3.11+ | `python --version` | https://python.org |
| pip | latest | `pip --version` | included with Python |
| Docker Desktop | 4.0+ | `docker --version` | https://docker.com |
| Docker Compose | 2.0+ | `docker compose version` | included with Docker Desktop |
| Git | 2.0+ | `git --version` | https://git-scm.com |
| Kaggle account | free | — | https://kaggle.com |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/pranshu1921/fraud-detection-xgboost-autoencoder.git
cd fraud-detection-xgboost-autoencoder
```

---

## Step 2 — Create a Python Virtual Environment

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Mac / Linux:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

Your terminal prompt should now show `(venv)` at the start.

---

## Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs approximately 40 packages including TensorFlow, XGBoost, FastAPI, Streamlit, MLflow, and Evidently. Expect 3 to 5 minutes.

Verify installation:

```bash
python -c "import xgboost, tensorflow, mlflow, evidently, fastapi, streamlit; print('All packages OK')"
```

Expected output: `All packages OK`

---

## Step 4 — Download the Dataset

### Option A: Kaggle CLI (recommended)

**4a. Set up Kaggle API credentials**

1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New Token"
4. This downloads a file called `kaggle.json`
5. Move it to the right location:

```bash
# Mac / Linux
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
copy %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

**4b. Accept the competition rules**

Go to https://www.kaggle.com/c/ieee-fraud-detection and click "Join Competition" (just to accept the data terms, no submission required).

**4c. Download and extract**

```bash
mkdir -p data
kaggle competitions download -c ieee-fraud-detection -p data/
cd data
unzip ieee-fraud-detection.zip
cd ..
```

### Option B: Manual download

1. Go to https://www.kaggle.com/c/ieee-fraud-detection/data
2. Download `train_transaction.csv` and `train_identity.csv`
3. Place both files in the `data/` folder

### Verify the data

```bash
ls -lh data/
# You should see:
# train_transaction.csv   (~470 MB)
# train_identity.csv      (~26 MB)
```

---

## Step 5 — Run Data Validation

```bash
cd src
python data_validation.py
cd ..
```

Expected output:
```
Loading data for validation...
  Transactions shape: (590540, 394)
  Identity shape:     (144233, 41)

Running transaction validation...
Running identity validation...

--- Validation Summary ---
Transactions: 5/5 checks passed
Identity:     3/3 checks passed

Overall validation: PASSED
```

If validation fails, check that both CSV files are in the `data/` folder.

---

## Step 6 — Train All Models

This is the main training step. It runs the full pipeline and logs everything to MLflow.

```bash
cd src
python train.py
cd ..
```

**What happens:**
1. Features are engineered and saved to `data/X_features.parquet`
2. Autoencoder trains on non-fraud transactions (~20 epochs)
3. XGBoost trains with early stopping on PR-AUC
4. Ensemble meta-learner is trained and evaluated
5. All artifacts are saved to `models/`
6. All runs are logged to `mlruns/`

**Expected runtime:** 25 to 40 minutes on a standard laptop (CPU only).

**Expected final output:**
```
--- Ensemble Comparison (PR-AUC) ---
  xgboost_only_pr_auc              : 0.8412
  autoencoder_only_pr_auc          : 0.6103
  weighted_ensemble_pr_auc         : 0.8619
  meta_learner_pr_auc              : 0.8721

Best approach: meta_learner_pr_auc (0.8721)

Training summary saved to models/training_summary.json
```

**Verify models were saved:**

```bash
ls models/
# Expected:
# autoencoder.keras
# ae_scaler.pkl
# ae_threshold.pkl
# ae_max_score.pkl
# xgboost_model.pkl
# meta_learner.pkl
# training_summary.json
```

---

## Step 7 — View MLflow Experiment Results

```bash
mlflow ui --backend-store-uri mlruns --port 5001
```

Open http://localhost:5001 in your browser.

You will see the `fraud_detection_ieee_cis` experiment with one run logged. Click the run to see:

- All hyperparameters logged
- PR-AUC, ROC-AUC, FPR@80recall metrics
- Comparison across all three model approaches
- Saved model artifacts

**Screenshot opportunity:** Take a screenshot of the MLflow experiment page showing your metrics. This is important for your LinkedIn post.

---

## Step 8 — Run Unit Tests

```bash
pytest tests/ -v --cov=src
```

Expected output:
```
tests/test_feature_engineering.py::TestVelocityFeatures::test_adds_log_amount PASSED
tests/test_feature_engineering.py::TestVelocityFeatures::test_log_amount_nonneg PASSED
tests/test_feature_engineering.py::TestVelocityFeatures::test_adds_tx_hour PASSED
...
---------- coverage: src/feature_engineering.py: 87% ----------

14 passed in 8.32s
```

---

## Step 9 — Generate Monitoring Reports

```bash
cd src
python monitor.py
cd ..
```

Expected output:
```
Training period:    472432 transactions
Production period:  118108 transactions

Generating data drift report...
Generating model performance report...

--- Monitoring Summary ---
Data Drift Detected:    False
Drifted Features:       4 / 20
Fraud Rate (train):     0.0348
Fraud Rate (prod sim):  0.0371
Fraud Rate Delta:       +0.0023

Reports saved to reports/
Drift within acceptable range. No retraining needed.
```

Open the HTML reports:

```bash
# Mac
open reports/data_drift_report.html
open reports/model_performance_report.html

# Windows
start reports/data_drift_report.html

# Linux
xdg-open reports/data_drift_report.html
```

**Screenshot opportunity:** Take a screenshot of both Evidently reports in your browser.

---

## Step 10 — Launch the Full Stack with Docker

Make sure Docker Desktop is running before this step.

```bash
docker-compose up --build
```

This builds both Docker images and starts three containers:
- `fraud_api` — FastAPI on port 8000
- `fraud_dashboard` — Streamlit on port 8501
- `fraud_mlflow` — MLflow UI on port 5001

First build takes 5 to 10 minutes. Subsequent starts are faster.

Wait until you see:
```
fraud_api        | INFO:     Application startup complete.
fraud_dashboard  | You can now view your Streamlit app in your browser.
```

Open in your browser:

| Service | URL | What to do |
|---|---|---|
| Dashboard | http://localhost:8501 | Click "Score 100 Transactions" |
| API Docs | http://localhost:8000/docs | Try the /predict endpoint |
| MLflow | http://localhost:5001 | View experiment runs |

**Screenshot opportunity:** Take screenshots of all three browser tabs. These are your primary proof screenshots.

---

## Step 11 — Test the API Manually

With the stack running, open a new terminal and run:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 2500.0,
    "ProductCD": "C",
    "card1": 4932,
    "card4": "visa",
    "card6": "credit",
    "P_emaildomain": "protonmail.com",
    "DeviceType": "mobile"
  }'
```

You will receive a JSON response with fraud probability, risk level, and top SHAP features.

You can also use the interactive Swagger UI at http://localhost:8000/docs to test without curl.

**Screenshot opportunity:** Screenshot the Swagger UI response. This shows end-to-end inference working.

---

## Step 12 — Shut Down

```bash
# Stop the Docker stack
docker-compose down

# Deactivate the virtual environment
deactivate
```

---

## Troubleshooting

**Docker: "port already in use"**
Change the host port in docker-compose.yml. For example change `"8000:8000"` to `"8001:8000"`.

**MLflow: no experiment showing**
Make sure you ran `python src/train.py` first. The `mlruns/` folder must exist before the MLflow container starts.

**API: 503 "Models not loaded"**
The API container could not find the model files. Make sure `models/` contains all artifacts from Step 6, and that the Docker volume mount in docker-compose.yml is pointing to the correct local path.

**TensorFlow: slow training**
This is expected on CPU. For faster training, use Google Colab (free GPU) and upload the dataset there. Training takes under 5 minutes on a T4 GPU.

**Great Expectations import error**
Try `pip install great-expectations==0.18.15` explicitly. Version 1.x has breaking changes from 0.18.x.

**ImportError in train.py**
Make sure you are running from inside the `src/` directory: `cd src && python train.py`

---

## Screenshots to Capture (Summary)

Capture these before submitting or sharing:

1. Terminal output of `python train.py` showing final ensemble PR-AUC
2. MLflow UI at http://localhost:5001 showing the experiment run with metrics
3. Streamlit dashboard at http://localhost:8501 after clicking "Score 100 Transactions"
4. Streamlit detail panel showing a flagged transaction with SHAP explanation
5. FastAPI Swagger UI at http://localhost:8000/docs showing a /predict response
6. Evidently data drift report (browser screenshot)
7. Evidently model performance report (browser screenshot)
8. GitHub Actions passing CI run (after pushing to GitHub)

---

## File Locations Quick Reference

| What | Where |
|---|---|
| Raw data | `data/train_transaction.csv`, `data/train_identity.csv` |
| Processed features | `data/X_features.parquet`, `data/y_labels.parquet` |
| Encoders | `data/encoders.pkl` |
| Autoencoder model | `models/autoencoder.keras` |
| XGBoost model | `models/xgboost_model.pkl` |
| Meta-learner | `models/meta_learner.pkl` |
| Training summary | `models/training_summary.json` |
| MLflow runs | `mlruns/` |
| Evidently reports | `reports/data_drift_report.html`, `reports/model_performance_report.html` |
| CI workflow | `.github/workflows/ci.yml` |
