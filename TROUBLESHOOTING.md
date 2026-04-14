# TROUBLESHOOTING

Common issues and exact fixes encountered during development and testing.
Covers Windows, AMD CPUs, GitBash, and package conflicts.

---

## Python and Environment Issues

### Segmentation fault when running Python in GitBash

**Symptom:**
```
$ python train.py
Segmentation fault
```

**Cause:** GitBash on Windows does not handle DLL calls correctly for compiled packages like PyTorch and XGBoost.

**Fix:** Never run Python in GitBash. Use Anaconda Prompt or Command Prompt for all Python commands. Use GitBash for Git commands only.

```
GitBash    → git add, git commit, git push
CMD/Anaconda Prompt → python, pip, uvicorn, streamlit, mlflow
```

---

### `conda` is not recognized in Command Prompt

**Symptom:**
```
'conda' is not recognized as an internal or external command
```

**Fix:** Open **Anaconda Prompt** instead of Command Prompt. Or run this once in Anaconda Prompt to add conda to PATH permanently:

```cmd
conda init cmd.exe
```

Then close and reopen Command Prompt.

---

### `pkg_resources` not found

**Symptom:**
```
ModuleNotFoundError: No module named 'pkg_resources'
```

**Fix:**
```bash
conda install setuptools -y
```

Verify:
```bash
python -c "import pkg_resources; print('OK')"
```

The deprecation warning that appears is harmless.

---

### TensorFlow DLL error or segfault on import

**Symptom:**
```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal
```
or
```
Segmentation fault
```

**Fix:** This project uses PyTorch, not TensorFlow. If you see this error it means the old TensorFlow version of `autoencoder.py` is still in your `src/` folder. Download the latest `autoencoder.py` from the repo which uses PyTorch.

---

### numpy and tensorflow version conflict

**Symptom:**
```
ERROR: Cannot install because these package versions have conflicting dependencies.
tensorflow-intel requires numpy<1.24
```

**Fix:** This project does not use TensorFlow. Remove it:

```bash
pip uninstall tensorflow tensorflow-intel tensorflow-cpu -y
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

---

## Kaggle and Data Issues

### 401 Unauthorized when running `kaggle competitions list`

**Symptom:**
```
401 - Unauthorized - Unauthenticated
```

**Cause:** `kaggle.json` is missing, in the wrong location, or has incorrect content.

**Fix:**

1. Go to https://www.kaggle.com/settings/account
2. Click **Create New API Token** (not "Create New Token" — those are different)
3. Move the downloaded file:

```bash
mkdir -p ~/.kaggle
cp /c/Users/YOUR_USERNAME/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

4. Verify:
```bash
cat ~/.kaggle/kaggle.json
# Should print: {"username":"...","key":"..."}
```

---

### Kaggle download shows "Skipping, found more recently modified local copy"

**Symptom:**
```
ieee-fraud-detection.zip: Skipping, found more recently modified local copy
```

**Cause:** A zip file already exists in the target folder.

**Fix:** This is not an error. The download was skipped because the file already exists. The extraction will still run. If the CSV files are missing after extraction, delete the zip and re-download:

```bash
rm data/ieee-fraud-detection.zip
kaggle competitions download -c ieee-fraud-detection -p data/
```

---

### CSV files extracted to wrong folder

**Symptom:**
```
FileNotFoundError: train_transaction.csv not found at data/train_transaction.csv
```

**Cause:** Running `train.py` from inside `src/` instead of the project root causes relative paths to resolve incorrectly.

**Fix:** Always run from the project root:

```bash
cd P:/fraud-detection-xgboost-autoencoder
python src/train.py
```

If files ended up in `src/data/`, move them:

```bash
mv src/data/train_transaction.csv data/train_transaction.csv
mv src/data/train_identity.csv data/train_identity.csv
rm -rf src/data/
```

---

## Training Issues

### ValueError: Input X contains infinity

**Symptom:**
```
ValueError: Input X contains infinity or a value too large for dtype('float64')
```

**Cause:** The `amt_deviation_from_card_mean` feature divides by standard deviation which is zero for cards with only one transaction, producing infinity.

**Fix:** Make sure `fill_missing_values` in `feature_engineering.py` replaces infinity before filling nulls:

```python
def fill_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df
```

---

### MLflow UnsupportedModelRegistryStoreURIException

**Symptom:**
```
UnsupportedModelRegistryStoreURIException: got unsupported URI 'P:\...\mlruns'
```

**Cause:** MLflow does not accept Windows absolute paths starting with a drive letter.

**Fix:** In `src/train.py`, make sure `MLFLOW_URI` is set to a simple relative path:

```python
MLFLOW_URI = "mlruns"
```

Not an `os.path.join` with `__file__`.

---

### PR-AUC seems low (~0.52)

**This is not a bug.** The IEEE-CIS dataset is one of the most challenging fraud detection datasets publicly available. The ROC-AUC of ~0.91 confirms strong discriminative power. PR-AUC in the 0.50 to 0.55 range is consistent with published academic benchmarks on this dataset due to extreme class imbalance (3.5% fraud rate) and complex anonymized features.

---

## API Issues

### FastAPI returns 503 on /predict and /model/info

**Symptom:**
```json
{"detail": "Models not loaded"}
```

**Cause:** Model files could not be loaded at startup.

**Fix:** Run this diagnostic to find which model is failing:

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from autoencoder import load_autoencoder_artifacts
from xgboost_model import load_xgboost_model
from ensemble import load_meta_learner
import json

xgb = load_xgboost_model('models')
print('XGBoost OK')
ae, scaler, threshold = load_autoencoder_artifacts('models')
print('Autoencoder OK')
meta = load_meta_learner('models')
print('Meta-learner OK')
with open('models/training_summary.json') as f:
    print('Summary OK:', json.load(f))
"
```

Make sure you are running uvicorn from the project root:

```bash
cd P:/fraud-detection-xgboost-autoencoder
uvicorn api.main:app --reload --port 8000
```

---

### FastAPI /predict returns 500 with feature names mismatch

**Symptom:**
```json
{"detail": "feature_names mismatch: expected ae_anomaly_score, ae_reconstruction_error in input data"}
```

**Cause:** The `preprocess_single` function in `api/main.py` is not adding the AE feature columns.

**Fix:** Make sure `preprocess_single` loads the reference feature columns from `data/X_features.parquet` and adds zero values for `ae_reconstruction_error` and `ae_anomaly_score` before passing to XGBoost.

---

### FastAPI /predict returns "previously unseen labels"

**Symptom:**
```json
{"detail": "y contains previously unseen labels: 'missing'"}
```

**Cause:** Label encoder does not have a "missing" class.

**Fix:** In `preprocess_single`, fall back to the first known class instead of "missing":

```python
if val not in known:
    val = le.classes_[0]
```

---

### FastAPI /predict returns "FraudAutoencoder has no attribute predict"

**Symptom:**
```json
{"detail": "'FraudAutoencoder' object has no attribute 'predict'"}
```

**Cause:** The API is calling `.predict()` which is a Keras method. This model uses PyTorch.

**Fix:** In `api/main.py`, replace the autoencoder inference block with:

```python
import torch
X_scaled = MODELS["ae_scaler"].transform(X_single).astype(np.float32)
X_tensor = torch.tensor(X_scaled)
with torch.no_grad():
    X_reconstructed = MODELS["ae"](X_tensor).numpy()
ae_error = float(np.mean(np.power(X_scaled - X_reconstructed, 2)))
```

---

## Monitoring Issues

### Evidently ImportError: DatasetMissingValuesSummaryMetric

**Symptom:**
```
ImportError: cannot import name 'DatasetMissingValuesSummaryMetric' from 'evidently.metrics'
```

**Cause:** Evidently changed their API in recent versions.

**Fix:** Remove `DatasetMissingValuesSummaryMetric` from the import and from the Report metrics list in `src/monitor.py`. Use only `DataDriftPreset()` and `ClassificationPreset()`.

---

### monitor.py fails with XGBoost feature names mismatch

**Symptom:**
```
ValueError: feature_names mismatch: expected ae_anomaly_score, ae_reconstruction_error
```

**Fix:** Add these lines at the start of `generate_model_performance_report` after loading the XGBoost model:

```python
X_train["ae_reconstruction_error"] = 0.0
X_train["ae_anomaly_score"] = 0.0
X_prod["ae_reconstruction_error"] = 0.0
X_prod["ae_anomaly_score"] = 0.0
```

---

## General Tips

**Always run Python commands from the project root:**
```bash
cd P:/fraud-detection-xgboost-autoencoder
python src/train.py        # correct
python src/monitor.py      # correct

cd src
python train.py            # can cause path issues
```

**Always activate the conda environment first:**
```bash
conda activate fraud-detection
```

**If something breaks after a fresh clone, run in this exact order:**
1. `pip install setuptools`
2. `pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple`
3. `python src/train.py` from the project root
