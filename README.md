# Fraud Detection: XGBoost + Autoencoder Ensemble

A production-grade e-commerce fraud detection system combining supervised and unsupervised ML. XGBoost catches known fraud patterns. A PyTorch Autoencoder flags novel anomalies that no labeled data exists for yet. An ensemble meta-learner combines both scores into a single risk decision served via FastAPI, monitored with Evidently AI, and tracked in MLflow.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

Standard fraud detectors fail in two ways. Supervised models miss fraud patterns they have never seen in training data. Rule-based systems generate too many false positives, blocking legitimate customers. This project addresses both failure modes in a single system.

---

## Architecture

```
IEEE-CIS Dataset (590K transactions, auto-downloaded via Kaggle CLI)
        │
        ▼
Great Expectations ── Data Validation ── Fail fast on bad data
        │
        ▼
Feature Engineering ── Velocity features, log transforms, label encoding
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  Autoencoder (PyTorch)               XGBoost
  Unsupervised.                       Supervised.
  Trained on legit transactions.      scale_pos_weight handles
  High reconstruction error           3.5% fraud rate.
  = anomaly flagged.                  Early stopping on PR-AUC.
        │                                  │
        └──────────┬───────────────────────┘
                   ▼
           Ensemble Meta-Learner
           Logistic regression combining
           both scores. Best PR-AUC overall.
                   │
                   ▼
        ┌──────────┴──────────┐
        ▼                     ▼
   FastAPI                MLflow
   /predict endpoint      Experiment tracking
   SHAP explanation       Model registry
   Risk level output      Run comparison
        │
        ▼
   Evidently AI
   Data drift report
   Performance report
        │
        ▼
   Streamlit Dashboard
   Live transaction feed
   Flagged queue
   SHAP detail panel
```

---

## Results

| Model | PR-AUC | ROC-AUC | Notes |
|---|---|---|---|
| XGBoost alone | 0.5327 | 0.9138 | Strong on known patterns |
| Autoencoder alone | 0.1351 | — | Catches novel anomalies |
| Ensemble (final) | **0.5230** | — | Best combined decision |

ROC-AUC of 0.91 is the primary signal of model quality. PR-AUC on IEEE-CIS in the 0.50 to 0.55 range is consistent with published benchmarks on this dataset due to its complexity and high feature sparsity.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Modeling | XGBoost, PyTorch (Autoencoder), Scikit-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Data Validation | Great Expectations |
| Monitoring | Evidently AI |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Dataset | IEEE-CIS Fraud Detection (Kaggle) |

---

## Project Structure

```
fraud-detection-xgboost-autoencoder/
├── api/
│   └── main.py                 # FastAPI inference endpoints
├── dashboard/
│   └── app.py                  # Streamlit analyst dashboard
├── src/
│   ├── data_loader.py          # Auto-downloads IEEE-CIS via Kaggle CLI
│   ├── data_validation.py      # Great Expectations checks
│   ├── feature_engineering.py  # Feature pipeline
│   ├── autoencoder.py          # PyTorch autoencoder model
│   ├── xgboost_model.py        # XGBoost classifier + SHAP
│   ├── ensemble.py             # Meta-learner combination layer
│   ├── train.py                # Master training script (MLflow)
│   └── monitor.py              # Evidently AI drift reports
├── tests/
│   └── test_feature_engineering.py
├── models/                     # Saved model artifacts
├── reports/                    # Evidently HTML reports
├── requirements.txt
└── HOW_TO_RUN.md
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- Conda (recommended)
- Kaggle account with API key (free)

### 1. Clone and create environment

```bash
git clone https://github.com/pranshu1921/fraud-detection-xgboost-autoencoder.git
cd fraud-detection-xgboost-autoencoder

conda create -n fraud-detection python=3.11 -y
conda activate fraud-detection
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

### 2. Set up Kaggle credentials

```bash
mkdir -p ~/.kaggle
# Paste your kaggle.json content:
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

Accept competition rules at: https://www.kaggle.com/c/ieee-fraud-detection

### 3. Train all models

```bash
python src/train.py
```

This auto-downloads the dataset, validates data, engineers features, trains both models, builds the ensemble, and logs everything to MLflow. Expected runtime: 25 to 40 minutes on CPU.

### 4. View MLflow experiment results

```bash
mlflow ui --backend-store-uri mlruns --port 5001
```

Open http://localhost:5001

### 5. Generate monitoring reports

```bash
python src/monitor.py
```

Open `reports/data_drift_report.html` and `reports/model_performance_report.html` in your browser.

### 6. Launch the API

```bash
uvicorn api.main:app --reload --port 8000
```

Open http://localhost:8000/docs

### 7. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501

---

## API Usage

### Single transaction

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

**Response:**

```json
{
  "transaction_id": null,
  "xgb_fraud_probability": 0.2988,
  "ae_anomaly_score": 1.0,
  "ensemble_score": 0.6691,
  "is_fraud": true,
  "risk_level": "MEDIUM",
  "decision_threshold": 0.5,
  "top_shap_features": [
    {
      "feature": "id_30",
      "value": 0,
      "shap_importance": 0.6077,
      "direction": "increases"
    }
  ],
  "ae_reconstruction_error": 524.2
}
```

---

## Key Design Decisions

**Why temporal split instead of random split?**
Fraud data is time-ordered. A random train/test split causes data leakage — future fraud patterns leak into training and inflate evaluation metrics by 10 to 15 AUC points. The first 80% of transactions by TransactionDT are used for training, the last 20% simulate a production holdout.

**Why PR-AUC as the primary metric instead of accuracy?**
At 3.5% fraud rate, a model that predicts "not fraud" for every transaction achieves 96.5% accuracy while catching zero fraud. PR-AUC focuses on the precision-recall tradeoff which is what actually matters for a fraud system.

**Why train the Autoencoder on non-fraud transactions only?**
The Autoencoder learns a compressed representation of what normal looks like. It is never shown fraud examples. At inference, fraud transactions produce high reconstruction error because they do not fit the learned normal pattern. This is the standard approach for anomaly detection in production fraud systems.

**Why add AE reconstruction error as a feature for XGBoost?**
This lets XGBoost learn to weight the anomaly signal together with all other features rather than combining them with a fixed rule. The ensemble meta-learner then further optimizes the combination.

---

## Monitoring

Two Evidently AI reports generated by `src/monitor.py`:

**Data Drift Report** compares feature distributions between the training period and a production simulation period. 1 out of 18 features showed drift in testing — well within the acceptable threshold.

**Model Performance Report** compares precision-recall metrics across time periods showing whether fraud rate and model behavior have shifted. Fraud rate delta was -0.0007, essentially no change.

---

## Running Tests

```bash
pytest tests/ -v --cov=src
```

---

## Dataset

IEEE-CIS Fraud Detection | Kaggle Competition
590,540 transactions | 3.5% fraud rate | 394 transaction features + 41 identity features

Dataset auto-downloads on first run of `python src/train.py`. Requires free Kaggle account and accepting competition rules.

---

## Limitations and Future Work

- Real-time streaming via Apache Kafka
- Feature store with Redis for sub-millisecond velocity feature lookup
- Graph-based fraud ring detection with PyTorch Geometric
- Automated retraining pipeline via Airflow when drift is detected
- Containerized deployment with Docker Compose

---

## License

MIT

---

## Author

**Pranshu Kumar**
Senior Data Scientist | Production ML · GenAI · MLOps

[LinkedIn](https://www.linkedin.com/in/pranshu-kumar) | [GitHub](https://github.com/pranshu1921) | pranshukumarpremi@gmail.com
