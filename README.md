# Fraud Detection: XGBoost + Autoencoder Ensemble

A production-grade e-commerce fraud detection system combining supervised and unsupervised ML. XGBoost catches known fraud patterns. An Autoencoder flags novel anomalies no labeled data exists for yet. An ensemble layer combines both into a single risk score served via FastAPI, monitored with Evidently AI, tracked in MLflow, and deployed with Docker Compose.

[![CI Pipeline](https://github.com/pranshu1921/fraud-detection-xgboost-autoencoder/actions/workflows/ci.yml/badge.svg)](https://github.com/pranshu1921/fraud-detection-xgboost-autoencoder/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

Standard fraud detectors fail in two ways. Supervised models miss fraud patterns they have never seen in training data. Rule-based systems generate too many false positives, blocking legitimate customers. This project addresses both failure modes in a single system.

---

## Architecture

```
IEEE-CIS Dataset (590K transactions)
        │
        ▼
Great Expectations ── Data Validation ── Fail fast on bad data
        │
        ▼
Feature Engineering ── Velocity features, log transforms, label encoding
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  Autoencoder                          XGBoost
  (unsupervised)                       (supervised)
  Trained on legit                     scale_pos_weight
  transactions only.                   handles 3.5% fraud rate.
  High reconstruction                  Early stopping on
  error = anomaly.                     PR-AUC.
        │                                  │
        └──────────┬───────────────────────┘
                   ▼
           Ensemble Layer
           (Logistic meta-learner)
           Combines both scores.
           Best PR-AUC of all three.
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
   Retraining trigger
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
| XGBoost (supervised) | ~0.84 | ~0.92 | Strong on known patterns |
| Autoencoder (unsupervised) | ~0.61 | ~0.78 | Catches novel fraud |
| Ensemble (final) | **~0.87** | **~0.93** | Best overall |

The ensemble catches approximately 23% more novel fraud patterns than XGBoost alone, at the cost of a small increase in false positives managed by the decision threshold.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Modeling | XGBoost, Keras/TensorFlow (Autoencoder), Scikit-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Data Validation | Great Expectations |
| Monitoring | Evidently AI |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Dataset | IEEE-CIS Fraud Detection (Kaggle) |

---

## Project Structure

```
fraud-detection-xgboost-autoencoder/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI: lint, test, docker build, drift check
├── api/
│   └── main.py                 # FastAPI inference endpoints
├── dashboard/
│   └── app.py                  # Streamlit analyst dashboard
├── src/
│   ├── data_validation.py      # Great Expectations checks
│   ├── feature_engineering.py  # Feature pipeline
│   ├── autoencoder.py          # Keras autoencoder model
│   ├── xgboost_model.py        # XGBoost classifier + SHAP
│   ├── ensemble.py             # Meta-learner combination layer
│   ├── train.py                # Master training script (MLflow)
│   └── monitor.py              # Evidently AI drift reports
├── tests/
│   └── test_feature_engineering.py
├── data/                       # Raw and processed data (gitignored)
├── models/                     # Saved model artifacts (gitignored)
├── reports/                    # Evidently HTML reports
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── requirements.txt
└── HOW_TO_RUN.md
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Kaggle account (free) for dataset download

### 1. Clone and install

```bash
git clone https://github.com/pranshu1921/fraud-detection-xgboost-autoencoder.git
cd fraud-detection-xgboost-autoencoder
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
# Option A: Kaggle CLI
kaggle competitions download -c ieee-fraud-detection -p data/
cd data && unzip ieee-fraud-detection.zip && cd ..

# Option B: Manual download
# Go to https://www.kaggle.com/c/ieee-fraud-detection/data
# Download train_transaction.csv and train_identity.csv
# Place both files in the data/ folder
```

### 3. Train all models

```bash
cd src
python train.py
```

This runs Great Expectations validation, engineers features, trains the Autoencoder, trains XGBoost, builds the ensemble, logs everything to MLflow, and saves all model artifacts to `models/`.

Expected runtime: 25 to 40 minutes depending on your machine.

### 4. Launch the full stack

```bash
docker-compose up --build
```

Open in your browser:

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Swagger UI | http://localhost:8000/docs |
| MLflow UI | http://localhost:5001 |

### 5. Generate monitoring reports

```bash
cd src
python monitor.py
# Open reports/data_drift_report.html and reports/model_performance_report.html
```

---

## API Usage

### Single transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 299.0,
    "ProductCD": "W",
    "card1": 13926,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "DeviceType": "desktop"
  }'
```

**Response:**

```json
{
  "transaction_id": null,
  "xgb_fraud_probability": 0.0821,
  "ae_anomaly_score": 0.4312,
  "ensemble_score": 0.1872,
  "is_fraud": false,
  "risk_level": "LOW",
  "decision_threshold": 0.5,
  "top_shap_features": [
    {
      "feature": "TransactionAmt_log",
      "value": 5.703,
      "shap_importance": 0.2341,
      "direction": "decreases"
    },
    {
      "feature": "card1_tx_count",
      "value": 14.0,
      "shap_importance": 0.1203,
      "direction": "decreases"
    },
    {
      "feature": "ae_reconstruction_error",
      "value": 0.0043,
      "shap_importance": 0.0891,
      "direction": "increases"
    }
  ],
  "ae_reconstruction_error": 0.004312
}
```

---

## Key Design Decisions

**Why temporal split instead of random split?**
Fraud data is time-ordered. Using a random train/test split causes data leakage: future fraud patterns leak into training and inflate evaluation metrics. The first 80% of transactions (by TransactionDT) are used for training and the last 20% simulate a production holdout.

**Why PR-AUC as the primary metric instead of accuracy?**
At 3.5% fraud rate, a model that predicts "not fraud" for every transaction achieves 96.5% accuracy while catching zero fraud. PR-AUC focuses on the precision-recall tradeoff which is what actually matters for a fraud system.

**Why train the Autoencoder on non-fraud transactions only?**
The Autoencoder learns a compressed representation of what normal looks like. It is never shown fraud examples. At inference, fraud transactions produce high reconstruction error because they do not fit the learned normal pattern. This is the standard approach for anomaly detection in production fraud systems.

**Why add the Autoencoder reconstruction error as a feature for XGBoost?**
This lets XGBoost learn to weight the anomaly signal together with all other features, rather than combining them with a fixed rule. The ensemble meta-learner then further optimizes the combination.

---

## Monitoring

Two Evidently AI reports are generated by `src/monitor.py`:

**Data Drift Report** compares feature distributions between the training period (first 80% of data) and a production simulation period (last 20%). If more than 30% of features show drift, a retraining warning is triggered in the CI pipeline.

**Model Performance Report** compares precision-recall metrics across both time periods, showing whether the fraud rate and model behavior have shifted. Fraud rate delta is logged as a monitoring metric.

View reports by opening the HTML files in `reports/` in any browser.

---

## CI/CD Pipeline

The GitHub Actions workflow runs on every push and PR:

1. **Lint** — black, isort, flake8
2. **Unit Tests** — pytest with coverage report
3. **Data Validation** — Great Expectations schema check
4. **Docker Build** — builds both images and runs a health check
5. **Drift Check** — runs Evidently reports and fails if drift exceeds threshold (scheduled daily at 6am UTC)

---

## Running Tests

```bash
pytest tests/ -v --cov=src
```

---

## Limitations and Future Work

This project uses the IEEE-CIS dataset which, while real-world inspired, is anonymized and static. A production system would add:

- Real-time feature computation with Redis or Feast
- Streaming ingestion via Apache Kafka
- Automated retraining via Airflow or Prefect when drift is detected
- Graph-based fraud ring detection with PyTorch Geometric
- Federated learning for cross-institutional pattern sharing

---

## Dataset

IEEE-CIS Fraud Detection | Kaggle Competition
590,540 transactions | 3.5% fraud rate | 433 features

Vesta Corporation provided the dataset for research purposes. The dataset includes transaction data from Vesta's fraud protection service and a wide range of features spanning card details, device information, time-based features, and behavioral signals.

---

## License

MIT

---

## Author

**Pranshu Kumar**
Senior Data Scientist | Production ML · GenAI · MLOps

[LinkedIn](https://www.linkedin.com/in/pranshu-kumar) | [GitHub](https://github.com/pranshu1921) | pranshukumarpremi@gmail.com
