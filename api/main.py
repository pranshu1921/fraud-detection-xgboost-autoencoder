"""
api/main.py
FastAPI inference endpoint for fraud detection.

Endpoints:
  POST /predict       - Score a single transaction
  POST /predict/batch - Score a batch of transactions
  GET  /health        - Health check
  GET  /model/info    - Model metadata
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib

# Add src to path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ensemble import ensemble_predict_single, load_meta_learner
from xgboost_model import get_top_shap_features
import shap


# -------------------------
# App Setup
# -------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="Two-stage fraud detection using XGBoost + Autoencoder ensemble",
    version="1.0.0",
)

# -------------------------
# Model Loading
# -------------------------

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "models")
)

def load_models():
    """Load all model artifacts at startup."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

        from autoencoder import load_autoencoder_artifacts, FraudAutoencoder
        from xgboost_model import load_xgboost_model
        from ensemble import load_meta_learner
        import json

        models = {}

        models["xgb"] = load_xgboost_model(MODEL_DIR)
        ae_model, ae_scaler, ae_threshold = load_autoencoder_artifacts(MODEL_DIR)
        models["ae"]           = ae_model
        models["ae_scaler"]    = ae_scaler
        models["ae_threshold"] = ae_threshold
        models["encoders"]     = joblib.load(
            os.path.join(os.path.dirname(__file__), "..", "data", "encoders.pkl")
        )
        models["meta"]         = load_meta_learner(MODEL_DIR)

        max_score_data = joblib.load(f"{MODEL_DIR}/ae_max_score.pkl")
        models["ae_max_score"] = max_score_data["ae_max_score"]

        import shap
        models["shap_explainer"] = shap.TreeExplainer(models["xgb"])

        with open(f"{MODEL_DIR}/training_summary.json") as f:
            models["training_summary"] = json.load(f)

        print("All models loaded successfully.")
        return models

    except Exception as e:
        print(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


# Load at startup
MODELS = load_models()


# -------------------------
# Request / Response Schemas
# -------------------------

class TransactionInput(BaseModel):
    TransactionAmt: float = Field(..., description="Transaction amount in USD", example=49.0)
    ProductCD:      str   = Field("W", description="Product code", example="W")
    card1:          int   = Field(0,   description="Card identifier 1", example=13926)
    card4:          Optional[str] = Field(None, description="Card network", example="visa")
    card6:          Optional[str] = Field(None, description="Card type", example="debit")
    P_emaildomain:  Optional[str] = Field(None, description="Purchaser email domain", example="gmail.com")
    addr1:          Optional[float] = Field(None, description="Billing zip code area", example=315.0)
    DeviceType:     Optional[str]  = Field(None, description="Device type", example="desktop")
    # Additional Vesta features (optional, default to 0)
    C1:  Optional[float] = Field(0.0)
    C2:  Optional[float] = Field(0.0)
    C6:  Optional[float] = Field(0.0)
    C11: Optional[float] = Field(0.0)
    D1:  Optional[float] = Field(0.0)
    V1:  Optional[float] = Field(0.0)
    V2:  Optional[float] = Field(0.0)
    V3:  Optional[float] = Field(0.0)


class FraudPrediction(BaseModel):
    transaction_id:        Optional[str]
    xgb_fraud_probability: float
    ae_anomaly_score:      float
    ensemble_score:        float
    is_fraud:              bool
    risk_level:            str
    decision_threshold:    float
    top_shap_features:     list
    ae_reconstruction_error: float


class HealthResponse(BaseModel):
    status:     str
    models_loaded: bool
    model_version: str


# -------------------------
# Helper Functions
# -------------------------

def preprocess_single(transaction: dict, encoders: dict) -> pd.DataFrame:
    """
    Convert a transaction dict to a full feature DataFrame
    matching the exact columns XGBoost was trained on.
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    # Load the feature column list from the saved parquet
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    X_ref = pd.read_parquet(
        os.path.join(data_dir, "X_features.parquet"),
        columns=None
    )
    # Get column names without ae columns (those get added after)
    base_cols = [
        c for c in X_ref.columns
        if c not in ["ae_reconstruction_error", "ae_anomaly_score"]
    ]

    # Start with a zero row matching all base columns
    row = {col: 0.0 for col in base_cols}

    # Fill in the values from the transaction input
    for key, val in transaction.items():
        if key in row and val is not None:
            row[key] = val

    df = pd.DataFrame([row])

    # Encode categorical columns using saved encoders
    cat_cols = [
        "ProductCD", "card4", "card6",
        "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo",
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    ]
    for col in cat_cols:
        if col in df.columns and col in encoders:
            le = encoders[col]
            val = str(df[col].values[0])
            known = set(le.classes_)
            if val not in known:
                val = le.classes_[0]  # fall back to first known class
            df[col] = le.transform([val])[0]
        elif col in df.columns:
            df[col] = 0

    # Apply same engineered features as training
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    df["amt_deviation_from_card_mean"] = 0.0
    df["card1_tx_count"]  = 1.0
    df["email_tx_count"]  = 1.0
    df["addr1_tx_count"]  = 1.0
    df["tx_hour"] = 12.0
    df["tx_day"]  = 0.0
    df["is_night"] = 0.0

    # Fill any nulls
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(0.0)

    return df[base_cols]


# -------------------------
# Endpoints
# -------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API and models are ready."""
    return HealthResponse(
        status="healthy" if MODELS else "degraded",
        models_loaded=bool(MODELS),
        model_version=MODELS.get("training_summary", {}).get("run_id", "unknown"),
    )


@app.get("/model/info")
def model_info():
    """Return model training metadata."""
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return MODELS.get("training_summary", {})


@app.post("/predict", response_model=FraudPrediction)
def predict(transaction: TransactionInput, transaction_id: Optional[str] = None):
    """
    Score a single transaction for fraud.
    Returns fraud probability, anomaly score, ensemble decision, and top SHAP features.
    """
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        tx_dict = transaction.model_dump()
        X_single = preprocess_single(tx_dict, MODELS["encoders"])

        # --- Autoencoder Score ---
        import torch
        X_scaled = MODELS["ae_scaler"].transform(X_single).astype(np.float32)
        X_tensor = torch.tensor(X_scaled)
        with torch.no_grad():
            X_reconstructed = MODELS["ae"](X_tensor).numpy()
        ae_error = float(np.mean(np.power(X_scaled - X_reconstructed, 2)))
        ae_score = ae_error / (MODELS["ae_threshold"] + 1e-10)

        # --- XGBoost Score ---
        # Add AE features to match training feature set
        X_single["ae_reconstruction_error"] = ae_error
        X_single["ae_anomaly_score"] = ae_score

        xgb_proba = float(MODELS["xgb"].predict_proba(X_single)[0, 1])

        # --- Ensemble Decision ---
        result = ensemble_predict_single(
            xgb_proba=xgb_proba,
            ae_anomaly_score=ae_score,
            ae_max_score=MODELS["ae_max_score"],
            meta_model=MODELS.get("meta"),
        )

        # --- SHAP Explanation ---
        top_features = get_top_shap_features(
            MODELS["shap_explainer"], X_single, top_n=3
        )

        return FraudPrediction(
            transaction_id=transaction_id,
            xgb_fraud_probability=result["xgb_fraud_probability"],
            ae_anomaly_score=result["ae_anomaly_score"],
            ensemble_score=result["ensemble_score"],
            is_fraud=result["is_fraud"],
            risk_level=result["risk_level"],
            decision_threshold=result["decision_threshold"],
            top_shap_features=top_features,
            ae_reconstruction_error=ae_error,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(transactions: list[TransactionInput]):
    """Score a batch of transactions. Returns list of predictions."""
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if len(transactions) > 500:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 500 transactions"
        )

    results = []
    for i, tx in enumerate(transactions):
        try:
            pred = predict(tx, transaction_id=str(i))
            results.append(pred)
        except Exception as e:
            results.append({"error": str(e), "index": i})

    return {"predictions": results, "count": len(results)}
