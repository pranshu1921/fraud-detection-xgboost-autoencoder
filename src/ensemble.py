"""
ensemble.py
Combines autoencoder anomaly scores with XGBoost fraud probabilities
into a final ensemble decision.

Two ensemble strategies:
  1. Weighted average (simple, interpretable)
  2. Logistic meta-learner (slightly more powerful, still explainable)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib
import os


def weighted_ensemble(
    xgb_proba: np.ndarray,
    ae_anomaly_score: np.ndarray,
    xgb_weight: float = 0.7,
    ae_weight: float = 0.3,
) -> np.ndarray:
    """
    Simple weighted average of XGBoost probability and normalized AE score.
    XGBoost gets higher weight as it is supervised and more precise.
    AE score is normalized to [0, 1] range before combining.
    """
    ae_normalized = np.clip(ae_anomaly_score / ae_anomaly_score.max(), 0, 1)
    ensemble_score = (xgb_weight * xgb_proba) + (ae_weight * ae_normalized)
    return ensemble_score


def train_meta_learner(
    xgb_proba: np.ndarray,
    ae_anomaly_score: np.ndarray,
    y_true: pd.Series,
    model_dir: str = "models",
) -> LogisticRegression:
    """
    Train a logistic regression meta-learner on the two base model outputs.
    This learns the optimal combination weight from data.
    """
    os.makedirs(model_dir, exist_ok=True)

    ae_normalized = np.clip(
        ae_anomaly_score / (ae_anomaly_score.max() + 1e-10), 0, 1
    )

    # Stack both scores as features for the meta-learner
    meta_features = np.column_stack([xgb_proba, ae_normalized])

    meta_model = LogisticRegression(
        class_weight="balanced", random_state=42, max_iter=1000
    )

    # Cross-validate the meta-learner
    cv_scores = cross_val_score(
        meta_model, meta_features, y_true,
        cv=5, scoring="average_precision"
    )
    print(f"Meta-learner CV PR-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    meta_model.fit(meta_features, y_true)

    joblib.dump(meta_model, f"{model_dir}/meta_learner.pkl")
    print(f"Saved meta-learner to {model_dir}/meta_learner.pkl")

    return meta_model


def evaluate_ensemble(
    xgb_proba: np.ndarray,
    ae_anomaly_score: np.ndarray,
    y_true: pd.Series,
    meta_model=None,
) -> dict:
    """
    Compare all three approaches: XGBoost alone, AE alone, and ensemble.
    """
    ae_normalized = np.clip(
        ae_anomaly_score / (ae_anomaly_score.max() + 1e-10), 0, 1
    )

    # XGBoost alone
    xgb_pr_auc = average_precision_score(y_true, xgb_proba)

    # AE alone
    ae_pr_auc = average_precision_score(y_true, ae_normalized)

    # Weighted ensemble
    ensemble_score = weighted_ensemble(xgb_proba, ae_anomaly_score)
    ensemble_pr_auc = average_precision_score(y_true, ensemble_score)

    results = {
        "xgboost_only_pr_auc": xgb_pr_auc,
        "autoencoder_only_pr_auc": ae_pr_auc,
        "weighted_ensemble_pr_auc": ensemble_pr_auc,
    }

    # Meta-learner if available
    if meta_model is not None:
        meta_features = np.column_stack([xgb_proba, ae_normalized])
        meta_proba = meta_model.predict_proba(meta_features)[:, 1]
        meta_pr_auc = average_precision_score(y_true, meta_proba)
        results["meta_learner_pr_auc"] = meta_pr_auc

    print("\n--- Ensemble Comparison (PR-AUC) ---")
    for name, score in results.items():
        print(f"  {name:<35}: {score:.4f}")

    best = max(results, key=results.get)
    print(f"\nBest approach: {best} ({results[best]:.4f})")

    return results


def ensemble_predict_single(
    xgb_proba: float,
    ae_anomaly_score: float,
    ae_max_score: float,
    meta_model=None,
    xgb_weight: float = 0.7,
    ae_weight: float = 0.3,
    threshold: float = 0.5,
) -> dict:
    """
    Make an ensemble prediction for a single transaction.
    Used by the FastAPI inference endpoint.
    """
    ae_normalized = min(ae_anomaly_score / (ae_max_score + 1e-10), 1.0)

    if meta_model is not None:
        meta_features = np.array([[xgb_proba, ae_normalized]])
        ensemble_score = float(meta_model.predict_proba(meta_features)[0, 1])
    else:
        ensemble_score = (xgb_weight * xgb_proba) + (ae_weight * ae_normalized)

    is_fraud = ensemble_score >= threshold
    risk_level = (
        "HIGH" if ensemble_score >= 0.7
        else "MEDIUM" if ensemble_score >= 0.4
        else "LOW"
    )

    return {
        "xgb_fraud_probability": round(xgb_proba, 4),
        "ae_anomaly_score": round(ae_normalized, 4),
        "ensemble_score": round(ensemble_score, 4),
        "is_fraud": bool(is_fraud),
        "risk_level": risk_level,
        "decision_threshold": threshold,
    }


def load_meta_learner(model_dir: str = "models"):
    """Load saved meta-learner. Returns None if not found."""
    path = f"{model_dir}/meta_learner.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None
