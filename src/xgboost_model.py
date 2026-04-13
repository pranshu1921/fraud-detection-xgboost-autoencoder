"""
xgboost_model.py
Trains an XGBoost classifier for supervised fraud detection.
Uses SHAP for explainability and handles class imbalance.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import shap
import joblib
import os


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    XGBoost scale_pos_weight = count(negative) / count(positive).
    Handles class imbalance natively without resampling.
    """
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    ratio = neg / pos
    print(f"Class ratio (neg/pos): {ratio:.1f}")
    return ratio


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    use_smote: bool = False,
    model_dir: str = "models",
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Train XGBoost with hyperparameters tuned for fraud detection.
    Returns (model, evaluation_metrics).
    """
    os.makedirs(model_dir, exist_ok=True)

    # Temporal split: first 80% for training, last 20% for evaluation
    # Important: do NOT use random split for time-series fraud data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Train fraud rate: {y_train.mean():.4f}")
    print(f"Test fraud rate:  {y_test.mean():.4f}")

    if use_smote:
        print("Applying SMOTE oversampling...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {len(X_train)} samples")

    scale_pos_weight = compute_scale_pos_weight(y_train)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_prob)

    # Save model
    joblib.dump(model, f"{model_dir}/xgboost_model.pkl")
    print(f"\nSaved XGBoost model to {model_dir}/xgboost_model.pkl")

    return model, metrics


def evaluate_model(y_true: pd.Series, y_prob: np.ndarray) -> dict:
    """Compute and print standard fraud detection metrics."""
    # Use 0.5 threshold for classification report
    y_pred = (y_prob >= 0.5).astype(int)

    pr_auc  = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    print("\n--- Model Evaluation ---")
    print(f"PR-AUC (primary):  {pr_auc:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Business metric: FPR at 80% recall
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    idx_80_recall = np.argmin(np.abs(recalls - 0.80))
    threshold_at_80_recall = thresholds[min(idx_80_recall, len(thresholds) - 1)]
    y_pred_80 = (y_prob >= threshold_at_80_recall).astype(int)
    fp_at_80_recall = ((y_pred_80 == 1) & (y_true == 0)).sum()
    total_legit = (y_true == 0).sum()
    fpr_at_80_recall = fp_at_80_recall / total_legit

    print(f"\nAt 80% recall threshold ({threshold_at_80_recall:.3f}):")
    print(f"  False Positive Rate: {fpr_at_80_recall:.4f}")
    print(f"  False Positives:     {fp_at_80_recall} out of {total_legit} legit transactions")

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "fpr_at_80_recall": fpr_at_80_recall,
        "threshold_at_80_recall": float(threshold_at_80_recall),
    }


def compute_shap_values(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    sample_size: int = 1000,
) -> tuple[shap.Explainer, np.ndarray]:
    """
    Compute SHAP values for a sample of transactions.
    Returns (explainer, shap_values).
    """
    print(f"\nComputing SHAP values on {sample_size} samples...")
    X_sample = X.sample(min(sample_size, len(X)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return explainer, shap_values, X_sample


def get_top_shap_features(
    explainer: shap.Explainer,
    X_single: pd.DataFrame,
    top_n: int = 3,
) -> list[dict]:
    """
    Get the top N SHAP features for a single transaction.
    Used by the FastAPI endpoint for explainability output.
    """
    shap_vals = explainer.shap_values(X_single)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # Take fraud class for binary classification

    feature_names = X_single.columns.tolist()
    shap_series = pd.Series(
        np.abs(shap_vals[0]), index=feature_names
    ).sort_values(ascending=False)

    top_features = []
    for feat, importance in shap_series.head(top_n).items():
        top_features.append({
            "feature": feat,
            "value": float(X_single[feat].values[0]),
            "shap_importance": float(importance),
            "direction": "increases" if shap_vals[0][feature_names.index(feat)] > 0 else "decreases",
        })

    return top_features


def load_xgboost_model(model_dir: str = "models") -> xgb.XGBClassifier:
    """Load saved XGBoost model."""
    return joblib.load(f"{model_dir}/xgboost_model.pkl")
