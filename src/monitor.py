"""
monitor.py
Generates Evidently AI drift monitoring reports.

Two reports:
  1. Data drift: compares feature distributions between
     training period and a "production" holdout period.
  2. Model performance: compares precision-recall metrics
     across time periods.

Usage:
    python src/monitor.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesSummaryMetric,
)


DATA_DIR  = "data"
MODEL_DIR = "models"
REPORT_DIR = "reports"


def load_data_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load features and split into training (first 80%) and
    production simulation (last 20%) periods.
    """
    X = pd.read_parquet(f"{DATA_DIR}/X_features.parquet")
    y = pd.read_parquet(f"{DATA_DIR}/y_labels.parquet")["isFraud"]

    split_idx = int(len(X) * 0.8)

    X_train = X.iloc[:split_idx].copy()
    X_prod  = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_prod  = y.iloc[split_idx:].copy()

    print(f"Training period:    {len(X_train)} transactions")
    print(f"Production period:  {len(X_prod)} transactions")

    return X_train, X_prod, y_train, y_prod


def generate_data_drift_report(
    X_train: pd.DataFrame,
    X_prod: pd.DataFrame,
    output_path: str,
    n_features: int = 20,
) -> dict:
    """
    Generate Evidently data drift report.
    Compares the top N most important features between time periods.
    Returns a dict with drift summary.
    """
    # Use a subset of features for readability
    # Prioritize the engineered features we know are meaningful
    priority_features = [
        "TransactionAmt", "TransactionAmt_log", "card1",
        "amt_deviation_from_card_mean", "card1_tx_count",
        "email_tx_count", "addr1_tx_count", "tx_hour", "tx_day",
        "ae_reconstruction_error", "ae_anomaly_score",
        "ProductCD", "card4", "card6", "P_emaildomain",
        "C1", "C2", "C6", "C11", "D1",
    ]

    # Filter to features that exist in both datasets
    features_to_use = [f for f in priority_features if f in X_train.columns]
    features_to_use = features_to_use[:n_features]

    reference = X_train[features_to_use].sample(
        min(5000, len(X_train)), random_state=42
    )
    current = X_prod[features_to_use].sample(
        min(5000, len(X_prod)), random_state=42
    )

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetMissingValuesSummaryMetric(),
    ])

    report.run(reference_data=reference, current_data=current)
    report.save_html(output_path)

    # Extract drift summary
    report_dict = report.as_dict()
    drift_metric = next(
        (m for m in report_dict["metrics"]
         if m["metric"] == "DatasetDriftMetric"),
        None
    )

    summary = {
        "report_path": output_path,
        "n_features_checked": len(features_to_use),
    }

    if drift_metric:
        result = drift_metric.get("result", {})
        summary["drift_detected"] = result.get("dataset_drift", False)
        summary["share_drifted"]  = result.get("share_of_drifted_columns", 0)
        summary["n_drifted"]      = result.get("number_of_drifted_columns", 0)

    return summary


def generate_model_performance_report(
    X_train: pd.DataFrame,
    X_prod: pd.DataFrame,
    y_train: pd.Series,
    y_prod: pd.Series,
    output_path: str,
) -> dict:
    """
    Generate Evidently model performance report.
    Compares predictions on training vs production period.
    Requires the trained XGBoost model.
    """
    import xgboost as xgb

    xgb_model = joblib.load(f"{MODEL_DIR}/xgboost_model.pkl")

    # Get predictions for both periods
    ref_proba = xgb_model.predict_proba(X_train.sample(
        min(5000, len(X_train)), random_state=42
    ))[:, 1]
    cur_proba = xgb_model.predict_proba(X_prod.sample(
        min(5000, len(X_prod)), random_state=42
    ))[:, 1]

    # Build DataFrames with predictions for Evidently
    ref_df = X_train.sample(min(5000, len(X_train)), random_state=42).copy()
    cur_df = X_prod.sample(min(5000, len(X_prod)), random_state=42).copy()

    ref_df["prediction"] = (ref_proba >= 0.5).astype(int)
    cur_df["prediction"] = (cur_proba >= 0.5).astype(int)

    ref_y = y_train.iloc[:len(ref_df)].values
    cur_y = y_prod.iloc[:len(cur_df)].values

    ref_df["target"] = ref_y
    cur_df["target"] = cur_y

    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=ref_df[["prediction", "target"]],
        current_data=cur_df[["prediction", "target"]],
    )
    report.save_html(output_path)

    summary = {
        "report_path": output_path,
        "reference_fraud_rate": float(y_train.mean()),
        "current_fraud_rate": float(y_prod.mean()),
        "fraud_rate_delta": float(y_prod.mean() - y_train.mean()),
    }

    return summary


def run_monitoring_pipeline():
    """Run both monitoring reports and print summary."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Loading data splits...")
    X_train, X_prod, y_train, y_prod = load_data_splits()

    print("\nGenerating data drift report...")
    drift_summary = generate_data_drift_report(
        X_train, X_prod,
        output_path=f"{REPORT_DIR}/data_drift_report.html",
    )

    print("\nGenerating model performance report...")
    perf_summary = generate_model_performance_report(
        X_train, X_prod, y_train, y_prod,
        output_path=f"{REPORT_DIR}/model_performance_report.html",
    )

    # Print summaries
    print("\n--- Monitoring Summary ---")
    print(f"Data Drift Detected:    {drift_summary.get('drift_detected', 'N/A')}")
    print(f"Drifted Features:       {drift_summary.get('n_drifted', 'N/A')} / {drift_summary.get('n_features_checked', 'N/A')}")
    print(f"Fraud Rate (train):     {perf_summary['reference_fraud_rate']:.4f}")
    print(f"Fraud Rate (prod sim):  {perf_summary['current_fraud_rate']:.4f}")
    print(f"Fraud Rate Delta:       {perf_summary['fraud_rate_delta']:+.4f}")

    print(f"\nReports saved to {REPORT_DIR}/")
    print(f"  {REPORT_DIR}/data_drift_report.html")
    print(f"  {REPORT_DIR}/model_performance_report.html")

    # Save monitoring summary as JSON for CI/CD consumption
    full_summary = {**drift_summary, **perf_summary}
    with open(f"{REPORT_DIR}/monitoring_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)

    # Trigger warning if significant drift detected
    if drift_summary.get("share_drifted", 0) > 0.3:
        print("\nWARNING: >30% of features show drift. Consider retraining.")
    else:
        print("\nDrift within acceptable range. No retraining needed.")

    return full_summary


if __name__ == "__main__":
    run_monitoring_pipeline()
