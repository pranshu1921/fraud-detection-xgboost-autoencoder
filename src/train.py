"""
train.py
Master training script. Runs the full pipeline:
  1. Data validation (Great Expectations)
  2. Feature engineering
  3. Autoencoder training
  4. XGBoost training
  5. Ensemble evaluation
  6. MLflow experiment logging and model registration

Usage:
    python src/train.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import joblib

from data_validation import run_validation_pipeline
from feature_engineering import run_feature_pipeline
from autoencoder import train_autoencoder, score_transactions
from xgboost_model import train_xgboost, compute_shap_values
from ensemble import train_meta_learner, evaluate_ensemble


# Config
DATA_DIR       = "data"
MODEL_DIR      = "models"
MLFLOW_URI     = "mlruns"
EXPERIMENT_NAME = "fraud_detection_ieee_cis"

TRANSACTION_PATH = f"{DATA_DIR}/train_transaction.csv"
IDENTITY_PATH    = f"{DATA_DIR}/train_identity.csv"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Step 1: Data Validation ---
    print("=" * 60)
    print("STEP 1: Data Validation")
    print("=" * 60)
    validation_passed = run_validation_pipeline(
        TRANSACTION_PATH, IDENTITY_PATH
    )
    if not validation_passed:
        print("ERROR: Data validation failed. Fix data issues before training.")
        sys.exit(1)

    # --- Step 2: Feature Engineering ---
    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering")
    print("=" * 60)
    X, y, encoders = run_feature_pipeline(
        TRANSACTION_PATH, IDENTITY_PATH, output_dir=DATA_DIR
    )

    # --- Step 3: MLflow Setup ---
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="full_pipeline") as run:
        print(f"\nMLflow Run ID: {run.info.run_id}")

        # Log dataset stats
        mlflow.log_params({
            "dataset": "IEEE-CIS Fraud Detection",
            "n_samples": len(X),
            "n_features": X.shape[1],
            "fraud_rate": float(y.mean()),
            "train_split": 0.80,
        })

        # --- Step 4: Train Autoencoder ---
        print("\n" + "=" * 60)
        print("STEP 4: Autoencoder Training")
        print("=" * 60)
        ae_model, ae_scaler, ae_threshold = train_autoencoder(
            X, y,
            epochs=20,
            batch_size=512,
            model_dir=MODEL_DIR,
        )

        mlflow.log_params({
            "ae_epochs": 20,
            "ae_batch_size": 512,
            "ae_architecture": "380-256-128-64-32-64-128-256-380",
            "ae_threshold_percentile": 95,
        })
        mlflow.log_metric("ae_threshold", ae_threshold)
        mlflow.tensorflow.log_model(ae_model, "autoencoder")

        # Score all transactions with AE
        print("\nScoring all transactions with autoencoder...")
        ae_scores = score_transactions(X, ae_model, ae_scaler, ae_threshold)

        # --- Step 5: Train XGBoost ---
        print("\n" + "=" * 60)
        print("STEP 5: XGBoost Training")
        print("=" * 60)

        # Add AE score as a feature for XGBoost
        X_with_ae = X.copy()
        X_with_ae["ae_reconstruction_error"] = ae_scores["ae_reconstruction_error"].values
        X_with_ae["ae_anomaly_score"]         = ae_scores["ae_anomaly_score"].values

        xgb_model, xgb_metrics = train_xgboost(
            X_with_ae, y,
            use_smote=False,
            model_dir=MODEL_DIR,
        )

        mlflow.log_params({
            "xgb_n_estimators": 500,
            "xgb_max_depth": 6,
            "xgb_learning_rate": 0.05,
            "xgb_use_smote": False,
        })
        mlflow.log_metrics({
            "xgb_pr_auc":  xgb_metrics["pr_auc"],
            "xgb_roc_auc": xgb_metrics["roc_auc"],
            "xgb_fpr_at_80_recall": xgb_metrics["fpr_at_80_recall"],
        })
        mlflow.sklearn.log_model(xgb_model, "xgboost")

        # --- Step 6: Ensemble ---
        print("\n" + "=" * 60)
        print("STEP 6: Ensemble Training and Evaluation")
        print("=" * 60)

        # Use test split (last 20%) for ensemble evaluation
        split_idx = int(len(X_with_ae) * 0.8)
        X_test  = X_with_ae.iloc[split_idx:]
        y_test  = y.iloc[split_idx:]

        xgb_proba_test = xgb_model.predict_proba(X_test)[:, 1]
        ae_score_test  = ae_scores["ae_anomaly_score"].values[split_idx:]

        meta_model = train_meta_learner(
            xgb_proba_test, ae_score_test, y_test, model_dir=MODEL_DIR
        )

        ensemble_results = evaluate_ensemble(
            xgb_proba_test, ae_score_test, y_test, meta_model
        )

        mlflow.log_metrics({
            "ensemble_weighted_pr_auc":   ensemble_results["weighted_ensemble_pr_auc"],
            "ensemble_meta_pr_auc":       ensemble_results.get("meta_learner_pr_auc", 0),
            "xgboost_only_pr_auc":        ensemble_results["xgboost_only_pr_auc"],
            "autoencoder_only_pr_auc":    ensemble_results["autoencoder_only_pr_auc"],
        })

        # --- Step 7: Save ae_max_score for inference ---
        ae_max_score = float(ae_scores["ae_anomaly_score"].max())
        joblib.dump({"ae_max_score": ae_max_score}, f"{MODEL_DIR}/ae_max_score.pkl")

        # --- Step 8: Register best model in MLflow ---
        best_pr_auc = ensemble_results.get(
            "meta_learner_pr_auc",
            ensemble_results["weighted_ensemble_pr_auc"]
        )
        mlflow.log_metric("final_pr_auc", best_pr_auc)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Final ensemble PR-AUC: {best_pr_auc:.4f}")
        print(f"MLflow run:            {run.info.run_id}")
        print(f"MLflow UI:             mlflow ui --backend-store-uri {MLFLOW_URI}")
        print(f"Models saved to:       {MODEL_DIR}/")

        # Save summary for downstream use
        summary = {
            "run_id": run.info.run_id,
            "ae_threshold": ae_threshold,
            "ae_max_score": ae_max_score,
            "xgb_pr_auc": xgb_metrics["pr_auc"],
            "ensemble_pr_auc": best_pr_auc,
        }
        with open(f"{MODEL_DIR}/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\nTraining summary saved to models/training_summary.json")


if __name__ == "__main__":
    main()
