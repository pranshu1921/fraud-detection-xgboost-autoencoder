"""
train.py
Master training script. Runs the full pipeline:
  1. Data loading (auto-downloads IEEE-CIS via Kaggle if not present)
  2. Data validation (Great Expectations)
  3. Feature engineering
  4. Autoencoder training (PyTorch)
  5. XGBoost training
  6. Ensemble training and evaluation
  7. MLflow experiment logging and model registration

Usage:
    cd src
    python train.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import joblib

# Add src to path so imports work when running from src/
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_data
from data_validation import run_validation_pipeline
from feature_engineering import run_feature_pipeline
from autoencoder import train_autoencoder, score_transactions
from xgboost_model import train_xgboost
from ensemble import train_meta_learner, evaluate_ensemble


# -------------------------
# Config
# -------------------------

DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR       = os.path.join(os.path.dirname(__file__), "..", "models")
MLFLOW_URI = "mlruns"
EXPERIMENT_NAME = "fraud_detection_ieee_cis"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # -------------------------
    # STEP 1: Load Data
    # -------------------------
    print("=" * 60)
    print("STEP 1: Data Loading")
    print("=" * 60)
    transactions, identity = load_data(
        transaction_path=os.path.join(DATA_DIR, "train_transaction.csv"),
        identity_path=os.path.join(DATA_DIR, "train_identity.csv"),
    )

    # -------------------------
    # STEP 2: Data Validation
    # -------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Data Validation")
    print("=" * 60)
    validation_passed = run_validation_pipeline(transactions, identity)
    if not validation_passed:
        print("ERROR: Data validation failed. Fix data issues before training.")
        sys.exit(1)

    # -------------------------
    # STEP 3: Feature Engineering
    # -------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Feature Engineering")
    print("=" * 60)
    X, y, encoders = run_feature_pipeline(
        transactions,
        identity,
        output_dir=DATA_DIR,
    )

    # -------------------------
    # STEP 4: MLflow Setup
    # -------------------------
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="full_pipeline") as run:
        print(f"\nMLflow Run ID: {run.info.run_id}")

        mlflow.log_params({
            "dataset":     "IEEE-CIS Fraud Detection (Kaggle)",
            "n_samples":   len(X),
            "n_features":  X.shape[1],
            "fraud_rate":  float(y.mean()),
            "train_split": 0.80,
        })

        # -------------------------
        # STEP 4: Train Autoencoder
        # -------------------------
        print("\n" + "=" * 60)
        print("STEP 4: Autoencoder Training (PyTorch)")
        print("=" * 60)
        ae_model, ae_scaler, ae_threshold = train_autoencoder(
            X, y,
            epochs=20,
            batch_size=512,
            model_dir=MODEL_DIR,
        )

        mlflow.log_params({
            "ae_epochs":               20,
            "ae_batch_size":           512,
            "ae_architecture":         "input->256->128->64->32->64->128->256->input",
            "ae_threshold_percentile": 95,
            "ae_backend":              "pytorch",
        })
        mlflow.log_metric("ae_threshold", ae_threshold)
        mlflow.pytorch.log_model(ae_model, "autoencoder")

        # Score all transactions with autoencoder
        print("\nScoring all transactions with autoencoder...")
        ae_scores = score_transactions(
            X, ae_model, ae_scaler, ae_threshold
        )

        # -------------------------
        # STEP 5: Train XGBoost
        # -------------------------
        print("\n" + "=" * 60)
        print("STEP 5: XGBoost Training")
        print("=" * 60)

        # Add AE scores as features for XGBoost
        X_with_ae = X.copy()
        X_with_ae["ae_reconstruction_error"] = ae_scores["ae_reconstruction_error"].values
        X_with_ae["ae_anomaly_score"]         = ae_scores["ae_anomaly_score"].values

        xgb_model, xgb_metrics = train_xgboost(
            X_with_ae, y,
            use_smote=False,
            model_dir=MODEL_DIR,
        )

        mlflow.log_params({
            "xgb_n_estimators":  500,
            "xgb_max_depth":     6,
            "xgb_learning_rate": 0.05,
            "xgb_use_smote":     False,
        })
        mlflow.log_metrics({
            "xgb_pr_auc":           xgb_metrics["pr_auc"],
            "xgb_roc_auc":          xgb_metrics["roc_auc"],
            "xgb_fpr_at_80_recall": xgb_metrics["fpr_at_80_recall"],
        })
        mlflow.sklearn.log_model(xgb_model, "xgboost")

        # -------------------------
        # STEP 6: Ensemble
        # -------------------------
        print("\n" + "=" * 60)
        print("STEP 6: Ensemble Training and Evaluation")
        print("=" * 60)

        split_idx      = int(len(X_with_ae) * 0.8)
        X_test         = X_with_ae.iloc[split_idx:]
        y_test         = y.iloc[split_idx:]
        xgb_proba_test = xgb_model.predict_proba(X_test)[:, 1]
        ae_score_test  = ae_scores["ae_anomaly_score"].values[split_idx:]

        meta_model = train_meta_learner(
            xgb_proba_test,
            ae_score_test,
            y_test,
            model_dir=MODEL_DIR,
        )

        ensemble_results = evaluate_ensemble(
            xgb_proba_test,
            ae_score_test,
            y_test,
            meta_model,
        )

        mlflow.log_metrics({
            "ensemble_weighted_pr_auc": ensemble_results["weighted_ensemble_pr_auc"],
            "ensemble_meta_pr_auc":     ensemble_results.get("meta_learner_pr_auc", 0),
            "xgboost_only_pr_auc":      ensemble_results["xgboost_only_pr_auc"],
            "autoencoder_only_pr_auc":  ensemble_results["autoencoder_only_pr_auc"],
        })

        # Save ae_max_score for inference
        ae_max_score = float(ae_scores["ae_anomaly_score"].max())
        joblib.dump(
            {"ae_max_score": ae_max_score},
            os.path.join(MODEL_DIR, "ae_max_score.pkl")
        )

        # Best metric
        best_pr_auc = ensemble_results.get(
            "meta_learner_pr_auc",
            ensemble_results["weighted_ensemble_pr_auc"],
        )
        mlflow.log_metric("final_pr_auc", best_pr_auc)

        # Save training summary
        summary = {
            "run_id":          run.info.run_id,
            "ae_threshold":    ae_threshold,
            "ae_max_score":    ae_max_score,
            "xgb_pr_auc":      xgb_metrics["pr_auc"],
            "ensemble_pr_auc": best_pr_auc,
        }
        summary_path = os.path.join(MODEL_DIR, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # -------------------------
        # Done
        # -------------------------
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Final ensemble PR-AUC: {best_pr_auc:.4f}")
        print(f"MLflow Run ID:         {run.info.run_id}")
        print(f"Models saved to:       {MODEL_DIR}/")
        print(f"\nNext steps:")
        print(f"  View MLflow UI:   mlflow ui --backend-store-uri mlruns")
        print(f"  Run monitoring:   python src/monitor.py")
        print(f"  Launch API:       uvicorn api.main:app --reload")
        print(f"  Launch dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
