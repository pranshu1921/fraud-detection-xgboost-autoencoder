"""
feature_engineering.py
Builds features from the IEEE-CIS Fraud Detection dataset.

Steps:
  1. Merge transaction and identity DataFrames on TransactionID
  2. Engineer velocity and time-based features
  3. Label encode categorical columns
  4. Fill missing values with median
  5. Save feature matrix to parquet

Usage:
    from feature_engineering import run_feature_pipeline
    X, y, encoders = run_feature_pipeline(transactions, identity)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os


# Columns to drop — IDs and raw time offset not useful as features
DROP_COLS = ["TransactionID", "TransactionDT"]

# Categorical columns to label encode
CAT_COLS = [
    "ProductCD",
    "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28",
    "id_29", "id_30", "id_31", "id_33", "id_34",
    "id_35", "id_36", "id_37", "id_38",
]


def merge_datasets(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left join identity onto transactions using TransactionID.
    Transactions without identity data keep NaN for identity columns.
    """
    print("Merging transaction and identity datasets...")
    df = transactions.merge(identity, on="TransactionID", how="left")
    print(f"Merged shape: {df.shape}")
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features based on transaction patterns.
    """
    print("Engineering features...")

    # Sort by time to preserve temporal order
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Time-based features
    # TransactionDT is seconds offset from a reference point
    df["tx_hour"] = (df["TransactionDT"] // 3600) % 24
    df["tx_day"]  = (df["TransactionDT"] // 86400) % 7
    df["is_night"] = (
        (df["tx_hour"] >= 22) | (df["tx_hour"] <= 5)
    ).astype(int)

    # Amount features
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # Amount deviation from card mean
    # Is this transaction unusually large or small for this card?
    card_mean = df.groupby("card1")["TransactionAmt"].transform("mean")
    card_std  = df.groupby("card1")["TransactionAmt"].transform("std").fillna(1)
    df["amt_deviation_from_card_mean"] = (
        (df["TransactionAmt"] - card_mean) / card_std
    )

    # Transaction count per card (proxy for velocity)
    df["card1_tx_count"] = df.groupby("card1")["TransactionAmt"].transform("count")

    # Transaction count per email domain
    df["email_tx_count"] = df.groupby(
        "P_emaildomain"
    )["TransactionAmt"].transform("count")

    # Transaction count per billing address
    df["addr1_tx_count"] = df.groupby(
        "addr1"
    )["TransactionAmt"].transform("count")

    return df


def encode_categoricals(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: dict = None,
) -> tuple:
    """
    Label encode categorical columns.
    fit=True  : fit new encoders (use during training)
    fit=False : use existing encoders (use during inference)
    Returns (df, encoders_dict).
    """
    if encoders is None:
        encoders = {}

    for col in CAT_COLS:
        if col not in df.columns:
            continue

        # Fill nulls before encoding
        df[col] = df[col].astype(str).fillna("missing")

        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                df[col] = 0
            else:
                known = set(le.classes_)
                df[col] = df[col].apply(
                    lambda x: x if x in known else "missing"
                )
                df[col] = le.transform(df[col])

    return df, encoders


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values and replace infinity in numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Replace infinity with NaN first, then fill with median
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    fit_encoders: bool = True,
    encoders: dict = None,
) -> tuple:
    """
    Full feature engineering pipeline on merged DataFrame.
    Returns (X, y, encoders).
    """
    df = add_velocity_features(df)
    df, encoders = encode_categoricals(df, fit=fit_encoders, encoders=encoders)
    df = fill_missing_values(df)

    # Separate target
    y = df["isFraud"].astype(int)

    # Drop columns not used as features
    cols_to_drop = [
        c for c in DROP_COLS + ["isFraud"]
        if c in df.columns
    ]
    X = df.drop(columns=cols_to_drop)

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    print(f"Feature matrix shape: {X.shape}")
    print(f"Fraud rate: {y.mean():.4f} ({y.sum():,} fraud / {len(y):,} total)")

    return X, y, encoders


def run_feature_pipeline(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
    output_dir: str = "data",
) -> tuple:
    """
    End-to-end pipeline:
      1. Merge datasets
      2. Build feature matrix
      3. Save to parquet
    Returns (X, y, encoders).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Merge
    df = merge_datasets(transactions, identity)

    # Build features
    X, y, encoders = build_feature_matrix(df, fit_encoders=True)

    # Save
    X.to_parquet(f"{output_dir}/X_features.parquet", index=False)
    y.to_frame().to_parquet(f"{output_dir}/y_labels.parquet", index=False)
    joblib.dump(encoders, f"{output_dir}/encoders.pkl")

    print(f"\nSaved to {output_dir}/")
    print(f"  X_features.parquet")
    print(f"  y_labels.parquet")
    print(f"  encoders.pkl")

    return X, y, encoders


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_data

    transactions, identity = load_data()
    X, y, encoders = run_feature_pipeline(transactions, identity)

    print("\nTop 10 feature names:")
    print(X.columns.tolist()[:10])
    print("\nSample row:")
    print(X.iloc[0])
