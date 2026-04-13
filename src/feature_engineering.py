"""
feature_engineering.py
Loads, merges, and engineers features from the IEEE-CIS dataset.
Produces a clean feature matrix ready for model training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os


# Columns to drop because they are IDs or too sparse to be useful
DROP_COLS = ["TransactionID", "TransactionDT"]

# Categorical columns that need label encoding
CAT_COLS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28",
    "id_29", "id_30", "id_31", "id_33", "id_34", "id_35",
    "id_36", "id_37", "id_38",
]


def load_and_merge(transaction_path: str, identity_path: str) -> pd.DataFrame:
    """Load both CSVs and merge on TransactionID."""
    print("Loading transaction data...")
    transactions = pd.read_csv(transaction_path)

    print("Loading identity data...")
    identity = pd.read_csv(identity_path)

    print("Merging on TransactionID...")
    df = transactions.merge(identity, on="TransactionID", how="left")
    print(f"Merged shape: {df.shape}")

    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transaction velocity features based on TransactionDT.
    TransactionDT is seconds from a reference point.
    We compute rolling counts per card over time windows.
    """
    print("Engineering velocity features...")
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Hour of day and day of week derived from TransactionDT
    df["tx_hour"] = (df["TransactionDT"] // 3600) % 24
    df["tx_day"]  = (df["TransactionDT"] // 86400) % 7

    # Transaction amount log transform (fraud amounts are skewed)
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # Amount deviation from card1 mean (is this transaction unusual for this card?)
    card_mean = df.groupby("card1")["TransactionAmt"].transform("mean")
    card_std  = df.groupby("card1")["TransactionAmt"].transform("std").fillna(1)
    df["amt_deviation_from_card_mean"] = (df["TransactionAmt"] - card_mean) / card_std

    # Transaction count per card1 (proxy for velocity without time window)
    df["card1_tx_count"] = df.groupby("card1")["TransactionAmt"].transform("count")

    # Transaction count per email domain
    df["email_tx_count"] = df.groupby("P_emaildomain")["TransactionAmt"].transform("count")

    # Transaction count per addr1
    df["addr1_tx_count"] = df.groupby("addr1")["TransactionAmt"].transform("count")

    return df


def encode_categoricals(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: dict = None
) -> tuple[pd.DataFrame, dict]:
    """
    Label encode categorical columns.
    If fit=True, fit new encoders. Otherwise use provided encoders.
    Returns (encoded_df, encoders_dict).
    """
    if encoders is None:
        encoders = {}

    for col in CAT_COLS:
        if col not in df.columns:
            continue

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
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "missing")
                df[col] = le.transform(df[col])

    return df, encoders


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values.
    Numeric columns: fill with median.
    Categorical columns: handled in encode step (filled with 'missing').
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    fit_encoders: bool = True,
    encoders: dict = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Full feature engineering pipeline.
    Returns (X, y, encoders).
    """
    df = add_velocity_features(df)
    df, encoders = encode_categoricals(df, fit=fit_encoders, encoders=encoders)
    df = fill_missing_values(df)

    # Separate target
    y = df["isFraud"].astype(int)

    # Drop columns not used as features
    cols_to_drop = [c for c in DROP_COLS + ["isFraud"] if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Fraud rate: {y.mean():.4f} ({y.sum()} fraud / {len(y)} total)")

    return X, y, encoders


def run_feature_pipeline(
    transaction_path: str,
    identity_path: str,
    output_dir: str = "data",
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    End-to-end: load, merge, engineer, and save features.
    """
    df = load_and_merge(transaction_path, identity_path)
    X, y, encoders = build_feature_matrix(df, fit_encoders=True)

    # Save processed features
    os.makedirs(output_dir, exist_ok=True)
    X.to_parquet(f"{output_dir}/X_features.parquet", index=False)
    y.to_frame().to_parquet(f"{output_dir}/y_labels.parquet", index=False)
    joblib.dump(encoders, f"{output_dir}/encoders.pkl")

    print(f"\nSaved features to {output_dir}/")
    return X, y, encoders


if __name__ == "__main__":
    X, y, encoders = run_feature_pipeline(
        transaction_path="data/train_transaction.csv",
        identity_path="data/train_identity.csv",
    )
