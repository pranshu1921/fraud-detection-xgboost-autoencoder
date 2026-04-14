"""
data_loader.py
Downloads the IEEE-CIS Fraud Detection dataset from Kaggle automatically.

Requirements:
  - kaggle.json placed in ~/.kaggle/ (one-time setup)
  - Competition rules accepted at kaggle.com/c/ieee-fraud-detection

First run: downloads ~677MB from Kaggle and saves to data/
All subsequent runs: loads directly from disk, no download needed.

Usage:
    from data_loader import load_data
    transactions, identity = load_data()
"""

import os
import subprocess
import zipfile
import pandas as pd


DATA_DIR              = "data"
TRANSACTION_PATH      = f"{DATA_DIR}/train_transaction.csv"
IDENTITY_PATH         = f"{DATA_DIR}/train_identity.csv"
COMPETITION_NAME      = "ieee-fraud-detection"
ZIP_PATH              = f"{DATA_DIR}/ieee-fraud-detection.zip"


def download_ieee_cis(data_dir: str = DATA_DIR) -> None:
    """
    Download IEEE-CIS dataset from Kaggle using the Kaggle CLI.
    Requires kaggle.json in ~/.kaggle/ and competition rules accepted.
    """
    os.makedirs(data_dir, exist_ok=True)

    print("Downloading IEEE-CIS Fraud Detection dataset from Kaggle...")
    print("This runs once (~677MB). Subsequent runs load from disk.")
    print("")

    result = subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", COMPETITION_NAME,
            "-p", data_dir,
        ],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Kaggle download failed.\n"
            "Make sure:\n"
            "  1. ~/.kaggle/kaggle.json exists with valid credentials\n"
            "  2. You accepted competition rules at:\n"
            "     https://www.kaggle.com/c/ieee-fraud-detection\n"
        )

    print("\nDownload complete. Extracting files...")
    extract_zip(data_dir)


def extract_zip(data_dir: str = DATA_DIR) -> None:
    """
    Extract the downloaded zip file.
    We only extract the two training files we need.
    """
    zip_path = f"{data_dir}/{COMPETITION_NAME}.zip"

    if not os.path.exists(zip_path):
        # Kaggle CLI sometimes saves with a different name
        zip_candidates = [
            f for f in os.listdir(data_dir)
            if f.endswith(".zip")
        ]
        if zip_candidates:
            zip_path = f"{data_dir}/{zip_candidates[0]}"
        else:
            raise FileNotFoundError(
                f"No zip file found in {data_dir}/ after download."
            )

    print(f"Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, "r") as z:
        all_files = z.namelist()
        print(f"Files in zip: {all_files}")

        # Extract only what we need
        files_to_extract = [
            f for f in all_files
            if "train_transaction" in f or "train_identity" in f
        ]

        for f in files_to_extract:
            print(f"  Extracting {f}...")
            z.extract(f, data_dir)

    print("Extraction complete.")


def load_data(
    transaction_path: str = TRANSACTION_PATH,
    identity_path: str = IDENTITY_PATH,
) -> tuple:
    """
    Main entry point. Returns (transactions_df, identity_df).

    Downloads automatically if files are not present locally.
    Loads from disk if files already exist.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download if either file is missing
    if not os.path.exists(transaction_path) or not os.path.exists(identity_path):
        print("Dataset not found locally. Starting download...")
        download_ieee_cis(DATA_DIR)

    # Verify files exist after download
    if not os.path.exists(transaction_path):
        raise FileNotFoundError(
            f"train_transaction.csv not found at {transaction_path}\n"
            f"Download may have failed. Run manually:\n"
            f"  kaggle competitions download -c ieee-fraud-detection -p data/"
        )
    if not os.path.exists(identity_path):
        raise FileNotFoundError(
            f"train_identity.csv not found at {identity_path}\n"
            f"Download may have failed. Run manually:\n"
            f"  kaggle competitions download -c ieee-fraud-detection -p data/"
        )

    # Load both files
    print(f"Loading {transaction_path}...")
    transactions = pd.read_csv(transaction_path)
    print(f"  Shape: {transactions.shape}")

    print(f"Loading {identity_path}...")
    identity = pd.read_csv(identity_path)
    print(f"  Shape: {identity.shape}")

    # Quick sanity check
    fraud_rate = transactions["isFraud"].mean()
    print(f"\nFraud rate:       {fraud_rate:.4f} ({transactions['isFraud'].sum():,} fraud cases)")
    print(f"Transaction cols: {transactions.shape[1]}")
    print(f"Identity cols:    {identity.shape[1]}")

    return transactions, identity


if __name__ == "__main__":
    transactions, identity = load_data()
    print("\nTransaction sample:")
    print(transactions.head(3))
    print("\nIdentity sample:")
    print(identity.head(3))
