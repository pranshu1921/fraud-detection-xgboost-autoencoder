"""
data_validation.py
Validates the raw IEEE-CIS transaction and identity datasets
using Great Expectations before any processing begins.
"""

import pandas as pd
import great_expectations as ge
from great_expectations.dataset import PandasDataset


def validate_transactions(df: pd.DataFrame) -> dict:
    """
    Run Great Expectations checks on the transaction dataframe.
    Returns a dict with pass/fail status and any failed expectations.
    """
    gdf = ge.from_pandas(df)

    results = []

    # Column existence checks
    required_cols = ["TransactionID", "isFraud", "TransactionAmt",
                     "ProductCD", "card1", "card4", "card6", "addr1",
                     "P_emaildomain", "TransactionDT"]
    for col in required_cols:
        r = gdf.expect_column_to_exist(col)
        results.append(("column_exists", col, r["success"]))

    # TransactionAmt must be positive
    r = gdf.expect_column_values_to_be_between(
        "TransactionAmt", min_value=0, mostly=0.99
    )
    results.append(("TransactionAmt_positive", "TransactionAmt", r["success"]))

    # isFraud must be 0 or 1
    r = gdf.expect_column_values_to_be_in_set("isFraud", [0, 1])
    results.append(("isFraud_binary", "isFraud", r["success"]))

    # TransactionID must be unique
    r = gdf.expect_column_values_to_be_unique("TransactionID")
    results.append(("TransactionID_unique", "TransactionID", r["success"]))

    # TransactionDT should be non-negative (seconds offset)
    r = gdf.expect_column_values_to_be_between(
        "TransactionDT", min_value=0, mostly=0.99
    )
    results.append(("TransactionDT_nonneg", "TransactionDT", r["success"]))

    failed = [(name, col) for name, col, passed in results if not passed]
    passed_count = sum(1 for _, _, p in results if p)

    summary = {
        "total_checks": len(results),
        "passed": passed_count,
        "failed_count": len(failed),
        "failed_checks": failed,
        "validation_passed": len(failed) == 0,
    }

    return summary


def validate_identity(df: pd.DataFrame) -> dict:
    """
    Run Great Expectations checks on the identity dataframe.
    """
    gdf = ge.from_pandas(df)
    results = []

    # TransactionID must exist and be non-null
    r = gdf.expect_column_to_exist("TransactionID")
    results.append(("column_exists", "TransactionID", r["success"]))

    r = gdf.expect_column_values_to_not_be_null("TransactionID")
    results.append(("TransactionID_not_null", "TransactionID", r["success"]))

    # DeviceType should only be known values if present
    if "DeviceType" in df.columns:
        r = gdf.expect_column_values_to_be_in_set(
            "DeviceType", ["desktop", "mobile", None], mostly=0.95
        )
        results.append(("DeviceType_valid", "DeviceType", r["success"]))

    failed = [(name, col) for name, col, passed in results if not passed]
    passed_count = sum(1 for _, _, p in results if p)

    summary = {
        "total_checks": len(results),
        "passed": passed_count,
        "failed_count": len(failed),
        "failed_checks": failed,
        "validation_passed": len(failed) == 0,
    }

    return summary


def run_validation_pipeline(
    transaction_path: str, identity_path: str, verbose: bool = True
) -> bool:
    """
    Load both files, run validations, print summary.
    Returns True if all checks pass.
    """
    print("Loading data for validation...")
    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)

    print(f"  Transactions shape: {transactions.shape}")
    print(f"  Identity shape:     {identity.shape}")

    print("\nRunning transaction validation...")
    t_results = validate_transactions(transactions)

    print("\nRunning identity validation...")
    i_results = validate_identity(identity)

    if verbose:
        print("\n--- Validation Summary ---")
        print(f"Transactions: {t_results['passed']}/{t_results['total_checks']} checks passed")
        if t_results["failed_checks"]:
            print(f"  FAILED: {t_results['failed_checks']}")

        print(f"Identity:     {i_results['passed']}/{i_results['total_checks']} checks passed")
        if i_results["failed_checks"]:
            print(f"  FAILED: {i_results['failed_checks']}")

    all_passed = t_results["validation_passed"] and i_results["validation_passed"]
    status = "PASSED" if all_passed else "FAILED"
    print(f"\nOverall validation: {status}")

    return all_passed


if __name__ == "__main__":
    run_validation_pipeline(
        transaction_path="data/train_transaction.csv",
        identity_path="data/train_identity.csv",
    )
