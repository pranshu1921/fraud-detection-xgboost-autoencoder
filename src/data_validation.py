"""
data_validation.py
Validates the IEEE-CIS transaction and identity DataFrames
using Great Expectations before any model training begins.

Checks:
  - Required columns exist
  - TransactionAmt is positive
  - isFraud is binary (0 or 1)
  - TransactionID is unique
  - TransactionDT is non-negative
  - No nulls in critical columns

Usage:
    from data_validation import run_validation_pipeline
    passed = run_validation_pipeline(transactions, identity)
"""

import pandas as pd
import great_expectations as ge


def validate_transactions(df: pd.DataFrame) -> dict:
    """
    Run Great Expectations checks on the transaction DataFrame.
    Returns a summary dict with pass/fail status.
    """
    gdf = ge.from_pandas(df)
    results = []

    # --- Required columns ---
    required_cols = [
        "TransactionID", "isFraud", "TransactionAmt",
        "TransactionDT", "ProductCD",
        "card1", "card4", "card6",
        "P_emaildomain",
    ]
    for col in required_cols:
        r = gdf.expect_column_to_exist(col)
        results.append(("column_exists", col, r["success"]))

    # --- TransactionAmt must be positive ---
    r = gdf.expect_column_values_to_be_between(
        "TransactionAmt", min_value=0, mostly=0.99
    )
    results.append(("TransactionAmt_positive", "TransactionAmt", r["success"]))

    # --- isFraud must be 0 or 1 ---
    r = gdf.expect_column_values_to_be_in_set("isFraud", [0, 1])
    results.append(("isFraud_binary", "isFraud", r["success"]))

    # --- TransactionID must be unique ---
    r = gdf.expect_column_values_to_be_unique("TransactionID")
    results.append(("TransactionID_unique", "TransactionID", r["success"]))

    # --- TransactionDT must be non-negative ---
    r = gdf.expect_column_values_to_be_between(
        "TransactionDT", min_value=0, mostly=0.99
    )
    results.append(("TransactionDT_nonneg", "TransactionDT", r["success"]))

    # --- TransactionAmt must not be null ---
    r = gdf.expect_column_values_to_not_be_null("TransactionAmt")
    results.append(("TransactionAmt_not_null", "TransactionAmt", r["success"]))

    # --- isFraud must not be null ---
    r = gdf.expect_column_values_to_not_be_null("isFraud")
    results.append(("isFraud_not_null", "isFraud", r["success"]))

    # --- Row count sanity check ---
    r = gdf.expect_table_row_count_to_be_between(
        min_value=100000, max_value=1000000
    )
    results.append(("row_count_sane", "table", r["success"]))

    failed = [(name, col) for name, col, passed in results if not passed]
    passed_count = sum(1 for _, _, p in results if p)

    return {
        "total_checks":      len(results),
        "passed":            passed_count,
        "failed_count":      len(failed),
        "failed_checks":     failed,
        "validation_passed": len(failed) == 0,
    }


def validate_identity(df: pd.DataFrame) -> dict:
    """
    Run Great Expectations checks on the identity DataFrame.
    Returns a summary dict with pass/fail status.
    """
    gdf = ge.from_pandas(df)
    results = []

    # --- TransactionID must exist ---
    r = gdf.expect_column_to_exist("TransactionID")
    results.append(("column_exists", "TransactionID", r["success"]))

    # --- TransactionID must not be null ---
    r = gdf.expect_column_values_to_not_be_null("TransactionID")
    results.append(("TransactionID_not_null", "TransactionID", r["success"]))

    # --- DeviceType must be known values if present ---
    if "DeviceType" in df.columns:
        r = gdf.expect_column_values_to_be_in_set(
            "DeviceType",
            ["desktop", "mobile", "nan", "None", None],
            mostly=0.95,
        )
        results.append(("DeviceType_valid", "DeviceType", r["success"]))

    # --- Row count sanity check ---
    r = gdf.expect_table_row_count_to_be_between(
        min_value=10000, max_value=500000
    )
    results.append(("row_count_sane", "table", r["success"]))

    failed = [(name, col) for name, col, passed in results if not passed]
    passed_count = sum(1 for _, _, p in results if p)

    return {
        "total_checks":      len(results),
        "passed":            passed_count,
        "failed_count":      len(failed),
        "failed_checks":     failed,
        "validation_passed": len(failed) == 0,
    }


def run_validation_pipeline(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
    verbose: bool = True,
) -> bool:
    """
    Run validation on both DataFrames.
    Returns True if all checks pass, False otherwise.
    Called by train.py before any model training begins.
    """
    print(f"Validating transactions ({transactions.shape[0]:,} rows)...")
    t_results = validate_transactions(transactions)

    print(f"Validating identity ({identity.shape[0]:,} rows)...")
    i_results = validate_identity(identity)

    if verbose:
        print("\n--- Validation Summary ---")
        print(
            f"Transactions: {t_results['passed']}/{t_results['total_checks']}"
            f" checks passed"
        )
        if t_results["failed_checks"]:
            print(f"  FAILED: {t_results['failed_checks']}")

        print(
            f"Identity:     {i_results['passed']}/{i_results['total_checks']}"
            f" checks passed"
        )
        if i_results["failed_checks"]:
            print(f"  FAILED: {i_results['failed_checks']}")

    all_passed = (
        t_results["validation_passed"] and
        i_results["validation_passed"]
    )
    status = "PASSED" if all_passed else "FAILED"
    print(f"\nOverall validation: {status}")

    return all_passed


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_data

    transactions, identity = load_data()
    run_validation_pipeline(transactions, identity)
