"""
tests/test_feature_engineering.py
Unit tests for the feature engineering pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import (
    add_velocity_features,
    encode_categoricals,
    fill_missing_values,
    build_feature_matrix,
)


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def sample_transactions():
    """Create a small synthetic transaction DataFrame for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "TransactionID": range(1, n + 1),
        "isFraud":        np.random.randint(0, 2, n),
        "TransactionDT":  np.arange(0, n * 100, 100),
        "TransactionAmt": np.abs(np.random.normal(100, 50, n)),
        "ProductCD":      np.random.choice(["W", "C", "R", "H", "S"], n),
        "card1":          np.random.randint(1000, 18000, n),
        "card4":          np.random.choice(["visa", "mastercard", None], n),
        "card6":          np.random.choice(["debit", "credit", None], n),
        "P_emaildomain":  np.random.choice(["gmail.com", "yahoo.com", None], n),
        "addr1":          np.random.choice([100.0, 200.0, np.nan], n),
    })
    return df


# -------------------------
# Tests
# -------------------------

class TestVelocityFeatures:
    def test_adds_log_amount(self, sample_transactions):
        result = add_velocity_features(sample_transactions.copy())
        assert "TransactionAmt_log" in result.columns

    def test_log_amount_nonneg(self, sample_transactions):
        result = add_velocity_features(sample_transactions.copy())
        assert (result["TransactionAmt_log"] >= 0).all()

    def test_adds_tx_hour(self, sample_transactions):
        result = add_velocity_features(sample_transactions.copy())
        assert "tx_hour" in result.columns
        assert result["tx_hour"].between(0, 23).all()

    def test_adds_card_count(self, sample_transactions):
        result = add_velocity_features(sample_transactions.copy())
        assert "card1_tx_count" in result.columns
        assert (result["card1_tx_count"] > 0).all()

    def test_adds_amt_deviation(self, sample_transactions):
        result = add_velocity_features(sample_transactions.copy())
        assert "amt_deviation_from_card_mean" in result.columns


class TestCategoricalEncoding:
    def test_encodes_productcd(self, sample_transactions):
        df = add_velocity_features(sample_transactions.copy())
        df, encoders = encode_categoricals(df, fit=True)
        assert df["ProductCD"].dtype in [np.int64, np.int32, object]
        assert "ProductCD" in encoders

    def test_consistent_encoding(self, sample_transactions):
        df1 = add_velocity_features(sample_transactions.copy())
        df1, encoders = encode_categoricals(df1, fit=True)

        df2 = add_velocity_features(sample_transactions.copy())
        df2, _ = encode_categoricals(df2, fit=False, encoders=encoders)

        pd.testing.assert_series_equal(df1["ProductCD"], df2["ProductCD"])

    def test_handles_missing_values(self, sample_transactions):
        df = add_velocity_features(sample_transactions.copy())
        df["card4"] = None  # All null
        df, encoders = encode_categoricals(df, fit=True)
        # Should not raise, all values become "missing" label
        assert "card4" in df.columns


class TestMissingValues:
    def test_no_nulls_after_fill(self, sample_transactions):
        df = add_velocity_features(sample_transactions.copy())
        df, _ = encode_categoricals(df, fit=True)
        df = fill_missing_values(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert df[numeric_cols].isnull().sum().sum() == 0


class TestBuildFeatureMatrix:
    def test_returns_correct_shapes(self, sample_transactions):
        X, y, encoders = build_feature_matrix(sample_transactions.copy())
        assert len(X) == len(y)
        assert len(y) == 100

    def test_target_is_binary(self, sample_transactions):
        X, y, encoders = build_feature_matrix(sample_transactions.copy())
        assert set(y.unique()).issubset({0, 1})

    def test_no_target_in_features(self, sample_transactions):
        X, y, encoders = build_feature_matrix(sample_transactions.copy())
        assert "isFraud" not in X.columns

    def test_no_id_in_features(self, sample_transactions):
        X, y, encoders = build_feature_matrix(sample_transactions.copy())
        assert "TransactionID" not in X.columns
