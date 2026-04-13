"""
autoencoder.py
Trains a deep autoencoder on non-fraud transactions only.
At inference time, high reconstruction error = likely anomaly.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_dim: int) -> keras.Model:
    """
    Build a symmetric autoencoder.
    Encoder compresses to a 32-dim bottleneck.
    Decoder reconstructs back to original dimension.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = layers.Dense(256, activation="relu", name="enc_1")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu", name="enc_2")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64,  activation="relu", name="enc_3")(x)
    bottleneck = layers.Dense(32, activation="relu", name="bottleneck")(x)

    # Decoder
    x = layers.Dense(64,  activation="relu", name="dec_3")(bottleneck)
    x = layers.Dense(128, activation="relu", name="dec_2")(x)
    x = layers.Dense(256, activation="relu", name="dec_1")(x)
    outputs = layers.Dense(input_dim, activation="linear", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(
    X: pd.DataFrame,
    y: pd.Series,
    epochs: int = 20,
    batch_size: int = 512,
    validation_split: float = 0.1,
    model_dir: str = "models",
) -> tuple[keras.Model, StandardScaler, float]:
    """
    Train autoencoder on non-fraud transactions only.
    Returns (model, scaler, anomaly_threshold).

    The threshold is set at the 95th percentile of reconstruction
    error on the training (non-fraud) data. Transactions above
    this threshold are flagged as anomalous.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Train only on legitimate transactions
    X_legit = X[y == 0].copy()
    print(f"Training autoencoder on {len(X_legit)} legitimate transactions...")

    # Scale features
    scaler = StandardScaler()
    X_legit_scaled = scaler.fit_transform(X_legit)

    # Build and train
    model = build_autoencoder(input_dim=X_legit_scaled.shape[1])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    )

    history = model.fit(
        X_legit_scaled, X_legit_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # Compute reconstruction error on training data
    X_legit_reconstructed = model.predict(X_legit_scaled, batch_size=batch_size)
    reconstruction_errors = np.mean(
        np.power(X_legit_scaled - X_legit_reconstructed, 2), axis=1
    )

    # Set threshold at 95th percentile of legitimate transaction errors
    threshold = float(np.percentile(reconstruction_errors, 95))
    print(f"\nAnomaly threshold (95th percentile): {threshold:.6f}")
    print(f"Mean reconstruction error (legit):   {reconstruction_errors.mean():.6f}")

    # Save artifacts
    model.save(f"{model_dir}/autoencoder.keras")
    joblib.dump(scaler, f"{model_dir}/ae_scaler.pkl")
    joblib.dump({"threshold": threshold}, f"{model_dir}/ae_threshold.pkl")

    print(f"\nSaved autoencoder artifacts to {model_dir}/")
    return model, scaler, threshold


def score_transactions(
    X: pd.DataFrame,
    model: keras.Model,
    scaler: StandardScaler,
    threshold: float,
    batch_size: int = 512,
) -> pd.DataFrame:
    """
    Score all transactions with the autoencoder.
    Returns a DataFrame with columns:
      - ae_reconstruction_error: raw error score
      - ae_anomaly_score: normalized score (0 to 1)
      - ae_is_anomaly: binary flag based on threshold
    """
    X_scaled = scaler.transform(X)
    X_reconstructed = model.predict(X_scaled, batch_size=batch_size, verbose=0)

    errors = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)

    # Normalize to 0-1 range using the threshold as the reference point
    # Scores above 1.0 are strongly anomalous
    normalized = errors / (threshold + 1e-10)

    scores = pd.DataFrame({
        "ae_reconstruction_error": errors,
        "ae_anomaly_score": np.clip(normalized, 0, 5),  # cap at 5x threshold
        "ae_is_anomaly": (errors > threshold).astype(int),
    })

    anomaly_rate = scores["ae_is_anomaly"].mean()
    print(f"Anomaly rate: {anomaly_rate:.4f} ({scores['ae_is_anomaly'].sum()} flagged)")

    return scores


def load_autoencoder_artifacts(model_dir: str = "models"):
    """Load saved autoencoder model, scaler, and threshold."""
    model = keras.models.load_model(f"{model_dir}/autoencoder.keras")
    scaler = joblib.load(f"{model_dir}/ae_scaler.pkl")
    threshold_data = joblib.load(f"{model_dir}/ae_threshold.pkl")
    threshold = threshold_data["threshold"]
    return model, scaler, threshold
