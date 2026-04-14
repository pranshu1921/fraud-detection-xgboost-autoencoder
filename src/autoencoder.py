"""
autoencoder.py
Trains a deep autoencoder on non-fraud transactions only.
At inference time, high reconstruction error = likely anomaly.

Backend: PyTorch (replaces TensorFlow for Windows compatibility)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FraudAutoencoder(nn.Module):
    """
    Symmetric autoencoder.
    Encoder compresses input to a 32-dim bottleneck.
    Decoder reconstructs back to original dimension.
    """
    def __init__(self, input_dim: int):
        super(FraudAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(
    X: pd.DataFrame,
    y: pd.Series,
    epochs: int = 20,
    batch_size: int = 512,
    validation_split: float = 0.1,
    model_dir: str = "models",
) -> tuple:
    """
    Train autoencoder on non-fraud transactions only.
    Returns (model, scaler, threshold).
    """
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cpu")

    X_legit = X[y == 0].copy()
    print(f"Training autoencoder on {len(X_legit)} legitimate transactions...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_legit).astype(np.float32)

    val_size = int(len(X_scaled) * validation_split)
    X_train_ae = X_scaled[:-val_size]
    X_val_ae   = X_scaled[-val_size:]

    train_dataset = TensorDataset(torch.tensor(X_train_ae))
    val_dataset   = TensorDataset(torch.tensor(X_val_ae))
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    input_dim = X_scaled.shape[1]
    model = FraudAutoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 3

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item() * len(batch)
        val_loss /= len(val_dataset)

        print(f"Epoch {epoch+1:2d}/{epochs} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{model_dir}/autoencoder_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(f"{model_dir}/autoencoder_best.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).to(device)
        X_reconstructed = model(X_tensor).cpu().numpy()

    errors = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
    threshold = float(np.percentile(errors, 95))

    print(f"\nAnomaly threshold (95th percentile): {threshold:.6f}")
    print(f"Mean reconstruction error (legit):   {errors.mean():.6f}")

    torch.save(model.state_dict(), f"{model_dir}/autoencoder.pt")
    joblib.dump({"input_dim": input_dim}, f"{model_dir}/autoencoder_config.pkl")
    joblib.dump(scaler, f"{model_dir}/ae_scaler.pkl")
    joblib.dump({"threshold": threshold}, f"{model_dir}/ae_threshold.pkl")

    print(f"Saved autoencoder artifacts to {model_dir}/")
    return model, scaler, threshold


def score_transactions(
    X: pd.DataFrame,
    model,
    scaler: StandardScaler,
    threshold: float,
    batch_size: int = 512,
) -> pd.DataFrame:
    """
    Score all transactions with the autoencoder.
    Returns DataFrame with reconstruction error, anomaly score, and binary flag.
    """
    device = torch.device("cpu")
    model.eval()

    X_scaled = scaler.transform(X).astype(np.float32)

    all_errors = []
    for i in range(0, len(X_scaled), batch_size):
        batch = torch.tensor(X_scaled[i:i+batch_size]).to(device)
        with torch.no_grad():
            reconstructed = model(batch).cpu().numpy()
        errors = np.mean(np.power(X_scaled[i:i+batch_size] - reconstructed, 2), axis=1)
        all_errors.extend(errors.tolist())

    errors = np.array(all_errors)
    normalized = np.clip(errors / (threshold + 1e-10), 0, 5)

    scores = pd.DataFrame({
        "ae_reconstruction_error": errors,
        "ae_anomaly_score":        normalized,
        "ae_is_anomaly":           (errors > threshold).astype(int),
    })

    anomaly_rate = scores["ae_is_anomaly"].mean()
    print(f"Anomaly rate: {anomaly_rate:.4f} ({scores['ae_is_anomaly'].sum()} flagged)")

    return scores


def load_autoencoder_artifacts(model_dir: str = "models"):
    """Load saved autoencoder model, scaler, and threshold."""
    config    = joblib.load(f"{model_dir}/autoencoder_config.pkl")
    scaler    = joblib.load(f"{model_dir}/ae_scaler.pkl")
    threshold = joblib.load(f"{model_dir}/ae_threshold.pkl")["threshold"]

    model = FraudAutoencoder(input_dim=config["input_dim"])
    model.load_state_dict(torch.load(f"{model_dir}/autoencoder.pt", map_location="cpu", weights_only=True))
    model.eval()

    return model, scaler, threshold
