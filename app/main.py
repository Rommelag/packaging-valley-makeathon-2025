# app/main.py
import pandas as pd
import numpy as np
from pathlib import Path
from model import LSTMAEConfig, LSTMAutoencoder
from detector import reconstruction_errors, calibrate_threshold, apply_threshold, ThresholdConfig

def test_ml_frameworks():
    import torch
    import tensorflow as tf

    print("TensorFlow version:", tf.__version__)
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))

    print("PyTorch version:", torch.__version__)
    print("Is CUDA available:", torch.cuda.is_available())

ART_DIR = Path("artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

def zscore_fit(X: np.ndarray):
    mu = X.mean(axis=(0,1), keepdims=True)
    sd = X.std(axis=(0,1), keepdims=True) + 1e-8
    return mu, sd

def zscore_transform(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd

def make_windows(values, seq_len, stride=1):
    n_feat = 1
    n_total = (len(values) - seq_len) // stride + 1
    X = np.zeros((n_total, seq_len, n_feat), dtype="float32")
    for i in range(n_total):
        start = i * stride
        X[i, :, 0] = values[start:start+seq_len]
    return X

def main():
    # -----------------------------
    # 1) Daten: good.csv einlesen
    # -----------------------------

    # CSV einlesen
    df = pd.read_csv("../data/good.csv", sep=";")
    values = df["value_dec"].astype("float32").values  # (N,)

    # Sliding Window mit variablem stride
    seq_len = 100
    stride = 1  # Kann als Parameter angepasst werden
    X = make_windows(values, seq_len, stride=stride)

    # Split in Train (reines Normal), Val (reines Normal)
    n_train = int(0.6 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]

    # Skalierung fitten NUR auf Train-Normal (kein Leakage)
    mu, sd = zscore_fit(X_train)
    X_train = zscore_transform(X_train, mu, sd)
    X_val   = zscore_transform(X_val,   mu, sd)

    # -----------------------------
    # 2) Modell trainieren
    # -----------------------------
    n_feat = 1  # Ein Feature: value_dec
    cfg = LSTMAEConfig(
        seq_len=seq_len, n_features=n_feat,
        latent_dim=64, encoder_units=(128, 64), decoder_units=(64, 128),
        dropout=0.1, bidirectional=False, learning_rate=1e-3, loss="mse"
    )
    ae = LSTMAutoencoder(cfg)
    ae.compile()
    ae.fit(X_train, X_val=X_val, epochs=30, batch_size=256, patience=5, verbose=1)

    # -----------------------------
    # 3) Threshold kalibrieren (Normal-Validierung)
    # -----------------------------
    # wähle Aggregation "mean" über die Zeit → ein Fehler pro Sequenz
    agg_time = "mean"
    val_err = reconstruction_errors(ae, X_val, agg_time=agg_time, batch_size=256)

    # Methode 1: p99-Perzentil    | Methode 2: robuste MAD
    thr_conf = ThresholdConfig(agg_time=agg_time, method="percentile", q=0.99)
    thr = calibrate_threshold(val_err, method=thr_conf.method, q=thr_conf.q, k=thr_conf.k)
    thr_conf.threshold = float(thr)

    print(f"[Calibration] threshold={thr_conf.threshold:.6f} (method={thr_conf.method}, q={thr_conf.q})")

    # -----------------------------
    # 4) Auf unbekannte Daten anwenden (bad.csv)
    # -----------------------------
    df_bad = pd.read_csv("../data/bad.csv", sep=";")
    values_bad = df_bad["value_dec"].astype("float32").values
    X_unk = make_windows(values_bad, seq_len, stride=stride)

    # gleiche Skalierung anwenden
    X_unk = zscore_transform(X_unk, mu, sd)

    unk_err = reconstruction_errors(ae, X_unk, agg_time=agg_time, batch_size=256)
    is_anom, scores = apply_threshold(unk_err, thr_conf.threshold)

    print(f"[Detection] Anteil Anomalien in Unbekannt (bad.csv): {is_anom.mean():.4f}")

    # -----------------------------
    # 5) Persistenz (Model + Threshold + Skalierung)
    # -----------------------------
    ae.save(ART_DIR / "lstm_ae")
    thr_conf.save(ART_DIR / "threshold.json")
    np.savez(ART_DIR / "scaler.npz", mu=mu, sd=sd)

    # Beispiel für Reload:
    # thr_loaded = ThresholdConfig.load(ART_DIR / "threshold.json")
    # sc = np.load(ART_DIR / "scaler.npz")
    # ae2 = LSTMAutoencoder.load(ART_DIR / "lstm_ae")

if __name__ == "__main__":
    # run LSTM-Autoencoder pipeline
    main()