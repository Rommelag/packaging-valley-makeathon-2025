# app/detector.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Tuple, Dict, Any, Optional
import json
import numpy as np
from model import LSTMAutoencoder

AggKind = Literal["mean", "max", "median"]

@dataclass
class ThresholdConfig:
    agg_time: AggKind = "mean"
    agg_feature: AggKind = "mean"   # falls du später Feature-seitig aggregieren willst
    method: Literal["percentile", "mad"] = "percentile"
    q: float = 0.99                 # Perzentil (z. B. 0.99)
    k: float = 3.0                  # MAD-Multiplikator (z. B. 3.0)
    threshold: float = 0.0          # wird nach Kalibrierung gesetzt

    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ThresholdConfig":
        obj = json.loads(Path(path).read_text())
        return cls(**obj)


def reconstruction_errors(
    ae: LSTMAutoencoder,
    X: np.ndarray,
    agg_time: AggKind = "mean",
    batch_size: int = 256,
    return_per_timestep: bool = False,
    error_kind: str = "mse",  # "mse" or "mae"
) -> np.ndarray:
    """
    Liefert Rekonstruktionsfehler (MSE oder MAE).
    - return_per_timestep=False: (batch,) – Sequenz-Score
    - return_per_timestep=True:  (batch, seq_len) – Fehler pro Zeitstempel
    """
    X_hat = ae.reconstruct(X, batch_size=batch_size)
    if error_kind == "mae":
        per_timestep = np.mean(np.abs(X - X_hat), axis=2)  # (batch, seq_len)
    else:
        per_timestep = np.mean((X - X_hat) ** 2, axis=2)  # (batch, seq_len)

    if return_per_timestep:
        return per_timestep

    if   agg_time == "mean":   seq_err = per_timestep.mean(axis=1)
    elif agg_time == "max":    seq_err = per_timestep.max(axis=1)
    elif agg_time == "median": seq_err = np.median(per_timestep, axis=1)
    else:
        raise ValueError(f"Unbekannte Aggregation: {agg_time}")

    return seq_err


def calibrate_threshold(
    errors: np.ndarray,
    method: Literal["percentile", "mad"] = "percentile",
    q: float = 0.99,
    k: float = 3.0,
) -> float:
    """
    Kalibriert einen Grenzwert aus Normal-Fehlern.
    - percentile:   Schwelle = Quantil q
    - mad (robust): Schwelle = median(err) + k * 1.4826 * MAD(err)
    """
    errors = np.asarray(errors).ravel()

    if method == "percentile":
        thr = float(np.quantile(errors, q))
        return thr

    if method == "mad":
        med = float(np.median(errors))
        mad = float(np.median(np.abs(errors - med))) + 1e-12
        thr = med + k * 1.4826 * mad
        return thr

    raise ValueError(f"Unbekannte Methode: {method}")


def apply_threshold(
    errors: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wendet den Schwellenwert an.
    Rückgabe:
      - is_anomaly: bool-Array (True = Anomalie)
      - scores:     die Eingangsfehler (zur Weiterverarbeitung)
    """
    err = np.asarray(errors).ravel()
    is_anom = err > threshold
    return is_anom, err