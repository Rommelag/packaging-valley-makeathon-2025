# app/model.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Tuple, Literal, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, optimizers, losses, metrics


AggKind = Literal["mean", "max", "median"]


@dataclass
class LSTMAEConfig:
    seq_len: int
    n_features: int
    latent_dim: int = 64
    encoder_units: Tuple[int, ...] = (128, 64)
    decoder_units: Tuple[int, ...] = (64, 128)
    dropout: float = 0.1
    bidirectional: bool = False
    learning_rate: float = 1e-3
    loss: str = "mse"  # "mse" | "mae" | "huber"
    # Initialisierung / reguläre Parameter kannst du hier ergänzen

    def get_config(self) -> dict:
        return asdict(self)


class LSTMAutoencoder:
    """
    LSTM-Autoencoder für multivariate Zeitreihen.

    Erwartet Eingaben mit Form (batch, seq_len, n_features).
    Der Autoencoder rekonstruiert die Eingabe; die Rekonstruktionsfehler
    eignen sich typischerweise als Anomalie-Score.

    Beispiel:
        cfg = LSTMAEConfig(seq_len=100, n_features=8)
        ae = LSTMAutoencoder(cfg)
        ae.compile()
        ae.fit(X_train, X_val=X_val)
        X_hat = ae.reconstruct(X_test)
        err  = ae.reconstruction_error(X_test)
    """

    def __init__(self, config: LSTMAEConfig):
        self.config = config
        self.model: Optional[Model] = None
        self._build()

    # --------------------------------------------------------------------- #
    # Model-Architektur
    # --------------------------------------------------------------------- #
    def _lstm_block(
        self, units: int, return_sequences: bool = True
    ) -> tf.keras.layers.Layer:
        c = self.config
        lstm = layers.LSTM(units, return_sequences=return_sequences)
        if c.bidirectional:
            return layers.Bidirectional(lstm, merge_mode="concat")
        return lstm

    def _build(self) -> None:
        c = self.config

        inputs = layers.Input(shape=(c.seq_len, c.n_features), name="inputs")

        x = inputs
        # Encoder
        for i, units in enumerate(c.encoder_units):
            x = self._lstm_block(units, return_sequences=True)(x)
            if c.dropout > 0:
                x = layers.Dropout(c.dropout, name=f"enc_drop_{i}")(x)

        # Latenter Zustand (Verdichtung auf Vektor)
        x = self._lstm_block(c.latent_dim, return_sequences=False)(x)
        latent = layers.Dense(c.latent_dim, activation="linear", name="latent")(x)

        # Decoder
        y = layers.RepeatVector(c.seq_len, name="repeat")(latent)
        for j, units in enumerate(c.decoder_units):
            y = self._lstm_block(units, return_sequences=True)(y)
            if c.dropout > 0:
                y = layers.Dropout(c.dropout, name=f"dec_drop_{j}")(y)

        outputs = layers.TimeDistributed(
            layers.Dense(c.n_features, activation="linear"),
            name="reconstruction",
        )(y)

        self.model = Model(inputs=inputs, outputs=outputs, name="lstm_autoencoder")

    # --------------------------------------------------------------------- #
    # Training/Inference
    # --------------------------------------------------------------------- #
    def compile(self) -> None:
        assert self.model is not None
        loss_map = {
            "mse": losses.MeanSquaredError(),
            "mae": losses.MeanAbsoluteError(),
            "huber": losses.Huber(),
        }
        metric_map = {
            "mse": metrics.MeanSquaredError(),
            "mae": metrics.MeanAbsoluteError(),
            "huber": metrics.MeanAbsoluteError(),  # Huber: MAE als Metrik
        }
        loss_fn = loss_map.get(self.config.loss, losses.MeanSquaredError())
        metric_fn = metric_map.get(self.config.loss, metrics.MeanSquaredError())
        opt = optimizers.Adam(learning_rate=self.config.learning_rate)
        self.model.compile(optimizer=opt, loss=loss_fn, metrics=[metric_fn])

    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 128,
        patience: int = 10,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:
        """
        Trainiert den Autoencoder. Nutzt EarlyStopping auf 'val_loss' wenn X_val vorhanden ist.
        """
        assert self.model is not None

        cbs: Iterable[callbacks.Callback] = []
        if X_val is not None:
            cbs = [
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=patience,
                    restore_best_weights=True,
                )
            ]

        history = self.model.fit(
            X_train,
            X_train,  # Ziel = Eingabe (Rekonstruktion)
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=list(cbs),
            verbose=verbose,
        )
        return history

    def reconstruct(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        assert self.model is not None
        return self.model.predict(X, batch_size=batch_size, verbose=0)

    def reconstruction_error(
        self,
        X: np.ndarray,
        agg_time: AggKind = "mean",
        agg_feature: AggKind = "mean",
        batch_size: int = 256,
        error_kind: str = "mse",  # "mse" or "mae"
    ) -> np.ndarray:
        """
        Berechnet pro Sample einen rekonstruktiven Fehler (MSE oder MAE über Zeit & Features).
        Du kannst getrennt über Zeit und Features aggregieren:
          - agg_time    ∈ {"mean","max","median"}
          - agg_feature ∈ {"mean","max","median"}
        Rückgabe: (batch,) — ein Score pro Sequenz
        """
        X_hat = self.reconstruct(X, batch_size=batch_size)
        if error_kind == "mae":
            err = np.mean(np.abs(X - X_hat), axis=2)  # (batch, seq_len)
        else:
            err = np.mean((X - X_hat) ** 2, axis=2)  # (batch, seq_len)
        err = _aggregate(err, axis=1, kind=agg_time)  # (batch,)
        return err

    # --------------------------------------------------------------------- #
    # Persistenz
    # --------------------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        """
        Speichert Keras-Model + Konfiguration.
        """
        assert self.model is not None
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model.keras")
        config_dict = tf.keras.utils.serialize_keras_object(self.config)
        import json
        (path / "config.json").write_text(json.dumps(config_dict, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LSTMAutoencoder":
        """
        Lädt Keras-Model + Konfiguration.
        """
        path = Path(path)
        config = tf.keras.utils.deserialize_keras_object((path / "config.json").read_text())
        inst = cls(config=config)  # baut ein neues Model
        inst.model = tf.keras.models.load_model(path / "model.keras")
        return inst


def _aggregate(x: np.ndarray, axis: int, kind: AggKind) -> np.ndarray:
    if kind == "mean":
        return np.mean(x, axis=axis)
    if kind == "max":
        return np.max(x, axis=axis)
    if kind == "median":
        return np.median(x, axis=axis)
    raise ValueError(f"Unbekannte Aggregation: {kind}")