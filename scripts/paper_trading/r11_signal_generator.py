"""
R11 Signal Generator — frozen HMM, never retrained.

Loads state/r11_hmm_model.pkl and predicts regime for a given feature vector.

Usage:
    gen = SignalGenerator()
    signal = gen.predict(feature_series)
    # → {"regime": "Bull", "p_bull": 0.847, "state": 1, "date": "2026-03-12"}
"""
from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STATE_DIR = Path(__file__).parent / "state"
MODEL_PKL = STATE_DIR / "r11_hmm_model.pkl"


class SignalGenerator:
    """Wraps the frozen R11 HMM. Thread-safe (stateless predict)."""

    def __init__(self, model_path: Path = MODEL_PKL) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Frozen model not found: {model_path}\n"
                "  Run: python scripts/paper_trading/freeze_r11_model.py"
            )
        with open(model_path, "rb") as fh:
            state = pickle.load(fh)

        self.model    = state["model"]
        self.scaler   = state["scaler"]
        self.mapa     = state["mapa"]          # {original → ordered (0=Bear,1=Bull)}
        self.features = state["features"]
        self.meta     = state["meta"]
        self._inv_mapa = {v: k for k, v in self.mapa.items()}  # {ordered → original}

        print(f"[SignalGenerator] Loaded frozen R11 HMM")
        print(f"  Frozen:    {self.meta.get('frozen_date', 'unknown')}")
        print(f"  Train:     {self.meta['train_start']} → {self.meta['train_end']}")
        print(f"  Features:  {self.features}")

    def predict(
        self,
        feature_input: pd.Series | dict,
        signal_date: str | None = None,
    ) -> dict:
        """
        Predict regime from a feature vector.

        Parameters
        ----------
        feature_input : pd.Series or dict
            Feature values keyed by feature name. Must contain all self.features.
        signal_date : str, optional
            Date string for the signal (e.g. "2026-03-12"). Defaults to today UTC.

        Returns
        -------
        dict with keys: regime, p_bull, p_bear, state, date, features_used
        """
        if isinstance(feature_input, dict):
            feature_input = pd.Series(feature_input)

        missing = [f for f in self.features if f not in feature_input.index]
        if missing:
            raise ValueError(f"SignalGenerator: missing features {missing}")

        x = feature_input[self.features].values.reshape(1, -1).astype(float)

        if np.any(np.isnan(x)):
            nan_feats = [self.features[i] for i in range(len(self.features)) if np.isnan(x[0, i])]
            raise ValueError(f"SignalGenerator: NaN in features {nan_feats}")

        x_scaled     = self.scaler.transform(x)
        raw_state    = int(self.model.predict(x_scaled)[0])
        ordered_state = self.mapa[raw_state]                    # 0=Bear, 1=Bull
        raw_proba    = self.model.predict_proba(x_scaled)[0]    # shape (n_states,)
        p_bull       = float(raw_proba[self._inv_mapa[1]])
        p_bear       = float(raw_proba[self._inv_mapa[0]])

        date_str = signal_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return {
            "regime":        "Bull" if ordered_state == 1 else "Bear",
            "state":         ordered_state,
            "p_bull":        round(p_bull, 4),
            "p_bear":        round(p_bear, 4),
            "date":          date_str,
            "features_used": self.features,
        }
