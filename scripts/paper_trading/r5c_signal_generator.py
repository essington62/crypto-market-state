"""
R5C Signal Generator — 3-state HMM (Bull / Sideways / Bear).

Loads data/05_models/r5c_hmm.pkl (raw GaussianHMM, no scaler).
Features: log_return, vol_short, vol_ratio, drawdown, volume_z, slope_21d
          (same formula as R11 — unscaled, model trained on raw values)

State identification at load time from model.means_[:, 0] (log_return):
    argmax → Bull  (highest positive return mean)
    argmin → Bear  (lowest / most negative return mean)
    remaining → Sideways

Usage:
    gen = R5CSignalGenerator(model_path=ROOT / "data/05_models/r5c_hmm.pkl")
    signal = gen.predict(daily_df, proba_window=10)
    # → {"regime": "Sideways", "prob_bull": 0.02, "prob_bear": 0.05,
    #    "prob_sideways": 0.93, "entropy": 0.22, "regime_age": 3}
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]
PROBA_WINDOW = 10   # sequence length for predict_proba


def _slope_normalized(arr: np.ndarray) -> float:
    n = len(arr)
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, arr, 1)[0])
    last = arr[-1]
    return slope / last if last != 0.0 else 0.0


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute R5C features from daily OHLCV. Matches specialist _compute_r11_features."""
    close  = df["close"]
    volume = df["volume"]

    log_ret  = np.log(close / close.shift(1))
    vol_s    = log_ret.rolling(7).std()
    vol_l    = log_ret.rolling(30).std()
    vol_r    = vol_s / vol_l.replace(0, np.nan)

    roll_max = close.rolling(90, min_periods=1).max()
    drawdown = close / roll_max - 1.0

    vol_mean = volume.rolling(30).mean()
    vol_std  = volume.rolling(30).std()
    volume_z = (volume - vol_mean) / vol_std.replace(0, np.nan)

    slope_21d = close.rolling(21).apply(_slope_normalized, raw=True)

    out = df.copy()
    out["log_return"] = log_ret
    out["vol_short"]  = vol_s
    out["vol_ratio"]  = vol_r
    out["drawdown"]   = drawdown
    out["volume_z"]   = volume_z
    out["slope_21d"]  = slope_21d
    return out


class R5CSignalGenerator:
    """Wraps the frozen R5C HMM (3 states). Thread-safe (stateless predict)."""

    def __init__(
        self,
        model_path: Path,
        proba_window: int = PROBA_WINDOW,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"R5C model not found: {model_path}")

        with open(model_path, "rb") as fh:
            self.model = pickle.load(fh)

        self.proba_window = proba_window
        self.features     = FEATURES

        # Identify states from log_return mean (means_[:, 0])
        lr_means = self.model.means_[:, 0]
        self.bull_state     = int(lr_means.argmax())
        self.bear_state     = int(lr_means.argmin())
        self.sideways_state = [
            i for i in range(self.model.n_components)
            if i != self.bull_state and i != self.bear_state
        ][0]

        self.state_names = {
            self.bull_state:     "Bull",
            self.bear_state:     "Bear",
            self.sideways_state: "Sideways",
        }

        print("[R5CSignalGenerator] Loaded R5C HMM (3 states)")
        print(f"  n_components: {self.model.n_components}")
        print(f"  Bull={self.bull_state}  Bear={self.bear_state}  Sideways={self.sideways_state}")
        print(f"  log_return means: {lr_means}")

    def predict(self, daily_df: pd.DataFrame, proba_window: int | None = None) -> dict:
        """
        Predict regime from daily OHLCV DataFrame.

        Parameters
        ----------
        daily_df : pd.DataFrame
            Daily OHLCV with at least columns: close, volume.
            Must have enough rows for feature computation (90+ recommended).
        proba_window : int, optional
            Override default sequence length. Defaults to self.proba_window.

        Returns
        -------
        dict with keys: regime, prob_bull, prob_bear, prob_sideways, entropy, regime_age
        """
        window = proba_window or self.proba_window
        feat_df = _compute_features(daily_df)
        valid   = feat_df.dropna(subset=self.features)

        if len(valid) < window:
            raise ValueError(
                f"[R5CSignalGenerator] Not enough valid rows: {len(valid)} < {window}"
            )

        recent = valid[self.features].iloc[-window:].astype(float).values
        states = self.model.predict(recent)
        probs  = self.model.predict_proba(recent)

        last_state = int(states[-1])
        last_probs = probs[-1]

        prob_bull     = float(last_probs[self.bull_state])
        prob_bear     = float(last_probs[self.bear_state])
        prob_sideways = float(last_probs[self.sideways_state])

        eps     = 1e-9
        entropy = float(-sum(p * np.log2(p + eps) for p in last_probs))

        return {
            "regime":       self.state_names[last_state],
            "prob_bull":    round(prob_bull,     4),
            "prob_bear":    round(prob_bear,     4),
            "prob_sideways": round(prob_sideways, 4),
            "entropy":      round(entropy,       4),
            "regime_age":   self._compute_regime_age(states),
        }

    def _compute_regime_age(self, states: np.ndarray) -> int:
        """Consecutive rows in the current state (proxy for regime age in days)."""
        current = states[-1]
        age = 1
        for s in reversed(states[:-1]):
            if s == current:
                age += 1
            else:
                break
        return age
