"""
R11 Signal Generator — frozen HMM, never retrained.

Loads state/r11_hmm_model.pkl and predicts regime for a given feature vector.

Usage:
    gen = SignalGenerator()
    signal = gen.predict(feature_series)
    # → {"regime": "Bull", "p_bull": 0.847, "state": 1, "date": "2026-03-12"}

Fixes applied (no retrain):
  BUG 1 — startprob_ bias: predict_proba is now called on a sliding window of the
           last PROBA_WINDOW rows from the OHLCV buffer rather than a single
           observation.  The length-1 posterior is dominated by startprob_; using a
           proper sequence lets the forward algorithm converge to emission-driven
           posteriors before the final step.

  BUG 2 — state label inversion: the frozen mapa sorted drawdown means ascending
           and assigned lowest→Bear/highest→Bull.  The debug diagnostic confirmed
           that the state with the LOWER (more negative) scaled drawdown has the
           HIGHER log_return (+0.082) and is therefore Bull.  The correct rule is
           lower drawdown mean → Bull(1), higher → Bear(0), which is the reverse of
           the original assignment.  The mapa is recomputed at load time from
           model.means_ so the fix survives a re-freeze without code changes.
"""
from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STATE_DIR  = Path(__file__).parent / "state"
MODEL_PKL  = STATE_DIR / "r11_hmm_model.pkl"

PROBA_WINDOW = 10   # default sequence length for predict_proba context


# ---------------------------------------------------------------------------
# State ordering helper
# ---------------------------------------------------------------------------

def _build_mapa(model, features: list[str]) -> dict[int, int]:
    """
    Recompute the {original_state → ordered_label} mapping from model.means_.

    Rule: sort states by drawdown mean ascending (lower = more negative = deeper
    pullback from peak).  Assign:
        lowest  drawdown mean → Bull (ordered 1)   ← recovering / still growing
        highest drawdown mean → Bear (ordered 0)   ← falling / at recent lows

    This is the inverse of the original freeze_r11_model assignment and matches
    what the debug diagnostic confirmed from actual means_ values.
    """
    feat_idx = features.index("drawdown") if "drawdown" in features else 0
    dd_means = {s: float(model.means_[s][feat_idx])
                for s in range(model.n_components)}
    # ascending sort → [state_with_lowest_dd, ..., state_with_highest_dd]
    asc_states = sorted(dd_means, key=dd_means.get)
    n = len(asc_states)
    # reverse assignment: rank 0 (lowest dd) → ordered n-1 (Bull=1 for 2 states)
    return {s: (n - 1 - rank) for rank, s in enumerate(asc_states)}


class SignalGenerator:
    """Wraps the frozen R11 HMM. Thread-safe (stateless predict)."""

    def __init__(
        self,
        model_path: Path = MODEL_PKL,
        proba_window: int = PROBA_WINDOW,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Frozen model not found: {model_path}\n"
                "  Run: python scripts/paper_trading/freeze_r11_model.py"
            )
        with open(model_path, "rb") as fh:
            state = pickle.load(fh)

        self.model        = state["model"]
        self.scaler       = state["scaler"]
        self.features     = state["features"]
        self.meta         = state["meta"]
        self.proba_window = proba_window

        # BUG 2 FIX: recompute mapa from model.means_ with correct ordering rule.
        # This overrides the stale mapa frozen in the pkl without retraining.
        self.mapa      = _build_mapa(self.model, self.features)
        self._inv_mapa = {v: k for k, v in self.mapa.items()}  # {ordered → original}

        print("[SignalGenerator] Loaded frozen R11 HMM")
        print(f"  Frozen:       {self.meta.get('frozen_date', 'unknown')}")
        print(f"  Train:        {self.meta['train_start']} → {self.meta['train_end']}")
        print(f"  Features:     {self.features}")
        print(f"  mapa:         {self.mapa}  (recomputed — Bear=0 Bull=1)")
        print(f"  proba_window: {self.proba_window} rows")

    # ── Context sequence builder ──────────────────────────────────────────

    def _build_context_X(
        self,
        feature_input: pd.Series,
        context_df: pd.DataFrame | None,
    ) -> np.ndarray:
        """
        Build a scaled sequence of shape (N, n_features) for predict_proba.

        If context_df is provided (caller-supplied feature DataFrame):
            - drop rows with NaN in self.features
            - take the last (proba_window - 1) rows as context
            - append the current feature_input row
            - scale and return the full sequence

        If context_df is None, falls back to single-row (shape 1, n_features).
        No disk I/O is performed inside this method.
        """
        current_df     = pd.DataFrame(
            [feature_input[self.features].values],
            columns=self.features,
        ).astype(float)
        current_scaled = self.scaler.transform(current_df)

        if context_df is None or context_df.empty:
            return current_scaled

        valid = context_df.dropna(subset=self.features)
        n_context = self.proba_window - 1
        if len(valid) == 0:
            return current_scaled

        context_rows   = valid[self.features].iloc[-n_context:].astype(float)
        context_scaled = self.scaler.transform(context_rows)
        return np.vstack([context_scaled, current_scaled])

    # ── Public API ────────────────────────────────────────────────────────

    def predict(
        self,
        feature_input: pd.Series | dict,
        signal_date: str | None = None,
        context_df: pd.DataFrame | None = None,
    ) -> dict:
        """
        Predict regime from a feature vector.

        Parameters
        ----------
        feature_input : pd.Series or dict
            Feature values keyed by feature name. Must contain all self.features.
        signal_date : str, optional
            Date string for the signal (e.g. "2026-03-12"). Defaults to today UTC.
        context_df : pd.DataFrame, optional
            Full feature DataFrame (already computed by the caller). The last
            (proba_window - 1) valid rows are used as sequence context so the
            HMM forward algorithm is emission-driven rather than startprob_-driven.
            When None, falls back to single-observation inference.

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
            nan_feats = [self.features[i] for i in range(len(self.features))
                         if np.isnan(x[0, i])]
            raise ValueError(f"SignalGenerator: NaN in features {nan_feats}")

        # Use caller-supplied context for multi-row sequence inference.
        X_seq = self._build_context_X(feature_input, context_df)  # (N, n_features)

        raw_state     = int(self.model.predict(X_seq)[-1])   # last step
        ordered_state = self.mapa[raw_state]                  # 0=Bear, 1=Bull
        raw_proba     = self.model.predict_proba(X_seq)[-1]  # last step posterior
        p_bull        = float(raw_proba[self._inv_mapa[1]])
        p_bear        = float(raw_proba[self._inv_mapa[0]])

        date_str = signal_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return {
            "regime":        "Bull" if ordered_state == 1 else "Bear",
            "state":         ordered_state,
            "p_bull":        round(p_bull, 4),
            "p_bear":        round(p_bear, 4),
            "date":          date_str,
            "features_used": self.features,
        }
