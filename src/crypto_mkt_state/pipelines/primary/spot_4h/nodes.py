"""
L3 Feature Engineering — BTC spot 4h.

Node: compute_spot_features_4h
  Input:  btc_spot_4h_clean  (L2 DatetimeIndex UTC, 4h frequency)
  Output: btc_spot_features_4h

Features
--------
returns_4h      : close.pct_change()
                  Single-candle log-return proxy. Used as momentum input.

volatility_24h  : returns_4h.rolling(6).std()
                  Realized volatility over the last 24h (6 × 4h candles).

volume_zscore   : (volume - volume.rolling(30).mean())
                   / volume.rolling(30).std()
                  Volume anomaly: > 2 = unusually active, < -2 = quiet.

buy_pressure    : taker_buy_quote_volume / quote_volume
                  Proportion of volume from buyer-initiated takers.
                  Clipped to [0, 1].  NaN when quote_volume = 0.

price_range_4h  : (high - low) / close
                  Normalised intra-candle range; proxy for intraday vol.

All L2 columns are preserved unchanged.
NaN permitted at series start (window burn-in) — never filled.
Pure function — no IO, no Kedro internals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_REQUIRED_L2 = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_quote_volume",
]


def compute_spot_features_4h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute L3 spot 4h features from the L2 clean DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        L2 clean DataFrame with DatetimeIndex UTC.  Must contain all
        columns listed in _REQUIRED_L2.

    Returns
    -------
    pd.DataFrame
        Original L2 columns plus 5 new feature columns.

    Raises
    ------
    ValueError
        If required L2 columns are missing or DataFrame is empty.
    """
    if df is None or df.empty:
        raise ValueError("[L3 spot_4h] Input DataFrame is None or empty.")

    missing = [c for c in _REQUIRED_L2 if c not in df.columns]
    if missing:
        raise ValueError(f"[L3 spot_4h] Missing required L2 columns: {missing}")

    out = df.copy()

    # ── returns_4h ──────────────────────────────────────────────────────────
    out["returns_4h"] = out["close"].pct_change()

    # ── volatility_24h ──────────────────────────────────────────────────────
    out["volatility_24h"] = out["returns_4h"].rolling(6).std()

    # ── volume_zscore ────────────────────────────────────────────────────────
    vol_mean = out["volume"].rolling(30).mean()
    vol_std  = out["volume"].rolling(30).std()
    out["volume_zscore"] = (out["volume"] - vol_mean) / vol_std.replace(0.0, np.nan)

    # ── buy_pressure ─────────────────────────────────────────────────────────
    qv = out["quote_volume"].replace(0.0, np.nan)
    out["buy_pressure"] = (out["taker_buy_quote_volume"] / qv).clip(0.0, 1.0)

    # ── price_range_4h ───────────────────────────────────────────────────────
    out["price_range_4h"] = (out["high"] - out["low"]) / out["close"].replace(0.0, np.nan)

    print(
        f"[L3 spot_4h] Features computed — shape: {out.shape} "
        f"({out.index[0].isoformat()} → {out.index[-1].isoformat()})"
    )
    return out
