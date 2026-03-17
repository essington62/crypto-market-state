"""
L3 Feature Engineering — CoinGlass derivatives 4h (BTC only).

Computes 5 strictly backward-looking features from the L2 clean DataFrame.

Features
--------
oi_momentum_12h     : % change in Binance OI over `oi_momentum_window` candles
                      (default 3 × 4h = 12h).  Positive = OI growing (leverage
                      expansion).  Negative = deleveraging.

oi_volatility_24h   : Rolling std of log OI returns over `oi_volatility_window`
                      candles (default 6 × 4h = 24h).  Captures OI instability.

funding_pressure    : OI-weighted funding rate normalized to a z-score over a
                      `funding_pressure_window`-candle rolling window.  > 0 means
                      longs pay more than usual; < 0 means unusually bearish cost.

funding_divergence  : Spread between OI-weighted and volume-weighted funding rates
                      (funding_oi_weighted − funding_vol_weighted).  Positive when
                      larger open-interest positions are more bullish than the
                      broader market.

leverage_expansion  : Log ratio of global-to-Binance OI
                      log(oi_aggregated / oi_binance).  Rising = leverage broadening
                      beyond Binance; falling = deleveraging concentrating on Binance.

All L2 columns are preserved unchanged.  NaN at the start of each series
(window burn-in) is permitted — never filled.

Forbidden: lookahead windows, cross-asset operations, label creation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_coinglass_derivatives_features(
    df: pd.DataFrame,
    oi_momentum_window: int,
    oi_volatility_window: int,
    funding_pressure_window: int,
) -> pd.DataFrame:
    """
    Compute L3 derivative features from the L2 clean DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        L2 clean DataFrame with DatetimeIndex UTC and columns:
        oi_binance, oi_aggregated, funding_binance,
        funding_oi_weighted, funding_vol_weighted.
    oi_momentum_window : int
        Number of candles for OI % change (e.g. 3 → 12h at 4h frequency).
    oi_volatility_window : int
        Rolling window for OI log-return std (e.g. 6 → 24h at 4h frequency).
    funding_pressure_window : int
        Rolling window for funding z-score normalization (e.g. 180 → 30 days).

    Returns
    -------
    pd.DataFrame
        Original L2 columns plus 5 new feature columns.

    Raises
    ------
    ValueError
        If any required L2 column is missing or DataFrame is empty.
    """
    required = ["oi_binance", "oi_aggregated", "funding_oi_weighted", "funding_vol_weighted"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[L3 derivatives_4h] Missing required L2 columns: {missing}"
        )
    if df is None or df.empty:
        raise ValueError("[L3 derivatives_4h] Input DataFrame is None or empty.")

    out = df.copy()

    # ── 1. oi_momentum_12h ───────────────────────────────────────────────────
    out["oi_momentum_12h"] = out["oi_binance"].pct_change(oi_momentum_window)

    # ── 2. oi_volatility_24h ────────────────────────────────────────────────
    log_oi_ret = np.log(
        out["oi_binance"] / out["oi_binance"].shift(1)
    )
    out["oi_volatility_24h"] = log_oi_ret.rolling(oi_volatility_window).std()

    # ── 3. funding_pressure (z-score of oi-weighted funding rate) ────────────
    fr = out["funding_oi_weighted"]
    fr_mean = fr.rolling(funding_pressure_window, min_periods=1).mean()
    fr_std  = fr.rolling(funding_pressure_window, min_periods=1).std()
    out["funding_pressure"] = (fr - fr_mean) / fr_std.replace(0.0, np.nan)

    # ── 4. funding_divergence ────────────────────────────────────────────────
    out["funding_divergence"] = (
        out["funding_oi_weighted"] - out["funding_vol_weighted"]
    )

    # ── 5. leverage_expansion ────────────────────────────────────────────────
    out["leverage_expansion"] = np.log(
        out["oi_aggregated"] / out["oi_binance"].replace(0.0, np.nan)
    )

    print(
        f"[L3 derivatives_4h] Features computed — shape: {out.shape} "
        f"({out.index[0].isoformat()} → {out.index[-1].isoformat()})"
    )

    return out
