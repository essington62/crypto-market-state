"""
Control layer — batch backtesting node.

Applies regime gate and timing gate to a DataFrame of historical signals,
producing allocation_final for performance evaluation.

Input columns required:
    allocation_raw   : model output (class 1 probability)
    r11_prob_bull    : R11 HMM bull probability
    r11_entropy      : R11 HMM entropy
    rsi_4h           : RSI 14-period (4h candles)
    bb_percent_b     : Bollinger %B (20-period, 2-std)
    volume_zscore    : volume z-score (50-period)

Gates are stateless (entry-only, no position tracking).
For stop loss / stop gain, use the paper trader (stateful).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_control_layer_batch(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Batch control layer for backtesting via Kedro.

    Same logic as the paper trader but operates on the full DataFrame.
    Gates are applied without position tracking (entry-only filter).

    Args:
        df:     DataFrame with allocation_raw + gate feature columns.
        params: Parameters from parameters.yml (modeling.control_layer section).

    Returns:
        DataFrame with added columns:
            regime_gate      : bool — True if regime allowed entry
            allocation_after_regime : float — allocation after regime gate
            timing_score     : float — timing gate score (0–1)
            timing_gate      : bool — True if timing allowed entry
            allocation_final : float — final allocation after all gates
    """
    df  = df.copy()
    cl  = params.get("control_layer", {})

    # ── Regime gate ───────────────────────────────────────────────────────────
    rg = cl.get("regime_gate", {})
    if rg.get("enabled", False):
        min_prob    = float(rg.get("min_prob_bull", 0.30))
        max_entropy = float(rg.get("max_entropy",   0.85))
        df["regime_gate"] = (
            df["r11_prob_bull"].fillna(0.0).ge(min_prob) &
            df["r11_entropy"].fillna(1.0).le(max_entropy)
        )
        df["allocation_after_regime"] = np.where(
            df["regime_gate"],
            df["allocation_raw"],
            0.0,
        )
    else:
        df["regime_gate"]             = True
        df["allocation_after_regime"] = df["allocation_raw"]

    # ── Timing gate ───────────────────────────────────────────────────────────
    tg = cl.get("timing_gate", {})
    if tg.get("enabled", False):
        w         = tg.get("weights", {"rsi": 0.4, "bollinger": 0.3, "volume": 0.3})
        rsi_t     = float(tg.get("rsi_threshold", 40))
        bb_t      = float(tg.get("bb_threshold",  0.30))
        min_score = float(tg.get("min_score",     0.30))

        rsi_comp = np.maximum(0.0, (rsi_t - df["rsi_4h"].fillna(rsi_t))     / rsi_t)
        bb_comp  = np.maximum(0.0, (bb_t  - df["bb_percent_b"].fillna(bb_t)) / bb_t)
        vol_comp = np.maximum(0.0, np.minimum(df["volume_zscore"].fillna(0.0) / 2.0, 1.0))

        df["timing_score"] = (
            w["rsi"]       * rsi_comp +
            w["bollinger"] * bb_comp  +
            w["volume"]    * vol_comp
        )
        df["timing_gate"]      = df["timing_score"].ge(min_score)
        df["allocation_final"] = np.where(
            df["timing_gate"],
            df["allocation_after_regime"],
            0.0,
        )
    else:
        df["timing_score"]     = 1.0
        df["timing_gate"]      = True
        df["allocation_final"] = df["allocation_after_regime"]

    # NaN safety
    df["allocation_final"] = df["allocation_final"].fillna(0.0)

    return df
