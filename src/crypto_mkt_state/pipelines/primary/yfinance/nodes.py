"""
L3 primary feature engineering nodes for Yahoo Finance macro data.

STRICT CONTRACT:
- date must already be datetime64[ns, UTC] (guaranteed by L2)
- L3 NEVER mutates or fixes the date column
- Fail-fast if the temporal contract is broken
"""

from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from crypto_mkt_state.utils.utils_l3_semantic import get_semantic_features


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_YF_ASSET_CATEGORIES = {"equity_index", "commodity"}
_YF_INDEX_CATEGORIES = {"volatility", "rates", "fx"}
_YF_SOURCE = "yfinance"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_l3_temporal_contract(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    L3 temporal contract: date comes from L2 (datetime64[ns, UTC]).
    L3 does NOT convert or normalize date; only sorts and removes duplicate dates.
    """
    if "date" not in df.columns:
        raise ValueError(
            f"YFinance L3: partition {symbol} missing 'date' column."
        )

    dtype = df["date"].dtype
    if not isinstance(dtype, pd.DatetimeTZDtype):
        raise ValueError(
            f"YFinance L3: date is not timezone-aware for {symbol}. "
            "Expected datetime64[ns, UTC] from L2."
        )

    if str(dtype.tz) != "UTC":
        raise ValueError(
            f"YFinance L3: date timezone is not UTC for {symbol} "
            f"(found {dtype.tz})."
        )

    # Sort and dedupe only; never pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    return df


# ---------------------------------------------------------------------------
# ASSETS
# ---------------------------------------------------------------------------
def build_yfinance_assets_primary_features(
    data: Dict[str, Callable[[], pd.DataFrame]],
    semantic_config: Dict,
) -> Dict[str, pd.DataFrame]:
    """
    L3 features for YFinance ASSETS only.

    Categories: equity_index, commodity
    Features: returns, rolling_std, zscore, momentum
    """
    output: Dict[str, pd.DataFrame] = {}

    for symbol, loader in data.items():
        df = loader() if callable(loader) else loader
        if df is None or df.empty:
            continue

        df = _validate_l3_temporal_contract(df.copy(), symbol)

        source = str(df["source"].iloc[0]).strip().lower()
        category = str(df["category"].iloc[0]).strip().lower()

        if source != _YF_SOURCE or category not in _YF_ASSET_CATEGORIES:
            continue

        asset_name = str(df["asset"].iloc[0])
        features_allowed = get_semantic_features(
            asset_name=asset_name,
            semantic_config=semantic_config,
        )

        if "close" not in df.columns:
            output[symbol] = df
            continue

        close = df["close"]

        # ---- Returns ----
        if "return_1d" in features_allowed:
            df["return_1d"] = close.pct_change(1, fill_method=None)
        if "return_5d" in features_allowed:
            df["return_5d"] = close.pct_change(5, fill_method=None)
        if "return_21d" in features_allowed:
            df["return_21d"] = close.pct_change(21, fill_method=None)

        # ---- Volatility (assets only; forbidden for indices) ----
        if "rolling_std_21" in features_allowed:
            df["rolling_std_21"] = close.pct_change(1, fill_method=None).rolling(21).std()
        if "rolling_std_63" in features_allowed:
            df["rolling_std_63"] = close.pct_change(1, fill_method=None).rolling(63).std()

        # ---- Z-score ----
        if "zscore_21" in features_allowed:
            df["zscore_21"] = (close - close.rolling(21).mean()) / close.rolling(21).std()
        if "zscore_63" in features_allowed:
            df["zscore_63"] = (close - close.rolling(63).mean()) / close.rolling(63).std()

        # ---- Momentum ----
        if "momentum_21" in features_allowed:
            df["momentum_21"] = close / close.shift(21) - 1
        if "momentum_63" in features_allowed:
            df["momentum_63"] = close / close.shift(63) - 1

        output[symbol] = df

    return output


# ---------------------------------------------------------------------------
# INDICES
# ---------------------------------------------------------------------------
def build_yfinance_indices_primary_features(
    data: Dict[str, Callable[[], pd.DataFrame]],
    semantic_config: Dict,
) -> Dict[str, pd.DataFrame]:
    """
    L3 features for YFinance INDICES only.

    Categories: volatility, rates, fx
    Features: value, deltas, zscore, rolling_mean
    """
    output: Dict[str, pd.DataFrame] = {}

    for symbol, loader in data.items():
        df = loader() if callable(loader) else loader
        if df is None or df.empty:
            continue

        df = _validate_l3_temporal_contract(df.copy(), symbol)

        source = str(df["source"].iloc[0]).strip().lower()
        category = str(df["category"].iloc[0]).strip().lower()

        if source != _YF_SOURCE or category not in _YF_INDEX_CATEGORIES:
            continue

        asset_name = str(df["asset"].iloc[0])
        features_allowed = get_semantic_features(
            asset_name=asset_name,
            semantic_config=semantic_config,
        )

        if "close" not in df.columns:
            output[symbol] = df
            continue

        close = df["close"]

        # ---- Level ----
        if "value" in features_allowed:
            df["value"] = close

        # ---- Deltas ----
        if "delta_1d" in features_allowed:
            df["delta_1d"] = close.diff(1)
        if "delta_21d" in features_allowed:
            df["delta_21d"] = close.diff(21)

        # ---- Z-score / rolling mean ----
        if "zscore_63" in features_allowed:
            df["zscore_63"] = (close - close.rolling(63).mean()) / close.rolling(63).std()
        if "rolling_mean_63" in features_allowed:
            df["rolling_mean_63"] = close.rolling(63).mean()

        output[symbol] = df

    return output
