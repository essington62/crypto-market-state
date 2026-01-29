"""
L3 primary feature engineering nodes for Yahoo Finance macro data.

This module is split into two nodes with strictly separated responsibilities:

1) build_yfinance_assets_primary_features
   - Processes ONLY assets: source=="yfinance", category in {equity_index, commodity}.
   - Features: return_*, rolling_std_*, zscore_*, momentum_* (only if in l3_semantic.assets).
   - Does NOT compute value, delta_*, rolling_mean_63.

2) build_yfinance_indices_primary_features
   - Processes ONLY indices: source=="yfinance", category in {volatility, rates, fx}.
   - Features: value, delta_1d, delta_21d, zscore_63, rolling_mean_63 (only if in l3_semantic.indices).
   - Does NOT compute return_*, rolling_std_*, momentum_*, log_return_*.

Both nodes use get_semantic_features(asset_name, semantic_config) and compute
ONLY features explicitly listed in parameters.yml (l3_semantic).
"""

from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from crypto_mkt_state.utils.utils_l3_semantic import get_semantic_features


# ---------------------------------------------------------------------------
# Constants: filters by spec
# ---------------------------------------------------------------------------
_YF_ASSET_CATEGORIES = {"equity_index", "commodity"}
_YF_INDEX_CATEGORIES = {"volatility", "rates", "fx"}
_YF_SOURCE = "yfinance"


def _normalize_and_validate(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Sort by date, drop duplicate dates, ensure required columns."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if "asset" not in df.columns:
        raise ValueError(
            f"YFinance partition {symbol} missing 'asset' column. "
            "L2 normalization should have added this."
        )
    return df


def build_yfinance_assets_primary_features(
    data: Dict[str, Callable[[], pd.DataFrame]],
    semantic_config: Dict,
) -> Dict[str, pd.DataFrame]:
    """
    Build L3 primary features for YFinance ASSETS only.

    Filter: source == "yfinance", category in {equity_index, commodity}.
    Examples: sp500, nasdaq, gold.

    Features (only if listed in l3_semantic.assets[asset].features):
    - return_1d, return_5d, return_21d
    - rolling_std_21, rolling_std_63
    - zscore_21, zscore_63
    - momentum_21, momentum_63

    This node does NOT compute: value, delta_1d, delta_21d, rolling_mean_63.
    """
    output: Dict[str, pd.DataFrame] = {}

    for symbol, loader in data.items():
        df = loader() if callable(loader) else loader
        if df is None or df.empty:
            continue

        df = _normalize_and_validate(df, symbol)
        source = str(df["source"].iloc[0]).strip().lower() if "source" in df.columns else ""
        category = str(df["category"].iloc[0]).strip().lower() if "category" in df.columns else ""

        if source != _YF_SOURCE or category not in _YF_ASSET_CATEGORIES:
            continue

        asset_name = str(df["asset"].iloc[0])
        try:
            features_allowed = get_semantic_features(
                asset_name=asset_name,
                semantic_config=semantic_config,
            )
        except ValueError as e:
            raise ValueError(
                f"YFinance asset partition {symbol} (asset={asset_name}): {e}"
            )

        if "close" not in df.columns:
            output[symbol] = df
            continue

        close = df["close"]

        # ---- Returns (assets only) ----
        if "return_1d" in features_allowed:
            df["return_1d"] = close.pct_change(1, fill_method=None)
        if "return_5d" in features_allowed:
            df["return_5d"] = close.pct_change(5, fill_method=None)
        if "return_21d" in features_allowed:
            df["return_21d"] = close.pct_change(21, fill_method=None)

        # ---- Volatility (rolling std of returns) ----
        if "rolling_std_21" in features_allowed:
            tmp_ret = close.pct_change(1, fill_method=None)
            df["rolling_std_21"] = tmp_ret.rolling(21).std()
        if "rolling_std_63" in features_allowed:
            tmp_ret = close.pct_change(1, fill_method=None)
            df["rolling_std_63"] = tmp_ret.rolling(63).std()

        # ---- Z-scores (level) ----
        if "zscore_21" in features_allowed:
            mean_21 = close.rolling(21).mean()
            std_21 = close.rolling(21).std()
            df["zscore_21"] = (close - mean_21) / std_21
        if "zscore_63" in features_allowed:
            mean_63 = close.rolling(63).mean()
            std_63 = close.rolling(63).std()
            df["zscore_63"] = (close - mean_63) / std_63

        # ---- Momentum ----
        if "momentum_21" in features_allowed:
            df["momentum_21"] = close / close.shift(21) - 1
        if "momentum_63" in features_allowed:
            df["momentum_63"] = close / close.shift(63) - 1

        output[symbol] = df

    return output


def build_yfinance_indices_primary_features(
    data: Dict[str, Callable[[], pd.DataFrame]],
    semantic_config: Dict,
) -> Dict[str, pd.DataFrame]:
    """
    Build L3 primary features for YFinance INDICES only.

    Filter: source == "yfinance", category in {volatility, rates, fx}.
    Examples: vix, dxy, us_10y_yield (from YFinance).

    Features (only if listed in l3_semantic.indices[asset].features):
    - value (level = close)
    - delta_1d, delta_21d
    - zscore_63
    - rolling_mean_63

    This node does NOT compute: return_*, rolling_std_*, momentum_*, log_return_*.
    VIX and other indices never get percent return features.
    """
    output: Dict[str, pd.DataFrame] = {}

    for symbol, loader in data.items():
        df = loader() if callable(loader) else loader
        if df is None or df.empty:
            continue

        df = _normalize_and_validate(df, symbol)
        source = str(df["source"].iloc[0]).strip().lower() if "source" in df.columns else ""
        category = str(df["category"].iloc[0]).strip().lower() if "category" in df.columns else ""

        if source != _YF_SOURCE or category not in _YF_INDEX_CATEGORIES:
            continue

        asset_name = str(df["asset"].iloc[0])
        try:
            features_allowed = get_semantic_features(
                asset_name=asset_name,
                semantic_config=semantic_config,
            )
        except ValueError as e:
            raise ValueError(
                f"YFinance index partition {symbol} (asset={asset_name}): {e}"
            )

        if "close" not in df.columns:
            output[symbol] = df
            continue

        close = df["close"]

        # ---- Level (index-only) ----
        if "value" in features_allowed:
            df["value"] = close

        # ---- Deltas (index-only) ----
        if "delta_1d" in features_allowed:
            df["delta_1d"] = close.diff(1)
        if "delta_21d" in features_allowed:
            df["delta_21d"] = close.diff(21)

        # ---- Z-score and rolling mean (index-only) ----
        if "zscore_63" in features_allowed:
            mean_63 = close.rolling(63).mean()
            std_63 = close.rolling(63).std()
            df["zscore_63"] = (close - mean_63) / std_63
        if "rolling_mean_63" in features_allowed:
            df["rolling_mean_63"] = close.rolling(63).mean()

        output[symbol] = df

    return output


def merge_yfinance_primary_partitions(
    assets_data: Dict[str, pd.DataFrame],
    indices_data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Merge asset and index L3 partitions into a single dict for downstream (e.g. L4).

    Keys are partition IDs (e.g. symbol); no key overlap between assets and indices.
    Accepts either Dict[str, pd.DataFrame] or Dict[str, Callable] (lazy load).
    """
    def _ensure_df(v):
        return v() if callable(v) else v

    out = {k: _ensure_df(v) for k, v in assets_data.items()}
    for k, v in indices_data.items():
        out[k] = _ensure_df(v)
    return out
