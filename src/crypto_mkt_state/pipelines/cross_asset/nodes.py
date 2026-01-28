"""
L4 cross-asset feature engineering nodes.

This module builds a single, non-partitioned dataset of global market state
features by combining:
- FRED macro series (rates, inflation, growth, liquidity)
- Yahoo Finance macro assets (equity indices, VIX, DXY, gold)

The goal is to produce one row per date describing the market regime in terms of:
- Risk / stress
- Liquidity / dollar
- Macro regime (rates, inflation, growth)
- Cross-asset relationships
- Volatility regime

Design:
- Pure functions (no IO, no Kedro Dataset usage)
- No resampling, no forward-fill, no timezone changes
- Inner joins on date only
- NaNs are allowed (burn-in, missing assets)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _load_partition_map(
    data: Dict[str, Callable[[], pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """
    Load a PartitionedDataset-like mapping into a dict[asset, DataFrame].

    Supports both callables and direct DataFrames.
    Ensures defensive copy, sorted dates and no duplicate dates.
    """
    by_asset: Dict[str, pd.DataFrame] = {}

    for _, loader in data.items():
        df = loader() if callable(loader) else loader
        if df is None or df.empty:
            continue

        df_norm = (
            df.copy()
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
        )

        asset = str(df_norm["asset"].iloc[0])
        by_asset[asset] = df_norm

    return by_asset


def _merge_on_date(
    base: Optional[pd.DataFrame],
    df: pd.DataFrame,
    columns: Dict[str, str],
) -> pd.DataFrame:
    """
    Merge selected columns from df into base using inner join on date.
    """
    subset = df[["date", *columns.keys()]].rename(columns=columns)

    if base is None:
        return subset.copy()

    return base.merge(subset, on="date", how="inner")


# ---------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------
def build_cross_asset_features(
    fred: Dict[str, Callable[[], pd.DataFrame]],
    yfinance: Dict[str, Callable[[], pd.DataFrame]],
) -> pd.DataFrame:
    """
    Build L4 cross-asset features describing the global market state.
    """
    fred_by_asset = _load_partition_map(fred)
    yf_by_asset = _load_partition_map(yfinance)

    cross: Optional[pd.DataFrame] = None

    # ==============================================================
    # 1. RISK / STRESS
    # ==============================================================
    vix_df = yf_by_asset.get("vix")
    sp500_df = yf_by_asset.get("sp500")

    if vix_df is not None:
        cross = _merge_on_date(
            cross,
            vix_df,
            {
                "close": "vix_level",
                "zscore_63": "vix_zscore_63",
            },
        )

    if sp500_df is not None:
        cross = _merge_on_date(
            cross,
            sp500_df,
            {
                "return_21d": "sp500_return_21d",
                "rolling_std_63": "equity_vol_63",
                "zscore_63": "sp500_zscore_63",
            },
        )

    if cross is None:
        return pd.DataFrame(columns=["date"])

    cross["equity_vol_risk_index"] = (
        cross["vix_zscore_63"] - cross["sp500_return_21d"]
        if {"vix_zscore_63", "sp500_return_21d"} <= set(cross.columns)
        else np.nan
    )

    # ==============================================================
    # 2. LIQUIDITY / DOLLAR
    # ==============================================================
    dxy_df = yf_by_asset.get("dxy")
    if dxy_df is not None:
        cross = _merge_on_date(
            cross,
            dxy_df,
            {"zscore_252": "dxy_zscore_252"},
        )

    # Yield curve (explicit)
    long_rate_df = None
    short_rate_df = None

    for asset, df in fred_by_asset.items():
        category = str(df["category"].iloc[0]).lower()
        name = asset.lower()

        if category == "yield_curve":
            if "10y" in name or "30y" in name:
                long_rate_df = df
            elif "2y" in name or "3m" in name:
                short_rate_df = df

    # Inflation
    inflation_df = next(
        (df for df in fred_by_asset.values()
         if str(df["category"].iloc[0]).lower() == "inflation"),
        None,
    )

    # Real rate proxy
    if long_rate_df is not None and inflation_df is not None:
        rr = long_rate_df[["date", "value"]].rename(columns={"value": "long_rate"})
        infl = inflation_df[["date", "zscore_252"]].rename(
            columns={"zscore_252": "inflation_zscore_252"}
        )
        rr = rr.merge(infl, on="date", how="inner")
        rr["real_rate_proxy"] = rr["long_rate"] - rr["inflation_zscore_252"]

        cross = _merge_on_date(
            cross,
            rr,
            {"real_rate_proxy": "real_rate_proxy"},
        )
    else:
        cross["real_rate_proxy"] = np.nan

    # ==============================================================
    # 3. MACRO REGIME (explicit growth roles)
    # ==============================================================
    growth_real_df = None
    growth_labor_df = None

    for asset, df in fred_by_asset.items():
        category = str(df["category"].iloc[0]).lower()
        if category == "growth_real":
            growth_real_df = df
        elif category == "growth_labor":
            growth_labor_df = df

    # Yield curve slope
    if long_rate_df is not None and short_rate_df is not None:
        yc = (
            long_rate_df[["date", "value"]]
            .rename(columns={"value": "long_rate"})
            .merge(
                short_rate_df[["date", "value"]].rename(
                    columns={"value": "short_rate"}
                ),
                on="date",
                how="inner",
            )
        )
        yc["yield_curve_slope"] = yc["long_rate"] - yc["short_rate"]

        cross = _merge_on_date(
            cross,
            yc,
            {"yield_curve_slope": "yield_curve_slope"},
        )
    else:
        cross["yield_curve_slope"] = np.nan

    # Growth vs inflation (INDPRO)
    if growth_real_df is not None and inflation_df is not None:
        gi = (
            growth_real_df[["date", "zscore_252"]]
            .rename(columns={"zscore_252": "growth_real_zscore_252"})
            .merge(
                inflation_df[["date", "zscore_252"]].rename(
                    columns={"zscore_252": "inflation_zscore_252"}
                ),
                on="date",
                how="inner",
            )
        )
        gi["growth_vs_inflation_score"] = (
            gi["growth_real_zscore_252"] - gi["inflation_zscore_252"]
        )

        cross = _merge_on_date(
            cross,
            gi,
            {"growth_vs_inflation_score": "growth_vs_inflation_score"},
        )
    else:
        cross["growth_vs_inflation_score"] = np.nan

    # Growth confirmation (INDPRO vs PAYEMS)
    if growth_real_df is not None and growth_labor_df is not None:
        gc = (
            growth_real_df[["date", "zscore_252"]]
            .rename(columns={"zscore_252": "growth_real_zscore_252"})
            .merge(
                growth_labor_df[["date", "zscore_252"]].rename(
                    columns={"zscore_252": "growth_labor_zscore_252"}
                ),
                on="date",
                how="inner",
            )
        )
        gc["growth_confirmation_gap"] = (
            gc["growth_real_zscore_252"] - gc["growth_labor_zscore_252"]
        )

        cross = _merge_on_date(
            cross,
            gc,
            {"growth_confirmation_gap": "growth_confirmation_gap"},
        )
    else:
        cross["growth_confirmation_gap"] = np.nan

    # ==============================================================
    # 4. CROSS-ASSET RELATIONSHIPS
    # ==============================================================
    gold_df = yf_by_asset.get("gold")
    if gold_df is not None and sp500_df is not None:
        ge = (
            gold_df[["date", "close"]].rename(columns={"close": "gold_close"})
            .merge(
                sp500_df[["date", "close"]].rename(
                    columns={"close": "sp500_close"}
                ),
                on="date",
                how="inner",
            )
        )
        ge["gold_vs_equity_ratio"] = ge["gold_close"] / ge["sp500_close"]

        cross = _merge_on_date(
            cross,
            ge,
            {"gold_vs_equity_ratio": "gold_vs_equity_ratio"},
        )
    else:
        cross["gold_vs_equity_ratio"] = np.nan

    cross["equity_vs_crypto_momentum"] = np.nan

    # ==============================================================
    # 5. VOLATILITY REGIME
    # ==============================================================
    cross["vol_regime_flag"] = (
        cross["sp500_zscore_63"] > 1.5
        if "sp500_zscore_63" in cross.columns
        else np.nan
    )

    return (
        cross.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )

