"""
L4 cross-asset regime layer.

This module does NOT compute any new statistics. It:
- Reads existing L3 columns (from FRED and YFinance primary)
- Validates that all l4.proxies exist and required signal columns are present
- Builds regime categorical/boolean columns from params:l4.regimes (thresholds only)
- Joins everything by date (inner join)

Contract:
- No rolling_*, zscore_*, return_*, volatility_* calculations
- Fail-fast if any proxy or required feature is missing
- All logic driven by params:l4 (proxies, regimes with source/signal/thresholds)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Partition loading
# ---------------------------------------------------------------------------
def _load_partition_map(
    data: Dict[str, Union[Callable[[], pd.DataFrame], pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """
    Load a PartitionedDataset-like mapping into a dict[asset_name, DataFrame].

    Keys are taken from df["asset"].iloc[0] so that FRED and YFinance
    partitions are indexed by canonical asset name (e.g. vix, sp500, cpi).
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
        asset = str(df_norm["asset"].iloc[0]).strip().lower()
        by_asset[asset] = df_norm

    return by_asset


def _safe_get_asset(
    data: Dict[str, pd.DataFrame],
    asset_name: str,
    context: str = "L4",
) -> pd.DataFrame:
    """
    Return the DataFrame for the given asset. Raise ValueError if missing.
    """
    key = asset_name.strip().lower()
    if key not in data:
        available = sorted(data.keys())
        raise ValueError(
            f"{context}: required asset '{asset_name}' not found in L3 data. "
            f"Available assets: {available}"
        )
    return data[key]


def _validate_required_columns(
    df: pd.DataFrame,
    columns: List[str],
    asset_name: str,
) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"L4: asset '{asset_name}' is missing required column(s): {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


# ---------------------------------------------------------------------------
# Regime helpers (threshold-based only; no new statistics)
# ---------------------------------------------------------------------------
def _compute_ternary_regime(
    series: pd.Series,
    high_threshold: float,
    low_threshold: float,
    high_label: str = "high",
    low_label: str = "low",
    neutral_label: str = "neutral",
) -> pd.Series:
    """
    Map a numeric series to a categorical regime: high, neutral, or low.

    Above high_threshold -> high_label; below low_threshold -> low_label; else neutral.
    NaN in input remains NaN in output.
    """
    out = pd.Series(index=series.index, dtype=object)
    out.loc[series > high_threshold] = high_label
    out.loc[series < low_threshold] = low_label
    out.loc[(series >= low_threshold) & (series <= high_threshold)] = neutral_label
    out.loc[series.isna()] = pd.NA
    return out


def _compute_binary_regime(
    series: pd.Series,
    threshold: float,
    high_label: str = "high",
    neutral_label: str = "neutral",
) -> pd.Series:
    """
    Map a numeric series to a binary regime: above threshold -> high, else neutral.
    """
    out = pd.Series(index=series.index, dtype=object)
    out.loc[series > threshold] = high_label
    out.loc[series <= threshold] = neutral_label
    out.loc[series.isna()] = pd.NA
    return out


# ---------------------------------------------------------------------------
# Merge by date (inner join)
# ---------------------------------------------------------------------------
def _merge_on_date(
    base: Optional[pd.DataFrame],
    df: pd.DataFrame,
    columns: Dict[str, str],
) -> pd.DataFrame:
    """
    Merge selected columns from df into base on date (inner join).
    """
    # date entra uma única vez, sempre
    subset = df[["date", *columns.keys()]].rename(columns=columns)
    subset = subset.loc[:, ~subset.columns.duplicated()]

    if base is None:
        return subset.copy()

    base = base.loc[:, ~base.columns.duplicated()]
    return base.merge(subset, on="date", how="inner")



# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------
def build_cross_asset_features(
    fred: Dict[str, Union[Callable[[], pd.DataFrame], pd.DataFrame]],
    yfinance: Dict[str, Union[Callable[[], pd.DataFrame], pd.DataFrame]],
    l4_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build L4 cross-asset regime features from L3 data only.

    - Validates that every asset in l4_config.proxies exists in L3 data.
    - For each regime in l4_config.regimes: loads source asset, validates signal
      column, applies thresholds, produces a categorical regime column.
    - Joins all series on date (inner). No new statistics are computed.

    Args:
        fred: Partitioned L3 FRED data (partition key -> loader or DataFrame).
        yfinance: Partitioned L3 YFinance data (partition key -> loader or DataFrame).
        l4_config: params:l4 (proxies, regimes with source/signal/thresholds).

    Returns:
        One DataFrame with columns: date, volatility_regime, dollar_regime,
        rates_regime, inflation_regime (and any further regimes defined in l4).

    Raises:
        ValueError: If any proxy is missing or any required signal column is absent.
    """
    fred_by_asset = _load_partition_map(fred)
    yf_by_asset = _load_partition_map(yfinance)
    all_assets: Dict[str, pd.DataFrame] = {**yf_by_asset, **fred_by_asset}

    proxies = l4_config.get("proxies") or {}
    regimes = l4_config.get("regimes") or {}
    validation = l4_config.get("validation") or {}
    require_proxies = validation.get("require_all_proxies", True)

    # 1) Validate all proxies exist (fail-fast)
    if require_proxies:
        for role, asset_name in proxies.items():
            _safe_get_asset(all_assets, asset_name, context=f"L4 proxy '{role}'")

    # 2) Build base by joining regime series on date
    cross: Optional[pd.DataFrame] = None

    for regime_name, regime_cfg in regimes.items():
        source_asset = regime_cfg.get("source")
        signal_col = regime_cfg.get("signal")
        if not source_asset or not signal_col:
            raise ValueError(
                f"L4 regime '{regime_name}' must have 'source' and 'signal' in params:l4.regimes"
            )

        df = _safe_get_asset(all_assets, source_asset, context=f"L4 regime '{regime_name}'")
        _validate_required_columns(df, ["date", signal_col], source_asset)

        series = df[signal_col]

        # Ternary or binary from params only (no new statistics)
        if "rising_threshold" in regime_cfg and "falling_threshold" in regime_cfg:
            high_t = float(regime_cfg["rising_threshold"])
            low_t = float(regime_cfg["falling_threshold"])
            regime_series = _compute_ternary_regime(
                series, high_t, low_t,
                high_label="rising", low_label="falling", neutral_label="neutral",
            )
        elif "strong_threshold" in regime_cfg and "weak_threshold" in regime_cfg:
            high_t = float(regime_cfg["strong_threshold"])
            low_t = float(regime_cfg["weak_threshold"])
            regime_series = _compute_ternary_regime(
                series, high_t, low_t,
                high_label="strong", low_label="weak", neutral_label="neutral",
            )
        elif "high_threshold" in regime_cfg and "low_threshold" in regime_cfg:
            high_t = float(regime_cfg["high_threshold"])
            low_t = float(regime_cfg["low_threshold"])
            regime_series = _compute_ternary_regime(
                series, high_t, low_t,
                high_label="high", low_label="low", neutral_label="neutral",
            )
        elif "high_threshold" in regime_cfg:
            regime_series = _compute_binary_regime(
                series, float(regime_cfg["high_threshold"]),
                high_label="high", neutral_label="neutral",
            )
        else:
            raise ValueError(
                f"L4 regime '{regime_name}' has no recognized threshold keys "
                "(high_threshold/low_threshold, strong_threshold/weak_threshold, "
                "rising_threshold/falling_threshold)."
            )

        regime_df = df[["date"]].copy()
        regime_df[f"{regime_name}_regime"] = regime_series.values
        cross = _merge_on_date(
            cross,
            regime_df,
            {f"{regime_name}_regime": f"{regime_name}_regime"},
        )

    if cross is None:
        return pd.DataFrame(columns=["date"])

    return (
        cross.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )
