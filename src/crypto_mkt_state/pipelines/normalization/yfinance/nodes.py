"""
L2 normalization nodes for Yahoo Finance macro data.

This module contains nodes that perform schema/metadata normalization only
for Yahoo Finance macro time series:
- Adds metadata columns (symbol, asset, category, source, interval, ingestion_ts)
- Ensures clean date index (sorted, duplicate dates removed)

No resampling, merging or value transformations are performed here.
Each asset (partition) is processed independently.
"""

from typing import Any, Callable, Dict

import pandas as pd


def normalize_yfinance_macro(
    indices_data: Dict[str, Any],
    assets_data: Dict[str, Any],
    yfinance_config: dict,
) -> Dict[str, pd.DataFrame]:
    """
    Normalize Yahoo Finance L1 (indices + assets) to L2 intermediate.

    Consumes yfinance_indices_raw and yfinance_assets_raw (semantic split at L1).
    Merges partitions and attaches metadata from params:yfinance.indices and
    params:yfinance.assets. No return/momentum/vol logic — schema only.

    Args:
        indices_data: Partitions from yfinance_indices_raw (ticker -> loader/df).
        assets_data: Partitions from yfinance_assets_raw (ticker -> loader/df).
        yfinance_config: params:yfinance (indices + assets lists with ticker, name, category).

    Returns:
        Dict[ticker, DataFrame] normalized, compatible with yfinance_macro_intermediate.
    """
    indices_meta = (yfinance_config or {}).get("indices") or []
    assets_meta = (yfinance_config or {}).get("assets") or []
    meta_by_ticker: Dict[str, dict] = {
        m["ticker"]: m for m in indices_meta + assets_meta if "ticker" in m
    }

    def _ensure_df(v: Any) -> pd.DataFrame:
        return v() if callable(v) else v

    data: Dict[str, Any] = {}
    for k, v in (indices_data or {}).items():
        data[k] = v
    for k, v in (assets_data or {}).items():
        if k in data:
            raise ValueError(
                f"L2 YFinance: duplicate partition key '{k}' between indices and assets."
            )
        data[k] = v

    normalized: Dict[str, pd.DataFrame] = {}

    for ticker, loader in data.items():
        if callable(loader):
            df = loader()
        else:
            df = loader  # type: ignore[assignment]

        if df is None or df.empty:
            continue

        df_norm = df.copy()

        # Keep only original yfinance OHLCV schema
        df_norm = df_norm[["date", "open", "high", "low", "close", "volume"]]

        # Attach metadata
        meta: dict[str, Any] = meta_by_ticker.get(ticker, {})
        category = meta.get("category")
        asset = meta.get("name")

        df_norm["symbol"] = ticker
        df_norm["asset"] = asset
        df_norm["category"] = category
        df_norm["source"] = "yfinance"
        df_norm["interval"] = "1d"
        df_norm["ingestion_ts"] = pd.Timestamp.utcnow()

        # Ensure clean time axis: sorted and without duplicate dates
        df_norm = df_norm.sort_values("date")
        df_norm = df_norm.drop_duplicates(subset=["date"], keep="last")

        normalized[ticker] = df_norm

    return normalized
