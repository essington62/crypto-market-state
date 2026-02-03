"""
L2 normalization nodes for Yahoo Finance data.

Schema-only normalization:
- Preserve OHLCV values
- Ensure date column is preserved as UTC
- Attach metadata (symbol, asset, category, source, interval, ingestion_ts)
"""

from typing import Any, Dict
import pandas as pd


def _normalize_yfinance_l2(
    data: Dict[str, Any],
    meta: list[dict],
) -> Dict[str, pd.DataFrame]:
    # build lookup once
    meta_by_symbol = {m["ticker"]: m for m in meta if "ticker" in m}

    out: Dict[str, pd.DataFrame] = {}

    for symbol, loader in data.items():
        df = loader() if callable(loader) else loader

        if df is None or df.empty:
            continue

        df = df.copy()

        # canonical OHLCV
        df = df[["date", "open", "high", "low", "close", "volume"]]

        # date: keep UTC, idempotent
        df["date"] = pd.to_datetime(df["date"], utc=True)

        meta_row = meta_by_symbol.get(symbol, {})

        df["symbol"] = symbol
        df["asset"] = meta_row.get("name")
        df["category"] = meta_row.get("category")
        df["source"] = "yfinance"
        df["interval"] = "1d"
        df["ingestion_ts"] = pd.Timestamp.utcnow()

        df = df.sort_values("date").drop_duplicates("date", keep="last")

        out[symbol] = df

    return out


# =========================
# PUBLIC NODES (WRAPPERS)
# =========================

def normalize_yfinance_indices(
    data: Dict[str, Any],
    meta: list[dict],
) -> Dict[str, pd.DataFrame]:
    """
    L2 normalization for Yahoo Finance INDICES.
    """
    return _normalize_yfinance_l2(data=data, meta=meta)


def normalize_yfinance_assets(
    data: Dict[str, Any],
    meta: list[dict],
) -> Dict[str, pd.DataFrame]:
    """
    L2 normalization for Yahoo Finance ASSETS.
    """
    return _normalize_yfinance_l2(data=data, meta=meta)
