"""
Kedro node for Yahoo Finance L1 ingestion.

L1 YFinance output is intentionally continuous daily (UTC): weekends and holidays
are filled via forward-fill so that all series have temporal parity with 24x7
assets (e.g. BTC). L2/L3/L4 do not perform calendar fill.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from crypto_mkt_state.clients.yfinance_client import fetch_yfinance_batch
from crypto_mkt_state.utils_temporal import enforce_l1_temporal_contract


def expand_to_continuous_daily(
    df: pd.DataFrame,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Expand a daily series with gaps (weekends/holidays) to continuous daily (UTC).

    Uses forward-fill only (never backfill). The upper bound of the daily range
    is the last timestamp present in the series, or `end_date` if provided
    (no datetime.now() — deterministic).

    Design:
    - L1 YFinance already outputs continuous daily; this helper implements that.
    - Forward-fill represents "market state" (last known value until next open),
      not an event. Original source granularity (business days only) is not
      preserved in L1; this is intentional so L2/L3/L4 need no calendar logic.

    Args:
        df: DataFrame with a 'date' column (datetime, UTC) and value columns.
        end_date: Optional upper bound (YYYY-MM-DD, UTC). If None, uses max(date).

    Returns:
        DataFrame reindexed to a full daily range (start=min(date), end=…), ffilled.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    start = out["date"].min()
    end_ts = out["date"].max()
    if end_date is not None:
        end_param = pd.to_datetime(end_date, utc=True).normalize()
        end_ts = min(end_ts, end_param)
    end = end_ts.normalize() if hasattr(end_ts, "normalize") else end_ts

    daily_index = pd.date_range(start=start, end=end, freq="D", tz="UTC")
    out = (
        out.set_index("date")
        .reindex(daily_index, method="ffill")
        .reset_index()
        .rename(columns={"index": "date"})
    )
    return out


def _load_yfinance_l1_impl(
    items: List[dict],
    start_date: str,
    interval: str,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Internal: load YFinance data and expand to continuous daily (UTC).
    Used by both load_yfinance_indices_l1 and load_yfinance_assets_l1.
    Pure function, no I/O beyond what the client does.
    """
    tickers = [item["ticker"] for item in items]

    raw = fetch_yfinance_batch(
        tickers=tickers,
        start_date=start_date,
    )

    output: Dict[str, pd.DataFrame] = {}

    for ticker, df in raw.items():
        if df is None or df.empty:
            output[ticker] = df if df is not None else pd.DataFrame()
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

        df = expand_to_continuous_daily(df, end_date=end_date)

        if df.empty:
            output[ticker] = df
            continue

        df = enforce_l1_temporal_contract(
            df=df,
            start_date=start_date,
            interval=interval,
        )

        output[ticker] = df

    return output


def load_yfinance_indices_l1(
    indices: List[dict],
    start_date: str,
    interval: str,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    L1 ingestion for Yahoo Finance MACRO INDICES only (e.g. VIX, DXY, GSPC).

    Semantic contract: indices are context (level, z-score, deltas). They must
    NEVER receive return_1d, momentum, or volatility features in L2/L3. This
    node only loads and expands to continuous daily (UTC); no feature logic.

    Same mechanical behavior as _load_yfinance_l1_impl: continuous daily,
    forward-fill weekends/holidays, UTC, pure function.
    """
    return _load_yfinance_l1_impl(
        items=indices,
        start_date=start_date,
        interval=interval,
        end_date=end_date,
    )


def load_yfinance_assets_l1(
    assets: List[dict],
    start_date: str,
    interval: str,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    L1 ingestion for Yahoo Finance TRADABLE ASSETS only (e.g. AAPL, MSFT).

    Semantic contract: assets are tradable; L2/L3 may compute return_1d,
    momentum, volatility. This node only loads and expands to continuous
    daily (UTC); no feature logic.
    """
    return _load_yfinance_l1_impl(
        items=assets,
        start_date=start_date,
        interval=interval,
        end_date=end_date,
    )
