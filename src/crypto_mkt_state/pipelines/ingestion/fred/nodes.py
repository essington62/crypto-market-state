from __future__ import annotations

from typing import Dict, List
import pandas as pd

from crypto_mkt_state.clients.fred_client import fetch_fred_batch
from crypto_mkt_state.utils.utils_temporal import enforce_l1_temporal_contract


def _infer_frequency(df: pd.DataFrame) -> str:
    """
    Infer frequency from median date diff.
    Returns: daily | weekly | monthly
    """
    if df.empty or len(df) < 3:
        return "daily"

    diffs = df["date"].diff().dropna()
    median_days = diffs.dt.days.median()

    if median_days <= 2:
        return "daily"
    elif 5 <= median_days <= 10:
        return "weekly"
    else:
        return "monthly"


def load_fred_l1(
    series: List[dict],
    start_date: str,
    interval: str,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    L1 FRED ingestion with automatic frequency routing.
    Preserves original frequency.
    """

    raw = fetch_fred_batch(
        series_ids=[s["id"] for s in series],
        start_date=start_date,
        end_date=None,
    )

    daily: Dict[str, pd.DataFrame] = {}
    weekly: Dict[str, pd.DataFrame] = {}
    monthly: Dict[str, pd.DataFrame] = {}

    for cfg in series:
        series_id = cfg["id"]
        df = raw.get(series_id)

        if df is None or df.empty:
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)

        df = (
            df.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

        df = enforce_l1_temporal_contract(
            df=df,
            start_date=start_date,
            interval=interval,
            assert_daily=False,
        )

        freq = _infer_frequency(df)

        if freq == "daily":
            daily[series_id] = df
        elif freq == "weekly":
            weekly[series_id] = df
        else:
            monthly[series_id] = df

    return {
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
    }