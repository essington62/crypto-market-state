from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List

import pandas as pd

from crypto_mkt_state.clients.fred_client import fetch_fred_series


def update_fred_incremental(
    macro_daily_raw: Dict[str, Callable[[], pd.DataFrame]],
    fred_series: List[dict],
    global_start_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Incrementally update FRED daily macro series (L1).

    Parameters
    ----------
    macro_daily_raw
        PartitionedDataset mapping series_id -> loader callable.
    fred_series
        List of series configs from params:fred.series (must contain 'id').
    global_start_date
        ISO date string used as initial start_date when a partition does not exist.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Updated partitions for macro_daily_incremental.
    """
    if not fred_series:
        raise ValueError("ingestion.fred.incremental: params:fred.series is empty.")

    start_date_utc = pd.to_datetime(global_start_date, utc=True).normalize()
    today_utc = datetime.now(timezone.utc).date()
    today_str = today_utc.isoformat()

    updated: Dict[str, pd.DataFrame] = {}

    for cfg in fred_series:
        series_id = cfg.get("id")
        if not series_id:
            continue

        # Load existing partition if present
        if series_id in macro_daily_raw:
            existing_df = macro_daily_raw[series_id]()
        else:
            existing_df = pd.DataFrame(columns=["date", "value"])

        existing_df = existing_df.copy()

        # Determine next start_date
        if not existing_df.empty and "date" in existing_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(existing_df["date"]):
                existing_df["date"] = pd.to_datetime(existing_df["date"], utc=True)
            last_date = existing_df["date"].max().date()
            next_start_date = (last_date + timedelta(days=1)).isoformat()
        else:
            next_start_date = start_date_utc.date().isoformat()

        # Already up-to-date
        if next_start_date > today_str:
            # Ensure contract: sorted, deduped, UTC
            if not existing_df.empty:
                existing_df["date"] = pd.to_datetime(existing_df["date"], utc=True)
                existing_df = (
                    existing_df.sort_values("date")
                    .drop_duplicates(subset=["date"], keep="last")
                    .reset_index(drop=True)
                )
                existing_df["value"] = existing_df["value"].astype("float64")
            updated[series_id] = existing_df
            continue

        # Fetch new observations from FRED (client already enforces L1 hygiene)
        new_df = fetch_fred_series(
            series_id=series_id,
            start_date=next_start_date,
            end_date=today_str,
        )

        # Merge existing + new
        if not existing_df.empty and not new_df.empty:
            merged = pd.concat([existing_df, new_df], axis=0, ignore_index=True)
        elif existing_df.empty and not new_df.empty:
            merged = new_df
        else:
            merged = existing_df

        if merged.empty:
            updated[series_id] = merged
            continue

        # Final L1 cleaning: date UTC, value float64, sort, dedupe
        merged["date"] = pd.to_datetime(merged["date"], utc=True)
        merged["value"] = pd.to_numeric(merged["value"], errors="coerce").astype("float64")

        merged = (
            merged.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

        updated[series_id] = merged

    return updated

