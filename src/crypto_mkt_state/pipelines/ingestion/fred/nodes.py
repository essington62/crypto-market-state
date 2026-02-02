from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

from crypto_mkt_state.clients.fred_client import fetch_fred_batch
from crypto_mkt_state.utils.utils_temporal import enforce_l1_temporal_contract


from datetime import datetime, timezone


def expand_monthly_to_daily(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand monthly macro series to daily frequency via forward-fill,
    replicating the last known value up to TODAY (UTC).

    L1 contract: continuous daily frequency (no date gaps), UTC, weekends
    filled, ZERO NaN in value. Forward-fill is applied explicitly after
    reindex; no backfill. Pure function, no I/O.
    """
    df = df.copy().sort_values("date")

    start = df["date"].min()
    end = pd.Timestamp(datetime.now(tz=timezone.utc).date(), tz="UTC")

    daily_index = pd.date_range(
        start=start,
        end=end,
        freq="D",
        tz="UTC",
    )

    out = (
        df.set_index("date")
        .reindex(daily_index, method="ffill")
        .reset_index()
        .rename(columns={"index": "date"})
    )

    # Explicit forward-fill so the last known value extends to the current date
    if "value" not in out.columns:
        raise ValueError("L1 FRED: missing 'value' column after daily expansion.")
    if out["value"].isna().any():
        out["value"] = out["value"].ffill()
    if out["value"].isna().any():
        raise ValueError(
            "L1 FRED: NaN in 'value' after forward-fill. "
            "Ensure raw series has at least one non-NaN value."
        )

    return out



def load_fred_l1(
    series: List[dict],
    start_date: str,
    interval: str,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    L1 ingestion for FRED macro data.

    Contract:
    - API already receives global.start_date
    - L1 validates and enforces the temporal cut
    - Monthly data expanded to daily macro state
    """

    raw = fetch_fred_batch(
        series_ids=[s["id"] for s in series],
        start_date=start_date,   # ✅ CORTE JÁ NA API
        end_date=end_date,
    )

    output: Dict[str, pd.DataFrame] = {}

    for cfg in series:
        series_id = cfg["id"]
        df = raw.get(series_id)

        if df is None or df.empty:
            output[series_id] = df
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

        # ✅ SEGURANÇA: enforce cut novamente
        df = enforce_l1_temporal_contract(
            df=df,
            start_date=start_date,
            interval=interval,
            assert_daily=False,
        )

        if df.empty:
            output[series_id] = df
            continue

        # ✅ EXPANSÃO MENSAL → DIÁRIO
        df_daily = expand_monthly_to_daily(df)

        output[series_id] = df_daily

    return output
