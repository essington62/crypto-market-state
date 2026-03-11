from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List

import pandas as pd
import yfinance as yf


def _download_yf_daily(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily data from Yahoo Finance for a single ticker."""
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c) for c in df.columns]

    if "Date" not in df.columns:
        raise ValueError(f"yfinance incremental: missing 'Date' column for ticker {ticker}")

    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)

    keep_cols = ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[keep_cols]

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    return df


def update_yfinance_incremental(
    spot_business_day_raw: Dict[str, Callable[[], pd.DataFrame]],
    yfinance_indices: List[dict],
    yfinance_assets: List[dict],
    global_start_date: str,
) -> Dict[str, pd.DataFrame]:

    start_date_utc = pd.to_datetime(global_start_date, utc=True).date()

    today_utc = datetime.now(timezone.utc).date()

    # Yahoo end_date is EXCLUSIVE → use tomorrow
    tomorrow_utc = today_utc + timedelta(days=1)
    tomorrow_str = tomorrow_utc.isoformat()

    updated: Dict[str, pd.DataFrame] = {}
    report_rows: List[Dict[str, object]] = []

    all_items: List[dict] = []
    all_items.extend(yfinance_indices or [])
    all_items.extend(yfinance_assets or [])

    for item in all_items:

        ticker = item.get("ticker")
        name = item.get("name")

        if not ticker or not name:
            continue

        partition_key = name

        if partition_key in spot_business_day_raw:
            existing_df = spot_business_day_raw[partition_key]()
        else:
            existing_df = pd.DataFrame(
                columns=["date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            )

        existing_df = existing_df.copy()

        if not existing_df.empty and "date" in existing_df.columns:
            existing_df["date"] = pd.to_datetime(existing_df["date"], utc=True)
            last_date = existing_df["date"].max().date()
            next_start_date = (last_date + timedelta(days=1)).isoformat()
        else:
            next_start_date = start_date_utc.isoformat()

        # already up to date
        if next_start_date >= tomorrow_str:

            updated[partition_key] = existing_df

            report_rows.append(
                {
                    "asset": partition_key,
                    "inserted_from": None,
                    "inserted_to": None,
                    "rows": 0,
                }
            )
            continue

        df_new = _download_yf_daily(
            ticker=ticker,
            start_date=next_start_date,
            end_date=tomorrow_str,
        )

        if not existing_df.empty and not df_new.empty:
            merged = pd.concat([existing_df, df_new], axis=0, ignore_index=True)
        elif existing_df.empty and not df_new.empty:
            merged = df_new
        else:
            merged = existing_df

        if merged.empty:

            updated[partition_key] = merged

            report_rows.append(
                {
                    "asset": partition_key,
                    "inserted_from": None,
                    "inserted_to": None,
                    "rows": 0,
                }
            )
            continue

        merged["date"] = pd.to_datetime(merged["date"], utc=True)

        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")

        merged = (
            merged.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

        updated[partition_key] = merged

        inserted_from = (
            pd.to_datetime(df_new["date"], utc=True).min().date().isoformat()
            if not df_new.empty
            else None
        )

        inserted_to = (
            pd.to_datetime(df_new["date"], utc=True).max().date().isoformat()
            if not df_new.empty
            else None
        )

        report_rows.append(
            {
                "asset": partition_key,
                "inserted_from": inserted_from,
                "inserted_to": inserted_to,
                "rows": int(len(df_new)),
            }
        )

    if report_rows:
        report_df = pd.DataFrame(report_rows)

        print("\n==============================================")
        print("YFINANCE INCREMENTAL UPDATE REPORT")
        print("==============================================")
        print(report_df.to_string(index=False))
        print("==============================================\n")

    return updated