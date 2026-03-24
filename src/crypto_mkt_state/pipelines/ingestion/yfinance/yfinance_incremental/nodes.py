from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List

import pandas as pd
import yfinance as yf

# Canonical lowercase column mapping for L1 yfinance output
_OHLCV_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize OHLCV columns to lowercase.
    Handles parquets that contain both lowercase (old) and uppercase (new) variants:
    fills lowercase NaN from uppercase, then drops the uppercase duplicates.
    """
    if df.empty:
        return df
    for up, lo in _OHLCV_RENAME.items():
        if up in df.columns:
            if lo in df.columns:
                # Patch rows where lowercase is NaN but uppercase has data
                mask = df[lo].isna() & df[up].notna()
                if mask.any():
                    df = df.copy()
                    df.loc[mask, lo] = df.loc[mask, up]
                df = df.drop(columns=[up])
            else:
                df = df.rename(columns={up: lo})
    return df


def _download_yf_daily(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Download daily data from Yahoo Finance for a single ticker.

    Handles MultiIndex columns, empty data, and data quality issues.
    """
    try:
        df = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        print(f"[YFINANCE ERROR] Failed to download {ticker}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns (common when downloading single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex: keep the first level (Price/Volume)
        df.columns = df.columns.get_level_values(0)

    # Reset index to make date a column
    df = df.reset_index()

    # Find date column (could be 'Date' or 'Datetime')
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', 'datetime']:
            date_col = col
            break

    if date_col is None:
        print(f"[YFINANCE WARN] No date column found for {ticker}")
        return pd.DataFrame()

    df = df.rename(columns={date_col: "date"})

    # Ensure date is UTC datetime
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # Keep only required columns (if they exist)
    required_cols = ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    existing_cols = [col for col in required_cols if col in df.columns]
    df = df[existing_cols]

    # Ensure all OHLCV columns are float64
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # For volatility indices (like ^VIX, ^MOVE, ^OVX), handle possible zeros
    # These are derived indices that can have zero volume but should have positive prices
    if ticker in ["^VIX", "^MOVE", "^OVX"]:
        # Volume can be zero for indices - that's acceptable
        # But ensure OHLC are positive
        ohlc_cols = ["Open", "High", "Low", "Close"]
        for col in ohlc_cols:
            if col in df.columns:
                # Replace any zeros or negatives with the previous valid value
                # (but only for volatility indices)
                mask = (df[col] <= 0) | df[col].isna()
                if mask.any():
                    # Forward fill from previous valid values
                    df[col] = df[col].mask(mask).ffill()
                    # If still NaN at start, use the next valid value
                    if len(df) > 0 and (df[col].iloc[0] <= 0 or pd.isna(df[col].iloc[0])):
                        df[col] = df[col].mask(df[col] <= 0).bfill()

    # Normalize to lowercase column names (L1 canonical format)
    df = _normalize_ohlcv_columns(df)

    # Remove rows where all OHLC are NaN
    ohlc_cols = ["open", "high", "low", "close"]
    ohlc_present = [col for col in ohlc_cols if col in df.columns]
    if ohlc_present:
        df = df.dropna(subset=ohlc_present, how="all")

    # Sort and deduplicate
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    return df


def _process_yfinance_items(
    existing_partitions: Dict[str, Callable[[], pd.DataFrame]],
    items: List[dict],
    global_start_date: str,
    group_label: str,
) -> Dict[str, pd.DataFrame]:
    """
    Core incremental update logic shared by both nodes.

    Loads existing partitions, fetches new rows, merges, deduplicates.
    Returns dict mapping partition_key → merged DataFrame.
    """
    start_date_utc = pd.to_datetime(global_start_date, utc=True).date()
    today_utc = datetime.now(timezone.utc).date()
    tomorrow_str = (today_utc + timedelta(days=1)).isoformat()

    updated: Dict[str, pd.DataFrame] = {}
    report_rows: List[Dict[str, object]] = []

    print(f"[YFINANCE INFO] Processing {len(items)} {group_label} items")

    for item in items:
        ticker = item.get("ticker")
        name = item.get("name")
        category = item.get("category", "unknown")

        if not ticker or not name:
            print(f"[YFINANCE WARN] Skipping item with missing ticker or name: {item}")
            continue

        partition_key = name

        print(f"[YFINANCE INFO] Processing {name} ({ticker}) - {category}")

        # Load existing partition if present
        if partition_key in existing_partitions:
            try:
                existing_df = existing_partitions[partition_key]()
                print(f"[YFINANCE INFO]   Loaded existing data: {len(existing_df) if existing_df is not None else 0} rows")
            except Exception as e:
                print(f"[YFINANCE ERROR]   Failed to load existing partition: {e}")
                existing_df = pd.DataFrame()
        else:
            existing_df = pd.DataFrame()
            print(f"[YFINANCE INFO]   No existing data, starting fresh")

        if not existing_df.empty:
            existing_df = existing_df.copy()
            # Normalize mixed-case columns from older parquets
            existing_df = _normalize_ohlcv_columns(existing_df)
            if "date" in existing_df.columns:
                existing_df["date"] = pd.to_datetime(existing_df["date"], utc=True)

        # Determine next start date for incremental fetch
        if not existing_df.empty and "date" in existing_df.columns:
            last_date = existing_df["date"].max().date()
            next_start_date = (last_date + timedelta(days=1)).isoformat()
        else:
            next_start_date = start_date_utc.isoformat()

        print(f"[YFINANCE INFO]   Next start date: {next_start_date}")

        # Already up to date
        if next_start_date >= tomorrow_str:
            print(f"[YFINANCE INFO]   Already up-to-date")
            if not existing_df.empty:
                updated[partition_key] = existing_df
            else:
                empty_df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
                empty_df["date"] = pd.to_datetime(empty_df["date"], utc=True)
                updated[partition_key] = empty_df
            report_rows.append({"asset": partition_key, "inserted_from": None, "inserted_to": None, "rows": 0})
            continue

        # Download new data
        try:
            df_new = _download_yf_daily(ticker=ticker, start_date=next_start_date, end_date=tomorrow_str)

            if df_new.empty:
                print(f"[YFINANCE WARN]   No data downloaded for {ticker}")
                if not existing_df.empty:
                    updated[partition_key] = existing_df
                else:
                    empty_df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
                    empty_df["date"] = pd.to_datetime(empty_df["date"], utc=True)
                    updated[partition_key] = empty_df
                report_rows.append({"asset": partition_key, "inserted_from": None, "inserted_to": None, "rows": 0})
                continue

            print(f"[YFINANCE INFO]   Downloaded {len(df_new)} new rows")

        except Exception as e:
            print(f"[YFINANCE ERROR]   Failed to download {ticker}: {e}")
            if not existing_df.empty:
                updated[partition_key] = existing_df
            else:
                empty_df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
                empty_df["date"] = pd.to_datetime(empty_df["date"], utc=True)
                updated[partition_key] = empty_df
            report_rows.append({"asset": partition_key, "inserted_from": None, "inserted_to": None, "rows": 0})
            continue

        # Merge existing and new data
        if not existing_df.empty and not df_new.empty:
            merged = pd.concat([existing_df, df_new], axis=0, ignore_index=True)
            print(f"[YFINANCE INFO]   Merged: {len(existing_df)} existing + {len(df_new)} new = {len(merged)} rows")
        elif existing_df.empty and not df_new.empty:
            merged = df_new
            print(f"[YFINANCE INFO]   Using only new data: {len(merged)} rows")
        elif not existing_df.empty and df_new.empty:
            merged = existing_df
            print(f"[YFINANCE INFO]   No new data, keeping existing: {len(merged)} rows")
        else:
            merged = pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
            print(f"[YFINANCE INFO]   No data available")

        if merged.empty:
            print(f"[YFINANCE WARN]   No data after merge")
            empty_df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
            empty_df["date"] = pd.to_datetime(empty_df["date"], utc=True)
            updated[partition_key] = empty_df
            report_rows.append({"asset": partition_key, "inserted_from": None, "inserted_to": None, "rows": 0})
            continue

        # Final cleaning
        merged["date"] = pd.to_datetime(merged["date"], utc=True)

        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")

        ohlc_cols = ["open", "high", "low", "close"]
        ohlc_present = [col for col in ohlc_cols if col in merged.columns]
        if ohlc_present:
            merged = merged.dropna(subset=ohlc_present, how="all")

        merged = (
            merged.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )

        updated[partition_key] = merged

        inserted_from = (
            pd.to_datetime(df_new["date"], utc=True).min().date().isoformat()
            if not df_new.empty else None
        )
        inserted_to = (
            pd.to_datetime(df_new["date"], utc=True).max().date().isoformat()
            if not df_new.empty else None
        )
        report_rows.append({
            "asset": partition_key,
            "inserted_from": inserted_from,
            "inserted_to": inserted_to,
            "rows": int(len(df_new)),
        })

        print(f"[YFINANCE INFO]   Final: {len(merged)} rows, range {merged['date'].min().date()} → {merged['date'].max().date()}")

    # Print report
    if report_rows:
        report_df = pd.DataFrame(report_rows)
        print("\n==============================================")
        print(f"YFINANCE INCREMENTAL UPDATE REPORT — {group_label.upper()}")
        print("==============================================")
        print(report_df.to_string(index=False))
        print("==============================================\n")

    print(f"[YFINANCE INFO] Processed {len(updated)} {group_label} items")
    return updated


def update_yfinance_indices(
    macro_daily_raw: Dict[str, Callable[[], pd.DataFrame]],
    yfinance_indices: List[dict],
    global_start_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Incrementally update Yahoo Finance macro indices (VIX, DXY).
    Reads from macro_daily_raw → writes to macro_daily_incremental.
    """
    return _process_yfinance_items(
        existing_partitions=macro_daily_raw,
        items=yfinance_indices or [],
        global_start_date=global_start_date,
        group_label="indices",
    )


def update_yfinance_assets(
    spot_business_day_raw: Dict[str, Callable[[], pd.DataFrame]],
    yfinance_assets: List[dict],
    global_start_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Incrementally update Yahoo Finance spot assets (SP500, Oil, HYG, etc.).
    Reads from spot_business_day_raw → writes to spot_business_day_incremental.
    """
    return _process_yfinance_items(
        existing_partitions=spot_business_day_raw,
        items=yfinance_assets or [],
        global_start_date=global_start_date,
        group_label="assets",
    )
