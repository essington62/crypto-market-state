from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

import time

import pandas as pd
import requests


COINGLASS_BASE_URL = "https://open-api-v4.coinglass.com"
MAX_RETRIES = 5
MAX_ITER = 10000
SLEEP_BETWEEN_CALLS = 0.5  # seconds


def _fetch_with_retry(path: str, query: dict, api_key: str) -> dict | None:
    """Low-level Coinglass GET with retry/backoff."""
    url = COINGLASS_BASE_URL + path
    headers = {
        "CG-API-KEY": api_key,
        "Accept": "application/json",
    }
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, params=query, timeout=30)
            status = resp.status_code
            if status == 200:
                return resp.json()

            # Simple retry on 429 / 5xx / 418
            if status in (429, 418) or 500 <= status < 600:
                last_error = RuntimeError(f"Coinglass HTTP {status}")
            else:
                resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            last_error = exc

        # Exponential backoff
        sleep_s = 2 ** attempt
        time.sleep(sleep_s)

    raise RuntimeError(f"Coinglass intraday request failed after retries: {last_error}")


def _normalize_intraday_df(
    raw_records: list[dict],
    ts_field: str,
    value_fields: List[str],
) -> pd.DataFrame:
    """Normalize generic intraday payload to L1 DataFrame."""
    if not raw_records:
        return pd.DataFrame()

    df = pd.DataFrame(raw_records)

    if ts_field not in df.columns:
        raise ValueError(f"Coinglass 1h: missing timestamp field '{ts_field}'.")

    # Auto-detect ms vs s; Coinglass typically uses ms
    ts_series = df[ts_field]
    if pd.api.types.is_integer_dtype(ts_series) or pd.api.types.is_float_dtype(ts_series):
        # Assume ms when values are large
        unit = "ms" if ts_series.iloc[0] > 1e12 else "s"
        df["timestamp"] = pd.to_datetime(ts_series.astype("int64"), unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts_series, utc=True)

    keep = [c for c in value_fields if c in df.columns]
    df = df[["timestamp"] + keep].copy()

    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    return df


def update_coinglass_1h_incremental(
    existing_data: Dict[str, pd.DataFrame] | Dict[str, object] | None,
    assets: List[str],
    start_date: str,
    interval: str,
) -> Dict[str, pd.DataFrame]:
    """
    Incrementally update Coinglass intraday (1h) derivatives per symbol.

    L1 contract:
    - Mirror API payload (no feature engineering)
    - UTC timestamps
    - float64 numerics
    - sorted and deduplicated by timestamp
    """
    if existing_data is None:
        existing_data = {}

    # Coinglass API key must be provided via environment variable (same as daily client).
    api_key = requests.utils.os.environ.get("COINGLASS_API_KEY")
    if not api_key:
        raise RuntimeError("COINGLASS_API_KEY not set for intraday ingestion.")

    start_ts_utc = pd.to_datetime(start_date, utc=True)
    now_utc = datetime.now(timezone.utc)

    updated: Dict[str, pd.DataFrame] = {}
    report_rows: list[dict[str, object]] = []

    for symbol in assets:
        # Existing per-symbol DataFrame if present
        df_existing_obj = existing_data.get(symbol)
        if callable(df_existing_obj):
            df_existing = df_existing_obj()
        elif isinstance(df_existing_obj, pd.DataFrame):
            df_existing = df_existing_obj
        else:
            df_existing = pd.DataFrame()

        df_existing = df_existing.copy()

        # Determine incremental start time
        if not df_existing.empty and "timestamp" in df_existing.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_existing["timestamp"]):
                df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], utc=True)
            last_ts = df_existing["timestamp"].max()
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            start_ts = last_ts + pd.Timedelta(milliseconds=1)
        else:
            start_ts = start_ts_utc

        if start_ts >= now_utc:
            # Already up-to-date: ensure L1 hygiene and continue
            if not df_existing.empty:
                df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], utc=True)
                for col in df_existing.columns:
                    if col == "timestamp":
                        continue
                    df_existing[col] = pd.to_numeric(df_existing[col], errors="coerce").astype("float64")
                df_existing = (
                    df_existing.sort_values("timestamp")
                    .drop_duplicates(subset=["timestamp"], keep="last")
                    .reset_index(drop=True)
                )
            updated[symbol] = df_existing
            report_rows.append(
                {
                    "symbol": symbol,
                    "inserted_from": None,
                    "inserted_to": None,
                    "rows": 0,
                }
            )
            continue

        # Placeholder for all intraday records; in a real implementation,
        # you would hit different endpoints (open_interest, funding_rate, etc.)
        # and merge on timestamp. Here we show a single generic call per asset.
        records: list[dict] = []

        # Reuse existing Coinglass derivatives/liquidations endpoint pattern
        # Contract: this block preserves raw fields; no feature engineering.
        raw = _fetch_with_retry(
            path="/api/futures/liquidation/aggregated-history",
            query={
                "symbol": symbol,
                "interval": interval,
                "start_time": int(start_ts.timestamp() * 1000),
                "end_time": int(now_utc.timestamp() * 1000),
            },
            api_key=api_key,
        )

        data = (raw or {}).get("data", [])
        if isinstance(data, list):
            records.extend(data)

        if not records:
            df_new = pd.DataFrame()
        else:
            # Assume API returns "timestamp" field and numeric columns for all endpoints
            value_fields = [c for c in records[0].keys() if c != "timestamp"]
            df_new = _normalize_intraday_df(records, ts_field="timestamp", value_fields=value_fields)

        # Merge existing + new
        if not df_existing.empty and not df_new.empty:
            df_merged = pd.concat([df_existing, df_new], axis=0, ignore_index=True)
        elif df_existing.empty and not df_new.empty:
            df_merged = df_new
        else:
            df_merged = df_existing

        if df_merged.empty:
            updated[symbol] = df_merged
            report_rows.append(
                {
                    "symbol": symbol,
                    "inserted_from": None,
                    "inserted_to": None,
                    "rows": 0,
                }
            )
            continue

        # Final L1 hygiene
        df_merged["timestamp"] = pd.to_datetime(df_merged["timestamp"], utc=True)
        for col in df_merged.columns:
            if col == "timestamp":
                continue
            df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce").astype("float64")

        df_merged = (
            df_merged.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )

        updated[symbol] = df_merged

        inserted_from = (
            df_new["timestamp"].min().isoformat() if not df_new.empty else None
        )
        inserted_to = (
            df_new["timestamp"].max().isoformat() if not df_new.empty else None
        )

        report_rows.append(
            {
                "symbol": symbol,
                "inserted_from": inserted_from,
                "inserted_to": inserted_to,
                "rows": int(len(df_new)),
            }
        )

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Execution report
    if report_rows:
        report_df = pd.DataFrame(report_rows)
        print("\n==============================================")
        print("COINGLASS 1H INCREMENTAL UPDATE REPORT")
        print("==============================================")
        print(report_df.to_string(index=False))
        print("==============================================\n")

    return updated

