"""
Production node — Coinglass intraday derivatives 4h incremental ingestion (L1).

Endpoint configuration is driven by conf/base/coinglass_endpoints.yml
(loaded via params:coinglass_api.endpoints).

Output: one partition per asset  (BTCUSDT.parquet, ETHUSDT.parquet)
        containing all configured endpoint fields merged on timestamp.
        Column format: {endpoint_name}__{canonical_field_name}

L1 contract:
- Mirror API payloads — no feature engineering, no aggregations
- UTC timestamps (datetime64[ns, UTC])
- float64 numerics via pd.to_numeric(..., errors="coerce")
- Sorted and deduplicated by timestamp
- NaN permitted where an endpoint has no data for a given candle
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import requests


COINGLASS_BASE_URL = "https://open-api-v4.coinglass.com"
_MAX_RETRIES = 5
_SLEEP_BETWEEN_ASSETS = 0.5
_SLEEP_BETWEEN_ENDPOINTS = 0.3


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fetch_with_retry(path: str, query: dict, api_key: str) -> dict | None:
    """Low-level Coinglass GET with retry/exponential backoff."""
    url = COINGLASS_BASE_URL + path
    headers = {"CG-API-KEY": api_key, "Accept": "application/json"}
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, params=query, timeout=30)
            status = resp.status_code
            if status == 200:
                return resp.json()
            if status in (429, 418) or 500 <= status < 600:
                last_error = RuntimeError(f"HTTP {status}")
            else:
                resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            last_error = exc

        time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Coinglass request failed after {_MAX_RETRIES} retries: {last_error}"
    )



def _fetch_endpoint_records(
    ep_config: dict,
    symbol: str,
    interval: str,
    exchange: str,
    api_key: str,
) -> list[dict]:
    """Fetch records for one endpoint in a single request.

    The Coinglass API (on the current plan) does not support time_params
    (start_time / end_time cause 400 "time error") and returns the most
    recent N candles where N is capped by the plan (~1080 for 4h).
    Pagination is therefore not applicable.
    """
    path = ep_config["path"]
    data_field = ep_config.get("response_format", {}).get("data_field", "data")

    # Build query without time params (they are not present in the YAML)
    query: dict = {}
    required = ep_config.get("required_params", {})
    query.update(required)

    if "symbol" in required:
        query["symbol"] = symbol
    if "exchange" in required:
        query["exchange"] = exchange
    if "exchange_list" in required:
        query["exchange_list"] = exchange
    if "interval" in required:
        query["interval"] = interval

    optional = ep_config.get("optional_params", {})
    query.update(optional)

    raw = _fetch_with_retry(path, query, api_key)
    records = (raw or {}).get(data_field, [])
    return records if isinstance(records, list) else []


def _normalize_endpoint_df(
    records: list[dict],
    ep_name: str,
    ep_config: dict,
) -> pd.DataFrame:
    """Normalize a list of raw API records to L1 DataFrame.

    Returns a DataFrame with:
        timestamp           — UTC datetime
        {ep_name}__{field}  — float64 for each value field in the fields mapping
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    fields_map: dict = ep_config.get("fields", {})  # canonical → api_field_name

    ts_api_field = fields_map.get("timestamp", "time")
    if ts_api_field not in df.columns:
        return pd.DataFrame()

    # Parse timestamp
    ts_series = df[ts_api_field]
    if pd.api.types.is_integer_dtype(ts_series) or pd.api.types.is_float_dtype(ts_series):
        unit = "ms" if ts_series.iloc[0] > 1e12 else "s"
        df["timestamp"] = pd.to_datetime(ts_series.astype("int64"), unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts_series, utc=True)

    # Map value fields → prefixed output columns
    keep = ["timestamp"]
    for canonical, api_field in fields_map.items():
        if canonical == "timestamp":
            continue
        if api_field in df.columns:
            col_out = f"{ep_name}__{canonical}"
            df[col_out] = pd.to_numeric(df[api_field], errors="coerce").astype("float64")
            keep.append(col_out)

    return (
        df[keep]
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _resolve_symbol(ep_config: dict, asset: str, coin: str) -> str:
    """Return pair symbol (BTCUSDT) or coin symbol (BTC) based on endpoint config.

    Detection: if the default symbol in required_params has more than 4 chars
    (e.g. "BTCUSDT" = 7), this is a pair-level endpoint → use asset.
    Otherwise (e.g. "BTC" = 3) it's a coin-level endpoint → use coin.
    """
    default_symbol = ep_config.get("required_params", {}).get("symbol", "")
    return asset if len(default_symbol) > 4 else coin


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def update_coinglass_4h_incremental(
    existing_data: Dict[str, pd.DataFrame] | None,
    assets: List[str],
    coins: List[str],
    interval: str,
    start_date: str,
    exchange: str,
    api_key: str,
    endpoints_config: dict,
) -> Dict[str, pd.DataFrame]:
    """
    Incrementally update Coinglass intraday (4h) derivatives per asset.

    Parameters
    ----------
    existing_data   : current PartitionedDataset (None on first run)
    assets          : pair-level symbols  ["BTCUSDT", "ETHUSDT"]
    coins           : coin-level symbols  ["BTC",     "ETH"]
    interval        : candle interval     "4h"
    start_date      : full-load start     "2024-01-01"
    exchange        : exchange filter     "Binance"
    api_key         : CG-API-KEY
    endpoints_config: dict loaded from coinglass_endpoints.yml → endpoints

    Returns
    -------
    Dict[str, DataFrame] partitioned by asset key (e.g. "BTCUSDT")
    """
    if existing_data is None:
        existing_data = {}

    asset_to_coin = dict(zip(assets, coins))
    now_utc = datetime.now(timezone.utc)
    start_ts_utc = pd.to_datetime(start_date, utc=True)

    updated: Dict[str, pd.DataFrame] = {}
    report_rows: list[dict] = []

    for asset, coin in asset_to_coin.items():

        # ── Resolve existing partition ──────────────────────────────────────
        df_existing_obj = existing_data.get(asset)
        if callable(df_existing_obj):
            df_existing = df_existing_obj()
        elif isinstance(df_existing_obj, pd.DataFrame):
            df_existing = df_existing_obj.copy()
        else:
            df_existing = pd.DataFrame()

        # ── Determine incremental start timestamp ───────────────────────────
        if not df_existing.empty and "timestamp" in df_existing.columns:
            df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], utc=True)
            last_ts = df_existing["timestamp"].max()
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            start_ts = last_ts + pd.Timedelta(milliseconds=1)
        else:
            start_ts = start_ts_utc

        if start_ts >= now_utc:
            updated[asset] = df_existing
            report_rows.append(
                {"asset": asset, "inserted_from": None, "inserted_to": None, "rows": 0}
            )
            continue

        # ── Fetch each configured endpoint ──────────────────────────────────
        endpoint_dfs: list[pd.DataFrame] = []

        for ep_name, ep_config in endpoints_config.items():
            symbol = _resolve_symbol(ep_config, asset, coin)

            try:
                records = _fetch_endpoint_records(
                    ep_config, symbol, interval, exchange, api_key
                )
                df_ep = _normalize_endpoint_df(records, ep_name, ep_config)
                if not df_ep.empty:
                    endpoint_dfs.append(df_ep)
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] {ep_name} / {asset}: {exc}")

            time.sleep(_SLEEP_BETWEEN_ENDPOINTS)

        # ── Merge all endpoint DataFrames on timestamp (outer join) ─────────
        if endpoint_dfs:
            df_new = endpoint_dfs[0]
            for df_ep in endpoint_dfs[1:]:
                df_new = pd.merge(df_new, df_ep, on="timestamp", how="outer")
            df_new = (
                df_new.sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="last")
                .reset_index(drop=True)
            )
        else:
            df_new = pd.DataFrame()

        # ── Merge existing + new ────────────────────────────────────────────
        if not df_existing.empty and not df_new.empty:
            df_merged = pd.concat([df_existing, df_new], ignore_index=True)
        elif not df_new.empty:
            df_merged = df_new
        else:
            df_merged = df_existing

        # ── Final L1 hygiene ────────────────────────────────────────────────
        if not df_merged.empty:
            df_merged["timestamp"] = pd.to_datetime(df_merged["timestamp"], utc=True)
            for col in df_merged.columns:
                if col != "timestamp":
                    df_merged[col] = (
                        pd.to_numeric(df_merged[col], errors="coerce").astype("float64")
                    )
            df_merged = (
                df_merged.sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="last")
                .reset_index(drop=True)
            )

        updated[asset] = df_merged

        inserted_from = df_new["timestamp"].min().isoformat() if not df_new.empty else None
        inserted_to = df_new["timestamp"].max().isoformat() if not df_new.empty else None
        report_rows.append(
            {
                "asset": asset,
                "inserted_from": inserted_from,
                "inserted_to": inserted_to,
                "rows": len(df_new),
            }
        )

        time.sleep(_SLEEP_BETWEEN_ASSETS)

    # ── Execution report ────────────────────────────────────────────────────
    if report_rows:
        df_report = pd.DataFrame(report_rows)
        print("\n==============================================")
        print("COINGLASS 4H INCREMENTAL UPDATE REPORT")
        print("==============================================")
        print(df_report.to_string(index=False))
        print("==============================================\n")

    return updated
