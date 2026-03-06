"""
L1 Ingestion — Coinglass V4 (derivatives + on-chain sentiment)

Confirmed available endpoints (validated 2026-03-05):
  liquidation/aggregated-history  → per-asset (BTC, ETH), exchange_list required
  fear-greed-history              → global index, parallel lists (time/data/price)
  bitcoin/bubble-index            → on-chain composite, date_string timestamp
  puell-multiple                  → miner metric, numeric timestamp
  ahr999                          → on-chain valuation, date_string timestamp

NOT available on current plan (all return 404):
  fundingRate, openInterest, globalLongShortAccountRatio,
  nupl, altcoin_season, stablecoin_mcap, etf_flows, etf_netassets

L1 Contract:
  - No transformations beyond schema normalization
  - Index: DatetimeIndex UTC normalized to midnight
  - No fill, no resampling, no feature engineering
  - API key via os.environ["COINGLASS_API_KEY"] only
  - Returns empty dict/DataFrame on failure — never raises
"""

import logging
import os
import time
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ── Base fetch with retry ──────────────────────────────────────────────────────

def _fetch(path: str, query: dict, cg_params: dict) -> dict | None:
    import requests

    url     = cg_params["api"]["base_url"] + path
    headers = {
        "CG-API-KEY": os.environ["COINGLASS_API_KEY"],
        "Accept":     "application/json",
    }
    for attempt in range(cg_params["retry_attempts"]):
        try:
            r = requests.get(url, headers=headers, params=query, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < cg_params["retry_attempts"] - 1:
                time.sleep(cg_params["retry_backoff"])
            else:
                logger.error("Failed %s (attempt %d): %s", path, attempt + 1, e)
                return None


# ── Timestamp parsing ──────────────────────────────────────────────────────────

def _ts_to_utc(val) -> pd.Timestamp:
    """Auto-detect Unix ms / Unix s / ISO date string → UTC Timestamp."""
    if isinstance(val, str):
        return pd.Timestamp(val, tz="UTC")
    if val > 1e12:
        return pd.Timestamp(int(val), unit="ms", tz="UTC")
    return pd.Timestamp(int(val), unit="s", tz="UTC")


def _to_df(
    records: list[dict],
    ts_field: str,
    value_fields: list[str],
    start_date: str,
) -> pd.DataFrame:
    """
    Convert list of dicts → DatetimeIndex UTC DataFrame.

    - Parses ts_field via _ts_to_utc (handles ms, s, or string)
    - Normalizes to midnight UTC
    - Keeps only value_fields that are present
    - Filters >= start_date
    - Deduplicates (keep="last"), sorts ascending
    - Casts all value columns to float64
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    idx = pd.DatetimeIndex([_ts_to_utc(v) for v in df[ts_field]]).normalize()
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx
    df.index.name = "timestamp"

    keep = [c for c in value_fields if c in df.columns]
    df = df[keep].copy()
    df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    return df


# ── Liquidations (per-asset, paginated) ───────────────────────────────────────

def fetch_liquidations(params: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """
    Fetch aggregated liquidations for BTC and ETH (exchange_list=ALL).

    Endpoint: /api/futures/liquidation/aggregated-history
    Params:   symbol, exchange_list (required), interval, start_time, end_time
    Paginates in 900-day chunks from start_date to today.

    Columns: aggregated_long_liquidation_usd, aggregated_short_liquidation_usd
    Returns: {"BTCUSDT": df, "ETHUSDT": df}
    """
    cg             = params["coinglass"]
    start_date     = cg["start_date"]
    sleep_sec      = cg["request_sleep"]
    exchange_list  = cg.get("liquidation_exchange_list", "ALL")
    chunk          = pd.Timedelta(days=900)

    VALUE_FIELDS = [
        "aggregated_long_liquidation_usd",
        "aggregated_short_liquidation_usd",
    ]

    result: dict[str, pd.DataFrame] = {}

    for asset in cg["assets"]:
        symbol = asset["symbol"]
        pair   = asset["pair"]

        t_start = pd.Timestamp(start_date).normalize().tz_localize("UTC")
        t_end   = pd.Timestamp.now("UTC").normalize()

        batches: list[pd.DataFrame] = []
        cur = t_start

        while cur < t_end:
            nxt = min(cur + chunk, t_end)
            raw = _fetch(
                "/api/futures/liquidation/aggregated-history",
                {
                    "symbol":        symbol,
                    "exchange_list": exchange_list,
                    "interval":      cg["interval"],
                    "start_time":    int(cur.timestamp() * 1000),
                    "end_time":      int(nxt.timestamp() * 1000),
                },
                cg,
            )
            if raw and raw.get("code") == "0":
                df_chunk = _to_df(raw.get("data", []), "time", VALUE_FIELDS, start_date)
                if not df_chunk.empty:
                    batches.append(df_chunk)
            cur = nxt
            time.sleep(sleep_sec)

        if batches:
            df_full = (
                pd.concat(batches)
                .pipe(lambda x: x[~x.index.duplicated(keep="last")].sort_index())
            )
            logger.info("liquidations/%s: %d rows (%s → %s)",
                        pair, len(df_full),
                        df_full.index.min().date(), df_full.index.max().date())
            result[pair] = df_full
        else:
            logger.warning("liquidations/%s: no data returned", pair)

    return result


# ── Fear & Greed (global) ─────────────────────────────────────────────────────

def fetch_fear_greed(params: dict[str, Any]) -> pd.DataFrame:
    """
    Fetch Crypto Fear & Greed Index (global, no symbol).

    Endpoint: /api/index/fear-greed-history
    Response: {data_list: [float,...], price_list: [float,...], time_list: [Unix ms,...]}

    Columns: fear_greed_value, btc_price
    """
    cg         = params["coinglass"]
    start_date = cg["start_date"]

    raw = _fetch("/api/index/fear-greed-history", {}, cg)
    if raw is None or raw.get("code") != "0":
        logger.warning("fear_greed: empty or error response")
        return pd.DataFrame()

    data       = raw.get("data", {})
    time_list  = data.get("time_list",  [])
    data_list  = data.get("data_list",  [])
    price_list = data.get("price_list", [])

    if not time_list or not data_list:
        logger.warning("fear_greed: empty lists in response")
        return pd.DataFrame()

    records = [
        {"time": t, "fear_greed_value": v, "btc_price": p}
        for t, v, p in zip(time_list, data_list, price_list)
    ]
    df = _to_df(records, "time", ["fear_greed_value", "btc_price"], start_date)
    logger.info("fear_greed: %d rows (%s → %s)",
                len(df), df.index.min().date(), df.index.max().date())
    return df


# ── Bitcoin Bubble Index (global) ─────────────────────────────────────────────

def fetch_bubble_index(params: dict[str, Any]) -> pd.DataFrame:
    """
    Fetch Bitcoin Bubble Index (on-chain composite).

    Endpoint: /api/index/bitcoin/bubble-index
    Timestamp: date_string ("YYYY-MM-DD")

    Columns: price, bubble_index, mining_difficulty,
             transaction_count, address_send_count
    """
    cg         = params["coinglass"]
    start_date = cg["start_date"]

    VALUE_FIELDS = [
        "price", "bubble_index", "mining_difficulty",
        "transaction_count", "address_send_count",
    ]

    raw = _fetch("/api/index/bitcoin/bubble-index", {}, cg)
    if raw is None or raw.get("code") != "0":
        logger.warning("bubble_index: empty or error response")
        return pd.DataFrame()

    df = _to_df(raw.get("data", []), "date_string", VALUE_FIELDS, start_date)
    logger.info("bubble_index: %d rows (%s → %s)",
                len(df), df.index.min().date(), df.index.max().date())
    return df


# ── Puell Multiple (global) ───────────────────────────────────────────────────

def fetch_puell_multiple(params: dict[str, Any]) -> pd.DataFrame:
    """
    Fetch Puell Multiple (miner daily revenue / 365d MA of daily issuance).

    Endpoint: /api/index/puell-multiple
    Timestamp: timestamp (numeric — auto-detected as Unix ms or s)

    Columns: price, puell_multiple
    """
    cg         = params["coinglass"]
    start_date = cg["start_date"]

    raw = _fetch("/api/index/puell-multiple", {}, cg)
    if raw is None or raw.get("code") != "0":
        logger.warning("puell_multiple: empty or error response")
        return pd.DataFrame()

    df = _to_df(raw.get("data", []), "timestamp", ["price", "puell_multiple"], start_date)
    logger.info("puell_multiple: %d rows (%s → %s)",
                len(df), df.index.min().date(), df.index.max().date())
    return df


# ── AHR999 (global) ───────────────────────────────────────────────────────────

def fetch_ahr999(params: dict[str, Any]) -> pd.DataFrame:
    """
    Fetch AHR999 Bitcoin valuation index.

    Endpoint: /api/index/ahr999
    Timestamp: date_string ("YYYY-MM-DD")

    Columns: average_price, ahr999_value
    """
    cg         = params["coinglass"]
    start_date = cg["start_date"]

    raw = _fetch("/api/index/ahr999", {}, cg)
    if raw is None or raw.get("code") != "0":
        logger.warning("ahr999: empty or error response")
        return pd.DataFrame()

    df = _to_df(
        raw.get("data", []),
        "date_string",
        ["average_price", "ahr999_value"],
        start_date,
    )
    logger.info("ahr999: %d rows (%s → %s)",
                len(df), df.index.min().date(), df.index.max().date())
    return df
