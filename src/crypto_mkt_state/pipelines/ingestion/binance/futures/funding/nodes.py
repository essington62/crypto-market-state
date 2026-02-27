"""
L1 raw layer: Binance USDT-M Perpetual Futures Funding Rates.

Contract:
- L1 = mirror of the API. No features, no derived metrics.
- Only minimal typing and schema normalization (rename fundingTime → open_time).
- Endpoint: GET /fapi/v1/fundingRate
- All timestamps in UTC.
- Forward pagination via startTime to ensure full historical coverage.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import pandas as pd
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

FUNDING_API_LIMIT_MAX = 1000
RATE_LIMIT_SLEEP_SEC = 0.4


def _fetch_funding_page(
    base_url: str,
    symbol: str,
    limit: int,
    start_time_ms: int,
) -> List[dict]:

    params = {
        "symbol": symbol,
        "limit": limit,
        "startTime": start_time_ms,
    }

    query = urlencode(params)
    url = f"{base_url.rstrip('/')}/fapi/v1/fundingRate?{query}"

    req = Request(url, method="GET")

    try:
        with urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise ValueError(
                    f"Binance Futures Funding L1: HTTP {resp.status} for {url}"
                )
            payload = resp.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        raise ValueError(
            f"Binance Futures Funding L1: request failed for {url}"
        ) from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Binance Futures Funding L1: invalid JSON response"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            "Binance Futures Funding L1: expected list response."
        )

    return data


def _fetch_symbol_funding_hist(
    base_url: str,
    symbol: str,
    start_date: str,
    limit: int,
) -> pd.DataFrame:

    start_dt = pd.to_datetime(start_date, utc=True)
    current_ms = int(start_dt.timestamp() * 1000)
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    request_limit = min(int(limit), FUNDING_API_LIMIT_MAX)

    all_rows: List[dict] = []
    iterations = 0
    max_iter = 50000

    while current_ms <= now_ms and iterations < max_iter:
        iterations += 1

        page = _fetch_funding_page(
            base_url=base_url,
            symbol=symbol,
            limit=request_limit,
            start_time_ms=current_ms,
        )

        if not page:
            break

        page_sorted = sorted(page, key=lambda x: int(x["fundingTime"]))
        all_rows.extend(page_sorted)

        max_ts_ms = int(page_sorted[-1]["fundingTime"])

        if len(page_sorted) < request_limit:
            break

        next_start = max_ts_ms + 1

        if next_start <= current_ms:
            break

        current_ms = next_start

        time.sleep(RATE_LIMIT_SLEEP_SEC)

    if not all_rows:
        raise ValueError(
            f"Binance Futures Funding L1: no data for {symbol}"
        )

    df = pd.DataFrame(all_rows)

    if "fundingTime" not in df.columns:
        raise ValueError(
            "Binance Futures Funding L1: missing fundingTime"
        )

    df = df.rename(columns={"fundingTime": "open_time"})

    df["open_time"] = pd.to_datetime(
        df["open_time"].astype("int64"),
        unit="ms",
        utc=True,
    )

    if "fundingRate" in df.columns:
        df["fundingRate"] = pd.to_numeric(
            df["fundingRate"], errors="coerce"
        ).astype("float64")

    if "markPrice" in df.columns:
        df["markPrice"] = pd.to_numeric(
            df["markPrice"], errors="coerce"
        ).astype("float64")

    df["symbol"] = str(symbol)
    df["extracted_at"] = pd.Timestamp.now(tz="UTC")

    df = (
        df.sort_values("open_time", ascending=True)
        .drop_duplicates(subset=["open_time"], keep="last")
        .reset_index(drop=True)
    )

    return df


def fetch_futures_funding_hist(
    symbols: List[str],
    start_date: str,
    base_url: str,
    limit: int,
) -> Dict[str, pd.DataFrame]:

    if not symbols:
        raise ValueError("Binance Futures Funding L1: symbols list empty.")

    result: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        result[symbol] = _fetch_symbol_funding_hist(
            base_url=base_url,
            symbol=symbol,
            start_date=start_date,
            limit=limit,
        )

    return result