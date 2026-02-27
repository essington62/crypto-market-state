"""
L1 raw layer: Binance USDT-M Perpetual Futures Klines.

Contract:
- L1 = mirror of the API. No features, no derived metrics.
- Only minimal typing and schema normalization.
- Endpoint: GET /fapi/v1/klines
- Timezone: all timestamps in UTC; open_time as datetime64[ns, UTC].
- Pagination: startTime-based pagination until now().
- Safe rate limiting.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import pandas as pd
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

KLINES_API_LIMIT_MAX = 1500
RATE_LIMIT_SLEEP_SEC = 0.5


def _fetch_klines_page(
    base_url: str,
    symbol: str,
    interval: str,
    limit: int,
    start_time_ms: int,
) -> List[list]:
    """Fetch one page of klines from Binance USDT-M Futures."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "startTime": start_time_ms,
    }

    query = urlencode(params)
    url = f"{base_url.rstrip('/')}/fapi/v1/klines?{query}"

    req = Request(url, method="GET")

    try:
        with urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise ValueError(
                    f"Binance Futures Klines L1: HTTP {resp.status} for URL {url}"
                )
            payload = resp.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        raise ValueError(
            f"Binance Futures Klines L1: request failed for URL {url}"
        ) from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Binance Futures Klines L1: invalid JSON response"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            "Binance Futures Klines L1: unexpected response format (expected list)."
        )

    return data


def _fetch_symbol_klines(
    base_url: str,
    symbol: str,
    interval: str,
    start_date: str,
    limit: int,
) -> pd.DataFrame:
    """Fetch full kline history for one symbol (paginated until now)."""
    start_dt = pd.to_datetime(start_date, utc=True)
    current_ms = int(start_dt.timestamp() * 1000)
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)

    request_limit = min(int(limit), KLINES_API_LIMIT_MAX)

    all_klines: List[list] = []
    max_iter = 50000
    iterations = 0

    while current_ms < now_ms and iterations < max_iter:
        iterations += 1

        page = _fetch_klines_page(
            base_url=base_url,
            symbol=symbol,
            interval=interval,
            limit=request_limit,
            start_time_ms=current_ms,
        )

        if not page:
            break

        all_klines.extend(page)

        last_open_time_ms = int(page[-1][0])
        next_start_ms = last_open_time_ms + 1

        if next_start_ms <= current_ms:
            break

        current_ms = next_start_ms
        time.sleep(RATE_LIMIT_SLEEP_SEC)

    if not all_klines:
        raise ValueError(
            f"Binance Futures Klines L1: no data returned for symbol {symbol}."
        )

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    df = pd.DataFrame(all_klines, columns=columns)

    # --- Type normalization ---
    df["open_time"] = pd.to_datetime(
        df["open_time"].astype("int64"), unit="ms", utc=True
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"].astype("int64"), unit="ms", utc=True
    )

    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    for col in float_cols:
        df[col] = df[col].astype("float64")

    df["number_of_trades"] = df["number_of_trades"].astype("int64")

    # --- Metadata ---
    df["symbol"] = symbol
    df["extracted_at"] = pd.Timestamp.utcnow()

    # --- Ordering and deduplication ---
    df = (
        df.sort_values("open_time")
        .drop_duplicates(subset=["open_time"], keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            f"Binance Futures Klines L1: DataFrame empty for symbol {symbol} after processing."
        )

    return df


def fetch_binance_futures_perpetual_klines(
    symbols: List[str],
    interval: str,
    start_date: str,
    base_url: str,
    limit: int,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch Klines history for each symbol (multi-symbol capable).

    Returns
    -------
    Dict[str, pd.DataFrame]
        Partition key (symbol) -> DataFrame (UTC, sorted, deduplicated).
    """
    if not symbols:
        raise ValueError("Binance Futures Klines L1: symbols list is empty.")

    result: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        result[symbol] = _fetch_symbol_klines(
            base_url=base_url,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            limit=limit,
        )

    return result