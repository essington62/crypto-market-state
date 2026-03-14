from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Dict, List

import time
import pandas as pd
import requests


BINANCE_SPOT_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

API_LIMIT = 1000
MAX_RETRIES = 5
MAX_ITER = 10000


def _fetch_binance_klines_with_retry(
    session: requests.Session,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": API_LIMIT,
    }

    url = f"{BINANCE_SPOT_BASE_URL}{KLINES_ENDPOINT}"
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=10)
            status = resp.status_code

            if status == 200:
                data = resp.json()

                if not data:
                    return pd.DataFrame()

                columns = [
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base_volume",
                    "taker_buy_quote_volume",
                    "ignore",
                ]

                return pd.DataFrame(data, columns=columns)

            if status in (429, 418) or 500 <= status < 600:
                last_error = RuntimeError(f"Binance HTTP {status}")
            else:
                resp.raise_for_status()

        except Exception as exc:
            last_error = exc

        time.sleep(2 ** attempt)

    raise RuntimeError(f"Binance request failed: {last_error}")


def _normalize_klines_df(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df

    df = df.copy()

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    if "trades" in df.columns:
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("int64")

    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])

    return df


def update_binance_spot_4h_incremental(
    existing_data: Dict[str, Callable[[], pd.DataFrame]] | None,
    spot_symbols: List[str],
    interval: str,
    global_start_date: str,
) -> Dict[str, pd.DataFrame]:

    if not spot_symbols:
        raise ValueError("binance_spot_4h_incremental: asset list empty")

    # IMPORTANT: handle empty partition dataset
    existing_data = existing_data or {}

    start_date_utc = pd.to_datetime(global_start_date, utc=True)

    now_utc = datetime.now(timezone.utc)

    # avoid requesting unfinished candle
    now_utc = now_utc.replace(minute=0, second=0, microsecond=0)

    now_ms = int(now_utc.timestamp() * 1000)

    updated: Dict[str, pd.DataFrame] = {}
    report_rows: list[dict] = []

    session = requests.Session()

    for symbol in spot_symbols:

        if symbol in existing_data:
            df_existing = existing_data[symbol]()
        else:
            df_existing = pd.DataFrame()

        df_existing = df_existing.copy()

        if not df_existing.empty and "open_time" in df_existing.columns:

            if not pd.api.types.is_datetime64_any_dtype(df_existing["open_time"]):
                df_existing["open_time"] = pd.to_datetime(df_existing["open_time"], utc=True)

            last_open = df_existing["open_time"].max()

            if last_open.tzinfo is None:
                last_open = last_open.tz_localize("UTC")

            start_ts = last_open + pd.Timedelta(milliseconds=1)

        else:
            start_ts = start_date_utc

        start_ms = int(start_ts.timestamp() * 1000)

        if start_ms >= now_ms:

            updated[symbol] = df_existing

            report_rows.append(
                dict(symbol=symbol, inserted_from=None, inserted_to=None, rows=0)
            )

            continue

        pages: list[pd.DataFrame] = []
        current_start_ms = start_ms
        iters = 0

        while current_start_ms < now_ms and iters < MAX_ITER:

            iters += 1

            raw = _fetch_binance_klines_with_retry(
                session=session,
                symbol=symbol,
                interval=interval,
                start_ms=current_start_ms,
                end_ms=now_ms,
            )

            if raw.empty:
                break

            page = _normalize_klines_df(raw)

            if page.empty:
                break

            page = page.sort_values("open_time")
            pages.append(page)

            last_open = page["open_time"].max()
            last_open_ms = int(last_open.timestamp() * 1000)

            if last_open_ms >= now_ms or len(page) < API_LIMIT:
                break

            current_start_ms = last_open_ms + 1

            time.sleep(0.2)

        if pages:
            df_new = pd.concat(pages, ignore_index=True)
        else:
            df_new = pd.DataFrame()

        if not df_existing.empty and not df_new.empty:
            merged = pd.concat([df_existing, df_new], ignore_index=True)
        elif df_existing.empty:
            merged = df_new
        else:
            merged = df_existing

        if merged.empty:

            updated[symbol] = merged

            report_rows.append(
                dict(symbol=symbol, inserted_from=None, inserted_to=None, rows=0)
            )

            continue

        merged = merged.sort_values("open_time")
        merged = merged.drop_duplicates("open_time", keep="last")

        updated[symbol] = merged

        inserted_from = (
            df_new["open_time"].min().isoformat() if not df_new.empty else None
        )

        inserted_to = (
            df_new["open_time"].max().isoformat() if not df_new.empty else None
        )

        report_rows.append(
            dict(
                symbol=symbol,
                inserted_from=inserted_from,
                inserted_to=inserted_to,
                rows=int(len(df_new)),
            )
        )

    session.close()

    report_df = pd.DataFrame(report_rows)

    print("\n==============================================")
    print("BINANCE 4H INCREMENTAL UPDATE REPORT")
    print("==============================================")
    print(report_df.to_string(index=False))
    print("==============================================\n")

    return updated