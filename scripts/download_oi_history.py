"""
Download Open Interest histórico do Coinglass V4.

Endpoint confirmado (2026-03-05):
  /api/futures/open-interest/history
  exchange=Binance, symbol=BTCUSDT, interval=1d
  start_time / end_time (Unix ms, snake_case)

Paginação em chunks de 900 dias para cobrir 2020-10-01 → hoje.
Campos: time (Unix ms), open, high, low, close (strings → float64)
close renomeado para open_interest_usd.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_oi_history.py
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

# ── Config ─────────────────────────────────────────────────────────────────────

KEY      = os.environ["COINGLASS_API_KEY"]
BASE_URL = "https://open-api-v4.coinglass.com"
HEADERS  = {"CG-API-KEY": KEY, "Accept": "application/json"}

EXCHANGE   = "Binance"
SYMBOL     = "BTCUSDT"
INTERVAL   = "1d"
START_DATE = "2020-10-01"
SLEEP_SEC  = 3.0
CHUNK_DAYS = 900

OUT_PATH = Path("data/01_raw/derivatives/coinglass/open_interest/BTCUSDT.parquet")

# ── Fetch one chunk ────────────────────────────────────────────────────────────

def fetch_chunk(start_ms: int, end_ms: int) -> list[dict]:
    params = {
        "exchange":   EXCHANGE,
        "symbol":     SYMBOL,
        "interval":   INTERVAL,
        "start_time": start_ms,
        "end_time":   end_ms,
    }
    for attempt in range(3):
        try:
            r = requests.get(
                BASE_URL + "/api/futures/open-interest/history",
                headers=HEADERS,
                params=params,
                timeout=30,
            )
            r.raise_for_status()
            d = r.json()
            if d.get("code") == "0":
                return d.get("data", [])
            print(f"  API error: code={d.get('code')} msg={d.get('msg','')}")
            return []
        except Exception as e:
            print(f"  Request error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(5)
    return []


# ── Paginate full history ──────────────────────────────────────────────────────

t_start = pd.Timestamp(START_DATE, tz="UTC").normalize()
t_end   = pd.Timestamp.now("UTC").normalize()
chunk   = pd.Timedelta(days=CHUNK_DAYS)

all_records: list[dict] = []
cur = t_start

print(f"Downloading OI: {EXCHANGE}/{SYMBOL} from {t_start.date()} → {t_end.date()}")
print(f"Chunk size: {CHUNK_DAYS} days | Expected batches: {int((t_end - t_start).days / CHUNK_DAYS) + 1}")

batch = 0
while cur < t_end:
    nxt       = min(cur + chunk, t_end)
    start_ms  = int(cur.timestamp() * 1000)
    end_ms    = int(nxt.timestamp() * 1000)
    batch    += 1

    print(f"  Batch {batch}: {cur.date()} → {nxt.date()} ...", end=" ", flush=True)
    records = fetch_chunk(start_ms, end_ms)
    print(f"{len(records)} rows")

    all_records.extend(records)
    cur = nxt
    time.sleep(SLEEP_SEC)

print(f"\nTotal raw records: {len(all_records)}")

# ── Parse ──────────────────────────────────────────────────────────────────────

if not all_records:
    print("ERROR: No data returned. Exiting.")
    raise SystemExit(1)

df = pd.DataFrame(all_records)

# Timestamp: Unix ms → midnight UTC
df.index = (
    pd.to_datetime(df["time"], unit="ms", utc=True)
    .dt.normalize()
)
df.index.name = "timestamp"
df = df.drop(columns=["time"])

# Cast OHLC to float64
for col in ["open", "high", "low", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

# Rename close → open_interest_usd
df = df.rename(columns={"close": "open_interest_usd"})
df = df[["open", "high", "low", "open_interest_usd"]]

# Deduplicate and sort
df = df[~df.index.duplicated(keep="last")].sort_index()

# Filter from start_date
df = df[df.index >= pd.Timestamp(START_DATE, tz="UTC")]

print(f"Shape:         {df.shape}")
print(f"Range:         {df.index.min().date()} → {df.index.max().date()}")
print(f"NaN:           {df.isna().sum().sum()}")
print(f"Weekend rows:  {(df.index.dayofweek >= 5).sum()}")
print(f"\nHead:\n{df.head(3)}")
print(f"\nTail:\n{df.tail(3)}")
print(f"\nDescribe:\n{df.describe()}")

# ── Save ───────────────────────────────────────────────────────────────────────

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, compression="snappy")
print(f"\nSaved: {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")
