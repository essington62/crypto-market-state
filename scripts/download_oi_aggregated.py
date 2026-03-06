"""
Download Open Interest Aggregated History (ALL exchanges) do Coinglass V4.

Endpoint validado no notebook:
  /api/futures/open-interest/aggregated-history
  symbol=BTC (não BTCUSDT), interval=1d, unit=usd
  start_time / end_time (Unix ms)

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_oi_aggregated.py
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

KEY     = os.environ["COINGLASS_API_KEY"]
BASE    = "https://open-api-v4.coinglass.com"
HEADERS = {"CG-API-KEY": KEY, "Accept": "application/json"}

START_MS = int(pd.Timestamp("2020-10-01", tz="UTC").timestamp() * 1000)
END_MS   = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

OUT = Path("data/01_raw/derivatives/coinglass/open_interest/BTCUSDT_aggregated.parquet")

# ── AÇÃO 1 — Download ──────────────────────────────────────────────────────────

time.sleep(2.5)
r = requests.get(
    BASE + "/api/futures/open-interest/aggregated-history",
    headers=HEADERS,
    params={
        "symbol":     "BTC",
        "interval":   "1d",
        "limit":      1000,
        "start_time": START_MS,
        "end_time":   END_MS,
        "unit":       "usd",
    },
    timeout=30,
)
resp = r.json()
data = resp.get("data", [])
print(f"code={resp.get('code')} | rows={len(data)}")

if not data:
    print(f"msg: {resp.get('msg', '')}")
    print("data_len=0 — parando conforme instrução.")
    sys.exit(1)

print("Primeiro:", data[0])
print("Ultimo:  ", data[-1])

# ── AÇÃO 2 — Parse e validar ──────────────────────────────────────────────────

df = pd.DataFrame(data)

df["date"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.normalize()
df = df.set_index("date").sort_index()
df = df.drop(columns=["time"])

for col in ["open", "high", "low", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

df = df.rename(columns={"close": "open_interest_usd"})
df = df[~df.index.duplicated(keep="last")]

print(f"\nShape:       {df.shape}")
print(f"Range:       {df.index.min().date()} -> {df.index.max().date()}")
print(f"Colunas:     {df.columns.tolist()}")
print(f"NaN:         {df.isna().sum().sum()}")
print(f"Fins semana: {len(df[df.index.dayofweek >= 5])}")
print(df.tail(3))

# ── AÇÃO 3 — Sanity check vs Binance only ─────────────────────────────────────

binance_path = Path("data/01_raw/derivatives/coinglass/open_interest/BTCUSDT.parquet")
df_binance = pd.read_parquet(binance_path)

merged = df_binance[["open_interest_usd"]].join(
    df[["open_interest_usd"]],
    how="inner",
    lsuffix="_binance",
    rsuffix="_all",
)
merged["binance_share"] = (
    merged["open_interest_usd_binance"] / merged["open_interest_usd_all"]
)

print(f"\nBinance share do mercado total:")
print(merged["binance_share"].describe().round(3))
print(f"\nShare atual (ultimo dia): {merged['binance_share'].iloc[-1]:.1%}")

violations = (merged["open_interest_usd_all"] < merged["open_interest_usd_binance"]).sum()
print(f"\nSanity check — aggregated >= binance always: "
      f"{'OK' if violations == 0 else f'FAIL ({violations} violations)'}")

# ── AÇÃO 4 — Salvar ───────────────────────────────────────────────────────────

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, compression="snappy")

df_check = pd.read_parquet(OUT)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {OUT}  ({OUT.stat().st_size // 1024} KB)")
