"""
Download Open Interest Aggregated Stablecoin Margin History do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_oi_stablecoin.py
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

OUT = Path("data/01_raw/derivatives/coinglass/open_interest/BTCUSDT_stablecoin.parquet")

BASE_PARAMS = {
    "symbol":     "BTC",
    "interval":   "1d",
    "limit":      1000,
    "start_time": START_MS,
    "end_time":   END_MS,
}

# ── AÇÃO 1 — Download (dois testes) ───────────────────────────────────────────

time.sleep(2.5)
r = requests.get(
    BASE + "/api/futures/open-interest/aggregated-stablecoin-history",
    headers=HEADERS,
    params={**BASE_PARAMS, "exchange_list": "Binance,OKX,Bybit"},
    timeout=30,
)
resp = r.json()
data = resp.get("data", [])
print(f"Binance+OKX+Bybit: code={resp.get('code')} | rows={len(data)}")
if data:
    print("Primeiro:", data[0])
    print("Ultimo:  ", data[-1])
else:
    print("msg:", resp.get("msg", ""))

time.sleep(2.5)
r2 = requests.get(
    BASE + "/api/futures/open-interest/aggregated-stablecoin-history",
    headers=HEADERS,
    params={**BASE_PARAMS, "exchange_list": "Binance"},
    timeout=30,
)
resp2 = r2.json()
data2 = resp2.get("data", [])
print(f"\nSo Binance: code={resp2.get('code')} | rows={len(data2)}")
if data2:
    print("Primeiro:", data2[0])
else:
    print("msg:", resp2.get("msg", ""))

if not data and not data2:
    print("\nrows=0 em ambos os testes — parando conforme instrução.")
    sys.exit(1)

# ── AÇÃO 2 — Parse e validar ──────────────────────────────────────────────────

best_data  = data if len(data) >= len(data2) else data2
best_label = "Binance+OKX+Bybit" if len(data) >= len(data2) else "Binance"
print(f"\nUsando: {best_label} | rows={len(best_data)}")

df = pd.DataFrame(best_data)

df["date"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.normalize()
df = df.set_index("date").sort_index()
df = df.drop(columns=["time"])

for col in df.select_dtypes(include="object").columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

df = df.rename(columns={"close": "oi_stablecoin_usd"})
df = df[~df.index.duplicated(keep="last")]

print(f"Shape:       {df.shape}")
print(f"Range:       {df.index.min().date()} -> {df.index.max().date()}")
print(f"Colunas:     {df.columns.tolist()}")
print(f"NaN:         {df.isna().sum().sum()}")
print(f"Fins semana: {len(df[df.index.dayofweek >= 5])}")
print(df.tail(3))

# ── AÇÃO 3 — Sanity check vs aggregated total ─────────────────────────────────

df_agg = pd.read_parquet(
    "data/01_raw/derivatives/coinglass/open_interest/BTCUSDT_aggregated.parquet"
)
merged = df_agg[["open_interest_usd"]].join(df[["oi_stablecoin_usd"]], how="inner")
merged["stablecoin_share"] = merged["oi_stablecoin_usd"] / merged["open_interest_usd"]

print(f"\nStablecoin share do OI total:")
print(merged["stablecoin_share"].describe().round(3))
print(f"Share atual: {merged['stablecoin_share'].iloc[-1]:.1%}")
print(f"Stablecoin sempre <= total: {(merged['oi_stablecoin_usd'] <= merged['open_interest_usd']).all()}")

# ── AÇÃO 4 — Salvar ───────────────────────────────────────────────────────────

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, compression="snappy")

df_check = pd.read_parquet(OUT)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {OUT}  ({OUT.stat().st_size // 1024} KB)")
