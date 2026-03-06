"""
Download Spot Aggregated Taker Buy/Sell Volume History do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_spot_aggregated_taker.py
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

key     = os.environ["COINGLASS_API_KEY"]
base    = "https://open-api-v4.coinglass.com"
headers = {"CG-API-KEY": key, "Accept": "application/json"}

START_MS = int(pd.Timestamp("2020-10-01", tz="UTC").timestamp() * 1000)
END_MS   = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

# ── AÇÃO 1 — Download ──────────────────────────────────────────────────────────

time.sleep(2.5)
r = requests.get(
    base + "/api/spot/aggregated-taker-buy-sell-volume/history",
    headers=headers,
    params={
        "exchange_list": "Binance,OKX,Bybit",
        "symbol":        "BTC",
        "interval":      "1d",
        "limit":         1000,
        "start_time":    START_MS,
        "end_time":      END_MS,
        "unit":          "usd",
    },
    timeout=30,
)
resp = r.json()
data = resp.get("data", [])
print(f"code={resp.get('code')} | rows={len(data)}")
if data:
    print("Primeiro:", data[0])
    print("Ultimo:  ", data[-1])
else:
    print("msg:", resp.get("msg", ""))
    print("rows=0 — parando conforme instrução.")
    sys.exit(1)

# ── AÇÃO 2 — Parse e validar ──────────────────────────────────────────────────

df = pd.DataFrame(data)

first_ts = df["time"].iloc[0]
unit_ts = "ms" if first_ts > 1e12 else "s"
print(f"Timestamp: {unit_ts}")

df["date"] = pd.to_datetime(df["time"], unit=unit_ts, utc=True).dt.normalize()
df = df.set_index("date").sort_index()
df = df.drop(columns=["time"])

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(float)

df = df[~df.index.duplicated(keep="last")]

print(f"Shape:       {df.shape}")
print(f"Range:       {df.index.min().date()} -> {df.index.max().date()}")
print(f"Colunas:     {df.columns.tolist()}")
print(f"NaN:         {df.isna().sum().sum()}")
print(f"Fins semana: {len(df[df.index.dayofweek >= 5])}")
print(f"\nColunas retornadas (verificar nomes exatos):")
print(df.head(2))

buy_col  = [c for c in df.columns if "buy"  in c.lower()]
sell_col = [c for c in df.columns if "sell" in c.lower()]

if buy_col and sell_col:
    buy_col  = buy_col[0]
    sell_col = sell_col[0]
    df["spot_buy_ratio"] = df[buy_col] / (df[buy_col] + df[sell_col])
    df["spot_cvd"]       = (df[buy_col] - df[sell_col]).cumsum()
    print(f"\nBuy ratio stats:")
    print(df["spot_buy_ratio"].describe().round(4))
    print(f"\nDias com buy_ratio > 0.55 (pressao compradora): {(df['spot_buy_ratio'] > 0.55).sum()}")
    print(f"Dias com buy_ratio < 0.45 (pressao vendedora):  {(df['spot_buy_ratio'] < 0.45).sum()}")
else:
    print(f"\nColunas buy/sell nao encontradas — verificar: {df.columns.tolist()}")

print(df.tail(3))

# ── AÇÃO 3 — Salvar (sem CVD/buy_ratio — serao recalculados no pipeline L2) ──

out = Path("data/01_raw/derivatives/coinglass/taker/spot_BTC_aggregated.parquet")
out.parent.mkdir(parents=True, exist_ok=True)

cols_raw = [c for c in df.columns if c not in ("spot_buy_ratio", "spot_cvd")]
df[cols_raw].to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
