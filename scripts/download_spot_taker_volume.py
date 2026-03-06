"""
Download Spot Taker Buy/Sell Volume History (mercado à vista) do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_spot_taker_volume.py
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

# ── AÇÃO 1 — Testar formato do interval diário ────────────────────────────────

working_interval = None
for interval in ["1d", "d1"]:
    time.sleep(2.5)
    r = requests.get(
        base + "/api/spot/taker-buy-sell-volume/history",
        headers=headers,
        params={
            "exchange":   "Binance",
            "symbol":     "BTCUSDT",
            "interval":   interval,
            "limit":      5,
            "start_time": START_MS,
            "end_time":   END_MS,
        },
        timeout=30,
    )
    resp = r.json()
    data = resp.get("data", [])
    print(f"interval={interval}: code={resp.get('code')} | rows={len(data)}")
    if data:
        print("  Primeiro:", data[0])
        print("  Ultimo:  ", data[-1])
        if working_interval is None:
            working_interval = interval

if working_interval is None:
    print("rows=0 em ambos os intervals — parando conforme instrução.")
    sys.exit(1)

print(f"\nInterval correto: {working_interval}")

# ── AÇÃO 2 — Download completo ────────────────────────────────────────────────

time.sleep(2.5)
r = requests.get(
    base + "/api/spot/taker-buy-sell-volume/history",
    headers=headers,
    params={
        "exchange":   "Binance",
        "symbol":     "BTCUSDT",
        "interval":   working_interval,
        "limit":      1000,
        "start_time": START_MS,
        "end_time":   END_MS,
    },
    timeout=30,
)
resp = r.json()
data = resp.get("data", [])
print(f"Download completo: code={resp.get('code')} | rows={len(data)}")
if data:
    print("Primeiro:", data[0])
    print("Ultimo:  ", data[-1])

if not data:
    print("rows=0 no download completo — parando.")
    sys.exit(1)

# ── AÇÃO 3 — Parse e validar ──────────────────────────────────────────────────

df = pd.DataFrame(data)
print(f"Colunas brutas: {df.columns.tolist()}")

first_ts = df["time"].iloc[0]
unit = "ms" if first_ts > 1e12 else "s"
print(f"Timestamp: {unit}")

df["date"] = pd.to_datetime(df["time"], unit=unit, utc=True).dt.normalize()
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
print(f"\nStats:")
print(df.describe().round(2))

buy_cols  = [c for c in df.columns if "buy"  in c.lower()]
sell_cols = [c for c in df.columns if "sell" in c.lower()]
if buy_cols and sell_cols:
    df["buy_ratio"] = df[buy_cols[0]] / (df[buy_cols[0]] + df[sell_cols[0]])
    df["cvd"]       = (df[buy_cols[0]] - df[sell_cols[0]]).cumsum()
    print(f"\nBuy ratio medio: {df['buy_ratio'].mean():.3f}")
    print(f"CVD range: {df['cvd'].min():.0f} -> {df['cvd'].max():.0f}")

print(df.tail(3))

# ── AÇÃO 4 — Salvar ───────────────────────────────────────────────────────────

out = Path("data/01_raw/spot/coinglass/BTCUSDT_taker_buy_sell.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
