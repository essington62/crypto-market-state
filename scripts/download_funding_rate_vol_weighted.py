"""
Download Funding Rate Vol-Weighted History (global, todas exchanges) do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_funding_rate_vol_weighted.py
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
    base + "/api/futures/funding-rate/vol-weight-history",
    headers=headers,
    params={
        "symbol":     "BTC",
        "interval":   "1d",
        "limit":      1000,
        "start_time": START_MS,
        "end_time":   END_MS,
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
unit = "ms" if first_ts > 1e12 else "s"
print(f"Timestamp: {unit} (valor={first_ts})")

df["date"] = pd.to_datetime(df["time"], unit=unit, utc=True).dt.normalize()
df = df.set_index("date").sort_index()
df = df.drop(columns=["time"])

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(float)

df = df.rename(columns={"close": "funding_rate_vol_weighted"})
df = df[~df.index.duplicated(keep="last")]

print(f"Shape:       {df.shape}")
print(f"Range:       {df.index.min().date()} -> {df.index.max().date()}")
print(f"Colunas:     {df.columns.tolist()}")
print(f"NaN:         {df.isna().sum().sum()}")
print(f"Fins semana: {len(df[df.index.dayofweek >= 5])}")
print(f"\nStats funding Vol-weighted:")
print(df["funding_rate_vol_weighted"].describe())
print(f"Dias negativos: {(df['funding_rate_vol_weighted'] < 0).sum()}")
print(df.tail(3))

# ── AÇÃO 3 — Comparar as 3 series de funding ──────────────────────────────────

df_raw = pd.read_parquet(
    "data/01_raw/derivatives/coinglass/funding/BTCUSDT_raw.parquet"
)
df_oi = pd.read_parquet(
    "data/01_raw/derivatives/coinglass/funding/BTCUSDT_oi_weighted.parquet"
)

merged = df_raw[["funding_rate"]].join(
    df_oi[["funding_rate_oi_weighted"]], how="inner"
).join(
    df[["funding_rate_vol_weighted"]], how="inner"
)

print(f"\nComparacao das 3 series de funding rate:")
print(f"  Rows em comum: {len(merged)}")
print(f"\nCorrelacoes:")
print(merged.corr().round(4))
print(f"\nDivergencia Vol vs OI (mean abs): {(merged['funding_rate_vol_weighted'] - merged['funding_rate_oi_weighted']).abs().mean():.6f}")
print(f"Divergencia Vol vs Raw (mean abs): {(merged['funding_rate_vol_weighted'] - merged['funding_rate']).abs().mean():.6f}")
print(f"\nDias onde Vol != OI (diff > 0.0001): {((merged['funding_rate_vol_weighted'] - merged['funding_rate_oi_weighted']).abs() > 0.0001).sum()}")

# ── AÇÃO 4 — Salvar ───────────────────────────────────────────────────────────

out = Path("data/01_raw/derivatives/coinglass/funding/BTCUSDT_vol_weighted.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
