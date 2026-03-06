"""
Download Funding Rate OI-Weighted History (global, todas exchanges) do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_funding_rate_oi_weighted.py
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

# ms (endpoint usa ms apesar da documentação indicar segundos)
START_S = int(pd.Timestamp("2020-10-01", tz="UTC").timestamp() * 1000)
END_S   = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

# ── AÇÃO 1 — Download ──────────────────────────────────────────────────────────

time.sleep(2.5)
r = requests.get(
    base + "/api/futures/funding-rate/oi-weight-history",
    headers=headers,
    params={
        "symbol":     "BTC",
        "interval":   "1d",
        "limit":      1000,
        "start_time": START_S,
        "end_time":   END_S,

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
print(f"Timestamp detectado como: {unit} (valor={first_ts})")

df["date"] = pd.to_datetime(df["time"], unit=unit, utc=True).dt.normalize()
df = df.set_index("date").sort_index()
df = df.drop(columns=["time"])

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(float)

df = df.rename(columns={"close": "funding_rate_oi_weighted"})
df = df[~df.index.duplicated(keep="last")]

print(f"Shape:       {df.shape}")
print(f"Range:       {df.index.min().date()} -> {df.index.max().date()}")
print(f"Colunas:     {df.columns.tolist()}")
print(f"NaN:         {df.isna().sum().sum()}")
print(f"Fins semana: {len(df[df.index.dayofweek >= 5])}")
print(f"\nStats funding OI-weighted:")
print(df["funding_rate_oi_weighted"].describe())
print(f"Dias negativos: {(df['funding_rate_oi_weighted'] < 0).sum()}")
print(df.tail(3))

# ── AÇÃO 3 — Comparar com raw (Binance only) ──────────────────────────────────

df_raw = pd.read_parquet(
    "data/01_raw/derivatives/coinglass/funding/BTCUSDT_raw.parquet"
)
merged = df_raw[["funding_rate"]].join(
    df[["funding_rate_oi_weighted"]], how="inner"
)
merged["divergence"] = merged["funding_rate_oi_weighted"] - merged["funding_rate"]

print(f"\nComparacao OI-weighted (global) vs Raw (Binance):")
print(f"  Rows em comum:         {len(merged)}")
print(f"  Divergencia mean abs:  {merged['divergence'].abs().mean():.6f}")
print(f"  Divergencia max abs:   {merged['divergence'].abs().max():.6f}")
print(f"  Correlacao:            {merged.corr().iloc[0,1]:.4f}")

# ── AÇÃO 4 — Salvar ───────────────────────────────────────────────────────────

out = Path("data/01_raw/derivatives/coinglass/funding/BTCUSDT_oi_weighted.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
