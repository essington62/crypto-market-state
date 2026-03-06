"""
Download Top Long/Short Account Ratio History (top traders) do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_long_short_top_accounts.py
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
    base + "/api/futures/top-long-short-account-ratio/history",
    headers=headers,
    params={
        "exchange":   "Binance",
        "symbol":     "BTCUSDT",
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
print(f"\nStats top traders:")
print(df.describe().round(2))
print(f"\nDias top traders majoritariamente short (long_percent < 50): {(df['top_account_long_percent'] < 50).sum()}")
print(df.tail(3))

# ── AÇÃO 3 — Comparar com Global ratio ────────────────────────────────────────

global_path = Path("data/01_raw/derivatives/coinglass/long_short_ratio/BTCUSDT.parquet")
if global_path.exists():
    df_global = pd.read_parquet(global_path)
    print(f"\nGlobal ratio colunas: {df_global.columns.tolist()}")
    ratio_col = [c for c in df_global.columns if "ratio" in c.lower()]
    if ratio_col:
        merged = df_global[ratio_col].join(
            df[["top_account_long_short_ratio"]], how="inner"
        )
        print(f"Correlacao global vs top traders: {merged.corr().iloc[0,1]:.4f}")
        print(f"Rows em comum: {len(merged)}")
else:
    print("Global ratio ainda nao disponivel — pular comparacao")

# ── AÇÃO 4 — Salvar ───────────────────────────────────────────────────────────

out = Path("data/01_raw/derivatives/coinglass/long_short_ratio/BTCUSDT_top_accounts.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
