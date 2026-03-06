"""
Download Top Long/Short Position Ratio History (posicoes dos top traders) do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_long_short_top_positions.py
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
    base + "/api/futures/top-long-short-position-ratio/history",
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
print(f"\nStats top positions:")
print(df.describe().round(2))
print(df.tail(3))

# ── AÇÃO 3 — Comparar accounts vs positions dos top traders ───────────────────

df_accounts = pd.read_parquet(
    "data/01_raw/derivatives/coinglass/long_short_ratio/BTCUSDT_top_accounts.parquet"
)

ratio_acc = [c for c in df_accounts.columns if "ratio" in c.lower()][0]
ratio_pos_candidates = [c for c in df.columns if "ratio" in c.lower()]
ratio_pos = ratio_pos_candidates[0] if ratio_pos_candidates else df.columns[-1]

merged = df_accounts[[ratio_acc]].join(
    df[[ratio_pos]], how="inner"
)
print(f"\nAccounts vs Positions (top traders):")
print(f"  Rows em comum:        {len(merged)}")
print(f"  Correlacao:           {merged.corr().iloc[0,1]:.4f}")
print(f"  Divergencia mean abs: {(merged[ratio_acc] - merged[ratio_pos]).abs().mean():.4f}")

# ── AÇÃO 4 — Salvar ───────────────────────────────────────────────────────────

out = Path("data/01_raw/derivatives/coinglass/long_short_ratio/BTCUSDT_top_positions.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
