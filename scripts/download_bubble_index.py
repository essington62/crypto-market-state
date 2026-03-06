"""
Download Bitcoin Bubble Index do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_bubble_index.py
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

# ── AÇÃO 1 — Download ──────────────────────────────────────────────────────────

time.sleep(2.5)
r = requests.get(
    base + "/api/index/bitcoin/bubble-index",
    headers=headers,
    timeout=30,
)
resp = r.json()
data = resp.get("data", [])
print(f"code={resp.get('code')} | rows={len(data)}")
if data:
    print("Primeiro:", data[0])
    print("Ultimo:  ", data[-1])
    print(f"\nCampos disponíveis: {list(data[0].keys())}")
else:
    print("msg:", resp.get("msg", ""))
    print("rows=0 — parando conforme instrução.")
    sys.exit(1)

# ── AÇÃO 2 — Parse e validar ──────────────────────────────────────────────────

df = pd.DataFrame(data)
print(f"\nColunas raw: {df.columns.tolist()}")

ts_col = None
for candidate in ["date_string", "timestamp", "time", "date"]:
    if candidate in df.columns:
        ts_col = candidate
        break
print(f"Coluna timestamp: {ts_col}")

if ts_col == "date_string":
    sample = df[ts_col].iloc[0]
    fmt = "%Y/%m/%d" if "/" in sample else "%Y-%m-%d"
    print(f"Formato detectado: {fmt} (sample={sample})")
    df["date"] = pd.to_datetime(df[ts_col], format=fmt, utc=True).dt.normalize()
else:
    first_ts = float(df[ts_col].iloc[0])
    unit = "ms" if first_ts > 1e12 else "s"
    print(f"Timestamp: {unit} (valor={first_ts})")
    df["date"] = pd.to_datetime(df[ts_col], unit=unit, utc=True).dt.normalize()

df = df.set_index("date").sort_index()

drop_cols = [c for c in df.columns if c in [ts_col, "price", "date_string", "timestamp", "time"]]
df = df.drop(columns=drop_cols, errors="ignore")

for col in df.select_dtypes(include="object").columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[~df.index.duplicated(keep="last")]

print(f"\nShape:   {df.shape}")
print(f"Range:   {df.index.min().date()} -> {df.index.max().date()}")
print(f"Colunas: {df.columns.tolist()}")
print(f"NaN:     {df.isna().sum().to_dict()}")
print(f"\nValores atuais (ultima linha):")
print(df.tail(1).T)

if "tweet_count" in df.columns:
    recent = df.loc["2023":, "tweet_count"]
    print(f"\ntweet_count em 2023+: zeros={(recent == 0).sum()} / total={len(recent)}")

# ── AÇÃO 3 — Salvar ───────────────────────────────────────────────────────────

out = Path("data/01_raw/derivatives/coinglass/indices/bubble_index.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
