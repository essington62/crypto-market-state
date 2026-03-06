"""
Download CGDI Index History do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_cgdi_index.py
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
    base + "/api/futures/cgdi-index/history",
    headers=headers,
    params={
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
    print(f"msg: {resp.get('msg', '')} — tentando sem params")
    time.sleep(2.5)
    r = requests.get(
        base + "/api/futures/cgdi-index/history",
        headers=headers,
        timeout=30,
    )
    resp = r.json()
    data = resp.get("data", [])
    print(f"[sem params] code={resp.get('code')} | rows={len(data)}")
    if data:
        print("Primeiro:", data[0])
        print("Ultimo:  ", data[-1])

if not data:
    print("rows=0 — parando conforme instrução.")
    sys.exit(1)

# ── AÇÃO 2 — Parse e validar ──────────────────────────────────────────────────

df = pd.DataFrame(data)
print(f"\nColunas brutas: {df.columns.tolist()}")

first_ts = float(df["time"].iloc[0])
unit = "ms" if first_ts > 1e12 else "s"
print(f"Timestamp: {unit}")

df["date"] = pd.to_datetime(df["time"], unit=unit, utc=True).dt.normalize()
df = df.set_index("date").sort_index()
df = df.drop(columns=["time"])
df["cgdi_index"] = df["cgdi_index_value"].astype(float)
df = df.drop(columns=["cgdi_index_value"])
df = df[~df.index.duplicated(keep="last")]

print(f"Shape:   {df.shape}")
print(f"Range:   {df.index.min().date()} -> {df.index.max().date()}")
print(f"Colunas: {df.columns.tolist()}")
print(f"NaN:     {df.isna().sum().sum()}")
print(f"\nCGDI stats:")
print(df["cgdi_index"].describe().round(2))
print(f"\nValor atual:    {df['cgdi_index'].iloc[-1]:.2f}")
print(f"Valor inicial:  {df['cgdi_index'].iloc[0]:.2f}")
print(f"Variacao total: {((df['cgdi_index'].iloc[-1] / df['cgdi_index'].iloc[0]) - 1) * 100:.1f}%")
print(df.tail(3))

# ── AÇÃO 3 — Salvar ───────────────────────────────────────────────────────────

out = Path("data/01_raw/derivatives/coinglass/indices/cgdi_index.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
