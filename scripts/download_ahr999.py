"""
Download AHR999 Index History do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_ahr999.py
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
    base + "/api/index/ahr999",
    headers=headers,
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
print(f"\nColunas brutas: {df.columns.tolist()}")

df["date"] = pd.to_datetime(
    df["date_string"], format="%Y/%m/%d", utc=True
).dt.normalize()
df = df.set_index("date").sort_index()

drop_cols = [c for c in ["date_string", "average_price", "current_value"] if c in df.columns]
df = df.drop(columns=drop_cols)

df["ahr999"] = df["ahr999_value"].astype(float)
df = df.drop(columns=["ahr999_value"])
df = df[~df.index.duplicated(keep="last")]

print(f"Shape:       {df.shape}")
print(f"Range:       {df.index.min().date()} -> {df.index.max().date()}")
print(f"Colunas:     {df.columns.tolist()}")
print(f"NaN:         {df.isna().sum().sum()}")
print(f"\nZonas AHR999:")
print(f"  Acumulacao forte (< 0.45):  {(df['ahr999'] < 0.45).sum()} dias")
print(f"  Acumulacao (0.45-1.2):      {((df['ahr999'] >= 0.45) & (df['ahr999'] < 1.2)).sum()} dias")
print(f"  Neutro (1.2-4.0):           {((df['ahr999'] >= 1.2) & (df['ahr999'] < 4.0)).sum()} dias")
print(f"  Topo de ciclo (> 4.0):      {(df['ahr999'] >= 4.0).sum()} dias")
print(f"\nValor atual: {df['ahr999'].iloc[-1]:.4f}")
print(df.tail(3))

# ── AÇÃO 3 — Salvar ───────────────────────────────────────────────────────────

out = Path("data/01_raw/derivatives/coinglass/indices/ahr999.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
