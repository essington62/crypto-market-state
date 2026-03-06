"""
Download Puell Multiple Index do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_puell_multiple.py
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
    base + "/api/index/puell-multiple",
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

first_ts = float(df["timestamp"].iloc[0])
unit = "ms" if first_ts > 1e12 else "s"
print(f"Timestamp: {unit} (valor={first_ts})")

df["date"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True).dt.normalize()
df = df.set_index("date").sort_index()

drop_cols = [c for c in ["timestamp", "price"] if c in df.columns]
df = df.drop(columns=drop_cols)

df["puell_multiple"] = df["puell_multiple"].astype(float)
df = df[~df.index.duplicated(keep="last")]

print(f"Shape:       {df.shape}")
print(f"Range:       {df.index.min().date()} -> {df.index.max().date()}")
print(f"Colunas:     {df.columns.tolist()}")
print(f"NaN:         {df.isna().sum().sum()}")
print(f"\nZonas Puell Multiple:")
print(f"  Subvalorizado - compra (< 0.5):    {(df['puell_multiple'] < 0.5).sum()} dias")
print(f"  Neutro (0.5 - 2.0):                {((df['puell_multiple'] >= 0.5) & (df['puell_multiple'] < 2.0)).sum()} dias")
print(f"  Mineradores vendendo (2.0 - 4.0):  {((df['puell_multiple'] >= 2.0) & (df['puell_multiple'] < 4.0)).sum()} dias")
print(f"  Topo de ciclo (> 4.0):             {(df['puell_multiple'] >= 4.0).sum()} dias")
print(f"\nValor atual: {df['puell_multiple'].iloc[-1]:.4f}")
print(df.tail(3))

# ── AÇÃO 3 — Salvar ───────────────────────────────────────────────────────────

out = Path("data/01_raw/derivatives/coinglass/indices/puell_multiple.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
