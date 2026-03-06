"""
Download Options Exchange OI History do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_options_oi.py
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

# ── AÇÃO 1 — Download com range=all ───────────────────────────────────────────

time.sleep(2.5)
r = requests.get(
    base + "/api/option/exchange-oi-history",
    headers=headers,
    params={
        "symbol": "BTC",
        "unit":   "USD",
        "range":  "all",
    },
    timeout=30,
)
resp = r.json()
print(f"code={resp.get('code')}")
print(f"Tipo de data: {type(resp.get('data'))}")
print(f"Keys disponiveis: {list(resp.keys())}")

data = resp.get("data", {})
if isinstance(data, list):
    print(f"Lista com {len(data)} items")
    print("Primeiro:", data[0] if data else "vazio")
elif isinstance(data, dict):
    print(f"Dict com keys: {list(data.keys())[:10]}")
    for k, v in data.items():
        print(f"  {k}: {type(v).__name__} len={len(v) if hasattr(v, '__len__') else 'N/A'}")
else:
    print(f"Estrutura inesperada: {type(data)}")
    print("resp completo:", resp)
    sys.exit(1)

# ── AÇÃO 2 — Parse ────────────────────────────────────────────────────────────

# Suporta tanto dateList+dataMap quanto time_list+data_map
if isinstance(data, dict):
    if "dateList" in data:
        dates   = pd.to_datetime(data["dateList"], unit="ms", utc=True).normalize()
        datamap = data.get("dataMap", {})
    elif "time_list" in data:
        dates   = pd.to_datetime(data["time_list"], unit="ms", utc=True).normalize()
        datamap = data.get("data_map", {})
    else:
        print(f"\nEstrutura desconhecida — keys: {list(data.keys())}")
        sys.exit(1)
else:
    print(f"\nEstrutura inesperada (não dict) — resp completo:")
    print(resp)
    sys.exit(1)

print(f"\nExchanges disponiveis: {list(datamap.keys())}")
print(f"Total de datas: {len(dates)}")
print(f"Primeira data: {dates[0].date()}")
print(f"Ultima data:   {dates[-1].date()}")

df = pd.DataFrame(index=dates)
df.index.name = "date"

for exchange, values in datamap.items():
    df[f"oi_{exchange.lower()}"] = values

df["oi_options_total_usd"] = df.sum(axis=1)
df = df[~df.index.duplicated(keep="last")]
df = df.sort_index()

# Filtrar apenas datas passadas (time_list pode incluir datas de vencimento futuras)
today = pd.Timestamp.now(tz="UTC").normalize()
n_future = (df.index > today).sum()
if n_future > 0:
    print(f"\nFiltradas {n_future} datas futuras (expiries de opcoes).")
    df = df[df.index <= today]

print(f"\nShape:   {df.shape}")
print(f"Range:   {df.index.min().date()} -> {df.index.max().date()}")
print(f"Colunas: {df.columns.tolist()}")
print(f"NaN:     {df.isna().sum().sum()}")
print(f"\nStats oi_options_total_usd:")
print(df["oi_options_total_usd"].describe().round(0))
print(df.tail(3))

# ── AÇÃO 3 — Salvar ───────────────────────────────────────────────────────────

out = Path("data/01_raw/derivatives/coinglass/options/BTC_oi_by_exchange.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
