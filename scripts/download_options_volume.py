"""
Download Options Exchange Volume History do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_options_volume.py
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
    base + "/api/option/exchange-volume-history",
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
data = resp.get("data", {})
print(f"Tipo de data: {type(data).__name__}")

if isinstance(data, dict):
    for k, v in data.items():
        print(f"  {k}: {type(v).__name__} len={len(v) if hasattr(v, '__len__') else 'N/A'}")
elif isinstance(data, list):
    print(f"Lista com {len(data)} items")
    print("Primeiro:", data[0] if data else "vazio")

# Se retornou vazio ou erro, tentar endpoint alternativo
if not data or (isinstance(data, dict) and not any(data.values())):
    print("\nTentando endpoint alternativo: exchange-vol-history")
    time.sleep(2.5)
    r = requests.get(
        base + "/api/option/exchange-vol-history",
        headers=headers,
        params={"symbol": "BTC", "unit": "USD", "range": "all"},
        timeout=30,
    )
    resp = r.json()
    print(f"Alt code={resp.get('code')}")
    data = resp.get("data", {})
    print(f"Alt tipo: {type(data).__name__}")
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"  {k}: {type(v).__name__} len={len(v) if hasattr(v, '__len__') else 'N/A'}")

# ── AÇÃO 2 — Parse ────────────────────────────────────────────────────────────

if not isinstance(data, dict):
    print(f"\nEstrutura inesperada — resp completo:")
    print(resp)
    sys.exit(1)

# Suporta dateList+dataMap e time_list+data_map
if "dateList" in data:
    dates   = pd.to_datetime(data["dateList"], unit="ms", utc=True).normalize()
    datamap = data.get("dataMap", {})
elif "time_list" in data:
    dates   = pd.to_datetime(data["time_list"], unit="ms", utc=True).normalize()
    datamap = data.get("data_map", {})
else:
    print(f"\nChaves desconhecidas: {list(data.keys())} — resp completo:")
    print(resp)
    sys.exit(1)

print(f"\nExchanges disponiveis: {list(datamap.keys())}")
print(f"Total de datas: {len(dates)}")
print(f"Primeira data:  {dates[0].date()}")
print(f"Ultima data:    {dates[-1].date()}")

df = pd.DataFrame(index=dates)
df.index.name = "date"

for exchange, values in datamap.items():
    df[f"vol_{exchange.lower()}"] = values

df["options_volume_total_usd"] = df.sum(axis=1, min_count=1)
df = df[~df.index.duplicated(keep="last")]
df = df.sort_index()

# Filtrar datas futuras
today = pd.Timestamp.now(tz="UTC").normalize()
n_future = (df.index > today).sum()
if n_future > 0:
    print(f"Filtradas {n_future} datas futuras.")
    df = df[df.index <= today]

# Comparar com OI options
oi_path = Path("data/01_raw/derivatives/coinglass/options/BTC_oi_by_exchange.parquet")
if oi_path.exists():
    df_oi = pd.read_parquet(oi_path)
    merged = df[["options_volume_total_usd"]].join(
        df_oi[["oi_options_total_usd"]], how="inner"
    )
    merged["vol_oi_ratio"] = merged["options_volume_total_usd"] / merged["oi_options_total_usd"]
    print(f"\nVol/OI ratio (rotatividade):")
    print(merged["vol_oi_ratio"].describe().round(3))
    print(f"Ratio atual: {merged['vol_oi_ratio'].iloc[-1]:.3f}")
else:
    print("\nOI options nao disponivel — pular comparacao")

print(f"\nShape:   {df.shape}")
print(f"Range:   {df.index.min().date()} -> {df.index.max().date()}")
print(f"Colunas: {df.columns.tolist()}")
print(f"NaN:     {df.isna().sum().sum()}")
print(df.tail(3))

# ── AÇÃO 3 — Salvar ───────────────────────────────────────────────────────────

out = Path("data/01_raw/derivatives/coinglass/options/BTC_volume_by_exchange.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
