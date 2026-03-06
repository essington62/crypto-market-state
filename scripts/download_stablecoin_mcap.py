"""
Download StableCoin MarketCap History do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_stablecoin_mcap.py
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
    base + "/api/index/stableCoin-marketCap-history",
    headers=headers,
    timeout=30,
)
resp = r.json()
print(f"code={resp.get('code')}")
data = resp.get("data", {})
print(f"Tipo: {type(data).__name__}")
if isinstance(data, dict):
    for k, v in data.items():
        print(f"  {k}: len={len(v)} | primeiro={v[0]} | ultimo={v[-1]}")
elif isinstance(data, list):
    print(f"Lista: {len(data)} items | primeiro={data[0]}")
else:
    print("Estrutura inesperada:", data)
    sys.exit(1)

# ── AÇÃO 2 — Parse ────────────────────────────────────────────────────────────

if isinstance(data, dict):
    time_list  = data.get("time_list", [])
    value_list = data.get("data_list", [])

    if not time_list:
        print("time_list vazio — parando.")
        sys.exit(1)

    first_ts = float(time_list[0])
    unit = "ms" if first_ts > 1e12 else "s"
    print(f"\nTimestamp: {unit} (valor={first_ts})")

    # data_list é lista de dicts {stablecoin: mcap_usd} — expandir em colunas
    dates = pd.to_datetime(time_list, unit=unit, utc=True).normalize()
    df = pd.DataFrame(value_list, index=dates)
    df.index.name = "date"
    df.columns = [f"mcap_{c.lower()}_usd" for c in df.columns]
    df = df.sort_index()
    # Total agregado
    df["stablecoin_mcap_usd"] = df.sum(axis=1, min_count=1)
    df = df[~df.index.duplicated(keep="last")]

elif isinstance(data, list):
    df = pd.DataFrame(data)
    first_ts = float(df.iloc[:, 0].iloc[0])
    unit = "ms" if first_ts > 1e12 else "s"
    print(f"\nTimestamp (fallback list): {unit}")
    df["date"] = pd.to_datetime(df.iloc[:, 0], unit=unit, utc=True).dt.normalize()
    df = df.set_index("date").sort_index()

print(f"Shape:   {df.shape}")
print(f"Range:   {df.index.min().date()} -> {df.index.max().date()}")
print(f"Colunas: {df.columns.tolist()}")
print(f"NaN:     {df.isna().sum().sum()}")
print(f"\nStablecoin MarketCap stats (USD):")
print(df["stablecoin_mcap_usd"].describe().apply(lambda x: f"${x/1e9:.1f}B"))
print(f"\nValor atual: ${df['stablecoin_mcap_usd'].iloc[-1]/1e9:.1f}B")
print(df.tail(3))

# ── AÇÃO 3 — Salvar ───────────────────────────────────────────────────────────

out = Path("data/01_raw/derivatives/coinglass/indices/stablecoin_mcap.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
