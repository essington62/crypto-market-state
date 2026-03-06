"""
Download Crypto Fear & Greed Index History do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_fear_greed.py
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
    base + "/api/index/fear-greed-history",
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
    print(f"Lista com {len(data)} items")
    print("Primeiro:", data[0])
else:
    print("Estrutura inesperada:", data)
    sys.exit(1)

# ── AÇÃO 2 — Parse ────────────────────────────────────────────────────────────

if isinstance(data, dict):
    time_list  = data.get("time_list",  [])
    value_list = data.get("data_list",  [])
    price_list = data.get("price_list", [])

    if not time_list:
        print("time_list vazio — parando.")
        sys.exit(1)

    first_ts = time_list[0]
    unit = "ms" if float(first_ts) > 1e12 else "s"
    print(f"\nTimestamp: {unit} (valor={first_ts})")

    df = pd.DataFrame({
        "time":       time_list,
        "fear_greed": value_list,
        "btc_price":  price_list,
    })

    df["date"] = pd.to_datetime(df["time"], unit=unit, utc=True).dt.normalize()
    df = df.set_index("date").sort_index()
    df = df.drop(columns=["time", "btc_price"])
    df["fear_greed"] = df["fear_greed"].astype(float)
    df = df[~df.index.duplicated(keep="last")]

elif isinstance(data, list):
    df = pd.DataFrame(data)
    first_ts = df.iloc[:, 0].iloc[0]
    unit = "ms" if float(first_ts) > 1e12 else "s"
    print(f"\nTimestamp (fallback list): {unit}")
    df["date"] = pd.to_datetime(df.iloc[:, 0], unit=unit, utc=True).dt.normalize()
    df = df.set_index("date").sort_index()

print(f"\nShape:       {df.shape}")
print(f"Range:       {df.index.min().date()} -> {df.index.max().date()}")
print(f"Colunas:     {df.columns.tolist()}")
print(f"NaN:         {df.isna().sum().sum()}")
print(f"\nDistribuicao Fear & Greed:")
print(f"  Extreme Fear  (0-25):   {((df['fear_greed'] >= 0)  & (df['fear_greed'] <= 25)).sum()} dias")
print(f"  Fear         (26-45):   {((df['fear_greed'] >= 26) & (df['fear_greed'] <= 45)).sum()} dias")
print(f"  Neutral      (46-55):   {((df['fear_greed'] >= 46) & (df['fear_greed'] <= 55)).sum()} dias")
print(f"  Greed        (56-75):   {((df['fear_greed'] >= 56) & (df['fear_greed'] <= 75)).sum()} dias")
print(f"  Extreme Greed (76-100): {((df['fear_greed'] >= 76) & (df['fear_greed'] <= 100)).sum()} dias")
print(df.tail(3))

# ── AÇÃO 3 — Salvar ───────────────────────────────────────────────────────────

out = Path("data/01_raw/derivatives/coinglass/indices/fear_greed.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, compression="snappy")

df_check = pd.read_parquet(out)
print(f"\nSalvo: {df_check.shape} | tz={df_check.index.tz}")
print(f"Path: {out}")
