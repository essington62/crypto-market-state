"""
Download Bitcoin ETF Flows History do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_etf_flows.py
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
    base + "/api/etf/bitcoin/flow-history",
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

# ── AÇÃO 2 — Parse ────────────────────────────────────────────────────────────

records_total     = []
records_breakdown = []

for row in data:
    ts   = row["timestamp"]
    unit = "ms" if float(ts) > 1e12 else "s"
    date = pd.to_datetime(ts, unit=unit, utc=True).normalize()

    records_total.append({
        "date":     date,
        "flow_usd": row["flow_usd"],
    })

    for etf in row.get("etf_flows", []):
        flow = etf.get("flow_usd")  # pode ser ausente quando ticker nao reportou naquele dia
        records_breakdown.append({
            "date":     date,
            "ticker":   etf["etf_ticker"],
            "flow_usd": flow,
        })

# DataFrame total diario
df_total = pd.DataFrame(records_total)
df_total = df_total.set_index("date").sort_index()
df_total = df_total[~df_total.index.duplicated(keep="last")]

# DataFrame breakdown — pivot por ticker
df_breakdown = pd.DataFrame(records_breakdown)
df_pivot = df_breakdown.pivot_table(
    index="date", columns="ticker",
    values="flow_usd", aggfunc="sum",
)
df_pivot.columns = [f"flow_{c.lower()}" for c in df_pivot.columns]
df_pivot = df_pivot.sort_index()

print(f"\nTotal diario:")
print(f"  Shape:  {df_total.shape}")
print(f"  Range:  {df_total.index.min().date()} -> {df_total.index.max().date()}")
print(f"  NaN:    {df_total.isna().sum().sum()}")

print(f"\nBreakdown por ticker:")
print(f"  Shape:   {df_pivot.shape}")
print(f"  Tickers: {df_pivot.columns.tolist()}")
print(f"  NaN:     {df_pivot.isna().sum().to_dict()}")

print(f"\nFlow total stats (USD):")
print(df_total["flow_usd"].describe().apply(lambda x: f"${x/1e6:.1f}M"))
print(f"\nDias com inflow  positivo: {(df_total['flow_usd'] > 0).sum()}")
print(f"Dias com outflow negativo: {(df_total['flow_usd'] < 0).sum()}")
print(f"\nMaior inflow:  ${df_total['flow_usd'].max()/1e6:.0f}M em {df_total['flow_usd'].idxmax().date()}")
print(f"Maior outflow: ${df_total['flow_usd'].min()/1e6:.0f}M em {df_total['flow_usd'].idxmin().date()}")
print(df_total.tail(3))

# ── AÇÃO 3 — Salvar ───────────────────────────────────────────────────────────

out_dir = Path("data/01_raw/derivatives/coinglass/etf")
out_dir.mkdir(parents=True, exist_ok=True)

out_t = out_dir / "BTC_flows_total.parquet"
df_total.to_parquet(out_t, compression="snappy")
print(f"\nSalvo: {out_t.name} | {df_total.shape}")

out_b = out_dir / "BTC_flows_by_ticker.parquet"
df_pivot.to_parquet(out_b, compression="snappy")
print(f"Salvo: {out_b.name} | {df_pivot.shape}")
