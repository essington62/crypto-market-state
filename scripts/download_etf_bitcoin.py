"""
Download ETF Bitcoin History para os principais tickers do Coinglass V4.

Run:
    export COINGLASS_API_KEY=<key>
    conda run -n crypto_market_state python scripts/download_etf_bitcoin.py
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

key     = os.environ["COINGLASS_API_KEY"]
base    = "https://open-api-v4.coinglass.com"
headers = {"CG-API-KEY": key, "Accept": "application/json"}

TICKERS = ["GBTC", "IBIT", "FBTC", "ARKB", "BITB"]

# ── AÇÃO 1 — Download todos os tickers ───────────────────────────────────────

raw = {}
for ticker in TICKERS:
    time.sleep(2.5)
    r = requests.get(
        base + "/api/etf/bitcoin/history",
        headers=headers,
        params={"ticker": ticker},
        timeout=30,
    )
    resp = r.json()
    data = resp.get("data", [])
    print(f"{ticker}: code={resp.get('code')} | rows={len(data)}")
    if data:
        print(f"  Primeiro: {data[0]}")
        print(f"  Ultimo:   {data[-1]}")
    else:
        print(f"  msg: {resp.get('msg', '')}")
    raw[ticker] = data

# ── AÇÃO 2 — Parse cada ticker ────────────────────────────────────────────────

def parse_etf(data, ticker):
    df = pd.DataFrame(data)
    ts_col = "assets_date" if "assets_date" in df.columns else "market_date"
    if ts_col not in df.columns:
        ts_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        ts_col = ts_candidates[0] if ts_candidates else df.columns[0]
    # Usar float() para suportar np.int64 e np.float64 (isinstance(np.int64, int) falha em Python 3)
    try:
        ts_val = float(df[ts_col].iloc[0])
        unit = "ms" if ts_val > 1e12 else "s"
        df["date"] = pd.to_datetime(df[ts_col], unit=unit, utc=True).dt.normalize()
    except (TypeError, ValueError):
        df["date"] = pd.to_datetime(df[ts_col], utc=True).dt.normalize()
    df = df.set_index("date").sort_index()
    ts_cols = [c for c in df.columns if "date" in c.lower()]
    df = df.drop(columns=ts_cols, errors="ignore")
    for col in df.select_dtypes(include="object").columns:
        if col not in ["name", "ticker"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[~df.index.duplicated(keep="last")]
    return df

dfs = {}
for ticker, data in raw.items():
    if not data:
        print(f"\n{ticker}: sem dados — ignorado")
        continue
    dfs[ticker] = parse_etf(data, ticker)
    df = dfs[ticker]
    print(f"\n{ticker}:")
    print(f"  Shape:  {df.shape}")
    print(f"  Range:  {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Cols:   {df.columns.tolist()}")
    print(f"  NaN:    {df.isna().sum().to_dict()}")

# ── AÇÃO 3 — Consolidar ───────────────────────────────────────────────────────

holdings_df   = pd.DataFrame()
netassets_df  = pd.DataFrame()

for ticker, df in dfs.items():
    if "btc_holdings" in df.columns:
        holdings_df[ticker] = df["btc_holdings"]
    if "net_assets" in df.columns:
        netassets_df[ticker] = df["net_assets"]

if not holdings_df.empty:
    holdings_df["total_btc_holdings"] = holdings_df.sum(axis=1, min_count=1)
    print(f"\nBTC Holdings consolidado:")
    print(f"  Shape: {holdings_df.shape}")
    print(f"  Range: {holdings_df.index.min().date()} -> {holdings_df.index.max().date()}")
    print(holdings_df.tail(3))

if not netassets_df.empty:
    netassets_df["total_net_assets_usd"] = netassets_df.sum(axis=1, min_count=1)
    print(f"\nNet Assets consolidado:")
    print(netassets_df[["total_net_assets_usd"]].tail(3))

# ── AÇÃO 4 — Salvar ───────────────────────────────────────────────────────────

out_dir = Path("data/01_raw/derivatives/coinglass/etf")
out_dir.mkdir(parents=True, exist_ok=True)

for ticker, df in dfs.items():
    out = out_dir / f"BTC_{ticker}.parquet"
    df.to_parquet(out, compression="snappy")
    print(f"Salvo: {out.name} | {df.shape}")

if not holdings_df.empty:
    out_h = out_dir / "BTC_holdings_consolidated.parquet"
    holdings_df.to_parquet(out_h, compression="snappy")
    print(f"Salvo: {out_h.name} | {holdings_df.shape}")

if not netassets_df.empty:
    out_n = out_dir / "BTC_netassets_consolidated.parquet"
    netassets_df.to_parquet(out_n, compression="snappy")
    print(f"Salvo: {out_n.name} | {netassets_df.shape}")
