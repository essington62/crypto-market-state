"""
Timeframe Analysis — Statistical analysis to determine optimal execution timeframe.

Compares 15min, 30min, 1h and 4h candles for BTC day-trade entry timing.
Data range: 2025-09 → 2026-03 (aligned with CoinGlass / training period).

Usage:
    python scripts/analysis/timeframe_analysis.py [--skip-download]

Outputs:
    data/01_raw/spot/crypto/15m/BTCUSDT_15m.parquet
    data/01_raw/spot/crypto/30m/BTCUSDT_30m.parquet
    data/01_raw/spot/crypto/1h/BTCUSDT_1h.parquet
    data/07_reports/timeframe_analysis_report.txt

NOT a Kedro pipeline — exploratory analysis script.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
DIR_15M    = ROOT / "data" / "01_raw" / "spot" / "crypto" / "15m"
DIR_30M    = ROOT / "data" / "01_raw" / "spot" / "crypto" / "30m"
DIR_1H     = ROOT / "data" / "01_raw" / "spot" / "crypto" / "1h"
DIR_4H     = ROOT / "data" / "01_raw" / "spot" / "crypto" / "4h"
DIR_REPORT = ROOT / "data" / "07_reports"

PARQUET_15M = DIR_15M / "BTCUSDT_15m.parquet"
PARQUET_30M = DIR_30M / "BTCUSDT_30m.parquet"
PARQUET_1H  = DIR_1H  / "BTCUSDT_1h.parquet"

for d in [DIR_15M, DIR_30M, DIR_1H, DIR_REPORT]:
    d.mkdir(parents=True, exist_ok=True)

# ── Binance Vision URLs ────────────────────────────────────────────────────────
BASE_MONTHLY = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m"
BASE_DAILY   = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/15m"

MONTHS = [
    "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02",
]
# March 2026 — daily files (month not yet complete)
DAILY_2026_03 = [f"2026-03-{d:02d}" for d in range(1, 26)]

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

NUMERIC_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume",
]


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Download & Parse
# ══════════════════════════════════════════════════════════════════════════════

def _download_zip(url: str, dest_dir: Path) -> list[pd.DataFrame]:
    """Download a zip from Binance Vision, extract CSV, return DataFrame list."""
    zip_path = Path("/tmp") / Path(url).name
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as exc:
        print(f"    ✗ {Path(url).name}: {exc}")
        return []

    dfs = []
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                if name.endswith(".csv"):
                    with z.open(name) as f:
                        text = f.read().decode("utf-8")
                    df = pd.read_csv(StringIO(text), header=None, names=COLUMNS)
                    dfs.append(df)
    except Exception as exc:
        print(f"    ✗ unzip {zip_path.name}: {exc}")
    finally:
        zip_path.unlink(missing_ok=True)

    return dfs


def download_15m() -> pd.DataFrame:
    """Download all 15m candles from Binance Vision, return concatenated DataFrame."""
    all_dfs = []

    print("[download] Monthly files …")
    for month in MONTHS:
        filename = f"BTCUSDT-15m-{month}.zip"
        url = f"{BASE_MONTHLY}/{filename}"
        print(f"  → {filename}")
        dfs = _download_zip(url, DIR_15M)
        all_dfs.extend(dfs)

    print("[download] Daily files for 2026-03 …")
    for day in DAILY_2026_03:
        filename = f"BTCUSDT-15m-{day}.zip"
        url = f"{BASE_DAILY}/{filename}"
        print(f"  → {filename}", end="")
        dfs = _download_zip(url, DIR_15M)
        if dfs:
            all_dfs.extend(dfs)
            print(" ✓")
        else:
            print()  # already printed ✗

    if not all_dfs:
        raise RuntimeError("[download] No data downloaded — check network / URLs.")

    df = pd.concat(all_dfs, ignore_index=True)

    # Drop header rows. Binance Vision 2025+ files use microseconds (μs) for timestamps.
    # Valid μs range: 2020-01-01 (1577836800000000) → 2030-01-01 (1893456000000000)
    _ot = pd.to_numeric(df["open_time"], errors="coerce")
    valid = _ot.notna() & _ot.between(1_577_836_800_000_000, 1_893_456_000_000_000)
    df = df[valid].copy()
    df["open_time"] = _ot[valid].astype("int64")

    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)

    # Convert microseconds → UTC datetime
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="us", utc=True)
    df = df.set_index("timestamp")

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df = df.drop(columns=["close_time", "ignore"], errors="ignore")
    return df


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    return df.resample(freq).agg({
        "open":                   "first",
        "high":                   "max",
        "low":                    "min",
        "close":                  "last",
        "volume":                 "sum",
        "quote_volume":           "sum",
        "trades":                 "sum",
        "taker_buy_volume":       "sum",
        "taker_buy_quote_volume": "sum",
    }).dropna(subset=["open"])


def load_4h_buffer() -> pd.DataFrame | None:
    """Load existing 4h buffer from paper trading state."""
    buffer_path = ROOT / "scripts" / "paper_trading" / "state" / "specialist_4h" / "ohlcv_4h_buffer.parquet"
    if buffer_path.exists():
        df = pd.read_parquet(buffer_path)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    # Fallback: L1 4h raw
    raw_4h = DIR_4H / "BTCUSDT.parquet"
    if raw_4h.exists():
        df = pd.read_parquet(raw_4h)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Indicator helpers
# ══════════════════════════════════════════════════════════════════════════════

PERIODS_IN_24H = {"15m": 96, "30m": 48, "1h": 24, "4h": 6}


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _bb_pct_b(close: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.Series:
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return (close - lower) / (upper - lower).replace(0, np.nan)


def _vol_zscore(volume: pd.Series, period: int = 50) -> pd.Series:
    mean = volume.rolling(period).mean()
    std  = volume.rolling(period).std().replace(0, np.nan)
    return (volume - mean) / std


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1 — Time-to-Target
# ══════════════════════════════════════════════════════════════════════════════

def time_to_target(df_15m: pd.DataFrame, targets: list[float] = None) -> pd.DataFrame:
    """
    For each 15m candle, measure time until price rises X%.
    Max look-forward: 24h (96 candles × 15min).
    """
    if targets is None:
        targets = [0.01, 0.015, 0.02]

    close      = df_15m["close"].values
    high       = df_15m["high"].values
    timestamps = df_15m.index
    n          = len(close)
    results    = []

    for i in range(n - 1):
        entry = close[i]
        for target in targets:
            target_price = entry * (1 + target)
            hit_j = None
            for j in range(i + 1, min(i + 97, n)):
                if high[j] >= target_price:
                    hit_j = j
                    break
            if hit_j is not None:
                minutes = (timestamps[hit_j] - timestamps[i]).total_seconds() / 60
                results.append({"target_pct": target, "time_minutes": minutes, "hit": True})
            else:
                results.append({"target_pct": target, "time_minutes": None, "hit": False})

    return pd.DataFrame(results)


def report_time_to_target(df_15m: pd.DataFrame) -> dict:
    print("\n[Analysis 1] Time-to-Target …")
    ttg = time_to_target(df_15m)
    out = {}
    for t in [0.01, 0.015, 0.02]:
        sub    = ttg[ttg["target_pct"] == t]
        hits   = sub[sub["hit"]]
        total  = len(sub)
        n_hit  = len(hits)
        hr     = n_hit / total if total > 0 else 0
        med    = hits["time_minutes"].median()
        p25    = hits["time_minutes"].quantile(0.25)
        p75    = hits["time_minutes"].quantile(0.75)
        out[t] = {"hit_rate": hr, "median_min": med, "p25_min": p25, "p75_min": p75, "n": total}
        print(f"  Target {t*100:.1f}%: hit_rate={hr*100:.1f}%  median={med:.0f}min  P25={p25:.0f}  P75={p75:.0f}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 2 — RSI Entry
# ══════════════════════════════════════════════════════════════════════════════

def rsi_entry_analysis(df: pd.DataFrame, tf_label: str, target_pct: float = 0.015) -> dict:
    df = df.copy()
    df["rsi"]              = _rsi(df["close"])
    periods_24h            = PERIODS_IN_24H[tf_label]
    df["max_fwd_return"]   = (
        df["high"].rolling(periods_24h).max().shift(-periods_24h) / df["close"] - 1
    )
    out = {}
    for thresh in [30, 35, 40]:
        sub = df[df["rsi"] < thresh].copy()
        n   = len(sub)
        hr  = (sub["max_fwd_return"] >= target_pct).mean() if n > 0 else 0.0
        out[thresh] = {"n": n, "hit_rate": hr}
    return out


def report_rsi_entry(dfs: dict) -> dict:
    print("\n[Analysis 2] RSI oversold hit rate (target=1.5%, 24h window):")
    results = {}
    for tf, df in dfs.items():
        r = rsi_entry_analysis(df, tf)
        results[tf] = r
        for thresh, vals in r.items():
            print(f"  {tf} RSI<{thresh}: hit_rate={vals['hit_rate']*100:.1f}%  n={vals['n']}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 3 — Bollinger %B Entry
# ══════════════════════════════════════════════════════════════════════════════

def bollinger_entry_analysis(df: pd.DataFrame, tf_label: str, target_pct: float = 0.015) -> dict:
    df = df.copy()
    df["bb_pct_b"]        = _bb_pct_b(df["close"])
    periods_24h           = PERIODS_IN_24H[tf_label]
    df["max_fwd_return"]  = (
        df["high"].rolling(periods_24h).max().shift(-periods_24h) / df["close"] - 1
    )
    out = {}
    for thresh in [0.1, 0.2, 0.3]:
        sub = df[df["bb_pct_b"] < thresh].copy()
        n   = len(sub)
        hr  = (sub["max_fwd_return"] >= target_pct).mean() if n > 0 else 0.0
        out[thresh] = {"n": n, "hit_rate": hr}
    return out


def report_bollinger_entry(dfs: dict) -> dict:
    print("\n[Analysis 3] Bollinger %B hit rate (target=1.5%, 24h window):")
    results = {}
    for tf, df in dfs.items():
        r = bollinger_entry_analysis(df, tf)
        results[tf] = r
        for thresh, vals in r.items():
            print(f"  {tf} BB<{thresh:.1f}: hit_rate={vals['hit_rate']*100:.1f}%  n={vals['n']}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 4 — Time-to-Stop-Loss
# ══════════════════════════════════════════════════════════════════════════════

def time_to_stop(df_15m: pd.DataFrame, stop_pct: float = 0.02) -> dict:
    close      = df_15m["close"].values
    low        = df_15m["low"].values
    timestamps = df_15m.index
    n          = len(close)
    times      = []

    for i in range(n - 1):
        entry       = close[i]
        stop_price  = entry * (1 - stop_pct)
        for j in range(i + 1, min(i + 97, n)):
            if low[j] <= stop_price:
                minutes = (timestamps[j] - timestamps[i]).total_seconds() / 60
                times.append(minutes)
                break

    r = pd.Series(times)
    return {
        "n_hits": len(times),
        "median_min": r.median(),
        "p10_min":    r.quantile(0.10),
        "p25_min":    r.quantile(0.25),
    }


def report_time_to_stop(df_15m: pd.DataFrame) -> dict:
    print("\n[Analysis 4] Time-to-Stop-Loss (15m resolution, 24h window):")
    out = {}
    for pct in [0.02, 0.03]:
        r = time_to_stop(df_15m, pct)
        out[pct] = r
        print(
            f"  Stop {pct*100:.0f}%: n_hits={r['n_hits']}  "
            f"P10={r['p10_min']:.0f}min  P25={r['p25_min']:.0f}min  median={r['median_min']:.0f}min"
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 5 — Combined Signal
# ══════════════════════════════════════════════════════════════════════════════

def combined_signal_analysis(
    df:           pd.DataFrame,
    tf_label:     str,
    target_pct:   float = 0.015,
    stop_pct:     float = 0.02,
) -> dict:
    df = df.copy()

    df["rsi"]      = _rsi(df["close"])
    df["bb_pct_b"] = _bb_pct_b(df["close"])
    df["vol_z"]    = _vol_zscore(df["volume"])

    periods_24h    = PERIODS_IN_24H[tf_label]
    df["max_up"]   = df["high"].rolling(periods_24h).max().shift(-periods_24h) / df["close"] - 1
    df["max_down"] = df["low"].rolling(periods_24h).min().shift(-periods_24h)  / df["close"] - 1

    signal  = (df["rsi"] < 35) & (df["bb_pct_b"] < 0.3) & (df["vol_z"] > 0)
    entries = df[signal].dropna(subset=["max_up", "max_down"])

    n = len(entries)
    if n == 0:
        return {"n": 0, "hit_rate": 0.0, "stop_rate": 0.0, "ratio": 0.0}

    hit_rate  = (entries["max_up"]   >= target_pct).mean()
    stop_rate = (entries["max_down"] <= -stop_pct).mean()
    ratio     = hit_rate / max(stop_rate, 0.01)

    return {"n": n, "hit_rate": hit_rate, "stop_rate": stop_rate, "ratio": ratio}


def report_combined_signal(dfs: dict) -> dict:
    print("\n[Analysis 5] Combined signal: RSI<35 AND BB<0.3 AND vol_z>0 (target=1.5%, stop=2%):")
    results = {}
    for tf, df in dfs.items():
        r = combined_signal_analysis(df, tf)
        results[tf] = r
        print(
            f"  {tf}: n={r['n']}  hit={r['hit_rate']*100:.1f}%  "
            f"stop={r['stop_rate']*100:.1f}%  ratio={r['ratio']:.2f}"
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Report
# ══════════════════════════════════════════════════════════════════════════════

def _best_tf(results: dict, key: str, sub_key=None) -> str:
    """Return timeframe with highest value for given key."""
    best_tf = max(
        results,
        key=lambda tf: (
            results[tf][key] if sub_key is None
            else (results[tf][key][sub_key] if key in results[tf] else 0.0)
        ),
    )
    return best_tf


def generate_report(
    ttg_results:      dict,
    rsi_results:      dict,
    bb_results:       dict,
    stop_results:     dict,
    combined_results: dict,
    df_range:         tuple,
) -> str:
    # Best RSI timeframe: highest hit_rate at RSI<35
    best_rsi_tf = max(rsi_results, key=lambda tf: rsi_results[tf].get(35, {}).get("hit_rate", 0))
    best_rsi_hr = rsi_results[best_rsi_tf].get(35, {}).get("hit_rate", 0)

    # Best BB timeframe: highest hit_rate at BB<0.2
    best_bb_tf = max(bb_results, key=lambda tf: bb_results[tf].get(0.2, {}).get("hit_rate", 0))
    best_bb_hr = bb_results[best_bb_tf].get(0.2, {}).get("hit_rate", 0)

    # Best combined timeframe: highest ratio
    best_comb_tf = max(combined_results, key=lambda tf: combined_results[tf].get("ratio", 0))
    best_comb    = combined_results[best_comb_tf]

    # Time-to-target 1.5%
    tgt_15 = ttg_results.get(0.015, {})
    med_min = tgt_15.get("median_min", float("nan"))
    p10_min = stop_results.get(0.02, {}).get("p10_min", float("nan"))

    # Recommend execution timeframe based on time-to-stop P10
    if not np.isnan(p10_min):
        if p10_min <= 20:
            exec_tf = "15min"
        elif p10_min <= 40:
            exec_tf = "30min"
        else:
            exec_tf = "1h"
    else:
        exec_tf = "15min"

    lines = [
        "═" * 55,
        "TIMEFRAME ANALYSIS — SUMMARY",
        f"Data range: {df_range[0]} → {df_range[1]}",
        "═" * 55,
        "",
        "Pergunta 1: Time-to-target (1.5%)",
        f"  Hit rate 24h  : {tgt_15.get('hit_rate', 0)*100:.1f}%",
        f"  Mediana       : {med_min:.0f} min",
        f"  P25 / P75     : {tgt_15.get('p25_min', 0):.0f} / {tgt_15.get('p75_min', 0):.0f} min",
        f"  → Candle recomendado: {exec_tf}",
        "",
        "Pergunta 2: RSI oversold hit rate (target=1.5%)",
        f"  → Melhor timeframe: {best_rsi_tf} (hit_rate={best_rsi_hr*100:.1f}% @ RSI<35)",
        *(
            f"    {tf}: RSI<35={rsi_results[tf].get(35,{}).get('hit_rate',0)*100:.1f}%  "
            f"RSI<40={rsi_results[tf].get(40,{}).get('hit_rate',0)*100:.1f}%"
            for tf in rsi_results
        ),
        "",
        "Pergunta 3: Bollinger %B hit rate (target=1.5%)",
        f"  → Melhor timeframe: {best_bb_tf} (hit_rate={best_bb_hr*100:.1f}% @ BB<0.2)",
        *(
            f"    {tf}: BB<0.2={bb_results[tf].get(0.2,{}).get('hit_rate',0)*100:.1f}%  "
            f"BB<0.3={bb_results[tf].get(0.3,{}).get('hit_rate',0)*100:.1f}%"
            for tf in bb_results
        ),
        "",
        "Pergunta 4: Time-to-Stop-Loss (2%)",
        f"  P10 = {p10_min:.0f} min  (10% dos stops mais rápidos)",
        f"  P25 = {stop_results.get(0.02,{}).get('p25_min', 0):.0f} min",
        f"  Median = {stop_results.get(0.02,{}).get('median_min', 0):.0f} min",
        f"  → Monitoramento mínimo: cada {exec_tf}",
        "",
        "Pergunta 5: Combinação ótima (RSI<35 AND BB<0.3 AND vol_z>0)",
        f"  → Melhor: {best_comb_tf}  "
        f"(hit={best_comb['hit_rate']*100:.1f}%  stop={best_comb['stop_rate']*100:.1f}%  ratio={best_comb['ratio']:.2f})",
        *(
            f"    {tf}: n={combined_results[tf]['n']}  "
            f"hit={combined_results[tf]['hit_rate']*100:.1f}%  "
            f"stop={combined_results[tf]['stop_rate']*100:.1f}%  "
            f"ratio={combined_results[tf]['ratio']:.2f}"
            for tf in combined_results
        ),
        "",
        "═" * 55,
        "RECOMENDAÇÃO FINAL:",
        f"  Execução / stop monitor : {exec_tf}",
        f"  Timing de entrada       : {best_comb_tf} (melhor ratio)",
        "  Modelo AB estratégico   : mantém 4h (decisor)",
        "═" * 55,
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(skip_download: bool = False) -> None:
    # ── Step 1: Download / load 15m data ──────────────────────────────────────
    if skip_download and PARQUET_15M.exists():
        print(f"[load] Loading cached 15m parquet: {PARQUET_15M}")
        df_15m = pd.read_parquet(PARQUET_15M)
        if df_15m.index.tz is None:
            df_15m.index = df_15m.index.tz_localize("UTC")
    else:
        print("[download] Fetching 15m candles from Binance Vision …")
        df_15m = download_15m()
        df_15m.to_parquet(PARQUET_15M)
        print(f"[save] 15m parquet: {len(df_15m)} candles  {df_15m.index[0]} → {df_15m.index[-1]}")

    # Filter to analysis window (Sep 2025 → Mar 2026)
    t_start = pd.Timestamp("2025-09-01", tz="UTC")
    t_end   = pd.Timestamp("2026-03-26", tz="UTC")
    df_15m  = df_15m.loc[t_start:t_end]
    print(f"[filter] 15m: {len(df_15m)} candles  {df_15m.index[0].date()} → {df_15m.index[-1].date()}")

    # ── Step 2: Resample ──────────────────────────────────────────────────────
    print("[resample] 30min …")
    df_30m = resample_ohlcv(df_15m, "30min")
    df_30m.to_parquet(PARQUET_30M)
    print(f"[save] 30m: {len(df_30m)} candles")

    print("[resample] 1h …")
    df_1h = resample_ohlcv(df_15m, "1h")
    df_1h.to_parquet(PARQUET_1H)
    print(f"[save] 1h: {len(df_1h)} candles")

    # Load 4h from paper trading buffer
    df_4h = load_4h_buffer()
    if df_4h is not None:
        df_4h = df_4h.loc[t_start:t_end]
        print(f"[load] 4h: {len(df_4h)} candles from buffer")
    else:
        print("[warn] 4h buffer not found — resampling from 15m")
        df_4h = resample_ohlcv(df_15m, "4h")

    dfs = {"15m": df_15m, "30m": df_30m, "1h": df_1h, "4h": df_4h}

    # ── Analyses ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("TIMEFRAME ANALYSIS — BTCUSDT  Sep/2025 → Mar/2026")
    print("═" * 55)

    ttg_results      = report_time_to_target(df_15m)
    rsi_results      = report_rsi_entry(dfs)
    bb_results       = report_bollinger_entry(dfs)
    stop_results     = report_time_to_stop(df_15m)
    combined_results = report_combined_signal(
        {tf: df for tf, df in dfs.items() if tf != "4h"}
    )

    # ── Report ────────────────────────────────────────────────────────────────
    report = generate_report(
        ttg_results, rsi_results, bb_results,
        stop_results, combined_results,
        (df_15m.index[0].date(), df_15m.index[-1].date()),
    )

    print("\n" + report)

    report_path = DIR_REPORT / "timeframe_analysis_report.txt"
    report_path.write_text(report)
    print(f"\n[save] Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timeframe Analysis — BTC 15m/30m/1h/4h")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download if parquet already exists",
    )
    args = parser.parse_args()
    main(skip_download=args.skip_download)
