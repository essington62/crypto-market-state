#!/usr/bin/env python3
"""
Data Quality Monitoring for L1 Raw Data Files.

This script checks all specified parquet files for:
- File existence
- Freshness (last date vs. today, using stale_days threshold)
- NaN counts in the last 5 rows

Outputs:
- Console report (tabulated)
- Text report in logs/ directory
- JSON status file for monitoring
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import tabulate

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE = "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state"
LOG_DIR = os.path.join(BASE, "logs")

# Files to check (all absolute paths confirmed)
FILES_TO_CHECK = [
    # BTC Spot
    {
        "name": "BTC spot daily",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/spot/crypto/daily_24x7/BTCUSDT.parquet",
        "stale_days": 2
    },
    {
        "name": "BTC spot 4h",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/spot/crypto/4h/BTCUSDT.parquet",
        "stale_days": 4
    },
    # Macro FRED
    {
        "name": "DGS2",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/macro/daily/DGS2.parquet",
        "stale_days": 5
    },
    {
        "name": "DGS10",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/macro/daily/DGS10.parquet",
        "stale_days": 5
    },
    {
        "name": "FEDFUNDS",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/macro/daily/FEDFUNDS.parquet",
        "stale_days": 45
    },
    {
        "name": "CPIAUCSL",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/macro/daily/CPIAUCSL.parquet",
        "stale_days": 60
    },
    {
        "name": "WALCL weekly",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/macro/weekly/WALCL.parquet",
        "stale_days": 21
    },
    {
        "name": "VIX",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/macro/daily/vix.parquet",
        "stale_days": 2
    },
    {
        "name": "DXY",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/macro/daily/dxy.parquet",
        "stale_days": 2
    },
    # Yahoo Finance
    {
        "name": "SP500",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/spot/business_day/sp500.parquet",
        "stale_days": 2
    },
    # CoinGlass Derivatives
    {
        "name": "CoinGlass deriv 4h",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/derivatives/coinglass/4h/BTCUSDT.parquet",
        "stale_days": 2
    },
    {
        "name": "CoinGlass funding OI",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/derivatives/coinglass/funding/BTCUSDT_oi_weighted.parquet",
        "stale_days": 2
    },
    {
        "name": "CoinGlass OI agg",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/derivatives/coinglass/open_interest/BTCUSDT_aggregated.parquet",
        "stale_days": 2
    },
    {
        "name": "CoinGlass fear greed",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/derivatives/coinglass/indices/fear_greed.parquet",
        "stale_days": 2
    },
    # Yahoo Finance — new tickers (L2 normalized)
    {
        "name": "Oil WTI",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/02_intermediate/spot/business_day/oil_wti.parquet",
        "stale_days": 2
    },
    {
        "name": "Oil Brent",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/02_intermediate/spot/business_day/oil_brent.parquet",
        "stale_days": 2
    },
    {
        "name": "MOVE Index",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/02_intermediate/spot/business_day/move_index.parquet",
        "stale_days": 6
    },
    {
        "name": "OVX",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/02_intermediate/spot/business_day/oil_volatility.parquet",
        "stale_days": 4
    },
    {
        "name": "Defense ETF",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/02_intermediate/spot/business_day/defense_etf.parquet",
        "stale_days": 2
    },
    {
        "name": "HYG",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/02_intermediate/spot/business_day/high_yield_bonds.parquet",
        "stale_days": 2
    },
    {
        "name": "TIP",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/02_intermediate/spot/business_day/tips_etf.parquet",
        "stale_days": 2
    },
    # CoinGlass Indices
    {
        "name": "CG ahr999",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/derivatives/coinglass/indices/ahr999.parquet",
        "stale_days": 2
    },
    {
        "name": "CG bubble index",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/derivatives/coinglass/indices/bubble_index.parquet",
        "stale_days": 2
    },
    {
        "name": "CG puell multiple",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/derivatives/coinglass/indices/puell_multiple.parquet",
        "stale_days": 2
    },
    {
        "name": "CG stablecoin mcap",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/derivatives/coinglass/indices/stablecoin_mcap.parquet",
        "stale_days": 2
    },
    {
        "name": "CG CDRI index",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/derivatives/coinglass/indices/cdri_index.parquet",
        "stale_days": 2
    },
    {
        "name": "CG CGDI index",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/derivatives/coinglass/indices/cgdi_index.parquet",
        "stale_days": 2
    },
    # CoinGlass Order Book
    {
        "name": "CoinGlass OB r1",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/orderbook/coinglass/4h/BTCUSDT_r1.parquet",
        "stale_days": 4
    },
    {
        "name": "CoinGlass OB r2",
        "path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/01_raw/orderbook/coinglass/4h/BTCUSDT_r2.parquet",
        "stale_days": 4
    },
]

# =============================================================================
# UTILITIES
# =============================================================================
def ensure_log_dir():
    """Create log directory if it doesn't exist."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


def get_utc_now():
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def get_last_date_from_dataframe(df):
    """
    Extrai última data do DataFrame.
    Tenta DatetimeIndex primeiro; fallback para colunas de data comuns.
    Retorna date object ou None.
    """
    idx = df.index

    if not isinstance(idx, pd.DatetimeIndex):
        # Fallback: procura coluna de data antes de tentar converter índice
        date_cols = ["date", "timestamp", "open_time", "time", "Date", "datetime"]
        found = next((c for c in date_cols if c in df.columns), None)
        if found:
            try:
                idx = pd.DatetimeIndex(pd.to_datetime(df[found]))
            except Exception:
                return None
        else:
            try:
                idx = pd.DatetimeIndex(pd.to_datetime(df.index))
            except Exception:
                return None

    # Normaliza timezone → UTC
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    last = idx.max()
    if pd.isna(last) or last.year < 2000:   # sanity check
        return None

    return last.date()



def compute_nan_last_rows(df, n=5):
    """
    Compute total NaN count in the last n rows of the DataFrame.
    Returns integer count.
    """
    if len(df) == 0:
        return 0
    last_rows = df.iloc[-n:]
    return int(last_rows.isna().sum().sum())


def check_file(file_info, today_utc):
    """
    Check a single file:
    - existence
    - load parquet
    - last date
    - staleness
    - NaN count in last 5 rows

    Returns dict with status, last_date, days_ago, nan_count, and optional error.
    """
    name = file_info["name"]
    path = file_info["path"]
    stale_days = file_info["stale_days"]

    result = {
        "name": name,
        "path": path,
        "stale_days": stale_days,
        "status": "UNKNOWN",
        "last_date": None,
        "days_ago": None,
        "nan_count": 0,
        "error": None,
    }

    # 1. Check file existence
    if not os.path.exists(path):
        result["status"] = "MISSING"
        result["error"] = "File not found"
        return result

    # 2. Load parquet
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        result["status"] = "MISSING"
        result["error"] = f"Failed to read parquet: {e}"
        return result

    if df.empty:
        result["status"] = "MISSING"
        result["error"] = "Empty DataFrame"
        return result

    # 3. Get last date
    last_date = get_last_date_from_dataframe(df)
    if last_date is None:
        result["status"] = "MISSING"
        result["error"] = "Index is not datetime-like or conversion failed"
        return result

    result["last_date"] = last_date

    # 4. Compute days ago
    days_ago = (today_utc.date() - last_date).days
    result["days_ago"] = days_ago

    # 5. Determine freshness
    if days_ago <= stale_days:
        result["status"] = "FRESH"
    else:
        result["status"] = "STALE"

    # 6. NaN count in last 5 rows
    result["nan_count"] = compute_nan_last_rows(df, n=5)

    return result


# =============================================================================
# REPORTING
# =============================================================================
def print_table(results, now_utc):
    """Print formatted table to console."""
    headers = ["Asset", "Last Date", "Days Ago", "Status", "NaN"]
    table_data = []

    for r in results:
        # Format status with checkmark if fresh
        if r["status"] == "FRESH":
            status_display = "✓ FRESH"
        elif r["status"] == "STALE":
            status_display = "✗ STALE"
        elif r["status"] == "MISSING":
            status_display = "✗ MISSING"
        else:
            status_display = r["status"]

        # Format last date
        last_date_str = r["last_date"].isoformat() if r["last_date"] else "-"
        days_ago_str = str(r["days_ago"]) if r["days_ago"] is not None else "-"

        table_data.append([
            r["name"],
            last_date_str,
            days_ago_str,
            status_display,
            r["nan_count"],
        ])

    # Print header with timestamp
    timestamp = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    print("═" * 60)
    print(f"DATA QUALITY REPORT — {timestamp}")
    print("═" * 60)

    print(tabulate.tabulate(table_data, headers=headers, tablefmt="simple"))

    # Print summary
    total = len(results)
    fresh = sum(1 for r in results if r["status"] == "FRESH")
    stale = sum(1 for r in results if r["status"] == "STALE")
    missing = sum(1 for r in results if r["status"] == "MISSING")

    print("─" * 60)
    print(f"Summary: {fresh}/{total} FRESH | {stale} STALE | {missing} MISSING")
    print("═" * 60)

    # Print warnings if any stale or missing
    if stale > 0 or missing > 0:
        print("\n⚠ WARNING: files need attention:")
        for r in results:
            if r["status"] in ("STALE", "MISSING"):
                if r["status"] == "STALE":
                    print(f"  - {r['name']}: last update {r['days_ago']} days ago (threshold: {r['stale_days']})")
                else:
                    print(f"  - {r['name']}: {r['error']}")


def save_text_report(results, now_utc):
    """Save formatted report to text file."""
    timestamp = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    date_str = now_utc.strftime("%Y-%m-%d")

    # Build report content
    lines = []
    lines.append("=" * 60)
    lines.append(f"DATA QUALITY REPORT — {timestamp}")
    lines.append("=" * 60)

    headers = ["Asset", "Last Date", "Days Ago", "Status", "NaN"]
    table_data = []
    for r in results:
        last_date_str = r["last_date"].isoformat() if r["last_date"] else "-"
        days_ago_str = str(r["days_ago"]) if r["days_ago"] is not None else "-"
        status_display = r["status"]
        table_data.append([
            r["name"],
            last_date_str,
            days_ago_str,
            status_display,
            r["nan_count"],
        ])

    # Manual formatting since tabulate might not be available in saved file
    col_widths = [25, 12, 8, 10, 5]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * 60)

    for row in table_data:
        row_line = "  ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        lines.append(row_line)

    lines.append("-" * 60)
    total = len(results)
    fresh = sum(1 for r in results if r["status"] == "FRESH")
    stale = sum(1 for r in results if r["status"] == "STALE")
    missing = sum(1 for r in results if r["status"] == "MISSING")
    lines.append(f"Summary: {fresh}/{total} FRESH | {stale} STALE | {missing} MISSING")
    lines.append("=" * 60)

    if stale > 0 or missing > 0:
        lines.append("\n⚠ WARNING: files need attention:")
        for r in results:
            if r["status"] in ("STALE", "MISSING"):
                if r["status"] == "STALE":
                    lines.append(f"  - {r['name']}: last update {r['days_ago']} days ago (threshold: {r['stale_days']})")
                else:
                    lines.append(f"  - {r['name']}: {r['error']}")

    # Write dated file
    dated_path = os.path.join(LOG_DIR, f"data_quality_{date_str}.txt")
    with open(dated_path, "w") as f:
        f.write("\n".join(lines))

    # Write last report (overwrite)
    last_path = os.path.join(LOG_DIR, "last_quality_report.txt")
    with open(last_path, "w") as f:
        f.write("\n".join(lines))


def save_json_status(results, now_utc):
    """Save JSON status file."""
    total = len(results)
    fresh = sum(1 for r in results if r["status"] == "FRESH")
    stale = sum(1 for r in results if r["status"] == "STALE")
    missing = sum(1 for r in results if r["status"] == "MISSING")

    warnings = []
    for r in results:
        if r["status"] == "STALE":
            warnings.append(f"{r['name']}: last update {r['days_ago']} days ago")
        elif r["status"] == "MISSING":
            warnings.append(f"{r['name']}: {r['error']}")

    status_data = {
        "status": "ok" if (stale == 0 and missing == 0) else "warning",
        "date": now_utc.strftime("%Y-%m-%d"),
        "time": now_utc.strftime("%H:%M:%S"),
        "timestamp_utc": now_utc.isoformat(),
        "total": total,
        "fresh": fresh,
        "stale": stale,
        "missing": missing,
        "warnings": warnings,
        "details": []
    }

    # Add per-file details
    for r in results:
        status_data["details"].append({
            "name": r["name"],
            "status": r["status"],
            "last_date": r["last_date"].isoformat() if r["last_date"] else None,
            "days_ago": r["days_ago"],
            "stale_threshold_days": r["stale_days"],
            "nan_last_5": r["nan_count"],
            "error": r["error"],
        })

    json_path = os.path.join(LOG_DIR, "last_update_status.json")
    with open(json_path, "w") as f:
        json.dump(status_data, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main entry point."""
    ensure_log_dir()
    now_utc = get_utc_now()

    # Check all files
    results = []
    for file_info in FILES_TO_CHECK:
        result = check_file(file_info, now_utc)
        results.append(result)

    # Print to console
    print_table(results, now_utc)

    # Save outputs
    save_text_report(results, now_utc)
    save_json_status(results, now_utc)


if __name__ == "__main__":
    # Always exit with 0 to avoid breaking cron jobs
    try:
        main()
    except Exception as e:
        # Log error to stderr but exit gracefully
        import sys
        print(f"CRITICAL ERROR in data_quality_report: {e}", file=sys.stderr)
        sys.exit(0)