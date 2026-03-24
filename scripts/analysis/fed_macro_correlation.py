"""
Fed Macro Correlation Analysis with BTC
========================================
Run from project root:
  conda activate crypto_market_state
  python scripts/analysis/fed_macro_correlation.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
from fredapi import Fred
from scipy import stats
import yfinance as yf

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("[WARN] tabulate not installed — using plain print. pip install tabulate")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BTC_PATH     = PROJECT_ROOT / "data/03_primary/spot/daily/BTCUSDT.parquet"

# Timezone-safe date boundaries — UTC from line 1
START = pd.Timestamp("2023-01-01", tz="UTC")
TODAY = pd.Timestamp.now(tz="UTC").normalize()

FOMC_DATES = pd.to_datetime([
    "2023-02-01", "2023-03-22", "2023-05-03",
    "2023-06-14", "2023-07-26", "2023-09-20",
    "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01",
    "2024-06-12", "2024-07-31", "2024-09-18",
    "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07",
    "2025-06-18", "2025-07-30", "2025-09-17",
    "2025-10-29", "2025-12-10",
    "2026-01-29", "2026-03-18",
], utc=True)

HAWKISH_THRESHOLD = 0.05  # percentage points on DGS2

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_table(rows, headers, title=None, fmt="simple"):
    if title:
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt=fmt, floatfmt="+.4f"))
    else:
        print("  " + "  ".join(str(h) for h in headers))
        for row in rows:
            print("  " + "  ".join(str(v) for v in row))
    print()


def _classify_surprise(bps):
    if pd.isna(bps):
        return "unknown"
    if bps > HAWKISH_THRESHOLD:
        return "hawkish"
    if bps < -HAWKISH_THRESHOLD:
        return "dovish"
    return "neutral"


def _zscore(series, window):
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / s.replace(0, np.nan)


def _corr_period(df, sig_col, target_col, start_str, end_str):
    mask = (df.index >= pd.Timestamp(start_str, tz="UTC")) & \
           (df.index <= pd.Timestamp(end_str,   tz="UTC"))
    sub  = df.loc[mask, [sig_col, target_col]].dropna()
    if len(sub) < 10:
        return np.nan
    return sub[sig_col].corr(sub[target_col])


def _pearsonr_safe(x, y):
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 10:
        return np.nan, np.nan
    r, p = stats.pearsonr(xy.iloc[:, 0], xy.iloc[:, 1])
    return r, p


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")

    # ── Credentials ──────────────────────────────────────────────────────────
    secrets_path = PROJECT_ROOT / "conf/local/secrets.yml"
    creds_path   = PROJECT_ROOT / "conf/local/credentials.yml"
    fred_key = None

    for path in [secrets_path, creds_path]:
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f)
            # Try api_keys.fred first, then fred.api_key
            try:
                fred_key = raw["api_keys"]["fred"]
            except (KeyError, TypeError):
                pass
            if fred_key is None:
                try:
                    fred_key = raw["fred"]["api_key"]
                except (KeyError, TypeError):
                    pass
            if fred_key:
                break

    if not fred_key:
        sys.exit("[ERROR] FRED API key not found in conf/local/secrets.yml or credentials.yml")

    # ── BTC ──────────────────────────────────────────────────────────────────
    if not BTC_PATH.exists():
        sys.exit(f"[ERROR] BTC parquet not found: {BTC_PATH}")

    btc_raw = pd.read_parquet(BTC_PATH)
    btc_raw.index = pd.to_datetime(btc_raw.index, utc=True)
    btc = btc_raw[["close", "log_return"]].copy()
    print(f"  BTC    : {btc.index[0].date()} → {btc.index[-1].date()}  ({len(btc)} rows)")

    # ── FRED ─────────────────────────────────────────────────────────────────
    fred_client = Fred(api_key=fred_key)

    def fetch_fred(sid):
        s = fred_client.get_series(sid, observation_start=START.strftime("%Y-%m-%d"))
        # FRED returns tz-naive index — localize explicitly
        s.index = pd.to_datetime(s.index).tz_localize("UTC")
        s.name  = sid
        return s

    dgs2     = fetch_fred("DGS2")
    dgs10    = fetch_fred("DGS10")
    dgs3mo   = fetch_fred("DGS3MO")
    fedfunds = fetch_fred("FEDFUNDS")
    cpi      = fetch_fred("CPIAUCSL")
    print(f"  FRED   : {dgs2.index[0].date()} → {dgs2.index[-1].date()}")

    # ── Yahoo Finance ─────────────────────────────────────────────────────────
    def fetch_yahoo(ticker, col="Close"):
        raw = yf.download(ticker, start=START.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False)
        s   = raw[col].squeeze()
        # Yahoo returns tz-naive index — localize explicitly
        s.index = pd.to_datetime(s.index).tz_localize("UTC")
        s.name  = ticker
        return s

    vix   = fetch_yahoo("^VIX")
    dxy   = fetch_yahoo("DX-Y.NYB")
    sp500 = fetch_yahoo("^GSPC")
    print(f"  Yahoo  : {vix.index[0].date()} → {vix.index[-1].date()}")

    # ── Align on business-day calendar (UTC) ─────────────────────────────────
    # Build naive range first, then localize — avoids bdate_range tz conflict
    start_naive = START.tz_localize(None)
    today_naive = TODAY.tz_localize(None)
    bday_idx = pd.bdate_range(start=start_naive, end=today_naive, freq="B").tz_localize("UTC")

    df = btc.reindex(bday_idx)
    for s in [dgs2, dgs10, dgs3mo, fedfunds, cpi, vix, dxy, sp500]:
        df = df.join(s.reindex(bday_idx), how="left")

    # Forward fill macro gaps (weekends / holidays / FRED release lag)
    macro_cols = ["DGS2", "DGS10", "DGS3MO", "FEDFUNDS", "CPIAUCSL",
                  "^VIX", "DX-Y.NYB", "^GSPC"]
    df[macro_cols] = df[macro_cols].ffill()

    # Rebuild log_return on aligned grid where missing
    df["log_return"] = df["log_return"].fillna(
        np.log(df["close"] / df["close"].shift(1))
    )

    print(f"  Aligned: {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} rows)")

    # Summary
    summary = [(c, df[c].notna().sum(), f"{df[c].notna().mean():.1%}") for c in df.columns]
    _print_table(summary, ["Column", "Non-NaN", "Coverage"], title="Data Summary")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Build signals
# ──────────────────────────────────────────────────────────────────────────────

def build_signals(df):
    print("Computing signals...")

    df = df.copy()

    df["rate_expectation"]      = df["DGS3MO"] - df["FEDFUNDS"]
    df["rate_expectation_ch30"] = df["rate_expectation"].diff(30)
    df["yield_curve_2y10y"]     = df["DGS10"] - df["DGS2"]
    df["dgs2_change_1d"]        = df["DGS2"].diff(1)
    df["vix_zscore_30d"]        = _zscore(df["^VIX"], 30)
    df["dxy_return_5d"]         = df["DX-Y.NYB"].pct_change(5)
    df["sp500_return_5d"]       = df["^GSPC"].pct_change(5)

    # CPI YoY vs 2% target (CPIAUCSL is monthly; ffill already applied)
    df["cpi_yoy"]       = df["CPIAUCSL"].pct_change(252) * 100   # ~252 bday ≈ 12m
    df["cpi_vs_target"] = df["cpi_yoy"] - 2.0

    # Forward BTC returns (intentional lookahead — analysis only)
    for h in [1, 3, 5]:
        df[f"btc_ret_{h}d"] = df["close"].pct_change(h).shift(-h)

    print(f"  Signals built. Columns: {list(df.columns)}\n")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FOMC event study
# ──────────────────────────────────────────────────────────────────────────────

def fomc_event_study(df):
    print("Running FOMC event study...")

    events = []
    for fd in FOMC_DATES:
        candidates = df.index[df.index >= fd]
        if len(candidates) == 0:
            continue
        obs = candidates[0]
        row = df.loc[obs]
        events.append({
            "fomc_date"    : fd.date(),
            "obs_date"     : obs.date(),
            "surprise_bps" : row["dgs2_change_1d"],
            "type"         : _classify_surprise(row["dgs2_change_1d"]),
            "rate_exp"     : row["rate_expectation"],
            "yield_curve"  : row["yield_curve_2y10y"],
            "vix_zscore"   : row["vix_zscore_30d"],
            "btc_ret_1d"   : row["btc_ret_1d"],
            "btc_ret_3d"   : row["btc_ret_3d"],
            "btc_ret_5d"   : row["btc_ret_5d"],
        })

    ev = pd.DataFrame(events)

    # ── Full table ────────────────────────────────────────────────────────────
    rows = []
    for _, r in ev.iterrows():
        rows.append([
            r["fomc_date"],
            f"{r['surprise_bps']:+.3f}" if not pd.isna(r["surprise_bps"]) else "N/A",
            r["type"].upper(),
            f"{r['btc_ret_1d']*100:+.2f}%" if not pd.isna(r["btc_ret_1d"]) else "N/A",
            f"{r['btc_ret_3d']*100:+.2f}%" if not pd.isna(r["btc_ret_3d"]) else "N/A",
            f"{r['btc_ret_5d']*100:+.2f}%" if not pd.isna(r["btc_ret_5d"]) else "N/A",
        ])
    _print_table(rows,
                 ["Date", "Surprise(pp)", "Type", "BTC_1d", "BTC_3d", "BTC_5d"],
                 title="FOMC Events — Fed Surprise vs BTC Reaction")

    # ── Averages by type ──────────────────────────────────────────────────────
    avg_rows = []
    for stype in ["hawkish", "neutral", "dovish"]:
        sub = ev[ev["type"] == stype]
        if len(sub) == 0:
            continue
        for horizon, col in [("1d", "btc_ret_1d"), ("3d", "btc_ret_3d"), ("5d", "btc_ret_5d")]:
            pass  # computed below

        m1   = sub["btc_ret_1d"].mean()
        m3   = sub["btc_ret_3d"].mean()
        m5   = sub["btc_ret_5d"].mean()
        # hit rate: hawkish/neutral → BTC fell; dovish → BTC rose
        s5   = sub["btc_ret_5d"].dropna()
        if stype == "dovish":
            hr = (s5 > 0).mean() if len(s5) > 0 else np.nan
        else:
            hr = (s5 < 0).mean() if len(s5) > 0 else np.nan

        avg_rows.append([
            stype.upper(),
            len(sub),
            f"{m1*100:+.2f}%" if not pd.isna(m1) else "N/A",
            f"{m3*100:+.2f}%" if not pd.isna(m3) else "N/A",
            f"{m5*100:+.2f}%" if not pd.isna(m5) else "N/A",
            f"{hr:.0%}"       if not pd.isna(hr)  else "N/A",
        ])

    _print_table(avg_rows,
                 ["Type", "N", "Avg_1d", "Avg_3d", "Avg_5d", "Hit_rate_5d"],
                 title="Average BTC Return by Surprise Type")

    # ── Today highlight ───────────────────────────────────────────────────────
    today_fd = pd.Timestamp("2026-03-18", tz="UTC")
    today_row = ev[ev["fomc_date"] == today_fd.date()]
    hawk_sub  = ev[ev["type"] == "hawkish"]["btc_ret_5d"].dropna()
    hawk_avg  = hawk_sub.mean() * 100 if len(hawk_sub) > 0 else np.nan

    print("─" * 70)
    print("  TODAY — FOMC 2026-03-18")
    print("─" * 70)
    if len(today_row) == 0:
        print("  No data found for 2026-03-18.")
    else:
        r = today_row.iloc[0]
        print(f"  Surprise (DGS2 change) : {r['surprise_bps']:+.3f} pp  → {r['type'].upper()}")
        print(f"  Rate expectation       : {r['rate_exp']:+.3f}  (>0 = market expects cut)")
        print(f"  Yield curve 2y-10y     : {r['yield_curve']:+.3f}")
        print(f"  VIX z-score (30d)      : {r['vix_zscore']:+.3f}" if not pd.isna(r["vix_zscore"]) else "  VIX z-score (30d)      : N/A")
        print(f"  BTC ret 1d             : {r['btc_ret_1d']*100:+.2f}%" if not pd.isna(r["btc_ret_1d"]) else "  BTC ret 1d             : N/A (future)")
        print(f"  BTC ret 3d             : {r['btc_ret_3d']*100:+.2f}%" if not pd.isna(r["btc_ret_3d"]) else "  BTC ret 3d             : N/A (future)")
        print(f"  BTC ret 5d             : {r['btc_ret_5d']*100:+.2f}%" if not pd.isna(r["btc_ret_5d"]) else "  BTC ret 5d             : N/A (future)")
        print(f"  Historical hawkish avg : {hawk_avg:+.2f}%")
    print()

    return ev


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Correlation analysis
# ──────────────────────────────────────────────────────────────────────────────

def correlation_analysis(df):
    print("Running correlation analysis...")

    target  = "btc_ret_5d"
    signals = [
        "rate_expectation",
        "rate_expectation_ch30",
        "yield_curve_2y10y",
        "dgs2_change_1d",
        "vix_zscore_30d",
        "dxy_return_5d",
        "sp500_return_5d",
        "cpi_vs_target",
    ]

    periods = {
        "2023"  : ("2023-01-01", "2023-12-31"),
        "2024"  : ("2024-01-01", "2024-12-31"),
        "2025+" : ("2025-01-01", "2026-12-31"),
    }

    results = []
    rows    = []

    for sig in signals:
        if sig not in df.columns:
            continue

        corr_full, p_val = _pearsonr_safe(df[sig], df[target])
        corr_by_period   = {
            k: _corr_period(df, sig, target, s, e)
            for k, (s, e) in periods.items()
        }
        significant = p_val < 0.05 if not pd.isna(p_val) else False

        results.append({
            "signal"     : sig,
            "corr_full"  : corr_full,
            "corr_2023"  : corr_by_period["2023"],
            "corr_2024"  : corr_by_period["2024"],
            "corr_2025+" : corr_by_period["2025+"],
            "p_value"    : p_val,
            "significant": significant,
        })

        def _fmt(v):
            return f"{v:+.4f}" if not pd.isna(v) else "N/A"

        rows.append([
            sig,
            _fmt(corr_full),
            _fmt(corr_by_period["2023"]),
            _fmt(corr_by_period["2024"]),
            _fmt(corr_by_period["2025+"]),
            f"{p_val:.4f}" if not pd.isna(p_val) else "N/A",
            "✓" if significant else "✗",
        ])

    _print_table(rows,
                 ["Signal", "Corr_full", "2023", "2024", "2025+", "p-value", "Sig?"],
                 title="Pearson Correlation with BTC 5d Forward Return")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Lead/lag analysis
# ──────────────────────────────────────────────────────────────────────────────

def lead_lag_analysis(df):
    print("Running lead/lag analysis...")

    target  = "btc_ret_1d"
    signals = [
        "rate_expectation",
        "rate_expectation_ch30",
        "yield_curve_2y10y",
        "dgs2_change_1d",
        "vix_zscore_30d",
        "dxy_return_5d",
        "sp500_return_5d",
        "cpi_vs_target",
    ]

    lag_range = range(-10, 11)
    results   = []
    rows      = []

    for sig in signals:
        if sig not in df.columns:
            continue

        best_lag, best_corr = 0, 0.0
        for lag in lag_range:
            # positive lag: signal leads BTC by `lag` days
            r, _ = _pearsonr_safe(df[sig], df[target].shift(-lag))
            if not pd.isna(r) and abs(r) > abs(best_corr):
                best_corr = r
                best_lag  = lag

        if best_lag > 0:
            interp = f"Signal leads BTC by {best_lag}d"
        elif best_lag < 0:
            interp = f"BTC leads signal by {abs(best_lag)}d"
        else:
            interp = "Contemporaneous"

        results.append({
            "signal"   : sig,
            "best_lag" : best_lag,
            "best_corr": best_corr,
        })
        rows.append([sig, best_lag, f"{best_corr:+.4f}", interp])

    _print_table(rows,
                 ["Signal", "Best_lag(days)", "Best_corr", "Interpretation"],
                 title="Lead/Lag Analysis (positive lag = signal leads BTC)")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Feature recommendation
# ──────────────────────────────────────────────────────────────────────────────

def feature_recommendation(corr_df, lag_df):
    print("Building feature recommendations...")

    merged = corr_df.merge(lag_df, on="signal", how="left")

    recommended = []
    not_recommended = []

    for _, row in merged.iterrows():
        sig      = row["signal"]
        c_full   = row["corr_full"]
        c_2023   = row["corr_2023"]
        c_2024   = row["corr_2024"]
        c_2025   = row["corr_2025+"]
        p_val    = row["p_value"]
        best_lag = row["best_lag"]
        best_corr= row["best_corr"]

        # Stability: consistent sign across periods AND |corr| > 0.05 in ≥2
        period_corrs = [v for v in [c_2023, c_2024, c_2025] if not pd.isna(v)]
        signs        = [np.sign(c) for c in period_corrs]
        sign_stable  = len(set(signs)) == 1 if signs else False
        strong_count = sum(abs(c) > 0.05 for c in period_corrs)

        stable        = sign_stable and strong_count >= 2
        sig_threshold = (not pd.isna(p_val)) and p_val < 0.10

        if stable and sig_threshold:
            recommended.append([
                sig,
                f"{c_full:+.4f}" if not pd.isna(c_full) else "N/A",
                best_lag,
                "Yes" if sign_stable else "No",
            ])
        else:
            reasons = []
            if not sign_stable:
                reasons.append("unstable sign across periods")
            if strong_count < 2:
                reasons.append(f"|corr|>0.05 in only {strong_count}/3 periods")
            if not sig_threshold:
                reasons.append(f"p={p_val:.3f} ≥ 0.10" if not pd.isna(p_val) else "p-value N/A")
            not_recommended.append([sig, "; ".join(reasons)])

    _print_table(recommended,
                 ["Feature", "Corr_full", "Best_lag(d)", "Sign_stable"],
                 title="RECOMMENDED for HMM v2 / Model 4h")

    _print_table(not_recommended,
                 ["Feature", "Reason"],
                 title="NOT RECOMMENDED")

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────────────────────

def save_results(merged_df):
    out_path = PROJECT_ROOT / "scripts/analysis/output/fed_macro_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_path, index=False)
    print(f"Results saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Fed Macro Correlation Analysis with BTC")
    print(f"  Period : {START.date()} → {TODAY.date()}")
    print("=" * 70)
    print()

    df      = load_data()
    df      = build_signals(df)
    _       = fomc_event_study(df)
    corr_df = correlation_analysis(df)
    lag_df  = lead_lag_analysis(df)
    merged  = feature_recommendation(corr_df, lag_df)
    save_results(merged)

    print("=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
