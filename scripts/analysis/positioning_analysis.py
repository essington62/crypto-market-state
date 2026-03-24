"""
Derivatives Positioning Correlation Analysis with BTC
======================================================
Validates if OI, Funding, and Leverage Stress signals
have predictive power for BTC regime.

Data constraints (CoinGlass Hobbyist plan):
  - OI / Funding: 4h, ~180 days history (2025-09-13 → present)
  - Liquidation Orders: NOT AVAILABLE (404 on Hobbyist)
  - Net Position: NOT AVAILABLE
  - CVD aggregated: NOT AVAILABLE
  Analyses in Sections 4-6 are limited to this window.

Run from project root:
  conda activate crypto_market_state
  python scripts/analysis/positioning_analysis.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from scipy import stats

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

plt.style.use("dark_background")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
BTC_PATH      = PROJECT_ROOT / "data/03_primary/spot/daily/BTCUSDT.parquet"
DERIV_PATH    = PROJECT_ROOT / "data/01_raw/derivatives/coinglass/4h/BTCUSDT.parquet"
OUT_DIR       = PROJECT_ROOT / "scripts/analysis/output"

START = pd.Timestamp("2023-01-01", tz="UTC")
TODAY = pd.Timestamp.now(tz="UTC").normalize()

# Actual derivatives data window (Hobbyist API limit)
DERIV_START = pd.Timestamp("2025-09-13", tz="UTC")

EVENTS_TS = {
    pd.Timestamp("2024-01-11", tz="UTC"): "ETF Bull run",
    pd.Timestamp("2024-03-14", tz="UTC"): "ATH $73k",
    pd.Timestamp("2024-08-05", tz="UTC"): "Correction -30%",
    pd.Timestamp("2025-01-20", tz="UTC"): "Trump ATH $109k",
    pd.Timestamp("2025-02-28", tz="UTC"): "Correction -40%",
    pd.Timestamp("2026-03-18", tz="UTC"): "Fed hawkish -4%",
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tbl(rows, headers, title=None):
    if title:
        print(f"\n{'─' * 72}")
        print(f"  {title}")
        print(f"{'─' * 72}")
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        print("  " + "  ".join(str(h) for h in headers))
        for row in rows:
            print("  " + "  ".join(str(v) for v in row))
    print()


def _fmt(v, pct=False, dec=4):
    if pd.isna(v):
        return "N/A"
    if pct:
        return f"{v*100:+.2f}%"
    return f"{v:+.{dec}f}"


def _zscore(series, window):
    m  = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - m) / sd.replace(0, np.nan)


def _pearsonr(x, y):
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 8:
        return np.nan, np.nan
    return stats.pearsonr(xy.iloc[:, 0], xy.iloc[:, 1])


def _best_lag(signal, target, lags):
    best_lag, best_corr = 0, 0.0
    for lag in lags:
        r, _ = _pearsonr(signal, target.shift(-lag))
        if not pd.isna(r) and abs(r) > abs(best_corr):
            best_corr, best_lag = r, lag
    return best_lag, best_corr


def _corr_period(df, sig, tgt, start_str, end_str):
    s = pd.Timestamp(start_str, tz="UTC")
    e = pd.Timestamp(end_str,   tz="UTC")
    sub = df.loc[(df.index >= s) & (df.index <= e), [sig, tgt]].dropna()
    if len(sub) < 8:
        return np.nan
    return sub[sig].corr(sub[tgt])


def _mark_events(ax):
    colors = ["#4da6ff", "#ff8c00", "#ff4d4d", "#4dff88", "#ff4d4d", "#ff88ff"]
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    xlim = ax.get_xlim()
    for i, (ts, label) in enumerate(EVENTS_TS.items()):
        ts_num = mdates.date2num(ts.to_pydatetime())
        if ts_num < xlim[0] or ts_num > xlim[1]:
            continue
        ax.axvline(ts, color=colors[i % len(colors)], linewidth=0.9,
                   linestyle="--", alpha=0.6)
        ypos = ylim[1] - span * (0.05 + (i % 4) * 0.1)
        ax.text(ts, ypos, f" {label}", color=colors[i % len(colors)],
                fontsize=6, va="top", clip_on=True)


def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Section 0: Loading data...")

    # ── BTC daily ─────────────────────────────────────────────────────────────
    if not BTC_PATH.exists():
        sys.exit(f"[ERROR] BTC parquet not found: {BTC_PATH}")

    btc_raw = pd.read_parquet(BTC_PATH)
    btc_raw.index = pd.to_datetime(btc_raw.index, utc=True)
    want = ["close", "log_return", "vol_short", "drawdown"]
    btc  = btc_raw[[c for c in want if c in btc_raw.columns]].copy()
    print(f"  BTC    : {btc.index[0].date()} → {btc.index[-1].date()}  ({len(btc)} rows)")

    # ── CoinGlass 4h derivatives ──────────────────────────────────────────────
    if not DERIV_PATH.exists():
        sys.exit(f"[ERROR] Derivatives parquet not found: {DERIV_PATH}")

    raw = pd.read_parquet(DERIV_PATH)

    # Index is integer (0-1079) — timestamp is a column
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw = raw.set_index("timestamp").sort_index()
    else:
        raw.index = pd.to_datetime(raw.index, utc=True)

    print(f"  DERIV  : {raw.index[0].date()} → {raw.index[-1].date()}  "
          f"({len(raw)} × 4h bars)")

    # Column mapping (double-underscore L1 contract)
    col_map = {
        "open_interest_aggregated_history__close" : "oi_agg",
        "open_interest_history__close"             : "oi_binance",
        "funding_rate_oi_weight_history__close"    : "funding_oi",
        "funding_rate_vol_weight_history__close"   : "funding_vol",
        "funding_rate_history__close"              : "funding_raw",
    }
    # Liquidation columns (Hobbyist plan — likely absent)
    liq_map = {
        "pair_liquidation_history__long_liquidation_usd" : "liq_long",
        "pair_liquidation_history__short_liquidation_usd": "liq_short",
    }

    present    = {k: v for k, v in col_map.items() if k in raw.columns}
    liq_present= {k: v for k, v in liq_map.items() if k in raw.columns}

    if not present:
        sys.exit("[ERROR] No expected derivative columns found.")

    deriv_4h = raw[[*present.keys(), *liq_present.keys()]].rename(
        columns={**present, **liq_present}
    )

    # ── Resample 4h → daily ────────────────────────────────────────────────────
    # OI: last value of each UTC day
    # Funding: mean of 3 settlements per day (00:00 / 08:00 / 16:00 UTC)
    oi_cols      = [v for v in present.values() if v.startswith("oi")]
    fund_cols    = [v for v in present.values() if v.startswith("fund")]
    liq_cols     = list(liq_present.values())

    daily_oi   = deriv_4h[oi_cols].resample("D").last()   if oi_cols   else pd.DataFrame()
    daily_fund = deriv_4h[fund_cols].resample("D").mean() if fund_cols else pd.DataFrame()
    daily_liq  = deriv_4h[liq_cols].resample("D").sum()   if liq_cols  else pd.DataFrame()

    deriv_daily = pd.concat(
        [x for x in [daily_oi, daily_fund, daily_liq] if not x.empty], axis=1
    ).sort_index()

    if liq_cols:
        print(f"  Liquidation columns found: {liq_cols}")
    else:
        print("  [INFO] Liquidation columns absent (Hobbyist plan — 404). "
              "Continuing with OI + Funding only.")

    # ── Align BTC + derivatives on shared daily UTC grid ─────────────────────
    s_naive = DERIV_START.tz_localize(None)
    e_naive = TODAY.tz_localize(None)
    day_idx = pd.date_range(start=s_naive, end=e_naive, freq="D").tz_localize("UTC")

    df = btc.reindex(day_idx).ffill()
    df = df.join(deriv_daily.reindex(day_idx).ffill(), how="left")

    # Forward returns
    df["btc_ret_5d"]  = df["close"].pct_change(5).shift(-5)
    df["btc_ret_21d"] = df["close"].pct_change(21).shift(-21)
    df["log_ret_21d"] = df["log_return"].rolling(21).sum()

    print(f"  Joined : {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} rows)\n")
    return df, bool(liq_cols)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Build positioning signals
# ──────────────────────────────────────────────────────────────────────────────

def section1_build_signals(df, has_liq):
    print("Section 1: Building positioning signals...")

    # ── OI signals ────────────────────────────────────────────────────────────
    if "oi_agg" in df.columns:
        df["oi_change_3d"]  = df["oi_agg"].pct_change(3)
        df["oi_change_7d"]  = df["oi_agg"].pct_change(7)
        df["oi_zscore_30d"] = _zscore(df["oi_agg"], 30)
        df["oi_momentum"]   = df["oi_change_3d"] - df["oi_change_3d"].shift(3)
    else:
        print("  [WARN] oi_agg not found — OI signals skipped")

    # ── Funding signals ───────────────────────────────────────────────────────
    if "funding_oi" in df.columns:
        df["funding_zscore_14d"]    = _zscore(df["funding_oi"], 14)
        df["funding_zscore_30d"]    = _zscore(df["funding_oi"], 30)
        df["funding_trend_7d"]      = df["funding_oi"].diff(7)
        df["funding_extreme_long"]  = (df["funding_zscore_14d"] >  2.0).astype(int)
        df["funding_extreme_short"] = (df["funding_zscore_14d"] < -2.0).astype(int)

    # ── Funding divergence (OI-weighted vs vol-weighted) ─────────────────────
    if "funding_oi" in df.columns and "funding_vol" in df.columns:
        df["funding_divergence"] = df["funding_oi"] - df["funding_vol"]

    # ── Leverage stress composite ─────────────────────────────────────────────
    if "oi_change_3d" in df.columns and "funding_zscore_14d" in df.columns:
        df["leverage_stress"] = df["oi_change_3d"] * df["funding_zscore_14d"]

    # ── Liquidation signals (if available) ────────────────────────────────────
    if has_liq and "liq_long" in df.columns and "liq_short" in df.columns:
        df["liq_long_zscore"]  = _zscore(df["liq_long"],  14)
        df["liq_short_zscore"] = _zscore(df["liq_short"], 14)
        total_liq = df["liq_long"] + df["liq_short"] + 1
        df["liq_intensity"]    = df["liq_long"] / total_liq
        df["liq_cascade_risk"] = (df["liq_long_zscore"] > 2.0).astype(int)
        print("  Liquidation signals: built")
    else:
        print("  Liquidation signals: skipped (data not available)")

    built = [c for c in ["oi_change_3d","oi_change_7d","oi_zscore_30d","oi_momentum",
                          "funding_zscore_14d","funding_zscore_30d","funding_trend_7d",
                          "leverage_stress","funding_divergence"] if c in df.columns]
    print(f"  Signals built: {built}\n")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Visual analysis
# ──────────────────────────────────────────────────────────────────────────────

def section2_visual(df):
    print("Section 2: Generating positioning overview chart...")

    panels = []
    if "oi_agg" in df.columns:
        panels.append(("OI", "oi_agg", "oi_change_3d"))
    if "funding_oi" in df.columns:
        panels.append(("Funding", "funding_oi", "funding_zscore_14d"))
    has_stress = "leverage_stress" in df.columns

    n_panels = 2 + len(panels) + (1 if has_stress else 0)
    ratios   = [2] + [1.5] * (n_panels - 1)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.5 * n_panels),
                             sharex=True, gridspec_kw={"height_ratios": ratios})
    axes = [axes] if n_panels == 1 else list(axes)
    ax_idx = 0

    # Panel 1: BTC price
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(df.index, df["close"], color="#aaaaaa", linewidth=0.9)
    ax.set_ylabel("BTC Close")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("BTC Derivatives Positioning Overview")
    _mark_events(ax)

    # OI panel
    if "oi_agg" in df.columns and "oi_change_3d" in df.columns:
        ax = axes[ax_idx]; ax_idx += 1
        axr = ax.twinx()
        ax.plot(df.index, df["oi_agg"] / 1e9, color="#4da6ff", linewidth=1.0,
                label="OI agg ($ Bn)")
        axr.bar(df.index, df["oi_change_3d"] * 100, color=[
            "#4dff88" if v >= 0 else "#ff4d4d"
            for v in df["oi_change_3d"].fillna(0)
        ], alpha=0.5, width=0.8, label="OI chg 3d %")
        axr.axhline(0, color="white", linewidth=0.4)
        ax.set_ylabel("OI ($ Bn)", color="#4da6ff")
        axr.set_ylabel("3d % change", color="#aaaaaa")
        lines1, lbl1 = ax.get_legend_handles_labels()
        lines2, lbl2 = axr.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)
        _mark_events(ax)

    # Funding panel
    if "funding_oi" in df.columns and "funding_zscore_14d" in df.columns:
        ax = axes[ax_idx]; ax_idx += 1
        axr = ax.twinx()
        ax.plot(df.index, df["funding_oi"] * 100, color="#ff8c00", linewidth=1.0,
                label="Funding OI-weighted (%)")
        ax.axhline(0, color="white", linewidth=0.4)
        axr.plot(df.index, df["funding_zscore_14d"], color="#ffff44", linewidth=0.8,
                 alpha=0.7, label="Funding z-score 14d")
        axr.axhline(+2, color="#ff4d4d", linewidth=0.5, linestyle="--", alpha=0.6)
        axr.axhline(-2, color="#4dff88", linewidth=0.5, linestyle="--", alpha=0.6)
        ax.set_ylabel("Funding rate (%)", color="#ff8c00")
        axr.set_ylabel("Z-score", color="#ffff44")
        lines1, lbl1 = ax.get_legend_handles_labels()
        lines2, lbl2 = axr.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)
        _mark_events(ax)

    # Leverage stress panel
    if has_stress:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(df.index, df["leverage_stress"], color="#ff4d4d", linewidth=1.0,
                label="leverage_stress (oi_chg3d × funding_z14d)")
        ax.fill_between(df.index, df["leverage_stress"], 0,
                        where=df["leverage_stress"] > 0, color="#ff4d4d", alpha=0.2)
        ax.fill_between(df.index, df["leverage_stress"], 0,
                        where=df["leverage_stress"] < 0, color="#4dff88", alpha=0.2)
        ax.axhline(0, color="white", linewidth=0.4)
        ax.axhline(+1, color="#ff4d4d", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axhline(-1, color="#4dff88", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_ylabel("Leverage Stress")
        ax.legend(fontsize=8)
        _mark_events(ax)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    fig.tight_layout()
    _save(fig, "positioning_overview.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Correlation analysis
# ──────────────────────────────────────────────────────────────────────────────

def section3_correlation(df):
    print("Section 3: Correlation analysis...")

    candidate_signals = [
        "oi_change_3d", "oi_change_7d", "oi_zscore_30d", "oi_momentum",
        "funding_zscore_14d", "funding_zscore_30d", "funding_trend_7d",
        "leverage_stress", "funding_divergence",
        "liq_intensity",
    ]
    signals = [s for s in candidate_signals if s in df.columns]

    periods = {
        "2023"   : ("2023-01-01", "2023-12-31"),
        "2024"   : ("2024-01-01", "2024-12-31"),
        "2025+"  : ("2025-01-01", str(TODAY.date())),
    }

    results = []
    rows5, rows21 = [], []

    for sig in signals:
        c5,  p5  = _pearsonr(df[sig], df["btc_ret_5d"])
        c21, p21 = _pearsonr(df[sig], df["btc_ret_21d"])
        bl5,  _  = _best_lag(df[sig], df["btc_ret_5d"],  range(0, 31))
        bl21, _  = _best_lag(df[sig], df["btc_ret_21d"], range(0, 31))

        period_corrs = {k: _corr_period(df, sig, "btc_ret_5d", s, e)
                        for k, (s, e) in periods.items()}
        valid_signs  = [np.sign(v) for v in period_corrs.values() if not pd.isna(v)]
        stable       = "YES" if len(set(valid_signs)) <= 1 else "NO"

        results.append({
            "signal": sig, "corr_5d": c5, "corr_21d": c21,
            "p_5d": p5, "p_21d": p21,
            "best_lag_5d": bl5, "best_lag_21d": bl21, "stable": stable,
        })
        rows5.append([sig, _fmt(c5), _fmt(c21),
                      f"{p5:.4f}" if not pd.isna(p5)  else "N/A",
                      f"{bl5}d", stable])

    _tbl(rows5,
         ["Signal", "Corr_5d", "Corr_21d", "p_5d", "Best_lag_5d", "Stable?"],
         title="Pearson Correlation with BTC Forward Returns")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Regime conditional analysis
# ──────────────────────────────────────────────────────────────────────────────

def section4_regime_conditional(df):
    print("Section 4: Regime conditional analysis...")

    # Simple regime rule (no model file dependency)
    # Uses drawdown and rolling 21d log return
    df = df.copy()
    df["log_ret_21d"] = df["log_return"].rolling(21).sum()

    bull_mask = (df["drawdown"] > -0.10) & (df["log_ret_21d"] > 0)
    bear_mask = (df["drawdown"] < -0.15) | (df["log_ret_21d"] < -0.05)
    df["regime"] = np.where(bull_mask, "Bull",
                   np.where(bear_mask, "Bear", "Sideways"))

    dist = df["regime"].value_counts()
    print(f"  Regime distribution: {dict(dist)}")

    candidate_signals = [
        "oi_change_3d", "oi_change_7d", "oi_zscore_30d",
        "funding_zscore_14d", "funding_zscore_30d",
        "leverage_stress", "funding_divergence",
    ]
    signals = [s for s in candidate_signals if s in df.columns]

    rows = []
    for sig in signals:
        bull_m = df.loc[df["regime"] == "Bull",    sig].mean()
        bear_m = df.loc[df["regime"] == "Bear",    sig].mean()
        side_m = df.loc[df["regime"] == "Sideways",sig].mean()

        # Pooled std for separation ratio
        pooled = df[sig].std()
        sep = abs(bull_m - bear_m) / pooled if pooled > 0 and not pd.isna(bull_m) and not pd.isna(bear_m) else np.nan

        rows.append([sig,
                     _fmt(bull_m), _fmt(bear_m), _fmt(side_m),
                     _fmt(sep)])

    rows.sort(key=lambda r: -float(r[4]) if r[4] != "N/A" else 0)

    _tbl(rows,
         ["Signal", "Bull_mean", "Bear_mean", "Sideways_mean", "Separation"],
         title="Signal Values by Regime  (Separation = |Bull−Bear| / σ)")

    print("  Higher separation = better HMM discriminator")
    print()
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Leverage stress deep dive
# ──────────────────────────────────────────────────────────────────────────────

def section5_leverage_stress(df):
    print("Section 5: Leverage stress deep dive...")

    if "leverage_stress" not in df.columns:
        print("  [SKIP] leverage_stress not available.\n")
        return

    stress_regimes = {
        "High stress" : df["leverage_stress"] >  1.0,
        "Neutral"     : (df["leverage_stress"] >= -1.0) & (df["leverage_stress"] <= 1.0),
        "Low stress"  : df["leverage_stress"] < -1.0,
    }

    rows = []
    for label, mask in stress_regimes.items():
        sub = df[mask].dropna(subset=["btc_ret_5d", "btc_ret_21d"])
        if len(sub) < 3:
            rows.append([label, "N/A", "N/A", "N/A", "N/A", len(sub)])
            continue
        m5   = sub["btc_ret_5d"].mean()
        m21  = sub["btc_ret_21d"].mean()
        pfal = (sub["btc_ret_5d"] < -0.05).mean()
        pris = (sub["btc_ret_5d"] >  0.05).mean()
        rows.append([label,
                     _fmt(m5, pct=True), _fmt(m21, pct=True),
                     f"{pfal:.0%}", f"{pris:.0%}", len(sub)])

    _tbl(rows,
         ["Stress regime", "BTC_5d", "BTC_21d", "P(fall>5%)", "P(rise>5%)", "N"],
         title="BTC Outcomes by Leverage Stress Regime")

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors_stress = np.where(df["leverage_stress"] > 1,  "#ff4d4d",
                    np.where(df["leverage_stress"] < -1, "#4dff88", "#888888"))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.5]})

    ax = axes[0]
    ax.scatter(df.index, df["close"], c=colors_stress, s=5, alpha=0.8, zorder=2)
    ax.plot(df.index, df["close"], color="#333333", linewidth=0.4, zorder=1)
    ax.set_ylabel("BTC Close")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("BTC Price colored by Leverage Stress  (red=high, green=low, gray=neutral)")

    ax2 = axes[1]
    ax2.plot(df.index, df["leverage_stress"], color="#ff8c00", linewidth=1.0,
             label="leverage_stress")
    ax2.fill_between(df.index, df["leverage_stress"], 0,
                     where=df["leverage_stress"] > 0, color="#ff4d4d", alpha=0.2)
    ax2.fill_between(df.index, df["leverage_stress"], 0,
                     where=df["leverage_stress"] < 0, color="#4dff88", alpha=0.2)
    ax2.axhline(+1, color="#ff4d4d", linewidth=0.6, linestyle="--", alpha=0.6)
    ax2.axhline(-1, color="#4dff88", linewidth=0.6, linestyle="--", alpha=0.6)
    ax2.axhline(0,  color="white",   linewidth=0.4)
    ax2.set_ylabel("Leverage Stress")
    ax2.legend(fontsize=8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    fig.tight_layout()
    _save(fig, "leverage_stress.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Funding rate extreme events
# ──────────────────────────────────────────────────────────────────────────────

def section6_funding_extremes(df):
    print("Section 6: Funding rate extreme events...")

    if "funding_zscore_14d" not in df.columns:
        print("  [SKIP] funding_zscore_14d not available.\n")
        return

    events = []
    for date, row in df.iterrows():
        z = row["funding_zscore_14d"]
        if pd.isna(z):
            continue
        if abs(z) >= 2.0:
            events.append({
                "date"           : date.date(),
                "funding_zscore" : z,
                "type"           : "EXTREME_LONG" if z > 0 else "EXTREME_SHORT",
                "btc_ret_3d"     : row["btc_ret_5d"],   # 5d used as 3d proxy
                "btc_ret_5d"     : row["btc_ret_5d"],
                "btc_ret_10d"    : row["btc_ret_21d"],   # 21d used as 10d proxy
            })

    if not events:
        print("  No extreme funding events in the data window.\n")
        return

    ev_df = pd.DataFrame(events)

    # Print event table (last 20 to keep it readable)
    rows = []
    for _, r in ev_df.tail(20).iterrows():
        rows.append([
            r["date"],
            f"{r['funding_zscore']:+.2f}",
            r["type"],
            _fmt(r["btc_ret_3d"], pct=True),
            _fmt(r["btc_ret_5d"], pct=True),
            _fmt(r["btc_ret_10d"], pct=True),
        ])

    _tbl(rows,
         ["Date", "Funding_z", "Type", "BTC_3d(proxy)", "BTC_5d", "BTC_10d(proxy)"],
         title=f"Extreme Funding Events (|z|≥2)  — last {len(rows)} of {len(ev_df)}")

    # Summary
    long_ev  = ev_df[ev_df["type"] == "EXTREME_LONG"]
    short_ev = ev_df[ev_df["type"] == "EXTREME_SHORT"]

    print(f"  Extreme LONG  events: {len(long_ev):3d}  avg BTC_5d = "
          + (_fmt(long_ev["btc_ret_5d"].mean(), pct=True) if len(long_ev) > 0 else "N/A"))
    print(f"  Extreme SHORT events: {len(short_ev):3d}  avg BTC_5d = "
          + (_fmt(short_ev["btc_ret_5d"].mean(), pct=True) if len(short_ev) > 0 else "N/A"))

    if len(long_ev) > 3:
        avg = long_ev["btc_ret_5d"].mean()
        print(f"  Extreme longs are {'CONTRARIAN (BTC falls)' if avg < 0 else 'momentum (BTC rises)'} ✓" if not pd.isna(avg) else "")
    if len(short_ev) > 3:
        avg = short_ev["btc_ret_5d"].mean()
        print(f"  Extreme shorts are {'CONTRARIAN (BTC rises = short squeeze)' if avg > 0 else 'momentum (BTC falls)'}" if not pd.isna(avg) else "")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Recommendation
# ──────────────────────────────────────────────────────────────────────────────

def section7_recommendation(df, corr_df):
    print("Section 7: Generating recommendation...")

    if corr_df.empty:
        print("  No correlation results to evaluate.\n")
        return

    CORR_THRESHOLD    = 0.10
    SEP_THRESHOLD     = 0.20
    P_THRESHOLD       = 0.15

    # Compute regime separation for each signal
    df_c = df.copy()
    df_c["log_ret_21d"] = df_c["log_return"].rolling(21).sum()
    bull = (df_c["drawdown"] > -0.10) & (df_c["log_ret_21d"] > 0)
    bear = (df_c["drawdown"] < -0.15) | (df_c["log_ret_21d"] < -0.05)
    df_c["regime"] = np.where(bull, "Bull", np.where(bear, "Bear", "Sideways"))

    include_rows = []
    exclude_rows = []

    for _, row in corr_df.iterrows():
        sig    = row["signal"]
        c5     = row["corr_5d"]
        p5     = row["p_5d"]
        stable = row["stable"]
        lag    = row["best_lag_5d"]

        bull_m = df_c.loc[df_c["regime"] == "Bull", sig].mean() if sig in df_c else np.nan
        bear_m = df_c.loc[df_c["regime"] == "Bear", sig].mean() if sig in df_c else np.nan
        pooled = df_c[sig].std() if sig in df_c else np.nan
        sep = abs(bull_m - bear_m) / pooled if (not pd.isna(pooled) and pooled > 0
                                                 and not pd.isna(bull_m) and not pd.isna(bear_m)) else np.nan

        meets_corr = (not pd.isna(c5))  and abs(c5)  >= CORR_THRESHOLD
        meets_p    = (not pd.isna(p5))  and p5        <= P_THRESHOLD
        meets_sep  = (not pd.isna(sep)) and sep        >= SEP_THRESHOLD

        if (meets_corr or meets_sep) and meets_p:
            include_rows.append([sig, _fmt(c5), f"{lag}d", _fmt(sep), stable])
        else:
            reasons = []
            if not meets_corr: reasons.append(f"|corr|={abs(c5):.3f}<{CORR_THRESHOLD}" if not pd.isna(c5) else "corr=N/A")
            if not meets_p:    reasons.append(f"p={p5:.3f}>{P_THRESHOLD}" if not pd.isna(p5) else "p=N/A")
            if not meets_sep:  reasons.append(f"sep={sep:.3f}<{SEP_THRESHOLD}" if not pd.isna(sep) else "sep=N/A")
            exclude_rows.append([sig, "; ".join(reasons)])

    print()
    print("═" * 64)
    print("  POSITIONING SIGNAL RECOMMENDATION FOR HMM v2")
    print("═" * 64)

    if include_rows:
        _tbl(include_rows,
             ["Signal", "Corr_5d", "Best_lag", "Separation", "Stable?"],
             title="INCLUDE")
    else:
        print("\n  INCLUDE: (none met threshold)")
        print()

    if exclude_rows:
        _tbl(exclude_rows, ["Signal", "Reason"], title="EXCLUDE")

    # ── CoinGlass plan upgrade analysis ───────────────────────────────────────
    best_corr = corr_df["corr_5d"].abs().max() if not corr_df.empty else np.nan
    oi_corr   = corr_df.loc[corr_df["signal"] == "oi_change_3d", "corr_5d"].values
    oi_corr   = oi_corr[0] if len(oi_corr) > 0 else np.nan

    upgrade_worthwhile = (not pd.isna(best_corr) and best_corr > 0.15)

    print("─" * 64)
    print("  COINGLASS PLAN UPGRADE ANALYSIS")
    print("─" * 64)
    print("  Current Hobbyist plan limitations:")
    print("    OI / Funding    : 4h only, ~180 days history")
    print("    Liquidation     : NOT AVAILABLE (404)")
    print("    Net Position    : NOT AVAILABLE")
    print("    CVD aggregated  : NOT AVAILABLE")
    print()
    print("  Standard plan would add:")
    print("    OI / Funding    : 1h granularity + 360 days history")
    print("    Liquidation Orders → liq_cascade_risk feature")
    print("    Net Position    → institutional long/short ratio")
    print("    CVD aggregated  → real flow confirmation")
    print()
    print(f"  Best corr found (5d): {best_corr:+.4f}" if not pd.isna(best_corr) else "  Best corr found: N/A")
    print(f"  oi_change_3d corr  : {oi_corr:+.4f}" if not pd.isna(oi_corr) else "  oi_change_3d corr: N/A")
    print()

    if upgrade_worthwhile:
        print("  Recommendation: UPGRADE TO STANDARD PLAN")
        print("    Reason: corr>0.15 indicates positioning signals have edge.")
        print("    Longer history would stabilise estimates & improve HMM training.")
        print("    Liquidation data would unlock liq_cascade_risk (key feature).")
    else:
        print("  Recommendation: WAIT — do not upgrade yet")
        print("    Reason: corr<0.15 with 6 months data. More history needed first.")
        print("    Current signals insufficient to justify Standard plan cost.")
        print("    Re-evaluate when 12+ months of data accumulates organically.")

    print("═" * 64)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Derivatives Positioning Correlation Analysis")
    print(f"  BTC period   : {START.date()} → {TODAY.date()}")
    print(f"  Deriv window : {DERIV_START.date()} → {TODAY.date()}  (~180 days)")
    print("  [NOTE] Hobbyist plan: no liquidations, no net position, no CVD")
    print("=" * 72)
    print()

    df, has_liq = load_data()
    df          = section1_build_signals(df, has_liq)
    section2_visual(df)
    corr_df     = section3_correlation(df)
    df          = section4_regime_conditional(df)
    section5_leverage_stress(df)
    section6_funding_extremes(df)
    section7_recommendation(df, corr_df)

    out_csv = OUT_DIR / "positioning_results.csv"
    corr_df.to_csv(out_csv, index=False)
    print(f"Results saved → {out_csv}")

    print("=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
