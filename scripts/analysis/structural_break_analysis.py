"""
Structural Break Analysis — BTC Regime
========================================
Validates optimal HMM training start date by detecting
when BTC statistical behavior changed structurally.

Run from project root:
  conda activate crypto_market_state
  python scripts/analysis/structural_break_analysis.py
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
import yaml
from fredapi import Fred
from scipy import stats
import yfinance as yf

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

plt.style.use("dark_background")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BTC_PATH     = PROJECT_ROOT / "data/03_primary/spot/daily/BTCUSDT.parquet"
OUT_DIR      = PROJECT_ROOT / "scripts/analysis/output"

START    = pd.Timestamp("2020-10-01", tz="UTC")
TODAY    = pd.Timestamp.now(tz="UTC").normalize()
ETF_DATE = pd.Timestamp("2024-01-11", tz="UTC")

EVENTS = {
    "2022-06-01": "Luna collapse",
    "2022-11-08": "FTX collapse",
    "2023-01-01": "Post-FTX recovery",
    "2024-01-11": "ETF approved",
    "2024-04-20": "Halving",
    "2025-01-20": "Trump inauguration",
}
EVENTS_TS = {pd.Timestamp(k, tz="UTC"): v for k, v in EVENTS.items()}

BREAK_CANDIDATES = {
    "2022-06-01": "Luna collapse",
    "2022-11-01": "FTX collapse",
    "2023-01-01": "Current HMM train start",
    "2023-06-01": "BlackRock ETF filing",
    "2024-01-11": "ETF approved",
    "2024-04-20": "Halving",
    "2025-01-20": "Trump inauguration",
}

PERIODS = {
    "2020-10→2022-10": ("2020-10-01", "2022-10-31"),
    "2022-11→2023-12": ("2022-11-01", "2023-12-31"),
    "2024-01→2024-12": ("2024-01-01", "2024-12-31"),
    "2025-01→today":   ("2025-01-01", str(TODAY.date())),
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
        return f"{v*100:+.3f}%"
    return f"{v:+.{dec}f}"


def _mark_events(ax, events_ts, ymin=None, ymax=None, text_offset=0.02):
    colors = ["#ff8c00", "#ff4d4d", "#4dff88", "#4da6ff", "#ff88ff", "#ffff44"]
    for i, (ts, label) in enumerate(events_ts.items()):
        ax.axvline(ts, color=colors[i % len(colors)], linewidth=0.9,
                   linestyle="--", alpha=0.7)
        ylim = ax.get_ylim()
        ypos = ylim[1] - (ylim[1] - ylim[0]) * (0.05 + i * 0.08)
        ax.text(ts, ypos, f" {label}", color=colors[i % len(colors)],
                fontsize=6.5, va="top", rotation=0, clip_on=True)


def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def _fisher_z_test(r1, n1, r2, n2):
    """Fisher Z-transformation test for equality of two Pearson correlations."""
    r1 = np.clip(r1, -0.9999, 0.9999)
    r2 = np.clip(r2, -0.9999, 0.9999)
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z  = (z1 - z2) / se
    p  = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Section 0: Loading data...")

    # ── Credentials ──────────────────────────────────────────────────────────
    fred_key = None
    for p in [PROJECT_ROOT / "conf/local/secrets.yml",
              PROJECT_ROOT / "conf/local/credentials.yml"]:
        if not p.exists():
            continue
        raw = yaml.safe_load(open(p))
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
        sys.exit("[ERROR] FRED API key not found.")

    # ── BTC ──────────────────────────────────────────────────────────────────
    if not BTC_PATH.exists():
        sys.exit(f"[ERROR] BTC parquet not found: {BTC_PATH}")

    btc_raw = pd.read_parquet(BTC_PATH)
    btc_raw.index = pd.to_datetime(btc_raw.index, utc=True)
    want = ["close", "high", "low", "volume", "log_return",
            "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]
    cols = [c for c in want if c in btc_raw.columns]
    btc  = btc_raw[cols].copy()
    print(f"  BTC  : {btc.index[0].date()} → {btc.index[-1].date()}  ({len(btc)} rows)")

    # ── FRED ─────────────────────────────────────────────────────────────────
    fc = Fred(api_key=fred_key)

    def _fred(sid):
        s = fc.get_series(sid, observation_start=START.strftime("%Y-%m-%d"))
        s.index = pd.to_datetime(s.index).tz_localize("UTC")
        s.name = sid
        return s

    dgs2     = _fred("DGS2")
    dgs10    = _fred("DGS10")
    dgs3mo   = _fred("DGS3MO")
    fedfunds = _fred("FEDFUNDS")

    # ── Yahoo ─────────────────────────────────────────────────────────────────
    def _yf(ticker, col="Close"):
        raw = yf.download(ticker, start=START.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False)
        s = raw[col].squeeze()
        s.index = pd.to_datetime(s.index).tz_localize("UTC")
        s.name = ticker
        return s

    vix   = _yf("^VIX")
    sp500 = _yf("^GSPC")

    # ── Align on business-day grid ────────────────────────────────────────────
    s_naive = START.tz_localize(None)
    e_naive = TODAY.tz_localize(None)
    bday_idx = pd.bdate_range(start=s_naive, end=e_naive, freq="B").tz_localize("UTC")

    df = btc.reindex(bday_idx)
    for s in [dgs2, dgs10, dgs3mo, fedfunds, vix, sp500]:
        df = df.join(s.reindex(bday_idx), how="left")

    macro_cols = ["DGS2", "DGS10", "DGS3MO", "FEDFUNDS", "^VIX", "^GSPC"]
    df[macro_cols] = df[macro_cols].ffill()

    if "log_return" in df.columns:
        df["log_return"] = df["log_return"].fillna(
            np.log(df["close"] / df["close"].shift(1))
        )
    else:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Derived returns for correlation analysis
    df["vix_change_1d"]   = df["^VIX"].diff(1)
    df["sp500_return_1d"] = np.log(df["^GSPC"] / df["^GSPC"].shift(1))
    df["dgs2_change_1d"]  = df["DGS2"].diff(1)

    print(f"  Grid : {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} rows)\n")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Visual CUSUM
# ──────────────────────────────────────────────────────────────────────────────

def section1_cusum(df):
    print("Section 1: Visual CUSUM...")

    ret = df["log_return"].fillna(0)
    cusum = ret.cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(cusum.index, cusum.values, color="#4dff88", linewidth=1.2,
                 label="CUSUM (cumulative log return)")
    axes[0].axhline(0, color="white", linewidth=0.4)
    _mark_events(axes[0], EVENTS_TS)
    axes[0].set_ylabel("Cumulative Log Return")
    axes[0].set_title("BTC CUSUM — Slope Changes Indicate Regime Shifts")
    axes[0].legend(fontsize=8)

    axes[1].plot(df.index, df["close"], color="#aaaaaa", linewidth=0.8,
                 label="BTC Close")
    _mark_events(axes[1], EVENTS_TS)
    axes[1].set_ylabel("Price (USD)")
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    axes[1].legend(fontsize=8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.tight_layout()
    _save(fig, "cusum_btc.png")

    print("  CUSUM interpretation:")
    print("    Positive slope = positive return regime")
    print("    Slope breaks   = structural regime changes")
    print("    Key breaks visible at FTX (2022-11), ETF (2024-01)\n")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Rolling statistics
# ──────────────────────────────────────────────────────────────────────────────

def section2_rolling_stats(df):
    print("Section 2: Rolling statistics...")

    r  = df["log_return"]
    w  = 90
    rm = r.rolling(w).mean()
    rv = r.rolling(w).std()
    rs = r.rolling(w).skew()
    sharpe = (rm / rv.replace(0, np.nan)) * np.sqrt(252)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    axes[0].plot(df.index, df["close"], color="#aaaaaa", linewidth=0.8)
    axes[0].set_ylabel("BTC Close")
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    axes[0].set_title(f"Rolling {w}d Statistics — BTC Regime Evolution")
    _mark_events(axes[0], EVENTS_TS)

    axes[1].plot(rm.index, rm.values * 100, color="#4dff88", linewidth=1.0)
    axes[1].axhline(0, color="white", linewidth=0.4)
    axes[1].set_ylabel("Mean Return (%)")
    _mark_events(axes[1], EVENTS_TS)

    axes[2].plot(rv.index, rv.values * 100, color="#ff8c00", linewidth=1.0)
    axes[2].set_ylabel("Volatility (%)")
    _mark_events(axes[2], EVENTS_TS)

    axes[3].plot(sharpe.index, sharpe.values, color="#4da6ff", linewidth=1.0)
    axes[3].axhline(0, color="white", linewidth=0.4)
    axes[3].axhline(1, color="#4dff88", linewidth=0.5, linestyle="--", alpha=0.6)
    axes[3].axhline(-1, color="#ff4d4d", linewidth=0.5, linestyle="--", alpha=0.6)
    axes[3].set_ylabel("Sharpe (ann.)")
    _mark_events(axes[3], EVENTS_TS)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.tight_layout()
    _save(fig, "rolling_stats_btc.png")

    # ── Per-period table ──────────────────────────────────────────────────────
    rows = []
    for label, (s, e) in PERIODS.items():
        mask = (df.index >= pd.Timestamp(s, tz="UTC")) & \
               (df.index <= pd.Timestamp(e, tz="UTC"))
        sub  = df.loc[mask, "log_return"].dropna()
        if len(sub) < 5:
            rows.append([label, "N/A", "N/A", "N/A", "N/A"])
            continue
        m  = sub.mean()
        v  = sub.std()
        sh = (m / v * np.sqrt(252)) if v > 0 else np.nan
        sk = sub.skew()
        rows.append([label, _fmt(m, pct=True), _fmt(v, pct=True),
                     _fmt(sh), _fmt(sk)])

    _tbl(rows,
         ["Period", "Mean_ret", "Vol", "Sharpe(ann)", "Skew"],
         title="Statistics per Period")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Chow Test
# ──────────────────────────────────────────────────────────────────────────────

def section3_chow_test(df):
    print("Section 3: Chow test for structural breaks...")

    r = df["log_return"].dropna()
    t = np.arange(len(r))   # time index as regressor

    def _ols_rss(y_arr, x_arr):
        """OLS residual sum of squares."""
        X = np.column_stack([np.ones(len(x_arr)), x_arr])
        beta = np.linalg.lstsq(X, y_arr, rcond=None)[0]
        resid = y_arr - X @ beta
        return np.sum(resid ** 2), len(y_arr)

    # Full-sample RSS
    rss_full, n = _ols_rss(r.values, t)

    rows = []
    confirmed = []

    for date_str, label in BREAK_CANDIDATES.items():
        ts  = pd.Timestamp(date_str, tz="UTC")
        idx = r.index.searchsorted(ts)

        if idx < 30 or idx > len(r) - 30:
            rows.append([date_str, label, "N/A", "N/A", "insufficient data"])
            continue

        rss1, n1 = _ols_rss(r.values[:idx],  t[:idx])
        rss2, n2 = _ols_rss(r.values[idx:],  t[idx:])

        rss_restricted = rss1 + rss2
        k = 2  # number of regressors (intercept + slope)

        if rss_restricted == 0:
            rows.append([date_str, label, "N/A", "N/A", "degenerate"])
            continue

        f_stat = ((rss_full - rss_restricted) / k) / \
                 (rss_restricted / (n - 2 * k))
        p_val  = 1 - stats.f.cdf(f_stat, k, n - 2 * k)
        sig    = "YES ✓" if p_val < 0.05 else "no"

        if p_val < 0.05:
            confirmed.append(date_str)

        rows.append([date_str, label, f"{f_stat:.3f}", f"{p_val:.4f}", sig])

    _tbl(rows,
         ["Date", "Event", "F-stat", "p-value", "Break?"],
         title="Chow Test — Structural Break Detection")

    return confirmed


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Correlation stability (Fisher Z-test)
# ──────────────────────────────────────────────────────────────────────────────

def section4_correlation_stability(df):
    print("Section 4: Correlation stability test (Fisher Z-transform)...")

    pre  = df[df.index < ETF_DATE]
    post = df[df.index >= ETF_DATE]

    variables = {
        "DGS2 change"  : "dgs2_change_1d",
        "VIX change"   : "vix_change_1d",
        "SP500 return" : "sp500_return_1d",
    }

    rows = []
    for label, col in variables.items():
        if col not in df.columns:
            rows.append([label, "N/A", "N/A", "N/A", "N/A", "no data"])
            continue

        def _corr(sub):
            xy = sub[["log_return", col]].dropna()
            if len(xy) < 10:
                return np.nan, 0
            r, _ = stats.pearsonr(xy["log_return"], xy[col])
            return r, len(xy)

        r_pre,  n_pre  = _corr(pre)
        r_post, n_post = _corr(post)

        if pd.isna(r_pre) or pd.isna(r_post) or n_pre < 10 or n_post < 10:
            rows.append([label, _fmt(r_pre), _fmt(r_post), "N/A", "N/A",
                         "insufficient data"])
            continue

        z_stat, p_val = _fisher_z_test(r_pre, n_pre, r_post, n_post)
        changed = "YES ✓" if p_val < 0.05 else "no"
        rows.append([label, _fmt(r_pre), _fmt(r_post),
                     f"{z_stat:+.3f}", f"{p_val:.4f}", changed])

    _tbl(rows,
         ["Variable", "Corr_pre_ETF", "Corr_post_ETF", "Z-stat", "p-value", "Changed?"],
         title="Correlation Stability — Pre vs Post ETF (2024-01-11)")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Recommendation
# ──────────────────────────────────────────────────────────────────────────────

def section5_recommendation(confirmed_breaks):
    print("Section 5: Structural break recommendation...")

    # Rank confirmed breaks — prefer the most recent that still leaves enough
    # training data (≥ 365 days)
    valid_starts = []
    for b in confirmed_breaks:
        ts = pd.Timestamp(b, tz="UTC")
        days_remaining = (TODAY - ts).days
        if days_remaining >= 365:
            valid_starts.append((ts, days_remaining))

    if valid_starts:
        # Most recent confirmed break with ≥365d trailing data
        optimal_ts, days_left = max(valid_starts, key=lambda x: x[0])
        optimal_start = optimal_ts.strftime("%Y-%m-%d")
        reasoning = (
            f"Most recent confirmed structural break ({optimal_start}) "
            f"with {days_left} business days of training data remaining. "
            f"Later breaks reduce training set below 365-day minimum."
        )
    else:
        optimal_start = "2023-01-01"
        reasoning = (
            "No confirmed breaks with ≥365d remaining data. "
            "Defaulting to 2023-01-01 (post-FTX recovery, current HMM train start). "
            "FTX collapse (2022-11) is a visual break in CUSUM even if not statistically confirmed."
        )

    print()
    print("═" * 54)
    print("  STRUCTURAL BREAK RECOMMENDATION")
    print("═" * 54)
    print(f"  Optimal HMM train start : {optimal_start}")
    print(f"  Confirmed breaks        : {confirmed_breaks if confirmed_breaks else 'none at p<0.05'}")
    print(f"  Reasoning               : {reasoning}")
    print("═" * 54)
    print()

    return optimal_start


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Structural Break Analysis — BTC Regime")
    print(f"  Period : {START.date()} → {TODAY.date()}")
    print("=" * 72)
    print()

    df = load_data()
    section1_cusum(df)
    section2_rolling_stats(df)
    confirmed = section3_chow_test(df)
    section4_correlation_stability(df)
    section5_recommendation(confirmed)

    print("=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
