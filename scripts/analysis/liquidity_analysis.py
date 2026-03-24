"""
Global Liquidity Correlation Analysis with BTC
================================================
Tests if Fed balance sheet (WALCL) and M2 money supply
have predictive power for BTC regime direction.

NOTE on global M2 limitation:
  True global M2 requires paid data (BIS, Bloomberg).
  This script uses WALCL (Fed balance sheet) as the
  primary free proxy for Fed liquidity cycles, and
  M2SL (US M2) as the domestic money supply proxy.
  ECB/PBOC balance sheets are not available via FRED
  without manual aggregation. Findings reflect US
  Fed liquidity only — not global M2.

Run from project root:
  conda activate crypto_market_state
  python scripts/analysis/liquidity_analysis.py
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

START = pd.Timestamp("2020-10-01", tz="UTC")
TODAY = pd.Timestamp.now(tz="UTC").normalize()

EVENTS_TS = {
    pd.Timestamp("2020-03-23", tz="UTC"): "COVID QE",
    pd.Timestamp("2021-11-03", tz="UTC"): "QE taper",
    pd.Timestamp("2022-06-01", tz="UTC"): "QT begins",
    pd.Timestamp("2023-03-12", tz="UTC"): "SVB liquidity",
    pd.Timestamp("2024-01-11", tz="UTC"): "ETF + QT slow",
    pd.Timestamp("2025-01-20", tz="UTC"): "Trump QT pause?",
}

REGIME_COLORS = {1: "#4dff88", 0: "#888888", -1: "#ff4d4d"}
REGIME_LABELS = {1: "FLOOD", 0: "NEUTRAL", -1: "DROUGHT"}

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


def _rolling_slope(series, window):
    """Annualised slope of OLS fit over a rolling window."""
    slopes = series.copy() * np.nan
    arr = series.values
    for i in range(window, len(arr) + 1):
        y = arr[i - window:i]
        if np.isnan(y).any():
            continue
        x = np.arange(window, dtype=float)
        b = np.polyfit(x, y, 1)[0]
        slopes.iloc[i - 1] = b
    return slopes


def _pearsonr(x, y):
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 10:
        return np.nan, np.nan
    return stats.pearsonr(xy.iloc[:, 0], xy.iloc[:, 1])


def _corr_period(df, sig_col, target_col, start_str, end_str):
    s = pd.Timestamp(start_str, tz="UTC")
    e = pd.Timestamp(end_str,   tz="UTC")
    sub = df.loc[(df.index >= s) & (df.index <= e), [sig_col, target_col]].dropna()
    if len(sub) < 10:
        return np.nan
    return sub[sig_col].corr(sub[target_col])


def _best_lag(signal, target, lags):
    """Return (best_lag, best_corr) scanning positive lags only (signal leads)."""
    best_lag, best_corr = 0, 0.0
    for lag in lags:
        r, _ = _pearsonr(signal, target.shift(-lag))
        if not pd.isna(r) and abs(r) > abs(best_corr):
            best_corr = r
            best_lag  = lag
    return best_lag, best_corr


def _mark_events(ax, ypos_frac=0.92):
    colors = ["#4da6ff", "#ff8c00", "#ff4d4d", "#4dff88", "#4da6ff", "#ff88ff"]
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for i, (ts, label) in enumerate(EVENTS_TS.items()):
        ax.axvline(ts, color=colors[i % len(colors)], linewidth=0.9,
                   linestyle="--", alpha=0.65)
        ypos = ylim[1] - span * (1 - ypos_frac + i * 0.07)
        ax.text(ts, ypos, f" {label}", color=colors[i % len(colors)],
                fontsize=6.5, va="top", clip_on=True)


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
    btc = btc_raw[["close", "log_return"]].copy()
    print(f"  BTC   : {btc.index[0].date()} → {btc.index[-1].date()}  ({len(btc)} rows)")

    # ── FRED ─────────────────────────────────────────────────────────────────
    fc = Fred(api_key=fred_key)

    def _fred(sid):
        s = fc.get_series(sid, observation_start=START.strftime("%Y-%m-%d"))
        # FRED returns tz-naive — localize explicitly
        s.index = pd.to_datetime(s.index).tz_localize("UTC")
        s.name  = sid
        return s

    walcl    = _fred("WALCL")     # weekly
    m2sl     = _fred("M2SL")      # monthly
    bogmbase = _fred("BOGMBASE")  # weekly
    dgs10    = _fred("DGS10")     # daily

    print(f"  WALCL : {walcl.index[0].date()} → {walcl.index[-1].date()}  ({len(walcl)} obs, weekly)")
    print(f"  M2SL  : {m2sl.index[0].date()} → {m2sl.index[-1].date()}  ({len(m2sl)} obs, monthly)")

    # ── Build daily grid (bdate_range, naive then localize) ───────────────────
    s_naive = START.tz_localize(None)
    e_naive = TODAY.tz_localize(None)
    bday_idx = pd.bdate_range(start=s_naive, end=e_naive, freq="B").tz_localize("UTC")

    df = btc.reindex(bday_idx)

    # Reindex and forward-fill each series (weekly/monthly → daily)
    for s in [walcl, m2sl, bogmbase, dgs10]:
        df = df.join(s.reindex(bday_idx), how="left")

    ff_cols = ["WALCL", "M2SL", "BOGMBASE", "DGS10"]
    df[ff_cols] = df[ff_cols].ffill()

    df["log_return"] = df["log_return"].fillna(
        np.log(df["close"] / df["close"].shift(1))
    )

    print(f"  Grid  : {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} rows)\n")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Build liquidity signals
# ──────────────────────────────────────────────────────────────────────────────

def section1_build_signals(df):
    print("Section 1: Building liquidity signals...")

    # WALCL is weekly data forward-filled to daily
    # Use business-day equivalent windows: 4w≈20d, 12w≈60d, 52w≈260d
    W4  = 20
    W12 = 60
    W52 = 260
    W8  = 40

    df["walcl_change_4w"]  = df["WALCL"].pct_change(W4)
    df["walcl_change_12w"] = df["WALCL"].pct_change(W12)
    df["walcl_zscore_52w"] = _zscore(df["WALCL"], W52)
    df["walcl_trend"]      = _rolling_slope(df["WALCL"], W8)

    # Liquidity regime
    df["liquidity_regime"] = np.where(df["walcl_change_12w"] >  0.01,  1,
                             np.where(df["walcl_change_12w"] < -0.01, -1, 0))

    # M2 signals (monthly → ffill to daily; 3m≈63d bday, 12m≈252d bday)
    df["m2_change_3m"] = df["M2SL"].pct_change(63)
    df["m2_yoy"]       = df["M2SL"].pct_change(252)
    df["m2_impulse"]   = df["m2_change_3m"].diff(1)

    # Combined liquidity score (weighted z-scores)
    m2_zscore = _zscore(df["m2_yoy"], W52)
    df["liquidity_score"] = (df["walcl_zscore_52w"] * 0.6 +
                             m2_zscore * 0.4)

    # Forward BTC returns (liquidity is slow — use 21d and 63d)
    df["btc_ret_21d"] = df["close"].pct_change(21).shift(-21)
    df["btc_ret_63d"] = df["close"].pct_change(63).shift(-63)

    regime_counts = pd.Series(df["liquidity_regime"]).value_counts()
    for r, lbl in REGIME_LABELS.items():
        n   = regime_counts.get(r, 0)
        pct = n / len(df)
        print(f"  {lbl:8s}: {n:4d} days ({pct:.0%})")
    print()

    return df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Visual analysis
# ──────────────────────────────────────────────────────────────────────────────

def section2_visual(df):
    print("Section 2: Generating liquidity chart...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2]})

    # ── Panel 1: BTC price colored by liquidity regime ────────────────────────
    ax = axes[0]
    for regime, color in REGIME_COLORS.items():
        mask = df["liquidity_regime"] == regime
        sub  = df[mask]
        ax.scatter(sub.index, sub["close"], s=3, color=color,
                   label=REGIME_LABELS[regime], alpha=0.7, zorder=2)
    ax.plot(df.index, df["close"], color="#333333", linewidth=0.5, zorder=1)
    ax.set_yscale("log")
    ax.set_ylabel("BTC Price (log scale)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("BTC Price vs Fed Liquidity Regime  (green=FLOOD, red=DROUGHT, gray=NEUTRAL)")
    ax.legend(fontsize=8, markerscale=3, loc="upper left")
    _mark_events(ax)

    # ── Panel 2: WALCL level + 12w change ─────────────────────────────────────
    ax2 = axes[1]
    ax2r = ax2.twinx()
    ax2.plot(df.index, df["WALCL"] / 1e3, color="#4da6ff", linewidth=1.1,
             label="WALCL (trillions USD)")
    ax2r.plot(df.index, df["walcl_change_12w"] * 100, color="#ff8c00",
              linewidth=0.9, alpha=0.8, label="12w % change")
    ax2r.axhline(1.0,  color="#4dff88", linewidth=0.6, linestyle="--", alpha=0.5)
    ax2r.axhline(-1.0, color="#ff4d4d", linewidth=0.6, linestyle="--", alpha=0.5)
    ax2r.axhline(0,    color="white",   linewidth=0.4)
    ax2.set_ylabel("WALCL ($ trillions)", color="#4da6ff")
    ax2r.set_ylabel("12w % change", color="#ff8c00")
    lines1, lbl1 = ax2.get_legend_handles_labels()
    lines2, lbl2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)
    _mark_events(ax2)

    # ── Panel 3: liquidity_score ───────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(df.index, df["liquidity_score"], color="#4dff88", linewidth=1.0,
             label="liquidity_score (WALCL×0.6 + M2×0.4 z-score)")
    ax3.fill_between(df.index, df["liquidity_score"], 0,
                     where=df["liquidity_score"] > 0, color="#4dff88", alpha=0.12)
    ax3.fill_between(df.index, df["liquidity_score"], 0,
                     where=df["liquidity_score"] < 0, color="#ff4d4d", alpha=0.12)
    ax3.axhline(0, color="white", linewidth=0.5)
    ax3.set_ylabel("Liquidity Score (z)")
    ax3.legend(fontsize=8)
    _mark_events(ax3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.tight_layout()
    _save(fig, "liquidity_btc.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Correlation and lag analysis
# ──────────────────────────────────────────────────────────────────────────────

def section3_correlation(df):
    print("Section 3: Correlation and lag analysis...")

    target  = "btc_ret_21d"
    signals = [
        "walcl_change_4w",
        "walcl_change_12w",
        "walcl_zscore_52w",
        "walcl_trend",
        "m2_change_3m",
        "m2_yoy",
        "liquidity_score",
    ]

    periods = {
        "20-22": ("2020-10-01", "2022-05-31"),
        "22-23": ("2022-06-01", "2023-12-31"),
        "24+":   ("2024-01-01", str(TODAY.date())),
    }

    lags = range(0, 61)   # up to 60 trading days ≈ 3 months

    results = []
    rows    = []

    for sig in signals:
        if sig not in df.columns:
            continue

        corr_full, p_val = _pearsonr(df[sig], df[target])
        period_corrs = {k: _corr_period(df, sig, target, s, e)
                        for k, (s, e) in periods.items()}
        best_lag, best_corr_lag = _best_lag(df[sig], df[target], lags)

        def _f(v):
            return _fmt(v) if not pd.isna(v) else "N/A"

        rows.append([
            sig,
            _f(corr_full),
            _f(period_corrs["20-22"]),
            _f(period_corrs["22-23"]),
            _f(period_corrs["24+"]),
            f"{p_val:.4f}" if not pd.isna(p_val) else "N/A",
            f"{best_lag}d",
        ])

        results.append({
            "signal"     : sig,
            "corr_full"  : corr_full,
            "corr_20_22" : period_corrs["20-22"],
            "corr_22_23" : period_corrs["22-23"],
            "corr_24_now": period_corrs["24+"],
            "p_value"    : p_val,
            "best_lag"   : best_lag,
            "best_corr_lag": best_corr_lag,
        })

    _tbl(rows,
         ["Signal", "Corr_full", "20-22", "22-23", "24+", "p-val", "Best_lag"],
         title="Pearson Correlation with BTC 21d Forward Return")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Regime analysis
# ──────────────────────────────────────────────────────────────────────────────

def section4_regime(df):
    print("Section 4: Regime return analysis...")

    rows = []
    regime_stats = {}

    for r_val, label in REGIME_LABELS.items():
        mask = df["liquidity_regime"] == r_val
        sub  = df[mask].dropna(subset=["btc_ret_21d", "btc_ret_63d", "log_return"])
        if len(sub) < 5:
            rows.append([label, "N/A", "N/A", "N/A", "N/A", len(sub)])
            continue

        m21  = sub["btc_ret_21d"].mean()
        m63  = sub["btc_ret_63d"].mean()
        hr   = (sub["btc_ret_21d"] > 0).mean()
        vol  = sub["log_return"].std() * np.sqrt(252)
        regime_stats[label] = {"mean_21d": m21, "series_21d": sub["btc_ret_21d"]}

        rows.append([label,
                     _fmt(m21, pct=True),
                     _fmt(m63, pct=True),
                     f"{hr:.0%}",
                     _fmt(vol, pct=True),
                     len(sub)])

    _tbl(rows,
         ["Regime", "BTC_21d", "BTC_63d", "Hit_rate", "Ann_vol", "N_days"],
         title="BTC Performance by Liquidity Regime")

    # T-test FLOOD vs DROUGHT
    flood   = regime_stats.get("FLOOD",   {}).get("series_21d", pd.Series(dtype=float))
    drought = regime_stats.get("DROUGHT", {}).get("series_21d", pd.Series(dtype=float))

    if len(flood) > 5 and len(drought) > 5:
        t_stat, p_val = stats.ttest_ind(flood.dropna(), drought.dropna(),
                                        equal_var=False)
        flood_mean   = flood.dropna().mean() * 100
        drought_mean = drought.dropna().mean() * 100
        delta        = flood_mean - drought_mean

        print(f"  T-test FLOOD vs DROUGHT (21d return):")
        print(f"    FLOOD  mean : {flood_mean:+.2f}%")
        print(f"    DROUGHT mean: {drought_mean:+.2f}%")
        print(f"    Delta       : {delta:+.2f}%")
        print(f"    t-statistic : {t_stat:+.3f}")
        print(f"    p-value     : {p_val:.4f}")

        sig = p_val < 0.05
        strong = abs(delta) > 5 and sig
        print(f"    Significant : {'YES ✓' if sig else 'no ✗'}")
        if strong:
            print(f"    → STRONG regime signal: delta>{5}% AND p<0.05")
        else:
            print(f"    → Weak or noisy regime signal")
    else:
        delta, t_stat, p_val = np.nan, np.nan, np.nan
        print("  Not enough data for t-test.")
    print()

    return delta, t_stat, p_val


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Lag distribution
# ──────────────────────────────────────────────────────────────────────────────

def section5_lag_distribution(df):
    print("Section 5: Lag distribution analysis...")

    target = "btc_ret_21d"
    lags   = range(0, 91)

    corrs_dir = []   # direction prediction (sign match)
    corrs_mag = []   # magnitude prediction (abs return)

    for lag in lags:
        r_dir, _ = _pearsonr(df["liquidity_score"], df[target].shift(-lag))
        r_mag, _ = _pearsonr(df["liquidity_score"],
                             df["log_return"].abs().rolling(21).mean().shift(-lag))
        corrs_dir.append(r_dir if not np.isnan(r_dir) else 0)
        corrs_mag.append(r_mag if not np.isnan(r_mag) else 0)

    best_lag_dir = int(np.argmax(np.abs(corrs_dir)))
    best_lag_mag = int(np.argmax(np.abs(corrs_mag)))

    print(f"  Direction best lag : {best_lag_dir}d  (corr={corrs_dir[best_lag_dir]:+.4f})")
    print(f"  Magnitude best lag : {best_lag_mag}d  (corr={corrs_mag[best_lag_mag]:+.4f})")
    print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    colors_bar = ["#4dff88" if c >= 0 else "#ff4d4d" for c in corrs_dir]
    ax.bar(list(lags), corrs_dir, color=colors_bar, alpha=0.8, width=0.8)
    ax.axhline(0,  color="white", linewidth=0.5)
    sig_band = 1.96 / np.sqrt(df["liquidity_score"].dropna().shape[0])
    ax.axhline(+sig_band, color="yellow", linewidth=0.7, linestyle=":",
               alpha=0.7, label=f"95% CI (±{sig_band:.3f})")
    ax.axhline(-sig_band, color="yellow", linewidth=0.7, linestyle=":", alpha=0.7)
    ax.axvline(best_lag_dir, color="cyan", linewidth=1.0, linestyle="--",
               alpha=0.7, label=f"Best lag = {best_lag_dir}d")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("liquidity_score → BTC 21d Direction Prediction (lag in trading days)")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    colors_bar2 = ["#4dff88" if c >= 0 else "#ff4d4d" for c in corrs_mag]
    ax2.bar(list(lags), corrs_mag, color=colors_bar2, alpha=0.8, width=0.8)
    ax2.axhline(0, color="white", linewidth=0.5)
    ax2.axhline(+sig_band, color="yellow", linewidth=0.7, linestyle=":", alpha=0.7)
    ax2.axhline(-sig_band, color="yellow", linewidth=0.7, linestyle=":", alpha=0.7)
    ax2.axvline(best_lag_mag, color="orange", linewidth=1.0, linestyle="--",
                alpha=0.7, label=f"Best lag = {best_lag_mag}d")
    ax2.set_xlabel("Lag (trading days)")
    ax2.set_ylabel("Pearson Correlation")
    ax2.set_title("liquidity_score → BTC Volatility Magnitude Prediction")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "liquidity_lag.png")

    return best_lag_dir, corrs_dir[best_lag_dir]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Comparison vs previously tested signals
# ──────────────────────────────────────────────────────────────────────────────

def section6_comparison(df):
    print("Section 6: Comparison vs previous signals...")

    # Rebuild yield_curve_2y10y and cpi_vs_target proxies if available
    # DGS10 was loaded; DGS2 not in this script — skip yield_curve if missing
    target = "btc_ret_21d"

    # Compute CPI proxy from FRED if needed — not loaded here, skip gracefully
    compare_sigs = {
        "liquidity_score"  : df.get("liquidity_score"),
        "walcl_change_12w" : df.get("walcl_change_12w"),
    }

    # Load previously saved results if available
    prev_csv = OUT_DIR / "fed_macro_results.csv"
    prev_best = {}
    if prev_csv.exists():
        prev = pd.read_csv(prev_csv).set_index("signal")
        for sig in ["yield_curve_2y10y", "cpi_vs_target"]:
            if sig in prev.index:
                prev_best[sig] = prev.loc[sig, "corr_full"]

    # Print comparison
    rows = []
    liq_corr, liq_p = _pearsonr(df["liquidity_score"], df[target])
    wc_corr,  wc_p  = _pearsonr(df["walcl_change_12w"], df[target])

    rows.append(["liquidity_score",   _fmt(liq_corr), f"{liq_p:.4f}", "this script"])
    rows.append(["walcl_change_12w",  _fmt(wc_corr),  f"{wc_p:.4f}", "this script"])
    for sig, corr in prev_best.items():
        rows.append([sig, _fmt(corr), "see fed_macro", "fed_macro_correlation.py"])

    _tbl(rows,
         ["Signal", "Corr_btc_21d", "p-value", "Source"],
         title="Liquidity vs Previously Tested Signals (BTC 21d return)")

    # Correlation matrix between signals
    matrix_cols = ["liquidity_score", "walcl_change_12w", target]
    avail = [c for c in matrix_cols if c in df.columns]
    corr_matrix = df[avail].dropna().corr()

    print("  Signal correlation matrix:")
    _tbl(
        [[row] + [_fmt(corr_matrix.loc[row, col]) for col in avail]
         for row in avail],
        [""] + avail,
    )

    # Redundancy check
    r_liq_wc, _ = _pearsonr(df["liquidity_score"], df["walcl_change_12w"])
    print(f"  liquidity_score × walcl_change_12w: {r_liq_wc:+.4f}")
    if abs(r_liq_wc) > 0.7:
        print("  → HIGH internal redundancy — walcl_change_12w is sufficient (simpler)")
    else:
        print("  → liquidity_score adds M2 information beyond WALCL alone")

    # Check against yield_curve if corr available
    if "yield_curve_2y10y" in prev_best:
        yc_corr = prev_best["yield_curve_2y10y"]
        if abs(liq_corr) > abs(yc_corr):
            print(f"  → liquidity_score ({liq_corr:+.4f}) STRONGER than yield_curve ({yc_corr:+.4f})")
        else:
            print(f"  → yield_curve ({yc_corr:+.4f}) remains stronger than liquidity_score ({liq_corr:+.4f})")
    print()

    return liq_corr, liq_p


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — HMM feature recommendation
# ──────────────────────────────────────────────────────────────────────────────

def section7_recommendation(df, corr_df, delta, t_stat, p_val_regime,
                             best_lag_dir, best_corr_dir, liq_corr, liq_p):
    print("Section 7: Generating recommendation...")

    # Find best single signal from corr_df
    if not corr_df.empty:
        best_row = corr_df.loc[corr_df["corr_full"].abs().idxmax()]
        best_sig  = best_row["signal"]
        best_corr = best_row["corr_full"]
        best_lag  = best_row["best_lag"]
        best_p    = best_row["p_value"]
    else:
        best_sig, best_corr, best_lag, best_p = "liquidity_score", liq_corr, best_lag_dir, liq_p

    # Stability: check if corr sign is consistent across periods
    sub = corr_df[corr_df["signal"] == best_sig] if not corr_df.empty else pd.DataFrame()
    if not sub.empty:
        period_corrs = [sub.iloc[0]["corr_20_22"], sub.iloc[0]["corr_22_23"],
                        sub.iloc[0]["corr_24_now"]]
        valid = [c for c in period_corrs if not pd.isna(c)]
        signs = [np.sign(c) for c in valid]
        stable = len(set(signs)) == 1
    else:
        stable = False

    flood_sub   = df[df["liquidity_regime"] ==  1]["btc_ret_21d"].dropna()
    drought_sub = df[df["liquidity_regime"] == -1]["btc_ret_21d"].dropna()
    flood_mean  = flood_sub.mean()  * 100 if len(flood_sub)  > 0 else np.nan
    drought_mean= drought_sub.mean()* 100 if len(drought_sub)> 0 else np.nan

    regime_strong = (not pd.isna(delta)) and abs(delta) > 5 and (not pd.isna(p_val_regime)) and p_val_regime < 0.05
    signal_sig    = (not pd.isna(best_p)) and best_p < 0.10

    include = regime_strong or signal_sig

    print()
    print("═" * 54)
    print("  LIQUIDITY SIGNAL RECOMMENDATION")
    print("═" * 54)
    print(f"  Best liquidity signal  : {best_sig}")
    print(f"  Corr with BTC 21d      : {best_corr:+.4f}  (p={best_p:.4f})" if not pd.isna(best_corr) else "  Corr with BTC 21d      : N/A")
    print(f"  Best lag (direction)   : {best_lag_dir} trading days (~{best_lag_dir//5} weeks)")
    print(f"  Stable across periods  : {'YES ✓' if stable else 'NO ✗'}")
    print()
    print(f"  FLOOD  regime BTC avg 21d : {flood_mean:+.2f}%"   if not pd.isna(flood_mean)  else "  FLOOD  regime : N/A")
    print(f"  DROUGHT regime BTC avg 21d: {drought_mean:+.2f}%" if not pd.isna(drought_mean) else "  DROUGHT regime : N/A")
    if not pd.isna(delta):
        print(f"  Delta                     : {delta:+.2f}%")
        print(f"  (t={t_stat:+.3f}, p={p_val_regime:.4f}  {'strong ✓' if regime_strong else 'weak ✗'})")
    print()

    if include:
        lag_feat = f"walcl_change_12w_lag{best_lag_dir:02d}"
        print(f"  Recommendation: INCLUDE in HMM v2")
        print(f"    Feature name   : {lag_feat}")
        print(f"    Engineering    : WALCL.pct_change(60).shift({best_lag_dir})")
        print(f"    Also consider  : liquidity_regime (-1/0/1) as regime gate")
    else:
        print(f"  Recommendation: EXCLUDE from HMM v2")
        print(f"    Reason: corr={best_corr:+.4f} not significant and regime delta weak")
        print(f"    Note: WALCL is a useful macro context but noisy at daily frequency")
        print(f"    Consider: use liquidity_regime as a meta-filter (not HMM feature)")

    print()
    print("  NOTE (M2 global limitation):")
    print("    This analysis uses WALCL (Fed only) + M2SL (US only).")
    print("    True global M2 (Fed+ECB+PBOC+BOJ) requires paid data.")
    print("    Global M2 historically shows stronger BTC correlation.")
    print("    Supplement with: https://fred.stlouisfed.org/ ECB BSI data")
    print("    or use CrossBorderCapital Global M2 proxy when available.")
    print("═" * 54)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Global Liquidity Correlation Analysis with BTC")
    print(f"  Period : {START.date()} → {TODAY.date()}")
    print("  NOTE: Uses WALCL+M2SL (US Fed only — not true global M2)")
    print("=" * 72)
    print()

    df      = load_data()
    df      = section1_build_signals(df)
    section2_visual(df)
    corr_df = section3_correlation(df)
    delta, t_stat, p_val_regime = section4_regime(df)
    best_lag_dir, best_corr_dir = section5_lag_distribution(df)
    liq_corr, liq_p = section6_comparison(df)
    section7_recommendation(df, corr_df, delta, t_stat, p_val_regime,
                            best_lag_dir, best_corr_dir, liq_corr, liq_p)

    # Save results
    out_csv = OUT_DIR / "liquidity_results.csv"
    corr_df.to_csv(out_csv, index=False)
    print(f"Results saved → {out_csv}")

    print("=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
