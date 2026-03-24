"""
Advanced BTC Regime Deep Analysis
===================================
Run from project root:
  conda activate crypto_market_state
  python scripts/analysis/regime_deep_analysis.py
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
# Constants — all UTC from line 1
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BTC_PATH     = PROJECT_ROOT / "data/03_primary/spot/daily/BTCUSDT.parquet"
OUT_DIR      = PROJECT_ROOT / "scripts/analysis/output"

START     = pd.Timestamp("2023-01-01", tz="UTC")
TODAY     = pd.Timestamp.now(tz="UTC").normalize()
ETF_DATE  = pd.Timestamp("2024-01-10", tz="UTC")

FOMC_DATES_TS = pd.to_datetime([
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

ROLL_WINDOW = 60   # days for rolling correlations

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


def _zscore(s, w):
    m = s.rolling(w).mean()
    sd = s.rolling(w).std()
    return (s - m) / sd.replace(0, np.nan)


def _pearsonr(x, y):
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 10:
        return np.nan, np.nan
    return stats.pearsonr(xy.iloc[:, 0], xy.iloc[:, 1])


def _fwd_ret(close, h):
    return close.pct_change(h).shift(-h)


def _mark_fomc(ax, ymin=None, ymax=None):
    for fd in FOMC_DATES_TS:
        ax.axvline(fd, color="gray", linewidth=0.35, linestyle="--", alpha=0.4)


def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")

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
    cols = [c for c in ["close", "high", "low", "volume", "log_return"]
            if c in btc_raw.columns]
    btc = btc_raw[cols].copy()
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
    print(f"  FRED : {dgs2.index[0].date()} → {dgs2.index[-1].date()}")

    # ── Yahoo ─────────────────────────────────────────────────────────────────
    def _yf(ticker, col="Close"):
        raw = yf.download(ticker, start=START.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False)
        s = raw[col].squeeze()
        s.index = pd.to_datetime(s.index).tz_localize("UTC")
        s.name = ticker
        return s

    vix   = _yf("^VIX")
    dxy   = _yf("DX-Y.NYB")
    sp500 = _yf("^GSPC")
    print(f"  Yahoo: {vix.index[0].date()} → {vix.index[-1].date()}")

    # ── Align (naive bdate_range → tz_localize) ───────────────────────────────
    s_naive = START.tz_localize(None)
    e_naive = TODAY.tz_localize(None)
    bday_idx = pd.bdate_range(start=s_naive, end=e_naive, freq="B").tz_localize("UTC")

    df = btc.reindex(bday_idx)
    for s in [dgs2, dgs10, dgs3mo, fedfunds, vix, dxy, sp500]:
        df = df.join(s.reindex(bday_idx), how="left")

    macro_cols = ["DGS2", "DGS10", "DGS3MO", "FEDFUNDS", "^VIX", "DX-Y.NYB", "^GSPC"]
    df[macro_cols] = df[macro_cols].ffill()

    if "log_return" in df.columns:
        df["log_return"] = df["log_return"].fillna(
            np.log(df["close"] / df["close"].shift(1))
        )
    else:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Derived 1d changes for macro
    df["dgs2_change_1d"]  = df["DGS2"].diff(1)
    df["dxy_return_1d"]   = np.log(df["DX-Y.NYB"] / df["DX-Y.NYB"].shift(1))
    df["vix_return_1d"]   = df["^VIX"].diff(1)
    df["sp500_return_1d"] = np.log(df["^GSPC"] / df["^GSPC"].shift(1))

    # Forward BTC returns
    for h in [1, 3, 5, 10]:
        df[f"btc_ret_{h}d"] = _fwd_ret(df["close"], h)

    print(f"  Grid : {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} rows)\n")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Rolling Regime Dependence
# ──────────────────────────────────────────────────────────────────────────────

def section1_rolling_correlation(df):
    print("=" * 72)
    print("  SECTION 1 — Rolling Regime Dependence")
    print("=" * 72)

    w = ROLL_WINDOW
    btc = df["log_return"]

    rc = pd.DataFrame(index=df.index)
    rc["BTC×DXY"]   = btc.rolling(w).corr(df["dxy_return_1d"])
    rc["BTC×DGS2"]  = btc.rolling(w).corr(df["dgs2_change_1d"])
    rc["BTC×VIX"]   = btc.rolling(w).corr(df["vix_return_1d"])
    rc["BTC×SP500"] = btc.rolling(w).corr(df["sp500_return_1d"])

    pre  = rc[rc.index < ETF_DATE]
    post = rc[rc.index >= ETF_DATE]

    rows = []
    for col in rc.columns:
        s       = rc[col].dropna()
        flips   = (np.sign(s).diff().abs() > 0).sum()
        rows.append([
            col,
            _fmt(s.mean()),
            _fmt(s.std()),
            f"{(s.abs() > 0.2).mean():.0%}",
            f"{(s > 0).mean():.0%}",
            int(flips),
            _fmt(pre[col].mean())  if len(pre[col].dropna()) > 0  else "N/A",
            _fmt(post[col].mean()) if len(post[col].dropna()) > 0 else "N/A",
        ])

    _tbl(rows,
         ["Signal", "Mean", "Std", "Persist>0.2", "Pct_pos", "Flips",
          "Pre-ETF", "Post-ETF"],
         title=f"Rolling {w}d Correlation — BTC vs Macro")

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors = ["#4da6ff", "#ff8c00", "#ff4d4d", "#4dff88"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    ax = axes[0]
    for (col, color) in zip(rc.columns, colors):
        ax.plot(rc.index, rc[col], color=color, linewidth=1.1, label=col, alpha=0.85)
    ax.axhline(0, color="white", linewidth=0.5)
    ax.axhline(+0.2, color="yellow", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axhline(-0.2, color="yellow", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axvline(ETF_DATE, color="cyan", linewidth=1.2, linestyle="--",
               label="ETF Approval (2024-01-10)")
    _mark_fomc(ax)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Pearson Correlation")
    ax.set_title(f"Rolling {w}d Correlation: BTC vs Macro")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = axes[1]
    ax2.plot(df.index, df["close"], color="#aaaaaa", linewidth=0.9)
    ax2.axvline(ETF_DATE, color="cyan", linewidth=1.2, linestyle="--")
    _mark_fomc(ax2)
    ax2.set_ylabel("BTC Close")
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.tight_layout()
    _save(fig, "rolling_correlation.png")

    return rc


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Volatility Clustering
# ──────────────────────────────────────────────────────────────────────────────

def section2_vol_clustering(df):
    print("=" * 72)
    print("  SECTION 2 — Volatility Clustering (Endogenous Regime)")
    print("=" * 72)

    df = df.copy()
    r  = df["log_return"]

    df["vol_7"]     = r.rolling(7).std()
    df["vol_30"]    = r.rolling(30).std()
    df["vol_ratio"] = df["vol_7"] / df["vol_30"].replace(0, np.nan)
    df["vol_of_vol"]= df["vol_7"].rolling(14).std()

    def _regime(vr):
        if pd.isna(vr):
            return "neutral"
        if vr > 1.2:
            return "expanding"
        if vr < 0.8:
            return "compressing"
        return "neutral"

    df["vol_regime"] = df["vol_ratio"].map(_regime)

    rows = []
    for state in ["expanding", "neutral", "compressing"]:
        mask = df["vol_regime"] == state
        sub  = df[mask]
        pct  = mask.mean()
        m5   = sub["btc_ret_5d"].mean()
        s5   = sub["btc_ret_5d"].std()
        hr5  = (sub["btc_ret_5d"] > 0).mean()
        rows.append([
            state,
            f"{pct:.0%}",
            _fmt(m5, pct=True),
            _fmt(s5, pct=True) if not pd.isna(s5) else "N/A",
            f"{hr5:.0%}" if not pd.isna(hr5) else "N/A",
        ])

    _tbl(rows,
         ["Vol Regime", "% Time", "Mean_5d", "Std_5d", "Hit_rate_5d"],
         title="Forward Returns by Volatility Regime")

    # Compression → big move predictor?
    corr_comp_bigmove, p_comp = _pearsonr(
        df["vol_ratio"],
        df["log_return"].abs().shift(-5).rolling(5).mean()
    )
    print(f"  vol_ratio vs abs(ret) [next 5d]: corr={corr_comp_bigmove:+.4f}  p={p_comp:.4f}")
    if not pd.isna(p_comp) and p_comp < 0.05:
        if corr_comp_bigmove > 0:
            print("  → Expanding vol predicts larger moves ahead  ✓")
        else:
            print("  → Compressing vol precedes larger moves  ✓")
    else:
        print("  → No statistically significant predictive link  ✗")
    print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df.index, df["close"], color="#aaaaaa", linewidth=0.9)
    axes[0].set_ylabel("BTC Close")
    axes[0].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    axes[0].set_title("Volatility Clustering & Regime")
    _mark_fomc(axes[0])

    axes[1].plot(df.index, df["vol_ratio"], color="#4da6ff", linewidth=1.0,
                 label="vol_ratio (7d/30d)")
    axes[1].axhline(1.0, color="white",  linewidth=0.5)
    axes[1].axhline(1.2, color="#ff4d4d", linewidth=0.7, linestyle="--",
                    label="Expanding (>1.2)")
    axes[1].axhline(0.8, color="#4dff88", linewidth=0.7, linestyle="--",
                    label="Compressing (<0.8)")
    axes[1].fill_between(df.index, df["vol_ratio"], 1.2,
                         where=df["vol_ratio"] > 1.2,
                         color="#ff4d4d", alpha=0.15)
    axes[1].fill_between(df.index, df["vol_ratio"], 0.8,
                         where=df["vol_ratio"] < 0.8,
                         color="#4dff88", alpha=0.15)
    axes[1].set_ylabel("vol_ratio")
    axes[1].legend(fontsize=8)
    _mark_fomc(axes[1])

    axes[2].plot(df.index, df["vol_of_vol"], color="#ff8c00", linewidth=0.9,
                 label="vol_of_vol (std of vol_7, 14d)")
    axes[2].set_ylabel("vol_of_vol")
    axes[2].legend(fontsize=8)
    _mark_fomc(axes[2])

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.tight_layout()
    _save(fig, "vol_clustering.png")

    return df[["vol_7", "vol_30", "vol_ratio", "vol_of_vol", "vol_regime"]]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Liquidity Regime
# ──────────────────────────────────────────────────────────────────────────────

def section3_liquidity_regime(df, vol_df):
    print("=" * 72)
    print("  SECTION 3 — Liquidity Regime")
    print("=" * 72)

    df = df.copy()
    df = df.join(vol_df, how="left")

    if "high" not in df.columns or "low" not in df.columns:
        print("  [WARN] high/low columns missing — skipping liquidity regime.\n")
        return pd.Series(dtype=float, name="liquidity_proxy")

    df["price_range"]     = (df["high"] - df["low"]) / df["close"]
    df["amplitude_7d"]    = df["price_range"].rolling(7).mean()
    df["return_abs"]      = df["log_return"].abs()
    df["liquidity_proxy"] = df["return_abs"].rolling(7).mean()

    df["volume_z"] = (
        (df["volume"] - df["volume"].rolling(30).mean()) /
        df["volume"].rolling(30).std().replace(0, np.nan)
    )

    q70 = df["amplitude_7d"].rolling(30).quantile(0.7)
    q30 = df["amplitude_7d"].rolling(30).quantile(0.3)

    df["fragile_regime"]      = ((df["amplitude_7d"] > q70) & (df["volume_z"] < -0.5)).astype(int)
    df["accumulation_regime"] = ((df["amplitude_7d"] < q30) & (df["volume_z"] > 0.5)).astype(int)

    def _liq_regime(row):
        if row["fragile_regime"] == 1:
            return "fragile"
        if row["accumulation_regime"] == 1:
            return "accumulation"
        return "neutral"

    df["liq_regime"] = df[["fragile_regime", "accumulation_regime"]].apply(_liq_regime, axis=1)

    rows = []
    interp = {
        "fragile":      "high amplitude + low volume = unstable",
        "accumulation": "tight range + high volume = institutional",
        "neutral":      "normal conditions",
    }
    for state in ["fragile", "neutral", "accumulation"]:
        mask = df["liq_regime"] == state
        sub  = df[mask]
        m5   = sub["btc_ret_5d"].mean()
        hr5  = (sub["btc_ret_5d"] > 0).mean()
        rows.append([
            state,
            f"{mask.mean():.0%}",
            _fmt(m5, pct=True),
            f"{hr5:.0%}" if not pd.isna(hr5) else "N/A",
            interp[state],
        ])

    _tbl(rows,
         ["Regime", "% Time", "Mean_5d", "Hit_rate_5d", "Interpretation"],
         title="Forward Returns by Liquidity Regime")

    # Redundancy check vs vol_ratio
    corr_liq_vol, p_lv = _pearsonr(df["liquidity_proxy"], df["vol_ratio"])
    print(f"  liquidity_proxy vs vol_ratio: corr={corr_liq_vol:+.4f}  p={p_lv:.4f}")
    if not pd.isna(corr_liq_vol) and abs(corr_liq_vol) > 0.7:
        print("  → HIGH redundancy (|corr|>0.7) — keep vol_ratio, drop liquidity_proxy")
    else:
        print("  → Complementary signals — both worth considering")
    print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df.index, df["close"], color="#aaaaaa", linewidth=0.9)
    axes[0].set_ylabel("BTC Close")
    axes[0].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    axes[0].set_title("Liquidity Regime Analysis")
    _mark_fomc(axes[0])

    axes[1].plot(df.index, df["amplitude_7d"], color="#4da6ff", linewidth=1.0,
                 label="amplitude_7d")
    axes[1].fill_between(df.index, df["amplitude_7d"], 0,
                         where=df["fragile_regime"] == 1,
                         color="#ff4d4d", alpha=0.25, label="Fragile")
    axes[1].fill_between(df.index, df["amplitude_7d"], 0,
                         where=df["accumulation_regime"] == 1,
                         color="#4dff88", alpha=0.25, label="Accumulation")
    axes[1].set_ylabel("amplitude_7d")
    axes[1].legend(fontsize=8)
    _mark_fomc(axes[1])

    axes[2].plot(df.index, df["volume_z"], color="#ff8c00", linewidth=0.9,
                 label="volume_z (30d)")
    axes[2].axhline(0, color="white", linewidth=0.5)
    axes[2].axhline(+0.5, color="#4dff88", linewidth=0.6, linestyle="--", alpha=0.7)
    axes[2].axhline(-0.5, color="#ff4d4d", linewidth=0.6, linestyle="--", alpha=0.7)
    axes[2].set_ylabel("volume_z")
    axes[2].legend(fontsize=8)
    _mark_fomc(axes[2])

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.tight_layout()
    _save(fig, "liquidity_regime.png")

    return df[["amplitude_7d", "liquidity_proxy", "fragile_regime",
               "accumulation_regime", "liq_regime"]]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Event Regime (FOMC Windows)
# ──────────────────────────────────────────────────────────────────────────────

def section4_fomc_event_regime(df, vol_df):
    print("=" * 72)
    print("  SECTION 4 — Event Regime (FOMC Windows)")
    print("=" * 72)

    df = df.copy()
    df = df.join(vol_df[["vol_7"]], how="left")

    # FOMC flag
    fomc_set = set(FOMC_DATES_TS.normalize())
    df["is_fomc"] = df.index.normalize().isin(fomc_set).astype(int)

    # Pre / post windows
    df["pre_fomc_3d"]  = df["is_fomc"].shift(-3).fillna(0)
    df["post_fomc_3d"] = df["is_fomc"].shift(1).fillna(0)
    for col in ["pre_fomc_3d", "post_fomc_3d"]:
        df[col] = df[col].rolling(3, min_periods=1).max().astype(int)

    df["fomc_window"] = (
        (df["pre_fomc_3d"] == 1) |
        (df["is_fomc"] == 1) |
        (df["post_fomc_3d"] == 1)
    ).astype(int)

    # Days to next FOMC
    fomc_list = sorted(FOMC_DATES_TS.tolist())

    def _days_to_next(d):
        future = [f for f in fomc_list if f > d]
        return int((min(future) - d).days) if future else 30

    df["days_to_fomc"]    = [_days_to_next(d) for d in df.index]
    df["fomc_proximity"]  = 1.0 / (df["days_to_fomc"] + 1)

    # Stats by window type
    windows = {
        "pre_fomc" : df["pre_fomc_3d"]  == 1,
        "fomc_day" : df["is_fomc"]       == 1,
        "post_fomc": df["post_fomc_3d"]  == 1,
        "normal"   : df["fomc_window"]   == 0,
    }

    rows = []
    for name, mask in windows.items():
        sub  = df[mask]
        m1   = sub["btc_ret_1d"].mean()
        m3   = sub["btc_ret_3d"].mean() if name != "fomc_day" else np.nan
        rows.append([
            name,
            _fmt(sub["vol_7"].mean(), dec=5) if "vol_7" in sub else "N/A",
            _fmt(m1, pct=True),
            _fmt(m3, pct=True) if not pd.isna(m3) else "—",
            len(sub),
        ])

    _tbl(rows,
         ["Window", "Vol_7_mean", "Mean_1d", "Mean_3d", "N_days"],
         title="Volatility and Returns Around FOMC")

    # t-test: vol around FOMC vs normal
    vol_fomc   = df.loc[df["fomc_window"] == 1, "vol_7"].dropna()
    vol_normal = df.loc[df["fomc_window"] == 0, "vol_7"].dropna()
    if len(vol_fomc) > 5 and len(vol_normal) > 5:
        t, p = stats.ttest_ind(vol_fomc, vol_normal, equal_var=False)
        print(f"  Vol t-test (FOMC window vs normal): t={t:+.3f}  p={p:.4f}")
        if p < 0.05:
            direction = "HIGHER" if vol_fomc.mean() > vol_normal.mean() else "LOWER"
            print(f"  → Vol is significantly {direction} around FOMC events  ✓")
        else:
            print("  → No significant vol difference around FOMC  ✗")
    print()

    return df[["is_fomc", "pre_fomc_3d", "post_fomc_3d",
               "fomc_window", "days_to_fomc", "fomc_proximity"]]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Calendar Effects
# ──────────────────────────────────────────────────────────────────────────────

def section5_calendar_effects(df):
    print("=" * 72)
    print("  SECTION 5 — Calendar Effects")
    print("=" * 72)

    df = df.copy()
    df["dow"]   = df.index.dayofweek        # 0=Mon, 4=Fri
    df["month"] = df.index.month
    df["is_eom"]= (df.index.day >= 26).astype(int)

    day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday",
                 3: "Thursday", 4: "Friday"}

    # ── By day of week ────────────────────────────────────────────────────────
    rows = []
    means_dow = []
    for d in range(5):
        sub  = df[df["dow"] == d]
        m    = sub["log_return"].mean()
        s    = sub["log_return"].std()
        hr   = (sub["log_return"] > 0).mean()
        means_dow.append(m)
        rows.append([day_names[d],
                     _fmt(m, pct=True), _fmt(s, pct=True),
                     f"{hr:.0%}", len(sub)])

    _tbl(rows,
         ["Day", "Mean_ret", "Std_ret", "Hit_rate", "N"],
         title="Returns by Day of Week")

    # Interpretation
    best_day  = max(range(5), key=lambda d: means_dow[d])
    worst_day = min(range(5), key=lambda d: means_dow[d])
    print(f"  Best day : {day_names[best_day]}  ({means_dow[best_day]*100:+.3f}%)")
    print(f"  Worst day: {day_names[worst_day]}  ({means_dow[worst_day]*100:+.3f}%)")
    mon = df[df["dow"] == 0]["log_return"].mean()
    mid = df[df["dow"].isin([1, 2, 3])]["log_return"].mean()
    fri = df[df["dow"] == 4]["log_return"].mean()
    if mon > mid:
        print("  → Monday strength suggests retail weekend activity")
    if mid > mon and mid > fri:
        print("  → Mid-week strength suggests institutional activity")
    print()

    # ── By month ──────────────────────────────────────────────────────────────
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    mrows = []
    means_month = []
    for m in range(1, 13):
        sub  = df[df["month"] == m]
        mean = sub["log_return"].mean()
        means_month.append(mean)
        mrows.append([month_names[m-1],
                      _fmt(mean, pct=True),
                      f"{(sub['log_return']>0).mean():.0%}" if len(sub)>0 else "N/A",
                      len(sub)])

    _tbl(mrows,
         ["Month", "Mean_ret", "Hit_rate", "N"],
         title="Returns by Month")

    # EOM effect
    eom_ret = df[df["is_eom"] == 1]["log_return"].mean()
    reg_ret = df[df["is_eom"] == 0]["log_return"].mean()
    print(f"  EOM (day>=26) mean return : {eom_ret*100:+.4f}%")
    print(f"  Regular days  mean return : {reg_ret*100:+.4f}%")
    print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    colors_pos = ["#4dff88" if m >= 0 else "#ff4d4d" for m in means_dow]
    axes[0].bar(list(day_names.values()), [m * 100 for m in means_dow],
                color=colors_pos, alpha=0.85)
    axes[0].axhline(0, color="white", linewidth=0.5)
    axes[0].set_ylabel("Mean Daily Return (%)")
    axes[0].set_title("Mean BTC Return by Day of Week")

    colors_m = ["#4dff88" if m >= 0 else "#ff4d4d" for m in means_month]
    axes[1].bar(month_names, [m * 100 for m in means_month],
                color=colors_m, alpha=0.85)
    axes[1].axhline(0, color="white", linewidth=0.5)
    axes[1].set_ylabel("Mean Daily Return (%)")
    axes[1].set_title("Mean BTC Return by Month")

    fig.tight_layout()
    _save(fig, "calendar_effects.png")

    return df[["dow", "month", "is_eom"]]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Feature Consolidation for HMM v2
# ──────────────────────────────────────────────────────────────────────────────

def section6_feature_consolidation(df, rc, vol_df, liq_df, fomc_df):
    print("=" * 72)
    print("  SECTION 6 — Feature Consolidation for HMM v2")
    print("=" * 72)

    # Merge all feature columns into main df
    full = df.copy()
    for extra in [vol_df, fomc_df]:
        for col in extra.columns:
            if col not in full.columns:
                full[col] = extra[col]
    if liq_df is not None and not liq_df.empty:
        for col in liq_df.columns:
            if col not in full.columns:
                full[col] = liq_df[col]

    # Rolling corr stability: std of rolling corr (low std = stable signal)
    for rc_col in rc.columns:
        full[f"rc_std_{rc_col}"] = rc[rc_col].rolling(90).std()

    target = "btc_ret_5d"

    candidates = {
        # (column_to_use, description)
        "vol_ratio"           : "Vol clustering — 7d/30d ratio",
        "vol_of_vol"          : "Volatility of volatility (14d std of vol_7)",
        "amplitude_7d"        : "Price range amplitude (7d mean)",
        "liquidity_proxy"     : "Abs return proxy for liquidity (7d)",
        "days_to_fomc"        : "Business days until next FOMC",
        "fomc_proximity"      : "1 / (days_to_fomc + 1)",
        "fomc_window"         : "Binary: within 3d of FOMC",
        "dgs2_change_1d"      : "Fed surprise proxy (from fed_macro_correlation.py)",
        "yield_curve_2y10y"   : "Yield curve slope (10y-2y)",
    }

    # Pull in from previous script results if available
    prev_csv = OUT_DIR / "fed_macro_results.csv"
    prev_corr = {}
    if prev_csv.exists():
        prev = pd.read_csv(prev_csv).set_index("signal")
        for sig in ["rate_expectation_ch30", "cpi_vs_target", "yield_curve_2y10y",
                    "dgs2_change_1d"]:
            if sig in prev.index:
                prev_corr[sig] = {
                    "corr_full": prev.loc[sig, "corr_full"],
                    "p_value"  : prev.loc[sig, "p_value"],
                }

    def _score_power(col):
        if col not in full.columns:
            return 0
        r, p = _pearsonr(full[col], full[target])
        if pd.isna(r):
            return 0
        if abs(r) > 0.10 and p < 0.05:
            return 3
        if abs(r) > 0.05 and p < 0.10:
            return 2
        if abs(r) > 0.02:
            return 1
        return 0

    def _score_stability(col):
        if col not in full.columns:
            return 0
        # Compute rolling 90d corr with target, measure its std
        rolling = full[col].rolling(90).corr(full[target])
        s = rolling.dropna().std()
        if pd.isna(s):
            return 1
        if s < 0.10:
            return 3
        if s < 0.20:
            return 2
        return 1

    def _score_relevance(col):
        # Does this feature vary meaningfully across regimes?
        # Use vol_regime as reference
        if col not in full.columns or "vol_regime" not in full.columns:
            return 1
        groups  = full.groupby("vol_regime")[col].mean()
        spread  = groups.max() - groups.min() if len(groups) > 1 else 0
        overall = full[col].std()
        if pd.isna(spread) or pd.isna(overall) or overall == 0:
            return 1
        ratio = spread / overall
        if ratio > 0.5:
            return 3
        if ratio > 0.2:
            return 2
        return 1

    rows = []
    records = []

    # Extra features from prior script
    extra_feats = {
        "rate_expectation_ch30": "30d trend in rate expectation (fed_macro)",
        "cpi_vs_target"        : "CPI YoY minus 2% target (fed_macro)",
    }
    all_candidates = {**candidates, **extra_feats}

    for col, desc in all_candidates.items():
        if col not in full.columns:
            # Try to get from prev_corr
            if col in prev_corr:
                pc = prev_corr[col]
                pw = 2 if abs(pc["corr_full"]) > 0.05 and pc["p_value"] < 0.10 else 1
            else:
                pw = 0
            st, rv = 1, 1
        else:
            pw = _score_power(col)
            st = _score_stability(col)
            rv = _score_relevance(col)

        total   = pw + st + rv
        include = "✓" if total >= 5 else "✗"

        rows.append([col, pw, st, rv, total, include])
        records.append({
            "feature"       : col,
            "description"   : desc,
            "score_power"   : pw,
            "score_stability": st,
            "score_relevance": rv,
            "total_score"   : total,
            "include"       : total >= 5,
        })

    # Sort by total descending
    rows.sort(key=lambda r: -r[4])

    _tbl(rows,
         ["Feature", "Power(0-3)", "Stability(0-3)", "Relevance(0-3)", "Total", "Include?"],
         title="Feature Consolidation — HMM v2 / Model 4h  (include if total ≥ 5)")

    rec  = [r[0] for r in rows if r[5] == "✓"]
    drop = [r[0] for r in rows if r[5] == "✗"]
    print(f"  RECOMMENDED  ({len(rec)}): {', '.join(rec)}")
    print(f"  NOT included ({len(drop)}): {', '.join(drop)}")
    print()

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Advanced BTC Regime Deep Analysis")
    print(f"  Period : {START.date()} → {TODAY.date()}")
    print("=" * 72)
    print()

    df      = load_data()

    rc      = section1_rolling_correlation(df)
    vol_df  = section2_vol_clustering(df)
    liq_df  = section3_liquidity_regime(df, vol_df)
    fomc_df = section4_fomc_event_regime(df, vol_df)
    _       = section5_calendar_effects(df)
    feat_df = section6_feature_consolidation(df, rc, vol_df, liq_df, fomc_df)

    out_csv = OUT_DIR / "regime_deep_analysis.csv"
    feat_df.to_csv(out_csv, index=False)
    print(f"Feature scores saved → {out_csv}")

    print("=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
