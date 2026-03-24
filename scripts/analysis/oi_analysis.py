"""
Open Interest Correlation Analysis with BTC Regime
====================================================
Deep analysis of OI transformations:
  - Which OI signal has the best predictive power?
  - Does OI lead or lag BTC price?
  - OI + Funding combined = stronger signal?
  - Is 6 months enough, or do we need Standard plan?

Data constraints (CoinGlass Hobbyist plan):
  - OI / Funding: 4h, ~180 days history (2025-09-13 → present)
  - Liquidation / Net Position / CVD: NOT AVAILABLE

Run from project root:
  conda activate crypto_market_state
  python scripts/analysis/oi_analysis.py
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
from tabulate import tabulate

# ── paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[2]
DERIV    = ROOT / "data/01_raw/derivatives/coinglass/4h/BTCUSDT.parquet"
BTC_D    = ROOT / "data/03_primary/spot/daily/BTCUSDT.parquet"
OUT_DIR  = ROOT / "scripts/analysis/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────
START = pd.Timestamp("2025-09-01", tz="UTC")
TODAY = pd.Timestamp.now(tz="UTC").normalize()

EVENTS = {
    pd.Timestamp("2026-01-20", tz="UTC"): "Trump ATH zone",
    pd.Timestamp("2026-02-03", tz="UTC"): "Correction begins",
    pd.Timestamp("2026-03-18", tz="UTC"): "Fed hawkish -4%",
}

HORIZONS = [1, 3, 5, 14]   # days


# ── helpers ────────────────────────────────────────────────────────────────
def zscore(s: pd.Series, window: int) -> pd.Series:
    mu  = s.rolling(window, min_periods=max(1, window // 2)).mean()
    sig = s.rolling(window, min_periods=max(1, window // 2)).std()
    return (s - mu) / sig.replace(0, np.nan)


def pearson_lag(x: pd.Series, y: pd.Series, max_lag: int = 21):
    """Return (corr, p, best_lag) scanning lag 0..max_lag."""
    best_corr, best_p, best_lag = np.nan, np.nan, 0
    for lag in range(0, max_lag + 1):
        xs = x.shift(lag)
        df = pd.concat([xs, y], axis=1).dropna()
        if len(df) < 20:
            continue
        c, p = stats.pearsonr(df.iloc[:, 0], df.iloc[:, 1])
        if np.isnan(best_corr) or abs(c) > abs(best_corr):
            best_corr, best_p, best_lag = c, p, lag
    return best_corr, best_p, best_lag


def mark_events(ax):
    xlim = ax.get_xlim()
    colors = ["#4da6ff", "#ff8c00", "#ff4d4d"]
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for i, (ts, label) in enumerate(EVENTS.items()):
        ts_num = mdates.date2num(ts.to_pydatetime())
        if ts_num < xlim[0] or ts_num > xlim[1]:
            continue
        ax.axvline(ts, color=colors[i % len(colors)], linewidth=0.9,
                   linestyle="--", alpha=0.6)
        ypos = ylim[1] - span * (0.05 + (i % 3) * 0.12)
        ax.text(ts, ypos, f" {label}", color=colors[i % len(colors)],
                fontsize=6.5, va="top", clip_on=True)


def fmt_ax_date(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis="x", labelsize=7, rotation=30)


# ── section 0: load data ───────────────────────────────────────────────────
def section0_load():
    print("\nSection 0: Loading data...")

    # ── CoinGlass 4h ──
    raw = pd.read_parquet(DERIV)
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw = raw.set_index("timestamp").sort_index()
    else:
        raw.index = pd.to_datetime(raw.index, utc=True)

    oi_agg_4h     = raw["open_interest_aggregated_history__close"].rename("oi_agg")
    oi_binance_4h = raw["open_interest_history__close"].rename("oi_binance")
    funding_4h    = raw["funding_rate_oi_weight_history__close"].rename("funding_oi")

    # resample 4h → daily
    oi_agg     = oi_agg_4h.resample("D").last()
    oi_binance = oi_binance_4h.resample("D").last()
    funding_oi = funding_4h.resample("D").mean()

    # ── BTC daily ──
    btc = pd.read_parquet(BTC_D)
    btc.index = pd.to_datetime(btc.index, utc=True)
    close    = btc["close"]
    drawdown = btc["drawdown"] if "drawdown" in btc.columns else (
        close / close.cummax() - 1
    )

    # inner join on date
    df = pd.concat([close, drawdown, oi_agg, oi_binance, funding_oi], axis=1).dropna(
        subset=["oi_agg"]
    )
    df = df.loc[df.index >= START]

    n_days = len(df)
    print(f"  CoinGlass 4h → daily : {oi_agg.index[0].date()} → {oi_agg.index[-1].date()}")
    print(f"  BTC daily            : {close.index[0].date()} → {close.index[-1].date()}")
    print(f"  Joined window        : {df.index[0].date()} → {df.index[-1].date()}  ({n_days} rows)")

    # ── sample size warning ──
    print()
    print("  " + "!" * 66)
    if n_days < 120:
        print(f"  [CRITICAL] Only {n_days} days — correlations UNRELIABLE. Need ≥120.")
    elif n_days < 180:
        print(f"  [WARNING]  {n_days} days — treat correlations with caution (need ≥180).")
    else:
        print(f"  [OK]       {n_days} days — sufficient for exploratory analysis.")
    print("  " + "!" * 66)

    return df


# ── section 1: build signals ───────────────────────────────────────────────
def section1_signals(df: pd.DataFrame) -> pd.DataFrame:
    print("\nSection 1: Building OI signals...")

    close      = df["close"]
    oi_agg     = df["oi_agg"]
    oi_binance = df["oi_binance"]
    funding    = df["funding_oi"]

    # ── raw changes ──
    df["oi_change_1d"]  = oi_agg.pct_change(1)
    df["oi_change_3d"]  = oi_agg.pct_change(3)
    df["oi_change_7d"]  = oi_agg.pct_change(7)
    df["oi_change_14d"] = oi_agg.pct_change(14)

    # ── price-adjusted (remove price trend bias) ──
    df["oi_price_ratio"]   = oi_agg / close
    df["oi_price_ratio_z"] = zscore(df["oi_price_ratio"], 30)

    # ── log-transform (stationarity) ──
    df["oi_log"]          = np.log(oi_agg)
    df["oi_log_diff_3d"]  = df["oi_log"].diff(3)

    # ── z-scores ──
    df["oi_zscore_14d"] = zscore(oi_agg, 14)
    df["oi_zscore_30d"] = zscore(oi_agg, 30)

    # ── momentum (acceleration) ──
    df["oi_momentum"] = df["oi_change_3d"] - df["oi_change_3d"].shift(3)

    # ── price × OI divergence (corrected sign) ──
    df["btc_change_3d"]  = close.pct_change(3)
    df["price_oi_div"]   = df["btc_change_3d"] - df["oi_change_3d"]
    # positive → price ↑, OI ↓ → short covering (bullish)
    # negative → price ↑, OI ↑ → fragile long buildup (bearish)

    # ── exchange divergence ──
    df["oi_exchange_div"] = oi_binance.pct_change(3) - oi_agg.pct_change(3)

    # ── combined with funding ──
    df["funding_zscore_14d"] = zscore(funding, 14)
    df["leverage_stress"]    = df["oi_change_3d"] * df["funding_zscore_14d"]
    df["leverage_expansion"] = df["oi_change_3d"] * (funding > 0).astype(int)

    # ── volatility of OI ──
    df["oi_vol_ratio"] = oi_agg.rolling(5).std() / oi_agg.rolling(30).std()

    # ── discrete signal (HMM-friendly) ──
    df["oi_extreme"] = (
        (df["oi_zscore_30d"] > 1.5).astype(int)
        - (df["oi_zscore_30d"] < -1.5).astype(int)
    )

    # ── forward returns (targets) ──
    for h in HORIZONS:
        df[f"btc_ret_{h}d"] = close.pct_change(h).shift(-h)

    built = [c for c in df.columns if c not in
             ["close", "drawdown", "oi_agg", "oi_binance", "funding_oi"]]
    print(f"  Signals built: {[s for s in built if not s.startswith('btc_ret')]}")
    return df


# ── section 2: visual overview ─────────────────────────────────────────────
def section2_visual(df: pd.DataFrame):
    print("\nSection 2: Generating OI overview chart...")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("BTC Open Interest — Deep Analysis", fontsize=13, fontweight="bold")

    close  = df["close"]
    oi_agg = df["oi_agg"]

    # Panel 1: BTC price
    ax = axes[0]
    ax.plot(close.index, close, color="#f7931a", linewidth=1.2)
    ax.set_ylabel("BTC Price (USD)", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(alpha=0.3)
    mark_events(ax)

    # Panel 2: OI aggregated level
    ax = axes[1]
    ax.fill_between(oi_agg.index, oi_agg / 1e9, alpha=0.4, color="#4da6ff")
    ax.plot(oi_agg.index, oi_agg / 1e9, color="#4da6ff", linewidth=0.9)
    ax.set_ylabel("OI Aggregated ($B)", fontsize=8)
    ax.grid(alpha=0.3)
    mark_events(ax)

    # Panel 3: oi_change_3d
    ax = axes[2]
    vals = df["oi_change_3d"] * 100
    colors = np.where(vals >= 0, "#4dff88", "#ff4d4d")
    ax.bar(vals.index, vals, color=colors, alpha=0.7, width=0.9)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_ylabel("OI Change 3d (%)", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    mark_events(ax)

    # Panel 4: price_oi_div
    ax = axes[3]
    div = df["price_oi_div"]
    ax.fill_between(div.index, div, 0,
                    where=div > 0, color="#4dff88", alpha=0.6,
                    label="Price↑ OI↓ (bullish)")
    ax.fill_between(div.index, div, 0,
                    where=div < 0, color="#ff4d4d", alpha=0.6,
                    label="Price↑ OI↑ (fragile)")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_ylabel("Price−OI Divergence", fontsize=8)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fmt_ax_date(ax)
    mark_events(ax)

    plt.tight_layout()
    out = OUT_DIR / "oi_overview.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── section 3: correlation analysis ───────────────────────────────────────
OI_SIGNALS = [
    "oi_change_1d", "oi_change_3d", "oi_change_7d", "oi_change_14d",
    "oi_price_ratio_z", "oi_log_diff_3d",
    "oi_zscore_14d", "oi_zscore_30d",
    "oi_momentum", "price_oi_div", "oi_exchange_div",
    "leverage_stress", "leverage_expansion",
    "oi_vol_ratio", "oi_extreme",
    "funding_zscore_14d",
]


def section3_correlation(df: pd.DataFrame) -> pd.DataFrame:
    print("\nSection 3: Correlation analysis...")
    rows = []
    for sig in OI_SIGNALS:
        if sig not in df.columns:
            continue
        x = df[sig]
        row = {"signal": sig}
        for h in HORIZONS:
            tgt = f"btc_ret_{h}d"
            if tgt not in df.columns:
                continue
            c, p, lag = pearson_lag(x, df[tgt], max_lag=21)
            row[f"corr_{h}d"] = round(c, 4) if not np.isnan(c) else np.nan
            row[f"p_{h}d"]    = round(p, 4) if not np.isnan(p) else np.nan
            row[f"lag_{h}d"]  = lag
        rows.append(row)

    res = pd.DataFrame(rows).set_index("signal")

    # display table — sort by |corr_5d|
    disp = res.copy()
    disp["abs_corr_5d"] = disp["corr_5d"].abs()
    disp = disp.sort_values("abs_corr_5d", ascending=False).drop(columns="abs_corr_5d")

    print()
    print("─" * 72)
    print("  Pearson Correlation with BTC Forward Returns  (sorted by |corr_5d|)")
    print("─" * 72)
    table_rows = []
    for sig, row in disp.iterrows():
        table_rows.append([
            sig,
            f"{row.get('corr_5d', np.nan):+.4f}",
            f"{row.get('p_5d', np.nan):.4f}",
            f"{int(row.get('lag_5d', 0))}d",
            f"{row.get('corr_14d', np.nan):+.4f}",
            f"{row.get('p_14d', np.nan):.4f}",
        ])
    print(tabulate(table_rows,
                   headers=["Signal", "Corr_5d", "p_5d", "BestLag", "Corr_14d", "p_14d"],
                   tablefmt="simple"))
    return res


# ── section 4: price×OI divergence deep dive ──────────────────────────────
def section4_divergence(df: pd.DataFrame):
    print("\nSection 4: Price×OI divergence deep dive...")

    div  = df["price_oi_div"]
    ret5 = df["btc_ret_5d"]

    # quartile buckets
    q25, q75 = div.quantile(0.25), div.quantile(0.75)
    labels = {
        "Low div (<Q25)":   div < q25,
        "Mid div (Q25-Q75)": (div >= q25) & (div <= q75),
        "High div (>Q75)":  div > q75,
    }

    table_rows = []
    for label, mask in labels.items():
        sub = ret5[mask].dropna()
        if len(sub) < 5:
            continue
        _, p = stats.ttest_1samp(sub, 0)
        table_rows.append([
            label,
            len(sub),
            f"{sub.mean() * 100:+.2f}%",
            f"{sub.std() * 100:.2f}%",
            f"{(sub > 0).mean() * 100:.0f}%",
            f"{p:.4f}",
        ])

    print()
    print("─" * 72)
    print("  BTC 5d Return by price_oi_div quartile")
    print("─" * 72)
    print(tabulate(table_rows,
                   headers=["Bucket", "N", "Mean_5d", "Std", "P(>0)", "p-val"],
                   tablefmt="simple"))

    # Divergence sign event study
    high_div = div > q75
    low_div  = div < q25
    print(f"\n  Divergence interpretation:")
    print(f"    High div (price↑ OI↓ = short covering): n={high_div.sum()} "
          f"→ avg BTC 5d = {ret5[high_div].mean() * 100:+.2f}%")
    print(f"    Low  div (price↑ OI↑ = fragile longs):  n={low_div.sum()} "
          f"→ avg BTC 5d = {ret5[low_div].mean() * 100:+.2f}%")


# ── section 5: regime separation ──────────────────────────────────────────
def section5_regime(df: pd.DataFrame):
    print("\nSection 5: Regime separation analysis...")

    close = df["close"]
    ret21 = close.pct_change(21)
    df    = df.copy()
    df["regime"] = np.where(ret21 > 0.05, "Bull",
                   np.where(ret21 < -0.05, "Bear", "Sideways"))

    counts = df["regime"].value_counts().to_dict()
    print(f"  Regime distribution: {counts}")

    table_rows = []
    for sig in OI_SIGNALS:
        if sig not in df.columns:
            continue
        bull = df.loc[df["regime"] == "Bull",     sig].dropna()
        bear = df.loc[df["regime"] == "Bear",     sig].dropna()
        side = df.loc[df["regime"] == "Sideways", sig].dropna()
        pooled_std = df[sig].std()
        if pooled_std == 0 or np.isnan(pooled_std):
            continue
        sep = abs(bull.mean() - bear.mean()) / pooled_std if len(bull) > 0 and len(bear) > 0 else np.nan
        table_rows.append([
            sig,
            f"{bull.mean():+.4f}" if len(bull) > 0 else "N/A",
            f"{bear.mean():+.4f}" if len(bear) > 0 else "N/A",
            f"{side.mean():+.4f}" if len(side) > 0 else "N/A",
            f"{sep:+.4f}" if not np.isnan(sep) else "N/A",
        ])

    table_rows.sort(key=lambda r: abs(float(r[4])) if r[4] != "N/A" else 0, reverse=True)

    print()
    print("─" * 72)
    print("  Signal Values by Regime  (Separation = |Bull−Bear| / σ)")
    print("─" * 72)
    print(tabulate(table_rows,
                   headers=["Signal", "Bull_mean", "Bear_mean", "Sideways_mean", "Separation"],
                   tablefmt="simple"))
    return df


# ── section 5.5: reversal detection ───────────────────────────────────────
def section55_reversal(df: pd.DataFrame):
    print("\nSection 5.5: Reversal detection...")

    ret5 = df["btc_ret_5d"]
    # future sign flip: does the 5d return change sign vs the prior 5d return?
    future_sign_flip = np.sign(ret5) != np.sign(ret5.shift(5))

    n_flip    = future_sign_flip.sum()
    n_nflip   = (~future_sign_flip).sum()
    print(f"  Regime transitions detected: {n_flip}  /  {len(future_sign_flip)} rows")

    check_sigs = ["oi_change_3d", "price_oi_div", "oi_vol_ratio",
                  "oi_momentum", "funding_zscore_14d", "leverage_stress"]

    table_rows = []
    for sig in check_sigs:
        if sig not in df.columns:
            continue
        flip_mean  = df.loc[future_sign_flip, sig].mean()
        nflip_mean = df.loc[~future_sign_flip, sig].mean()
        diff       = flip_mean - nflip_mean
        _, p = stats.ttest_ind(
            df.loc[future_sign_flip, sig].dropna(),
            df.loc[~future_sign_flip, sig].dropna(),
            equal_var=False
        ) if df.loc[future_sign_flip, sig].dropna().shape[0] > 1 else (np.nan, np.nan)
        table_rows.append([sig,
                           f"{flip_mean:+.4f}",
                           f"{nflip_mean:+.4f}",
                           f"{diff:+.4f}",
                           f"{p:.4f}" if not np.isnan(p) else "N/A"])

    print()
    print("─" * 72)
    print("  OI signal values BEFORE regime flip vs stable periods")
    print("─" * 72)
    print(tabulate(table_rows,
                   headers=["Signal", "At_flip", "Stable", "Diff", "p-val (Welch)"],
                   tablefmt="simple"))
    print("\n  Interpretation:")
    print("    At_flip > Stable  → signal elevated before reversals (leading indicator)")
    print("    p < 0.10          → statistically meaningful difference")


# ── section 6: sample size warning ────────────────────────────────────────
def section6_sample_size(df: pd.DataFrame):
    print("\nSection 6: Sample size assessment...")

    n  = len(df)
    print()
    print("  " + "═" * 60)
    print(f"  Window: {df.index[0].date()} → {df.index[-1].date()}  ({n} days)")
    print()
    if n < 120:
        print("  STATUS: CRITICAL — Insufficient data for reliable inference")
        print("  Minimum required: 120 days. All results UNRELIABLE.")
    elif n < 180:
        print("  STATUS: WARNING — Borderline sample size")
        print("  Treat significant results as preliminary signals only.")
    else:
        print("  STATUS: OK — Minimum threshold for exploratory analysis met")
        print("  Results are directional; confirm at 12+ months.")

    print()
    print(f"  Effective observations per signal:")
    for sig in ["oi_change_3d", "price_oi_div", "oi_vol_ratio"]:
        if sig in df.columns:
            n_valid = df[sig].dropna().shape[0]
            print(f"    {sig:<25s}: {n_valid} valid rows")

    print()
    print("  STANDARD PLAN DECISION GATE:")
    print("  Re-evaluate CoinGlass upgrade when:")
    print("    1. Organic data hits 12+ months  (≈2026-09)")
    print("    2. Any signal clears corr>0.15 AND p<0.10")
    print("  " + "═" * 60)


# ── section 7: recommendation ─────────────────────────────────────────────
def section7_recommendation(df_raw: pd.DataFrame, corr_res: pd.DataFrame):
    print("\nSection 7: Generating recommendation...")

    INCLUDE_CORR_MIN = 0.10
    INCLUDE_SEP_MIN  = 0.30

    # compute regime separation for recommendation
    close = df_raw["close"]
    ret21 = close.pct_change(21)
    regime = np.where(ret21 > 0.05, "Bull",
              np.where(ret21 < -0.05, "Bear", "Sideways"))

    sep_map = {}
    for sig in OI_SIGNALS:
        if sig not in df_raw.columns:
            continue
        bull_vals = df_raw.loc[regime == "Bull",  sig].dropna()
        bear_vals = df_raw.loc[regime == "Bear",  sig].dropna()
        pooled_std = df_raw[sig].std()
        if pooled_std > 0 and len(bull_vals) > 0 and len(bear_vals) > 0:
            sep_map[sig] = abs(bull_vals.mean() - bear_vals.mean()) / pooled_std

    include_rows = []
    exclude_rows = []

    for sig in OI_SIGNALS:
        if sig not in corr_res.index:
            continue
        row   = corr_res.loc[sig]
        corr5 = abs(row.get("corr_5d", np.nan))
        p5    = row.get("p_5d", np.nan)
        lag5  = row.get("lag_5d", 0)
        sep   = sep_map.get(sig, np.nan)

        meets_corr = (not np.isnan(corr5)) and corr5 >= INCLUDE_CORR_MIN
        meets_sep  = (not np.isnan(sep))   and sep  >= INCLUDE_SEP_MIN

        if meets_corr and meets_sep:
            include_rows.append([
                sig,
                f"{row.get('corr_5d', np.nan):+.4f}",
                f"{int(lag5)}d",
                f"{sep:+.4f}",
            ])
        else:
            reasons = []
            if not meets_corr:
                reasons.append(f"|corr_5d|={corr5:.3f}<{INCLUDE_CORR_MIN}")
            if not meets_sep:
                sep_str = f"{sep:.3f}" if not np.isnan(sep) else "N/A"
                reasons.append(f"sep={sep_str}<{INCLUDE_SEP_MIN}")
            exclude_rows.append([sig, "; ".join(reasons)])

    print()
    print("═" * 68)
    print("  OPEN INTEREST SIGNAL RECOMMENDATION")
    print(f"  Criteria: |corr_5d| ≥ {INCLUDE_CORR_MIN}  AND  separation ≥ {INCLUDE_SEP_MIN}")
    print("═" * 68)

    print()
    print("─" * 68)
    print("  INCLUDE")
    print("─" * 68)
    if include_rows:
        print(tabulate(include_rows,
                       headers=["Signal", "Corr_5d", "BestLag", "Separation"],
                       tablefmt="simple"))
    else:
        print("  None — no OI signal meets both criteria with current data.")

    print()
    print("─" * 68)
    print("  EXCLUDE")
    print("─" * 68)
    if exclude_rows:
        print(tabulate(exclude_rows, headers=["Signal", "Reason"], tablefmt="simple"))

    print()
    print("─" * 68)
    print("  COINGLASS STANDARD PLAN — UPGRADE DECISION")
    print("─" * 68)
    best_corr_sig = corr_res["corr_5d"].abs().idxmax() if "corr_5d" in corr_res else None
    best_corr_val = corr_res.loc[best_corr_sig, "corr_5d"] if best_corr_sig else np.nan

    print(f"  Best |corr_5d| found : {abs(best_corr_val):.4f}  ({best_corr_sig})")
    print(f"  Upgrade threshold    : corr > 0.15  (arbitrary — same as positioning)")
    print()
    if abs(best_corr_val) >= 0.15:
        print("  Recommendation: CONSIDER upgrade — at least one OI signal crosses")
        print("  the 0.15 threshold. Evaluate Standard plan ROI before committing.")
    else:
        print("  Recommendation: WAIT — no OI signal clears corr>0.15.")
        print("  Re-evaluate organically at 12+ months of history (≈2026-09).")
        print("  Standard plan adds liquidation data — may change picture, but")
        print("  existing OI/Funding signals are insufficient to justify cost now.")

    print("═" * 68)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 70)
    print("  Open Interest Correlation Analysis")
    print(f"  Window: {START.date()} → {TODAY.date()}")
    print("  [NOTE] CoinGlass Hobbyist: OI + Funding only. No liquidations.")
    print("=" * 70)

    df        = section0_load()
    df        = section1_signals(df)
    section2_visual(df)
    corr_res  = section3_correlation(df)
    section4_divergence(df)
    df_regime = section5_regime(df)
    section55_reversal(df)
    section6_sample_size(df)
    section7_recommendation(df, corr_res)

    out_csv = OUT_DIR / "oi_results.csv"
    corr_res.to_csv(out_csv)
    print(f"\nResults saved → {out_csv}")
    print("=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
