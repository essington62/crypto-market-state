"""
R11 Baseline Re-run — BTC HMM regime model, frozen configuration.

Frozen R11 Config:
  n_states        : 2
  covariance_type : diag
  n_iter          : 1000
  random_state    : 42
  features        : [log_return, vol_short, vol_ratio, drawdown, volume_z, slope_21d]
  state_ordering  : drawdown ascending (most negative = Bear, least negative = Bull)
  TC              : 12 bps round-trip
  entry           : same-day
  data_layer      : data/03_primary/spot/daily/

Splits: read from conf/base/parameters.yml (walkforward.split_3)

Outputs:
  data/06_reports/r11_baseline.json
  data/06_reports/r11_baseline.md
  data/06_reports/plots/r11_baseline_equity_curve.png
  data/06_reports/plots/r11_baseline_regime_sequence.png
  data/06_reports/plots/r11_baseline_drawdown.png
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
CONF_FILE   = ROOT / "conf" / "base" / "parameters.yml"
DATA_INPUT  = ROOT / "data" / "03_primary" / "spot" / "daily"
REPORTS_DIR = ROOT / "data" / "06_reports"
PLOTS_DIR   = REPORTS_DIR / "plots"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── R11 Frozen Config ──────────────────────────────────────────────────────────
# These values are frozen and must not be changed.
R11_CONFIG = {
    "n_states":        2,
    "covariance_type": "diag",
    "n_iter":          1000,
    "random_state":    42,
    "features":        ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"],
    "state_ordering":  "drawdown",     # most negative drawdown = Bear
    "entry":           "same_day",
    "data_layer":      "03_primary",
}

TC         = 0.0012   # 12 bps round-trip
PERIODS_YR = 252      # daily data


# ── Config loader ──────────────────────────────────────────────────────────────
def _load_split3() -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, object]:
    """Load split_3 dates from parameters.yml."""
    with open(CONF_FILE) as fh:
        cfg = yaml.safe_load(fh)

    split3 = next(s for s in cfg["walkforward"]["splits"] if s["name"] == "split_3")
    train_start = pd.Timestamp(str(split3["train_start"]), tz="UTC")
    train_end   = pd.Timestamp(str(split3["train_end"]),   tz="UTC")
    test_start  = pd.Timestamp(str(split3["test_start"]),  tz="UTC")
    test_end_raw = split3["test_end"]
    return train_start, train_end, test_start, test_end_raw


# ── HMM state ordering ─────────────────────────────────────────────────────────
def _order_by_drawdown(model: GaussianHMM, features: list[str]) -> dict[int, int]:
    """
    Order states by mean drawdown (ascending).
    State with most negative drawdown → Bear (0).
    State with least negative drawdown → Bull (1).
    Fallback: log_return ascending if drawdown not in features.
    """
    if "drawdown" in features:
        idx = features.index("drawdown")
    else:
        idx = features.index("log_return") if "log_return" in features else 0
    vals    = {s: model.means_[s][idx] for s in range(model.n_components)}
    ordered = sorted(vals, key=vals.get)   # ascending: Bear first
    return {orig: new for new, orig in enumerate(ordered)}


# ── Run durations ──────────────────────────────────────────────────────────────
def _run_durations(states: np.ndarray, target: int) -> float:
    durations, count = [], 0
    for s in states:
        if s == target:
            count += 1
        else:
            if count > 0:
                durations.append(count)
                count = 0
    if count > 0:
        durations.append(count)
    return float(np.mean(durations)) if durations else 0.0


# ── Backtest ───────────────────────────────────────────────────────────────────
def _run_backtest(
    test_df: pd.DataFrame,
    states: np.ndarray,
) -> dict:
    """
    Same-day entry: signal on day t → position applied on day t.
    Bull (state=1) → Long, Bear (state=0) → Exit.
    TC applied on position changes.
    """
    log_ret  = test_df["log_return"].values
    n        = len(log_ret)
    position = (states == 1).astype(float)

    prev_pos   = np.concatenate([[0.0], position[:-1]])
    tc_cost    = TC * np.abs(position - prev_pos)
    net_ret    = position * log_ret - tc_cost
    bnh_ret    = log_ret

    equity_strat = pd.Series((1 + net_ret).cumprod(), index=test_df.index)
    equity_bnh   = pd.Series((1 + bnh_ret).cumprod(), index=test_df.index)

    def _dd(eq):
        peak = eq.cummax()
        return (eq - peak) / peak

    dd_strat = _dd(equity_strat)
    dd_bnh   = _dd(equity_bnh)

    n_years = n / PERIODS_YR
    cagr    = float(equity_strat.iloc[-1] ** (1 / n_years) - 1)
    sharpe  = float((net_ret.mean() / (net_ret.std() + 1e-10)) * np.sqrt(PERIODS_YR))
    max_dd  = float(dd_strat.min())

    active     = net_ret[net_ret != 0]
    hit_rate   = float((active > 0).mean()) if len(active) > 0 else float("nan")
    gains      = active[active > 0].sum()
    losses     = abs(active[active < 0].sum())
    pf         = float(gains / losses) if losses > 0 else float("nan")
    n_trades   = int((np.abs(position - prev_pos) > 0).sum())

    return {
        "cagr":          cagr,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "hit_rate":      hit_rate,
        "profit_factor": pf,
        "n_trades":      n_trades,
        "n_obs":         n,
        "equity_strat":  equity_strat,
        "equity_bnh":    equity_bnh,
        "dd_strat":      dd_strat,
        "dd_bnh":        dd_bnh,
        "net_ret":       net_ret,
        "position":      position,
    }


# ── Plots ──────────────────────────────────────────────────────────────────────
def _plot_equity_curve(bt: dict, train_start: pd.Timestamp, test_start: pd.Timestamp) -> Path:
    eq   = bt["equity_strat"]
    bnh  = bt["equity_bnh"]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("R11 Baseline — Equity Curve (BTC, test 2025-01-01→)", fontsize=12)

    ax.plot(eq.index, eq.values, color="steelblue", linewidth=1.4,
            label=f"R11  CAGR={bt['cagr']:+.2%}  Sharpe={bt['sharpe']:+.3f}")
    ax.plot(bnh.index, bnh.values, color="gray", linewidth=0.9, linestyle="--", alpha=0.7,
            label="BTC Buy & Hold")
    ax.axhline(1, color="black", linewidth=0.5, linestyle=":")
    ax.axvline(test_start, color="orange", linewidth=1.0, linestyle="--", alpha=0.8, label="Test start")
    ax.set_ylabel("Equity (base 1)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = PLOTS_DIR / "r11_baseline_equity_curve.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def _plot_regime_sequence(
    test_df: pd.DataFrame,
    states: np.ndarray,
    bt: dict,
    test_start: pd.Timestamp,
) -> Path:
    idx = test_df.index[:len(states)]
    eq  = bt["equity_strat"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("R11 Baseline — Regime Sequence (BTC, test 2025-01-01→)", fontsize=12)

    bull_mask = states == 1
    ax1.plot(idx, eq.values, color="steelblue", linewidth=1.0, label="Strategy equity")
    ax1.fill_between(idx, 1, eq.values, where=bull_mask,
                     color="green", alpha=0.20, label="Bull regime")
    ax1.fill_between(idx, 1, eq.values, where=~bull_mask,
                     color="red",   alpha=0.15, label="Bear regime")
    ax1.axhline(1, color="black", linewidth=0.5, linestyle=":")
    ax1.set_ylabel("Equity")
    ax1.legend(fontsize=9)

    ax2.step(idx, states, where="post", color="navy", linewidth=0.8)
    ax2.fill_between(idx, 0, states, step="post", color="green", alpha=0.4, label="Bull (1)")
    ax2.fill_between(idx, states, 1, step="post", color="red",   alpha=0.2, label="Bear (0)")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Bear", "Bull"])
    ax2.set_ylabel("Regime")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = PLOTS_DIR / "r11_baseline_regime_sequence.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def _plot_drawdown(bt: dict) -> Path:
    dd_s = bt["dd_strat"]
    dd_b = bt["dd_bnh"]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("R11 Baseline — Drawdown (BTC, test 2025-01-01→)", fontsize=12)

    ax.fill_between(dd_s.index, dd_s.values, 0, color="steelblue", alpha=0.5,
                    label=f"R11 max DD={bt['max_drawdown']:+.2%}")
    ax.fill_between(dd_b.index, dd_b.values, 0, color="gray", alpha=0.25,
                    label="BTC B&H")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = PLOTS_DIR / "r11_baseline_drawdown.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


# ── Report writers ─────────────────────────────────────────────────────────────
def _write_json(payload: dict) -> Path:
    def _clean(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    clean = {k: _clean(v) for k, v in payload.items()}
    out = REPORTS_DIR / "r11_baseline.json"
    with open(out, "w") as fh:
        json.dump(clean, fh, indent=2, default=str)
    return out


def _write_markdown(payload: dict, model: GaussianHMM, mapa: dict, features: list[str]) -> Path:
    bt          = payload
    trans       = model.transmat_
    idx_bear    = next(k for k, v in mapa.items() if v == 0)
    idx_bull    = next(k for k, v in mapa.items() if v == 1)
    lr_idx      = features.index("log_return")

    lines = [
        "# R11 Baseline — BTC (Frozen)",
        "",
        f"**Generated:** {date.today().isoformat()}",
        f"**Status:** FROZEN BASELINE — do not modify",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| n_states | 2 |",
        f"| covariance_type | diag |",
        f"| n_iter | 1000 |",
        f"| random_state | 42 |",
        f"| features | {', '.join(features)} |",
        f"| state_ordering | drawdown ascending (most negative = Bear) |",
        f"| TC | 12 bps round-trip |",
        f"| entry | same-day |",
        f"| data_layer | 03_primary |",
        "",
        "## Periods",
        "",
        f"| Period | Start | End | N days |",
        f"|--------|-------|-----|--------|",
        f"| Train | {payload['train_start']} | {payload['train_end']} | {payload['n_train']} |",
        f"| Test  | {payload['test_start']}  | {payload['test_end']}  | {payload['n_test']}  |",
        "",
        "## HMM Diagnostics",
        "",
        f"| State | Label | Mean log_return |",
        f"|-------|-------|-----------------|",
        f"| {idx_bear} | Bear | {model.means_[idx_bear][lr_idx]:+.6f} |",
        f"| {idx_bull} | Bull | {model.means_[idx_bull][lr_idx]:+.6f} |",
        "",
        "**Transition matrix:**",
        "```",
        f"Bear→Bear: {trans[idx_bear, idx_bear]:.4f}   Bear→Bull: {trans[idx_bear, idx_bull]:.4f}",
        f"Bull→Bear: {trans[idx_bull, idx_bear]:.4f}   Bull→Bull: {trans[idx_bull, idx_bull]:.4f}",
        "```",
        "",
        "## Regime Summary (test period)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Bull days | {payload['bull_days']} ({payload['bull_pct']:.1%}) |",
        f"| Bear days | {payload['bear_days']} ({payload['bear_pct']:.1%}) |",
        f"| Transitions | {payload['transitions']} |",
        f"| Avg Bull duration | {payload['bull_duration']:.1f} days |",
        f"| Avg Bear duration | {payload['bear_duration']:.1f} days |",
        "",
        "## Strategy Metrics (test period, TC=12bps, next-day entry)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| CAGR | {payload['cagr']:+.2%} |",
        f"| Sharpe | {payload['sharpe']:+.3f} |",
        f"| Max Drawdown | {payload['max_drawdown']:+.2%} |",
        f"| Hit Rate | {payload['hit_rate']:+.2%} |",
        f"| Profit Factor | {payload['profit_factor']:.3f} |",
        f"| N Trades | {payload['n_trades']} |",
        "",
        "## Plots",
        "",
        "- `plots/r11_baseline_equity_curve.png`",
        "- `plots/r11_baseline_regime_sequence.png`",
        "- `plots/r11_baseline_drawdown.png`",
    ]

    out = REPORTS_DIR / "r11_baseline.md"
    out.write_text("\n".join(lines))
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Dates from config ─────────────────────────────────────────────────────
    train_start, train_end, test_start, test_end_raw = _load_split3()

    # ── Load data ─────────────────────────────────────────────────────────────
    btc_path = DATA_INPUT / "BTCUSDT.parquet"
    if not btc_path.exists():
        print(f"ERROR: {btc_path} not found.")
        print("  Run: conda run -n crypto_market_state kedro run --pipeline primary.spot.crypto")
        sys.exit(1)

    df = pd.read_parquet(btc_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    test_end = df.index.max() if test_end_raw is None else pd.Timestamp(str(test_end_raw), tz="UTC")

    features = R11_CONFIG["features"]

    # ── Pre-run validation ────────────────────────────────────────────────────
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"ERROR: Required features missing from data: {missing}")
        sys.exit(1)

    if df.index.tz is None or str(df.index.tz) != "UTC":
        print(f"ERROR: Index timezone is {df.index.tz}, expected UTC")
        sys.exit(1)

    if df.index.min() > pd.Timestamp("2023-01-01", tz="UTC"):
        print(f"ERROR: Data starts at {df.index.min().date()}, need data from 2023-01-01")
        sys.exit(1)

    # ── Prepare train / test ──────────────────────────────────────────────────
    df_clean  = df.sort_index().dropna(subset=features)
    train_df  = df_clean.loc[train_start:train_end]
    test_df   = df_clean.loc[test_start:test_end].dropna(subset=["log_return"])

    if len(train_df) < 30:
        print(f"ERROR: Train set only {len(train_df)} rows after dropna")
        sys.exit(1)

    # ── Train HMM ────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test  = scaler.transform(test_df[features])

    model = GaussianHMM(
        n_components=R11_CONFIG["n_states"],
        covariance_type=R11_CONFIG["covariance_type"],
        n_iter=R11_CONFIG["n_iter"],
        random_state=R11_CONFIG["random_state"],
    )
    model.fit(X_train)

    mapa       = _order_by_drawdown(model, features)
    raw_states = model.predict(X_test)
    states     = np.array([mapa[s] for s in raw_states])

    # ── Regime stats ──────────────────────────────────────────────────────────
    bull_days   = int((states == 1).sum())
    bear_days   = int((states == 0).sum())
    n_test      = len(states)
    transitions = int((np.diff(states) != 0).sum())
    bull_dur    = _run_durations(states, 1)
    bear_dur    = _run_durations(states, 0)

    # ── Backtest ──────────────────────────────────────────────────────────────
    bt = _run_backtest(test_df, states)

    # ── HMM diagnostic values ─────────────────────────────────────────────────
    lr_idx   = features.index("log_return")
    idx_bear = next(k for k, v in mapa.items() if v == 0)
    idx_bull = next(k for k, v in mapa.items() if v == 1)
    trans    = model.transmat_

    # ── Console report ────────────────────────────────────────────────────────
    sep = "=" * 50
    div = "-" * 50
    print(f"\n{sep}")
    print("R11 BASELINE — BTC")
    print(sep)
    print(f"Train period : {train_start.date()} → {train_end.date()}  (N={len(train_df)} days)")
    print(f"Test period  : {test_start.date()} → {test_end.date()}  (N={n_test} days)")
    print(f"Features     : {features}")
    print(f"n_states     : {R11_CONFIG['n_states']}")
    print(f"cov_type     : {R11_CONFIG['covariance_type']}")
    print(f"random_state : {R11_CONFIG['random_state']}")
    print(div)
    print("HMM Diagnostics:")
    print(f"  State {idx_bear} mean log_return : {model.means_[idx_bear][lr_idx]:+.6f}")
    print(f"  State {idx_bull} mean log_return : {model.means_[idx_bull][lr_idx]:+.6f}")
    print(f"  Bear state assigned to  : State {idx_bear}")
    print(f"  Bull state assigned to  : State {idx_bull}")
    print( "  Transition matrix       :")
    print(f"    [[{trans[0,0]:.4f}  {trans[0,1]:.4f}]")
    print(f"     [{trans[1,0]:.4f}  {trans[1,1]:.4f}]]")
    print(div)
    print("Regime Summary (test period):")
    print(f"  Bull days   : {bull_days}  ({bull_days/n_test:.1%})")
    print(f"  Bear days   : {bear_days}  ({bear_days/n_test:.1%})")
    print(f"  Transitions : {transitions}")
    print(f"  Avg Bull duration : {bull_dur:.1f} days")
    print(f"  Avg Bear duration : {bear_dur:.1f} days")
    print(div)
    print("Strategy Metrics (test period, TC=12bps):")
    print(f"  CAGR          : {bt['cagr']:+.2%}")
    print(f"  Sharpe        : {bt['sharpe']:+.3f}")
    print(f"  Max Drawdown  : {bt['max_drawdown']:+.2%}")
    print(f"  Hit Rate      : {bt['hit_rate']:+.2%}")
    print(f"  Profit Factor :  {bt['profit_factor']:.3f}")
    print(f"  N Trades      : {bt['n_trades']}")
    print(sep)

    # ── Success criteria check ────────────────────────────────────────────────
    checks = [
        ("Sharpe ≥ 0.70",      bt["sharpe"]        >= 0.70),
        ("CAGR ≥ 10%",         bt["cagr"]           >= 0.10),
        ("MaxDD ≥ -16%",       bt["max_drawdown"]   >= -0.16),
        ("Bull drawdown > Bear drawdown (less negative)",
         model.means_[idx_bull][features.index("drawdown")]
         > model.means_[idx_bear][features.index("drawdown")]),
    ]
    print("\nSuccess Criteria:")
    all_pass = True
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {label}")

    if not all_pass:
        print("\nWARNING: One or more success criteria failed.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    p1 = _plot_equity_curve(bt, train_start, test_start)
    p2 = _plot_regime_sequence(test_df, states, bt, test_start)
    p3 = _plot_drawdown(bt)
    print(f"  Saved: {p1.name}")
    print(f"  Saved: {p2.name}")
    print(f"  Saved: {p3.name}")

    # ── Save JSON / MD ────────────────────────────────────────────────────────
    payload = {
        "asset":          "BTC",
        "generated":      str(date.today()),
        "frozen_baseline": True,
        "train_start":    str(train_start.date()),
        "train_end":      str(train_end.date()),
        "test_start":     str(test_start.date()),
        "test_end":       str(test_end.date()),
        "n_train":        len(train_df),
        "n_test":         n_test,
        "features":       features,
        "n_states":       R11_CONFIG["n_states"],
        "covariance_type": R11_CONFIG["covariance_type"],
        "random_state":   R11_CONFIG["random_state"],
        "TC_bps":         TC * 10_000,
        "entry":          "same_day",
        "state_ordering": "drawdown_ascending",
        "data_layer":     R11_CONFIG["data_layer"],
        # regime
        "bull_days":      bull_days,
        "bear_days":      bear_days,
        "bull_pct":       bull_days / n_test,
        "bear_pct":       bear_days / n_test,
        "transitions":    transitions,
        "bull_duration":  bull_dur,
        "bear_duration":  bear_dur,
        # hmm
        "bear_state_idx": idx_bear,
        "bull_state_idx": idx_bull,
        "bear_mean_logret": float(model.means_[idx_bear][lr_idx]),
        "bull_mean_logret": float(model.means_[idx_bull][lr_idx]),
        "transmat":       trans.tolist(),
        # strategy
        "cagr":           bt["cagr"],
        "sharpe":         bt["sharpe"],
        "max_drawdown":   bt["max_drawdown"],
        "hit_rate":       bt["hit_rate"],
        "profit_factor":  bt["profit_factor"],
        "n_trades":       bt["n_trades"],
    }

    json_path = _write_json(payload)
    md_path   = _write_markdown(payload, model, mapa, features)
    print(f"\nOutputs:")
    print(f"  {json_path.relative_to(ROOT)}")
    print(f"  {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
