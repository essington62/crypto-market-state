"""
R11 Position Sizing — MaxDD Reduction Analysis.

Strategies:
  0 — Baseline          : 100% Bull / 0% Bear
  1 — Fixed Fractional  : f× Bull (f=0.25, 0.50, 0.75)
  2 — Vol Targeting     : σ_target / σ_20d_ann (capped at 1), Bear=0
  3 — Regime Confidence : P(Bull|obs) as size, Bear=0  (+ floor variant)
  4 — Drawdown Control  : scale back on drawdown thresholds
  5 — Hybrid            : Vol Targeting + Drawdown Control overlay

Frozen R11 config (do not modify):
  data_path  : data/03_primary/spot/daily/BTCUSDT.parquet
  cov_type   : diag  |  ordering : drawdown ascending
  features   : [log_return, vol_short, vol_ratio, drawdown, volume_z, slope_21d]
  train      : 2023-01-01 → 2024-12-31  |  test : 2025-01-01 →
  TC         : 12 bps round-trip  |  entry : same-day

Criteria for success: MaxDD > -10%, Sharpe > 0.60, CAGR > 8%

Outputs:
  data/06_reports/r11_position_sizing.json
  data/06_reports/r11_position_sizing_report.md
  data/06_reports/plots/r11_sizing_equity_curves.png
  data/06_reports/plots/r11_sizing_sharpe_vs_maxdd.png
  data/06_reports/plots/r11_sizing_drawdown.png
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
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

# ── Frozen R11 Config ──────────────────────────────────────────────────────────
R11_FEATURES     = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]
R11_N_STATES     = 2
R11_COV_TYPE     = "diag"
R11_N_ITER       = 1000
R11_RANDOM_STATE = 42

TC         = 0.0012   # 12 bps round-trip
PERIODS_YR = 252

# ── Success criteria ───────────────────────────────────────────────────────────
MAXDD_TARGET  = -0.10   # MaxDD must be > -10% (less than 10% down)
SHARPE_FLOOR  = 0.60
CAGR_FLOOR    = 0.08

# ── Strategy colours (stable across plots) ────────────────────────────────────
STRAT_COLORS = {
    "0": "#555555",   # gray
    "1": "#2196F3",   # blue
    "2": "#4CAF50",   # green
    "3": "#FF9800",   # orange
    "4": "#F44336",   # red
    "5": "#9C27B0",   # purple
}


# ── Config loader ──────────────────────────────────────────────────────────────
def _load_split3():
    with open(CONF_FILE) as fh:
        cfg = yaml.safe_load(fh)
    split3 = next(s for s in cfg["walkforward"]["splits"] if s["name"] == "split_3")
    ts = lambda k: pd.Timestamp(str(split3[k]), tz="UTC")
    return ts("train_start"), ts("train_end"), ts("test_start"), split3["test_end"]


# ── HMM state ordering ─────────────────────────────────────────────────────────
def _ordenar_estados(model: GaussianHMM, features: list[str]) -> dict[int, int]:
    idx = features.index("drawdown") if "drawdown" in features else 0
    vals = {s: model.means_[s][idx] for s in range(model.n_components)}
    ordered = sorted(vals, key=vals.get)
    return {orig: new for new, orig in enumerate(ordered)}


# ── Setup: train HMM once, return everything needed for sizing ─────────────────
def _setup() -> dict:
    train_start, train_end, test_start, test_end_raw = _load_split3()

    df = pd.read_parquet(DATA_INPUT / "BTCUSDT.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    test_end = df.index.max() if test_end_raw is None else pd.Timestamp(str(test_end_raw), tz="UTC")

    features = [f for f in R11_FEATURES if f in df.columns]
    df_c = df.sort_index().dropna(subset=features)
    train_df = df_c.loc[train_start:train_end]
    test_df  = df_c.loc[test_start:test_end].dropna(subset=["log_return"])

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test  = scaler.transform(test_df[features])

    model = GaussianHMM(
        n_components=R11_N_STATES, covariance_type=R11_COV_TYPE,
        n_iter=R11_N_ITER, random_state=R11_RANDOM_STATE,
    )
    model.fit(X_train)

    mapa       = _ordenar_estados(model, features)
    raw_states = model.predict(X_test)
    states     = np.array([mapa[s] for s in raw_states])

    # Posterior P(Bull | obs_t) — needed for Strategy 3
    raw_proba  = model.predict_proba(X_test)
    inv_mapa   = {v: k for k, v in mapa.items()}
    bull_prob  = raw_proba[:, inv_mapa[1]]

    # Annualised 20-day rolling vol computed on full history → no NaN at test start
    vol_daily = df_c["log_return"].rolling(20).std()
    vol_ann   = (vol_daily * np.sqrt(PERIODS_YR)).reindex(test_df.index).values

    return {
        "states":    states,
        "bull_prob": bull_prob,
        "log_ret":   test_df["log_return"].values,
        "vol_ann":   vol_ann,
        "test_idx":  test_df.index,
        "n_train":   len(train_df),
        "n_test":    len(test_df),
        "train_start": train_start,
        "test_start":  test_start,
        "test_end":    test_end,
    }


# ── Generic backtest ───────────────────────────────────────────────────────────
def _backtest(positions: np.ndarray, log_ret: np.ndarray, test_idx: pd.DatetimeIndex) -> dict:
    """Standard backtest — same-day, proportional TC on position changes."""
    prev_pos = np.concatenate([[0.0], positions[:-1]])
    tc_cost  = TC * np.abs(positions - prev_pos)
    net_ret  = positions * log_ret - tc_cost
    equity   = (1 + net_ret).cumprod()
    peak     = np.maximum.accumulate(equity)
    dd       = (equity - peak) / peak

    n       = len(net_ret)
    n_years = n / PERIODS_YR
    cagr    = float(equity[-1] ** (1 / n_years) - 1)
    sharpe  = float((net_ret.mean() / (net_ret.std() + 1e-10)) * np.sqrt(PERIODS_YR))
    max_dd  = float(dd.min())

    active   = net_ret[net_ret != 0]
    hit_rate = float((active > 0).mean()) if len(active) > 0 else float("nan")
    gains    = active[active > 0].sum()
    losses   = abs(active[active < 0].sum())
    pf       = float(gains / losses) if losses > 0 else float("nan")
    n_trades = int((np.abs(positions - prev_pos) > 0).sum())

    return {
        "cagr":          cagr,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "hit_rate":      hit_rate,
        "profit_factor": pf,
        "n_trades":      n_trades,
        "avg_position":  float(positions.mean()),
        "_equity":       pd.Series(equity, index=test_idx),
        "_dd":           pd.Series(dd, index=test_idx),
    }


# ── Path-dependent iterative simulation (Strategies 4 & 5) ────────────────────
def _simulate_iterative(
    base_pos_fn,       # callable(t, states, vol_ann, curr_dd) → float in [0,1]
    states: np.ndarray,
    log_ret: np.ndarray,
    test_idx: pd.DatetimeIndex,
    **kwargs,
) -> dict:
    """
    Iterative backtest where position sizing depends on the live equity curve.
    TC included in equity evolution so drawdown state reflects real costs.
    """
    n        = len(log_ret)
    positions = np.zeros(n)
    equity    = np.empty(n)
    eq        = 1.0
    peak      = 1.0
    prev_pos  = 0.0

    for t in range(n):
        curr_dd = (eq / peak) - 1.0
        pos     = float(base_pos_fn(t, states, curr_dd, **kwargs))
        pos     = max(0.0, min(1.0, pos))

        tc   = TC * abs(pos - prev_pos)
        ret  = pos * log_ret[t] - tc
        eq   = eq * (1.0 + ret)
        peak = max(peak, eq)

        positions[t] = pos
        equity[t]    = eq
        prev_pos     = pos

    peak_arr = np.maximum.accumulate(equity)
    dd_arr   = (equity - peak_arr) / peak_arr

    # Recompute net_ret from positions for Sharpe / hit-rate / PF
    prev_pos_arr = np.concatenate([[0.0], positions[:-1]])
    tc_arr       = TC * np.abs(positions - prev_pos_arr)
    net_ret      = positions * log_ret - tc_arr

    n_years = n / PERIODS_YR
    cagr    = float(equity[-1] ** (1 / n_years) - 1)
    sharpe  = float((net_ret.mean() / (net_ret.std() + 1e-10)) * np.sqrt(PERIODS_YR))
    max_dd  = float(dd_arr.min())

    active   = net_ret[net_ret != 0]
    hit_rate = float((active > 0).mean()) if len(active) > 0 else float("nan")
    gains    = active[active > 0].sum()
    losses   = abs(active[active < 0].sum())
    pf       = float(gains / losses) if losses > 0 else float("nan")
    n_trades = int((np.abs(positions - prev_pos_arr) > 0).sum())

    return {
        "cagr":          cagr,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "hit_rate":      hit_rate,
        "profit_factor": pf,
        "n_trades":      n_trades,
        "avg_position":  float(positions.mean()),
        "_equity":       pd.Series(equity, index=test_idx),
        "_dd":           pd.Series(dd_arr,  index=test_idx),
    }


# ── Strategy position functions ────────────────────────────────────────────────

# Strategy 1 — Fixed Fractional
def _pos_fixed(states: np.ndarray, f: float) -> np.ndarray:
    return np.where(states == 1, f, 0.0)

# Strategy 2 — Vol Targeting
def _pos_vol_target(states: np.ndarray, vol_ann: np.ndarray, sigma_target: float) -> np.ndarray:
    safe_vol = np.where(vol_ann > 0, vol_ann, np.nan)
    size     = np.clip(sigma_target / safe_vol, 0.0, 1.0)
    size     = np.where(np.isnan(size), 0.0, size)
    return np.where(states == 1, size, 0.0)

# Strategy 3 — Regime Confidence
def _pos_confidence(states: np.ndarray, bull_prob: np.ndarray, floor: float | None) -> np.ndarray:
    if floor is not None:
        size = np.where(bull_prob >= floor, bull_prob, floor)
    else:
        size = bull_prob
    return np.where(states == 1, size, 0.0)

# Strategy 4 — Drawdown Control (iterative position function)
def _pos_fn_dd_control(t, states, curr_dd, *, threshold):
    base = 1.0 if states[t] == 1 else 0.0
    if curr_dd <= -2 * threshold:
        return 0.0
    if curr_dd <= -threshold:
        return 0.5 * base
    return base

# Strategy 5 — Hybrid (iterative)
def _pos_fn_hybrid(t, states, curr_dd, *, vol_ann, sigma_target, dd_threshold):
    base = 1.0 if states[t] == 1 else 0.0
    if base == 0.0:
        return 0.0
    # Vol targeting size
    v = vol_ann[t]
    vt_size = float(np.clip(sigma_target / v, 0.0, 1.0)) if (v > 0 and not np.isnan(v)) else 0.0
    # Drawdown overlay
    if curr_dd <= -2 * dd_threshold:
        return 0.0
    if curr_dd <= -dd_threshold:
        return 0.5 * vt_size
    return vt_size


# ── Pareto frontier ────────────────────────────────────────────────────────────
def _pareto_indices(rows: list[dict]) -> list[int]:
    """
    Indices of Pareto-optimal rows (maximize both Sharpe and MaxDD).
    A point is dominated if another has higher-or-equal Sharpe AND
    higher-or-equal MaxDD (less negative) with at least one strict.
    """
    idxs = []
    for i, r in enumerate(rows):
        dominated = any(
            j != i
            and rows[j]["sharpe"]       >= r["sharpe"]
            and rows[j]["max_drawdown"] >= r["max_drawdown"]
            and (rows[j]["sharpe"] > r["sharpe"] or rows[j]["max_drawdown"] > r["max_drawdown"])
            for j in range(len(rows))
        )
        if not dominated:
            idxs.append(i)
    return idxs


# ── Meets criteria ─────────────────────────────────────────────────────────────
def _meets(r: dict) -> bool:
    return (
        r["max_drawdown"] > MAXDD_TARGET
        and r["sharpe"]   > SHARPE_FLOOR
        and r["cagr"]     > CAGR_FLOOR
    )


# ── Plots ──────────────────────────────────────────────────────────────────────
def _plot_equity_curves(results: list[dict], test_idx: pd.DatetimeIndex) -> Path:
    good     = [r for r in results if _meets(r)]
    baseline = next(r for r in results if r["strategy_id"] == "0")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                   gridspec_kw={"height_ratios": [3, 1.5]}, sharex=True)
    fig.suptitle("R11 Position Sizing — Equity Curves (strategies meeting criteria)", fontsize=12)

    # Baseline always shown
    eq_b = baseline["_equity"]
    ax1.plot(eq_b.index, eq_b.values, color=STRAT_COLORS["0"], linewidth=1.2, linestyle="--",
             alpha=0.7, label=f"0—Baseline  Sh={baseline['sharpe']:+.3f} DD={baseline['max_drawdown']:+.2%}")
    ax2.fill_between(eq_b.index, baseline["_dd"].values, 0,
                     color=STRAT_COLORS["0"], alpha=0.25, label="Baseline")

    for r in good:
        if r["strategy_id"] == "0":
            continue
        color = STRAT_COLORS.get(r["strategy_id"], "#333333")
        lbl   = f"{r['label']}  Sh={r['sharpe']:+.3f} DD={r['max_drawdown']:+.2%}"
        ax1.plot(r["_equity"].index, r["_equity"].values, color=color, linewidth=1.0, label=lbl)
        ax2.fill_between(r["_dd"].index, r["_dd"].values, 0,
                         color=color, alpha=0.25, label=r["label"])

    ax1.axhline(1, color="black", linewidth=0.4, linestyle=":")
    ax1.set_ylabel("Equity (base 1)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.legend(fontsize=8, loc="upper left")

    ax2.axhline(MAXDD_TARGET, color="red", linewidth=0.8, linestyle="--", alpha=0.5,
                label=f"Target MaxDD {MAXDD_TARGET:.0%}")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = PLOTS_DIR / "r11_sizing_equity_curves.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def _plot_sharpe_vs_maxdd(results: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("R11 Position Sizing — Sharpe vs MaxDD (Pareto Frontier)", fontsize=12)

    pareto_idx = _pareto_indices(results)

    # Target zone
    ax.axhline(SHARPE_FLOOR,  color="green", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(MAXDD_TARGET,  color="green", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.fill_betweenx(
        [SHARPE_FLOOR, 100], MAXDD_TARGET, 0,
        color="green", alpha=0.06, label="Target zone"
    )

    # All points
    for i, r in enumerate(results):
        color  = STRAT_COLORS.get(r["strategy_id"], "#333333")
        marker = "D" if _meets(r) else "o"
        size   = 90 if _meets(r) else 50
        ax.scatter(r["max_drawdown"], r["sharpe"], color=color,
                   marker=marker, s=size, zorder=3)
        ax.annotate(r["params"], (r["max_drawdown"], r["sharpe"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")

    # Pareto frontier line
    p_rows = sorted([results[i] for i in pareto_idx], key=lambda x: x["max_drawdown"])
    if len(p_rows) >= 2:
        ax.plot([r["max_drawdown"] for r in p_rows],
                [r["sharpe"] for r in p_rows],
                color="black", linewidth=1.2, linestyle=":", zorder=2,
                label="Pareto frontier")

    # Legend patches
    legend_handles = [
        mpatches.Patch(color=STRAT_COLORS[k], label=v)
        for k, v in {
            "0": "0—Baseline", "1": "1—Fixed", "2": "2—Vol Target",
            "3": "3—Confidence", "4": "4—DD Control", "5": "5—Hybrid",
        }.items()
    ]
    legend_handles += [
        plt.Line2D([0], [0], color="black", linestyle=":", label="Pareto frontier"),
        mpatches.Patch(color="green", alpha=0.3, label="Target zone"),
        plt.scatter([], [], marker="D", color="gray", s=80, label="Meets criteria"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    ax.set_xlabel("Max Drawdown (higher = less negative = better)")
    ax.set_ylabel("Sharpe Ratio")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.tight_layout()

    out = PLOTS_DIR / "r11_sizing_sharpe_vs_maxdd.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def _plot_drawdown(results: list[dict]) -> Path:
    good     = [r for r in results if _meets(r)]
    baseline = next(r for r in results if r["strategy_id"] == "0")

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("R11 Position Sizing — Drawdown (strategies meeting criteria)", fontsize=12)

    ax.fill_between(baseline["_dd"].index, baseline["_dd"].values, 0,
                    color=STRAT_COLORS["0"], alpha=0.3, label=f"Baseline {baseline['max_drawdown']:+.2%}")
    for r in good:
        if r["strategy_id"] == "0":
            continue
        color = STRAT_COLORS.get(r["strategy_id"], "#333333")
        ax.plot(r["_dd"].index, r["_dd"].values, color=color, linewidth=0.9,
                label=f"{r['label']}  {r['max_drawdown']:+.2%}")

    ax.axhline(MAXDD_TARGET, color="red", linewidth=1.0, linestyle="--",
               label=f"Target {MAXDD_TARGET:.0%}")
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = PLOTS_DIR / "r11_sizing_drawdown.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


# ── Reports ────────────────────────────────────────────────────────────────────
def _write_json(results: list[dict], best: dict | None, meta: dict) -> Path:
    def _clean(r):
        return {k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in r.items() if not k.startswith("_")}

    out = REPORTS_DIR / "r11_position_sizing.json"
    with open(out, "w") as fh:
        json.dump({
            "meta":       meta,
            "best":       _clean(best) if best else None,
            "all_results": [_clean(r) for r in results],
        }, fh, indent=2, default=str)
    return out


def _write_markdown(results: list[dict], best: dict | None) -> Path:
    def pct(v): return f"{v:+.2%}" if isinstance(v, float) and not np.isnan(v) else "N/A"
    def flt(v): return f"{v:.3f}"  if isinstance(v, float) and not np.isnan(v) else "N/A"
    def fl2(v): return f"{v:.2f}"  if isinstance(v, float) and not np.isnan(v) else "N/A"

    lines = [
        "# R11 Position Sizing — Analysis Report",
        "",
        f"**Generated:** {date.today().isoformat()}",
        f"**Criteria:** MaxDD > {MAXDD_TARGET:.0%}  |  Sharpe > {SHARPE_FLOOR}  |  CAGR > {CAGR_FLOOR:.0%}",
        "",
        "## Results Table",
        "",
        "| Strategy | Params | Sharpe | CAGR | MaxDD | Meets? | Avg Pos |",
        "|----------|--------|--------|------|-------|--------|---------|",
    ]
    for r in results:
        ok = "✓ YES" if _meets(r) else "no"
        lines.append(
            f"| {r['label']} | {r['params']} | {flt(r['sharpe'])} | "
            f"{pct(r['cagr'])} | {pct(r['max_drawdown'])} | {ok} | "
            f"{pct(r['avg_position'])} |"
        )
    lines.append("")

    if best:
        lines += [
            "## Recommended Config",
            "",
            f"**{best['label']}** — params: `{best['params']}`",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Sharpe | {flt(best['sharpe'])} |",
            f"| CAGR | {pct(best['cagr'])} |",
            f"| Max Drawdown | {pct(best['max_drawdown'])} |",
            f"| Hit Rate | {pct(best['hit_rate'])} |",
            f"| Profit Factor | {fl2(best['profit_factor'])} |",
            f"| N Trades | {best['n_trades']} |",
            f"| Avg Position | {pct(best['avg_position'])} |",
            "",
            "**Justification:** Highest Sharpe among combinations satisfying MaxDD > "
            f"{MAXDD_TARGET:.0%}, Sharpe > {SHARPE_FLOOR}, CAGR > {CAGR_FLOOR:.0%}.",
            "",
        ]
    else:
        lines += ["## Recommended Config", "", "No combination met all criteria.", ""]

    lines += [
        "## Plots",
        "",
        "- `plots/r11_sizing_equity_curves.png`",
        "- `plots/r11_sizing_sharpe_vs_maxdd.png`",
        "- `plots/r11_sizing_drawdown.png`",
    ]

    out = REPORTS_DIR / "r11_position_sizing_report.md"
    out.write_text("\n".join(lines))
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Setting up R11 (training HMM)...")
    s = _setup()
    states    = s["states"]
    bull_prob = s["bull_prob"]
    log_ret   = s["log_ret"]
    vol_ann   = s["vol_ann"]
    test_idx  = s["test_idx"]

    print(f"Train: {s['train_start'].date()} → N={s['n_train']}  |  "
          f"Test: {s['test_start'].date()} → N={s['n_test']}")
    print(f"Bull days: {int((states==1).sum())} ({(states==1).mean():.1%})  "
          f"Bear days: {int((states==0).sum())} ({(states==0).mean():.1%})\n")

    results: list[dict] = []

    def _add(strat_id, label, params, positions_or_bt):
        if isinstance(positions_or_bt, np.ndarray):
            bt = _backtest(positions_or_bt, log_ret, test_idx)
        else:
            bt = positions_or_bt
        results.append({
            "strategy_id": strat_id,
            "label": label,
            "params": params,
            **{k: v for k, v in bt.items()},
        })
        m = "✓" if _meets(bt) else " "
        print(f"  [{m}] {label:35s} {params:20s}  "
              f"Sh={bt['sharpe']:+.3f}  CAGR={bt['cagr']:+.2%}  DD={bt['max_drawdown']:+.2%}")

    # ── Strategy 0 — Baseline ─────────────────────────────────────────────────
    print("Strategy 0 — Baseline")
    _add("0", "0 — Baseline", "100%",
         np.where(states == 1, 1.0, 0.0))

    # ── Strategy 1 — Fixed Fractional ─────────────────────────────────────────
    print("Strategy 1 — Fixed Fractional")
    for f in [0.25, 0.50, 0.75]:
        _add("1", "1 — Fixed Fractional", f"f={f:.2f}",
             _pos_fixed(states, f))

    # ── Strategy 2 — Vol Targeting ────────────────────────────────────────────
    print("Strategy 2 — Vol Targeting")
    for st in [0.10, 0.15, 0.20]:
        _add("2", "2 — Vol Targeting", f"σ={st:.0%}",
             _pos_vol_target(states, vol_ann, st))

    # ── Strategy 3 — Regime Confidence ───────────────────────────────────────
    print("Strategy 3 — Regime Confidence")
    _add("3", "3 — Confidence (raw)", "P(Bull)",
         _pos_confidence(states, bull_prob, floor=None))
    _add("3", "3 — Confidence (floor)", "P(Bull) floor=0.3",
         _pos_confidence(states, bull_prob, floor=0.3))

    # ── Strategy 4 — Drawdown Control ─────────────────────────────────────────
    print("Strategy 4 — Drawdown Control")
    for thr in [0.05, 0.08, 0.10]:
        bt = _simulate_iterative(
            _pos_fn_dd_control, states, log_ret, test_idx, threshold=thr
        )
        _add("4", "4 — DD Control", f"DD={thr:.0%}", bt)

    # ── Strategy 5 — Hybrid ───────────────────────────────────────────────────
    print("Strategy 5 — Hybrid (Vol Target + DD Control)")
    bt = _simulate_iterative(
        _pos_fn_hybrid, states, log_ret, test_idx,
        vol_ann=vol_ann, sigma_target=0.15, dd_threshold=0.08,
    )
    _add("5", "5 — Hybrid", "σ=15% + DD=8%", bt)

    # ── Summary table ──────────────────────────────────────────────────────────
    meeting = [r for r in results if _meets(r)]
    print(f"\n{'='*75}")
    print(f"RESULTS — {len(meeting)}/{len(results)} combinations meet all criteria")
    print(f"{'='*75}")
    hdr = f"{'Strategy':<35} {'Params':<20} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'OK':>4}"
    print(hdr)
    print("-" * 75)
    for r in results:
        ok = "YES" if _meets(r) else "---"
        print(f"{r['label']:<35} {r['params']:<20} "
              f"{r['sharpe']:>+7.3f} {r['cagr']:>+8.2%} "
              f"{r['max_drawdown']:>+8.2%} {ok:>4}")
    print("=" * 75)

    # Best = highest Sharpe among meeting criteria
    best = max(meeting, key=lambda r: r["sharpe"]) if meeting else None
    if best:
        print(f"\nRecommended: {best['label']} — {best['params']}")
        print(f"  Sharpe={best['sharpe']:+.3f}  CAGR={best['cagr']:+.2%}  "
              f"MaxDD={best['max_drawdown']:+.2%}  AvgPos={best['avg_position']:.1%}")

    # ── Pareto frontier ────────────────────────────────────────────────────────
    pareto_idx = _pareto_indices(results)
    print(f"\nPareto frontier ({len(pareto_idx)} points):")
    for i in sorted(pareto_idx, key=lambda i: results[i]["max_drawdown"]):
        r = results[i]
        print(f"  {r['label']:35s} {r['params']:20s}  "
              f"Sh={r['sharpe']:+.3f}  DD={r['max_drawdown']:+.2%}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    p1 = _plot_equity_curves(results, test_idx)
    p2 = _plot_sharpe_vs_maxdd(results)
    p3 = _plot_drawdown(results)
    for p in [p1, p2, p3]:
        print(f"  Saved: {p.name}")

    # ── Save reports ───────────────────────────────────────────────────────────
    meta = {
        "generated":     str(date.today()),
        "test_start":    str(s["test_start"].date()),
        "test_end":      str(s["test_end"].date()),
        "n_test":        s["n_test"],
        "features":      R11_FEATURES,
        "covariance_type": R11_COV_TYPE,
        "TC_bps":        TC * 10_000,
        "criteria":      {"maxdd_target": MAXDD_TARGET, "sharpe_floor": SHARPE_FLOOR,
                          "cagr_floor": CAGR_FLOOR},
    }
    json_path = _write_json(results, best, meta)
    md_path   = _write_markdown(results, best)
    print(f"\nOutputs:")
    print(f"  {json_path.relative_to(ROOT)}")
    print(f"  {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
