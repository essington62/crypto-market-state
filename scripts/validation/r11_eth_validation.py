"""
R11 ETH Validation — HMM regime model applied to ETH, compared vs BTC.

Training:  split_3 config  (train 2023-01-01 → 2024-12-31,
                             test  2025-01-01 → present)
Strategy:  Bull → Long, Bear → Exit
TC:        0.12% round-trip (12 bps)

Outputs:
  data/06_reports/r11_eth_vs_btc.json
  data/06_reports/r11_eth_vs_btc_report.md
  data/06_reports/plots/r11_eth_regime_sequence.png
  data/06_reports/plots/r11_eth_equity_curve.png
  data/06_reports/plots/r11_eth_drawdown.png
  data/06_reports/plots/r11_cross_asset_regime_agreement.png

Prerequisite: ETHUSDT.parquet must exist in data/04_model_input/spot/daily/
  If missing: kedro run --pipeline primary.spot.crypto
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

TC         = 0.0012   # 12 bps round-trip
PERIODS_YR = 252      # daily data


# ── Config loader ──────────────────────────────────────────────────────────────
def _load_config() -> tuple[dict, dict]:
    """Load modeling and walkforward params from parameters.yml."""
    with open(CONF_FILE) as fh:
        cfg = yaml.safe_load(fh)

    modeling_params    = cfg["modeling"]["regime_hmm"]
    walkforward_params = cfg["walkforward"]
    return modeling_params, walkforward_params


# ── HMM helpers (mirrored from pipeline nodes) ─────────────────────────────────
def _ordenar_estados(model: GaussianHMM, features: list[str]) -> dict[int, int]:
    """Order states by drawdown (ascending). Bear=0, Bull=n_states-1."""
    n_states = model.n_components
    if "drawdown" in features:
        idx = features.index("drawdown")
        vals = {s: model.means_[s][idx] for s in range(n_states)}
        ordered = sorted(vals, key=vals.get)
    else:
        idx = features.index("log_return") if "log_return" in features else 0
        ordered = sorted(range(n_states), key=lambda s: model.means_[s][idx])
    return {orig: new for new, orig in enumerate(ordered)}


def _run_durations(states: np.ndarray, target_state: int) -> float:
    durations, count = [], 0
    for s in states:
        if s == target_state:
            count += 1
        else:
            if count > 0:
                durations.append(count)
                count = 0
    if count > 0:
        durations.append(count)
    return float(np.mean(durations)) if durations else 0.0


def _train_and_predict(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    features: list[str],
    modeling_params: dict,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Train HMM on [train_start, train_end] and predict [test_start, test_end].
    Returns (test_states_ordered, bull_prob_test, effective_features).
    """
    effective_features = [f for f in features if f in df.columns]
    if not effective_features:
        raise ValueError(f"No configured features found in DataFrame columns: {df.columns.tolist()}")

    df = df.sort_index().dropna(subset=effective_features)

    train_df = df.loc[train_start:train_end]
    test_df  = df.loc[test_start:test_end]

    if len(train_df) < 30:
        raise ValueError(f"Train set too small: {len(train_df)} rows")
    if len(test_df) < 5:
        raise ValueError(f"Test set too small: {len(test_df)} rows")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[effective_features])
    X_test  = scaler.transform(test_df[effective_features])

    model = GaussianHMM(
        n_components=modeling_params["n_states"],
        covariance_type=modeling_params["covariance_type"],
        n_iter=modeling_params["n_iter"],
        random_state=modeling_params["random_state"],
    )
    model.fit(X_train)

    mapa       = _ordenar_estados(model, effective_features)
    raw_states = model.predict(X_test)
    states     = np.array([mapa[s] for s in raw_states])

    raw_proba  = model.predict_proba(X_test)
    bull_state = modeling_params["n_states"] - 1
    inv_mapa   = {new_s: orig_s for orig_s, new_s in mapa.items()}
    bull_prob  = raw_proba[:, inv_mapa[bull_state]]

    return states, bull_prob, effective_features


# ── Backtest ───────────────────────────────────────────────────────────────────
def _run_backtest(
    df: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    states: np.ndarray,
    horizon_days: int,
) -> dict:
    """
    Daily backtest: Bull → Long, Bear → Exit.
    Returns metrics dict and series for equity/drawdown.
    """
    test_df = df.sort_index().loc[test_start:test_end].copy()
    test_df = test_df.dropna(subset=["log_return"])

    n = min(len(test_df), len(states))
    test_df = test_df.iloc[:n]
    sigs    = states[:n]

    bull_state = 1
    position   = (sigs == bull_state).astype(float)  # 1=Long, 0=exit

    # 5-day forward return (regime evaluation metric)
    fwd_ret = sum(test_df["log_return"].shift(-k) for k in range(1, horizon_days + 1))

    # Daily trading returns
    daily_ret  = test_df["log_return"].values
    prev_pos   = np.concatenate([[0.0], position[:-1]])
    trade_flag = np.abs(position - prev_pos)
    tc_cost    = TC * trade_flag
    net_ret    = position * daily_ret - tc_cost

    equity     = pd.Series((1 + net_ret).cumprod(), index=test_df.index)
    bnh_ret    = daily_ret
    bnh_equity = pd.Series((1 + bnh_ret).cumprod(), index=test_df.index)

    def drawdown_series(eq: pd.Series) -> pd.Series:
        peak = eq.cummax()
        return (eq - peak) / peak

    dd_strat = drawdown_series(equity)
    dd_bnh   = drawdown_series(bnh_equity)

    # Sharpe
    n_years    = len(net_ret) / PERIODS_YR
    final_eq   = float(equity.iloc[-1])
    cagr       = final_eq ** (1 / n_years) - 1 if n_years > 0 else float("nan")
    sharpe     = (net_ret.mean() / (net_ret.std() + 1e-10)) * np.sqrt(PERIODS_YR)
    max_dd     = float(dd_strat.min())
    active     = net_ret[net_ret != 0]
    hit_rate   = float((active > 0).mean()) if len(active) > 0 else float("nan")
    gains      = active[active > 0].sum()
    losses     = abs(active[active < 0].sum())
    pf         = gains / losses if losses > 0 else float("nan")
    n_trades   = int((trade_flag > 0).sum())

    # Regime metrics
    bull_mask = sigs == 1
    bear_mask = sigs == 0
    E_bull   = fwd_ret[bull_mask].mean() if bull_mask.sum() > 0 else float("nan")
    E_bear   = fwd_ret[bear_mask].mean() if bear_mask.sum() > 0 else float("nan")
    delta_5d = E_bull - E_bear if not (np.isnan(E_bull) or np.isnan(E_bear)) else float("nan")
    bull_dur = _run_durations(sigs, 1)
    bear_dur = _run_durations(sigs, 0)
    bull_pct = float(bull_mask.sum() / n)

    return {
        "metrics": {
            "CAGR":          cagr,
            "Sharpe":        sharpe,
            "Max_Drawdown":  max_dd,
            "Hit_Rate":      hit_rate,
            "Profit_Factor": pf,
            "n_trades":      n_trades,
            "n_obs":         n,
            "delta_5d":      float(delta_5d),
            "E_bull":        float(E_bull),
            "E_bear":        float(E_bear),
            "bull_duration": bull_dur,
            "bear_duration": bear_dur,
            "bull_pct":      bull_pct,
        },
        "equity":     equity,
        "bnh_equity": bnh_equity,
        "dd_strat":   dd_strat,
        "dd_bnh":     dd_bnh,
        "sigs":       sigs,
        "index":      test_df.index,
    }


# ── Plots ──────────────────────────────────────────────────────────────────────
def _plot_regime_sequence(results: dict, asset: str, test_start: pd.Timestamp) -> Path:
    idx    = results["index"]
    states = results["sigs"]
    eq     = results["equity"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle(f"{asset} HMM Regime Sequence — test from {test_start.date()}", fontsize=12)

    # Price proxy: equity curve
    ax1.plot(idx, eq.values, color="steelblue", linewidth=1.0, label="Strategy equity")
    bull_mask = states == 1
    ax1.fill_between(idx, 0, eq.values, where=bull_mask,
                     color="green", alpha=0.2, label="Bull")
    ax1.fill_between(idx, 0, eq.values, where=~bull_mask,
                     color="red", alpha=0.2, label="Bear")
    ax1.set_ylabel("Equity")
    ax1.legend(fontsize=9)

    # State series
    ax2.step(idx, states, where="post", color="navy", linewidth=0.8)
    ax2.fill_between(idx, 0, states, step="post", color="green", alpha=0.4, label="Bull (1)")
    ax2.fill_between(idx, states, 1, step="post", color="red", alpha=0.2, label="Bear (0)")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Bear", "Bull"])
    ax2.set_ylabel("Regime")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = PLOTS_DIR / f"r11_{asset.lower()}_regime_sequence.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def _plot_equity_curves(btc_res: dict, eth_res: dict, test_start: pd.Timestamp) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle(f"HMM Strategy Equity Curves — BTC vs ETH (test from {test_start.date()})", fontsize=12)

    m_btc = btc_res["metrics"]
    m_eth = eth_res["metrics"]

    ax.plot(btc_res["equity"].index, btc_res["equity"].values, color="orange",
            linewidth=1.2, label=f"BTC  CAGR={m_btc['CAGR']:+.1%}  Sharpe={m_btc['Sharpe']:.2f}")
    ax.plot(eth_res["equity"].index, eth_res["equity"].values, color="steelblue",
            linewidth=1.2, label=f"ETH  CAGR={m_eth['CAGR']:+.1%}  Sharpe={m_eth['Sharpe']:.2f}")
    ax.plot(btc_res["bnh_equity"].index, btc_res["bnh_equity"].values, color="orange",
            linewidth=0.7, linestyle="--", alpha=0.5, label="BTC B&H")
    ax.plot(eth_res["bnh_equity"].index, eth_res["bnh_equity"].values, color="steelblue",
            linewidth=0.7, linestyle="--", alpha=0.5, label="ETH B&H")
    ax.axhline(1, color="black", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Equity (base 1)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = PLOTS_DIR / "r11_eth_equity_curve.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def _plot_drawdown(btc_res: dict, eth_res: dict, test_start: pd.Timestamp) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(f"HMM Strategy Drawdown — BTC vs ETH (test from {test_start.date()})", fontsize=12)

    m_btc = btc_res["metrics"]
    m_eth = eth_res["metrics"]

    ax.fill_between(btc_res["dd_strat"].index, btc_res["dd_strat"].values, 0,
                    color="orange", alpha=0.4, label=f"BTC max DD={m_btc['Max_Drawdown']:+.2%}")
    ax.fill_between(eth_res["dd_strat"].index, eth_res["dd_strat"].values, 0,
                    color="steelblue", alpha=0.4, label=f"ETH max DD={m_eth['Max_Drawdown']:+.2%}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = PLOTS_DIR / "r11_eth_drawdown.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def _plot_regime_agreement(btc_res: dict, eth_res: dict, test_start: pd.Timestamp) -> Path:
    """Cross-asset regime agreement: overlay BTC and ETH state series."""
    btc_idx    = btc_res["index"]
    eth_idx    = eth_res["index"]
    btc_states = pd.Series(btc_res["sigs"], index=btc_idx, name="BTC")
    eth_states = pd.Series(eth_res["sigs"], index=eth_idx, name="ETH")

    aligned = pd.concat([btc_states, eth_states], axis=1).dropna()
    agree_bull = (aligned["BTC"] == 1) & (aligned["ETH"] == 1)
    agree_bear = (aligned["BTC"] == 0) & (aligned["ETH"] == 0)
    disagree   = ~(agree_bull | agree_bear)
    agreement_pct = float((agree_bull | agree_bear).mean())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(
        f"Cross-Asset Regime Agreement — BTC vs ETH (test from {test_start.date()})\n"
        f"Agreement: {agreement_pct:.1%}",
        fontsize=12,
    )

    ax1.step(aligned.index, aligned["BTC"].values, where="post",
             color="orange", linewidth=1.0, label="BTC")
    ax1.step(aligned.index, aligned["ETH"].values + 0.05, where="post",
             color="steelblue", linewidth=1.0, linestyle="--", label="ETH (offset +0.05)")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Bear", "Bull"])
    ax1.set_ylabel("Regime")
    ax1.legend(fontsize=9)

    # Agreement bars
    colors = np.where(agree_bull.values, "green",
             np.where(agree_bear.values, "red", "gray"))
    for i, (ts, col) in enumerate(zip(aligned.index, colors)):
        ax2.axvspan(
            ts,
            aligned.index[i + 1] if i + 1 < len(aligned) else ts + pd.Timedelta(days=1),
            color=col, alpha=0.4,
        )
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="green", alpha=0.4, label="Both Bull"),
        Patch(facecolor="red",   alpha=0.4, label="Both Bear"),
        Patch(facecolor="gray",  alpha=0.4, label="Disagree"),
    ]
    ax2.legend(handles=legend_handles, fontsize=9)
    ax2.set_ylabel("Agreement")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = PLOTS_DIR / "r11_cross_asset_regime_agreement.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


# ── Report writers ─────────────────────────────────────────────────────────────
def _write_json(btc_metrics: dict, eth_metrics: dict, meta: dict) -> Path:
    """Serialize metrics to JSON (NaN → null)."""
    def _clean(d):
        return {k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in d.items()}

    payload = {
        "meta":        meta,
        "BTC_metrics": _clean(btc_metrics),
        "ETH_metrics": _clean(eth_metrics),
    }
    out = REPORTS_DIR / "r11_eth_vs_btc.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return out


def _write_markdown(btc_metrics: dict, eth_metrics: dict, meta: dict) -> Path:
    def pct(v): return f"{v:+.2%}" if v is not None and not np.isnan(v) else "N/A"
    def flt(v): return f"{v:.3f}"  if v is not None and not np.isnan(v) else "N/A"
    def days(v): return f"{v:.1f}d" if v is not None and not np.isnan(v) else "N/A"

    lines = [
        "# R11 ETH vs BTC — HMM Regime Validation",
        "",
        f"**Generated:** {date.today().isoformat()}",
        f"**Train period:** {meta['train_start']} → {meta['train_end']}",
        f"**Test period:** {meta['test_start']} → {meta['test_end']}",
        f"**Features:** {', '.join(meta['features'])}",
        f"**TC:** 0.12% round-trip  |  Strategy: Bull→Long, Bear→Exit",
        "",
        "## Regime Metrics",
        "",
        "| Metric | BTC | ETH |",
        "|--------|-----|-----|",
        f"| delta_5d | {pct(btc_metrics['delta_5d'])} | {pct(eth_metrics['delta_5d'])} |",
        f"| E_bull (5d) | {pct(btc_metrics['E_bull'])} | {pct(eth_metrics['E_bull'])} |",
        f"| E_bear (5d) | {pct(btc_metrics['E_bear'])} | {pct(eth_metrics['E_bear'])} |",
        f"| Bull duration | {days(btc_metrics['bull_duration'])} | {days(eth_metrics['bull_duration'])} |",
        f"| Bear duration | {days(btc_metrics['bear_duration'])} | {days(eth_metrics['bear_duration'])} |",
        f"| Bull % | {pct(btc_metrics['bull_pct'])} | {pct(eth_metrics['bull_pct'])} |",
        "",
        "## Trading Strategy Metrics",
        "",
        "| Metric | BTC | ETH |",
        "|--------|-----|-----|",
        f"| CAGR | {pct(btc_metrics['CAGR'])} | {pct(eth_metrics['CAGR'])} |",
        f"| Sharpe | {flt(btc_metrics['Sharpe'])} | {flt(eth_metrics['Sharpe'])} |",
        f"| Max Drawdown | {pct(btc_metrics['Max_Drawdown'])} | {pct(eth_metrics['Max_Drawdown'])} |",
        f"| Hit Rate | {pct(btc_metrics['Hit_Rate'])} | {pct(eth_metrics['Hit_Rate'])} |",
        f"| Profit Factor | {flt(btc_metrics['Profit_Factor'])} | {flt(eth_metrics['Profit_Factor'])} |",
        f"| N Trades | {btc_metrics['n_trades']} | {eth_metrics['n_trades']} |",
        f"| N Obs | {btc_metrics['n_obs']} | {eth_metrics['n_obs']} |",
        "",
        "## Plots",
        "",
        "- `plots/r11_btc_regime_sequence.png`",
        "- `plots/r11_eth_regime_sequence.png`",
        "- `plots/r11_eth_equity_curve.png`",
        "- `plots/r11_eth_drawdown.png`",
        "- `plots/r11_cross_asset_regime_agreement.png`",
    ]

    out = REPORTS_DIR / "r11_eth_vs_btc_report.md"
    out.write_text("\n".join(lines))
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Config ────────────────────────────────────────────────────────────────
    modeling_params, walkforward_params = _load_config()
    features      = modeling_params["features"]
    horizon_days  = walkforward_params["horizon_days"]

    # Use split_3 (last split: train 2023-01-01→2024-12-31, test 2025-01-01→present)
    split3 = next(s for s in walkforward_params["splits"] if s["name"] == "split_3")

    train_start = pd.Timestamp(str(split3["train_start"]), tz="UTC")
    train_end   = pd.Timestamp(str(split3["train_end"]),   tz="UTC")
    test_start  = pd.Timestamp(str(split3["test_start"]),  tz="UTC")
    test_end_raw = split3["test_end"]

    # ── File checks ───────────────────────────────────────────────────────────
    btc_path = DATA_INPUT / "BTCUSDT.parquet"
    eth_path = DATA_INPUT / "ETHUSDT.parquet"

    if not btc_path.exists():
        print(f"ERROR: BTC L3 file not found: {btc_path}")
        print("  Run: conda run -n crypto_market_state kedro run --pipeline primary.spot.crypto")
        sys.exit(1)

    if not eth_path.exists():
        print(f"ERROR: ETH L3 file not found: {eth_path}")
        print("  Run: conda run -n crypto_market_state kedro run --pipeline primary.spot.crypto")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading BTC model input...")
    btc_df = pd.read_parquet(btc_path)
    if not isinstance(btc_df.index, pd.DatetimeIndex):
        btc_df.index = pd.to_datetime(btc_df.index, utc=True)
    elif btc_df.index.tz is None:
        btc_df.index = btc_df.index.tz_localize("UTC")

    print("Loading ETH model input...")
    eth_df = pd.read_parquet(eth_path)
    if not isinstance(eth_df.index, pd.DatetimeIndex):
        eth_df.index = pd.to_datetime(eth_df.index, utc=True)
    elif eth_df.index.tz is None:
        eth_df.index = eth_df.index.tz_localize("UTC")

    # ── Schema check ──────────────────────────────────────────────────────────
    btc_cols = set(btc_df.columns)
    eth_cols = set(eth_df.columns)
    missing_in_eth = btc_cols - eth_cols
    if missing_in_eth:
        print(f"WARNING: columns in BTC but not ETH: {sorted(missing_in_eth)}")
    missing_features = [f for f in features if f not in eth_df.columns]
    if missing_features:
        print(f"WARNING: features missing from ETH: {missing_features} — will be skipped")

    test_end = (
        eth_df.index.max()
        if test_end_raw is None
        else pd.Timestamp(str(test_end_raw), tz="UTC")
    )

    print(f"\nConfig:")
    print(f"  Train:    {train_start.date()} → {train_end.date()}")
    print(f"  Test:     {test_start.date()} → {test_end.date()}")
    print(f"  Features: {features}")
    print(f"  n_states: {modeling_params['n_states']}")

    # ── Train and predict ─────────────────────────────────────────────────────
    print("\nTraining HMM on BTC...")
    btc_states, btc_bull_prob, btc_features = _train_and_predict(
        btc_df, train_start, train_end, test_start, test_end, features, modeling_params,
    )
    print(f"  BTC: {btc_states.sum()} Bull days / {len(btc_states)} total")

    print("Training HMM on ETH...")
    eth_states, eth_bull_prob, eth_features = _train_and_predict(
        eth_df, train_start, train_end, test_start, test_end, features, modeling_params,
    )
    print(f"  ETH: {eth_states.sum()} Bull days / {len(eth_states)} total")

    # ── Backtests ─────────────────────────────────────────────────────────────
    print("\nRunning BTC backtest...")
    btc_res = _run_backtest(btc_df, test_start, test_end, btc_states, horizon_days)

    print("Running ETH backtest...")
    eth_res = _run_backtest(eth_df, test_start, test_end, eth_states, horizon_days)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HMM R11 REGIME VALIDATION — BTC vs ETH")
    print("=" * 60)
    header = f"{'Metric':<22} {'BTC':>16} {'ETH':>16}"
    print(header)
    print("-" * 60)

    m_btc = btc_res["metrics"]
    m_eth = eth_res["metrics"]

    rows = [
        ("delta_5d",       f"{m_btc['delta_5d']:+.4f}",     f"{m_eth['delta_5d']:+.4f}"),
        ("E_bull (5d)",    f"{m_btc['E_bull']:+.4f}",        f"{m_eth['E_bull']:+.4f}"),
        ("E_bear (5d)",    f"{m_btc['E_bear']:+.4f}",        f"{m_eth['E_bear']:+.4f}"),
        ("Bull duration",  f"{m_btc['bull_duration']:.1f}d", f"{m_eth['bull_duration']:.1f}d"),
        ("Bear duration",  f"{m_btc['bear_duration']:.1f}d", f"{m_eth['bear_duration']:.1f}d"),
        ("CAGR",           f"{m_btc['CAGR']:+.2%}",          f"{m_eth['CAGR']:+.2%}"),
        ("Sharpe",         f"{m_btc['Sharpe']:+.3f}",         f"{m_eth['Sharpe']:+.3f}"),
        ("Max Drawdown",   f"{m_btc['Max_Drawdown']:+.2%}",   f"{m_eth['Max_Drawdown']:+.2%}"),
        ("Hit Rate",       f"{m_btc['Hit_Rate']:+.2%}",       f"{m_eth['Hit_Rate']:+.2%}"),
        ("Profit Factor",  f"{m_btc['Profit_Factor']:.3f}",   f"{m_eth['Profit_Factor']:.3f}"),
        ("N Trades",       str(m_btc['n_trades']),             str(m_eth['n_trades'])),
        ("N Obs (test)",   str(m_btc['n_obs']),                str(m_eth['n_obs'])),
    ]
    for label, v_btc, v_eth in rows:
        print(f"{label:<22} {v_btc:>16} {v_eth:>16}")
    print("=" * 60)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    p1 = _plot_regime_sequence(btc_res, "BTC", test_start)
    p2 = _plot_regime_sequence(eth_res, "ETH", test_start)
    p3 = _plot_equity_curves(btc_res, eth_res, test_start)
    p4 = _plot_drawdown(btc_res, eth_res, test_start)
    p5 = _plot_regime_agreement(btc_res, eth_res, test_start)
    print(f"  Saved: {p1.name}")
    print(f"  Saved: {p2.name}")
    print(f"  Saved: {p3.name}")
    print(f"  Saved: {p4.name}")
    print(f"  Saved: {p5.name}")

    # ── JSON + Markdown ───────────────────────────────────────────────────────
    meta = {
        "train_start": str(train_start.date()),
        "train_end":   str(train_end.date()),
        "test_start":  str(test_start.date()),
        "test_end":    str(test_end.date()),
        "features":    btc_features,
        "n_states":    modeling_params["n_states"],
        "TC_bps":      TC * 10_000,
        "horizon_days": horizon_days,
    }

    json_path = _write_json(m_btc, m_eth, meta)
    md_path   = _write_markdown(m_btc, m_eth, meta)
    print(f"\nOutputs:")
    print(f"  {json_path.relative_to(ROOT)}")
    print(f"  {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
