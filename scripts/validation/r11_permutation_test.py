"""
R11 Permutation Test — Statistical significance of BTC HMM regime signals.

Three complementary permutation strategies:
  A — Signal Shuffle       : randomly permute regime labels
  B — Circular Block Shift : shift signal array by random offset (circular)
  C — Stationary Block Bootstrap : resample signal blocks (geometric length, mean=21d)

Primary stat:   Sharpe Ratio
Secondary stats: CAGR, Profit Factor

Usage:
    python scripts/validation/r11_permutation_test.py [--n 5000]

Outputs:
    data/06_reports/r11_permutation_results.json
    data/06_reports/r11_permutation_report.md
    data/06_reports/plots/r11_perm_strategy_a.png
    data/06_reports/plots/r11_perm_strategy_b.png
    data/06_reports/plots/r11_perm_strategy_c.png
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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

TC            = 0.0012    # 12 bps round-trip
PERIODS_YR    = 252       # daily data
MEAN_BLOCK    = 21        # stationary bootstrap mean block size (days)
OUTLIER_MULT  = 1.5       # flag permutation if sharpe > OUTLIER_MULT × real


# ── R11 Frozen Config (do not change) ─────────────────────────────────────────
R11_FEATURES     = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]
R11_N_STATES     = 2
R11_COV_TYPE     = "diag"
R11_N_ITER       = 1000
R11_RANDOM_STATE = 42


# ── Config loader (splits only) ────────────────────────────────────────────────
def _load_config() -> dict:
    with open(CONF_FILE) as fh:
        cfg = yaml.safe_load(fh)
    return cfg["walkforward"]


# ── HMM helpers ────────────────────────────────────────────────────────────────
def _ordenar_estados(model: GaussianHMM, features: list[str]) -> dict[int, int]:
    n_states = model.n_components
    if "drawdown" in features:
        idx = features.index("drawdown")
        vals = {s: model.means_[s][idx] for s in range(n_states)}
        ordered = sorted(vals, key=vals.get)
    else:
        idx = features.index("log_return") if "log_return" in features else 0
        ordered = sorted(range(n_states), key=lambda s: model.means_[s][idx])
    return {orig: new for new, orig in enumerate(ordered)}


def _fit_and_predict_r11(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Train R11 HMM (frozen config) on train window, predict test window.
    Returns (test_states, test_log_returns, test_index).
    """
    features = [f for f in R11_FEATURES if f in df.columns]
    df = df.sort_index().dropna(subset=features)

    train_df = df.loc[train_start:train_end]
    test_df  = df.loc[test_start:test_end].dropna(subset=["log_return"])

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test  = scaler.transform(test_df[features])

    model = GaussianHMM(
        n_components=R11_N_STATES,
        covariance_type=R11_COV_TYPE,
        n_iter=R11_N_ITER,
        random_state=R11_RANDOM_STATE,
    )
    model.fit(X_train)

    mapa       = _ordenar_estados(model, features)
    raw_states = model.predict(X_test)
    states     = np.array([mapa[s] for s in raw_states])
    returns    = test_df["log_return"].values

    return states, returns, test_df.index


# ── Backtest ───────────────────────────────────────────────────────────────────
def _backtest_metrics(signals: np.ndarray, daily_ret: np.ndarray) -> dict[str, float]:
    """
    Bull→Long (position=1), Bear→Exit (position=0).
    TC applied on position changes. Returns Sharpe, CAGR, Profit Factor.
    """
    n        = len(signals)
    position = (signals == 1).astype(float)
    prev_pos = np.concatenate([[0.0], position[:-1]])
    tc_cost  = TC * np.abs(position - prev_pos)
    net_ret  = position * daily_ret - tc_cost

    equity  = (1 + net_ret).cumprod()
    n_years = n / PERIODS_YR
    cagr    = float(equity[-1] ** (1 / n_years) - 1) if n_years > 0 else float("nan")
    sharpe  = float((net_ret.mean() / (net_ret.std() + 1e-10)) * np.sqrt(PERIODS_YR))

    active  = net_ret[net_ret != 0]
    gains   = active[active > 0].sum()
    losses  = abs(active[active < 0].sum())
    pf      = float(gains / losses) if losses > 0 else float("nan")

    return {"sharpe": sharpe, "cagr": cagr, "profit_factor": pf}


# ── Permutation strategies ─────────────────────────────────────────────────────
def _permute_signal_shuffle(signals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Strategy A: randomly shuffle regime labels."""
    perm = signals.copy()
    rng.shuffle(perm)
    return perm


def _permute_circular_shift(signals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Strategy B: circular shift by random offset."""
    offset = rng.integers(1, len(signals))
    return np.roll(signals, offset)


def _permute_block_bootstrap(
    signals: np.ndarray,
    rng: np.random.Generator,
    mean_block: int = MEAN_BLOCK,
) -> np.ndarray:
    """
    Strategy C: stationary block bootstrap.
    Block lengths ~ Geometric(p = 1/mean_block), starts sampled uniformly.
    Preserves local temporal dependencies at regime-length scale.
    """
    n   = len(signals)
    p   = 1.0 / mean_block
    out = np.empty(n, dtype=signals.dtype)
    i   = 0
    while i < n:
        start      = int(rng.integers(0, n))
        block_len  = int(rng.geometric(p))
        for j in range(block_len):
            if i >= n:
                break
            out[i] = signals[(start + j) % n]
            i += 1
    return out


# ── Run permutation test for one strategy ──────────────────────────────────────
def _run_permutation(
    signals: np.ndarray,
    daily_ret: np.ndarray,
    permute_fn,
    n_perms: int,
    strategy_label: str,
    seed: int = 42,
) -> dict:
    """
    Run n_perms permutations using permute_fn.
    Returns dict with null distributions and p-values for all stats.
    """
    rng          = np.random.default_rng(seed)
    real_metrics = _backtest_metrics(signals, daily_ret)
    stat_keys    = ["sharpe", "cagr", "profit_factor"]

    null: dict[str, list[float]] = {k: [] for k in stat_keys}

    print(f"\n  {strategy_label} ({n_perms} permutations):")
    for i in range(n_perms):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"    iteration {i + 1}/{n_perms}...")

        perm_sig     = permute_fn(signals, rng)
        perm_metrics = _backtest_metrics(perm_sig, daily_ret)
        for k in stat_keys:
            val = perm_metrics[k]
            null[k].append(val if not np.isnan(val) else 0.0)

    results = {}
    for k in stat_keys:
        arr      = np.array(null[k])
        real_val = real_metrics[k]
        perm_arr = arr[~np.isnan(arr)]

        p_value = float((perm_arr >= real_val).mean()) if len(perm_arr) > 0 else float("nan")
        z_score = float(
            (real_val - perm_arr.mean()) / (perm_arr.std() + 1e-10)
        ) if len(perm_arr) > 0 else float("nan")
        outlier_pct = float(
            (perm_arr >= OUTLIER_MULT * real_val).mean()
        ) if real_val > 0 else float("nan")

        results[k] = {
            "stat_real":    real_val,
            "perm_mean":    float(perm_arr.mean()),
            "perm_std":     float(perm_arr.std()),
            "p_value":      p_value,
            "z_score":      z_score,
            "significant":  p_value < 0.05,
            "pct_beat":     float((perm_arr >= real_val).mean()),
            "outlier_flag": bool(outlier_pct > 0) if not np.isnan(outlier_pct) else False,
            "_null_dist":   perm_arr.tolist(),
        }

    return results


# ── Plot null distributions ────────────────────────────────────────────────────
def _plot_null_distributions(
    results: dict,
    strategy_label: str,
    strategy_code: str,
    n_perms: int,
) -> Path:
    stat_labels = {
        "sharpe":        "Sharpe Ratio",
        "cagr":          "CAGR",
        "profit_factor": "Profit Factor",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Null Distribution — {strategy_label}\n"
        f"(n={n_perms} permutations, R11 BTC test 2025-01-01→)",
        fontsize=11,
    )

    for ax, (stat_key, stat_name) in zip(axes, stat_labels.items()):
        r      = results[stat_key]
        null   = np.array(r["_null_dist"])
        real   = r["stat_real"]
        p_val  = r["p_value"]
        z_sc   = r["z_score"]
        sig    = r["significant"]
        color  = "steelblue"

        ax.hist(null, bins=60, color=color, alpha=0.6, density=True, label="Null dist")
        ax.axvline(real, color="crimson", linewidth=2.0,
                   label=f"Real: {real:.4f}")
        ax.axvline(np.mean(null), color="gray", linewidth=1.0, linestyle="--",
                   label=f"Perm mean: {np.mean(null):.4f}")

        # Shade p-value region
        tail_vals = null[null >= real]
        if len(tail_vals) > 0:
            kde_x = np.linspace(null.min(), null.max(), 300)
            from scipy.stats import gaussian_kde
            try:
                kde   = gaussian_kde(null)
                kde_y = kde(kde_x)
                mask  = kde_x >= real
                ax.fill_between(kde_x[mask], kde_y[mask], alpha=0.3,
                                color="crimson", label=f"p={p_val:.4f}")
            except Exception:
                pass

        sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.set_title(f"{stat_name}\np={p_val:.4f} ({sig_marker})  z={z_sc:.2f}", fontsize=10)
        ax.set_xlabel(stat_name, fontsize=9)
        ax.set_ylabel("Density" if ax == axes[0] else "", fontsize=9)
        ax.legend(fontsize=8)

        border_color = "green" if sig else "orange"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)

    plt.tight_layout()
    out = PLOTS_DIR / f"r11_perm_{strategy_code}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


# ── Reports ────────────────────────────────────────────────────────────────────
def _write_json(all_results: dict, meta: dict) -> Path:
    # Strip _null_dist (too large for JSON report)
    clean = {}
    for strat, strat_res in all_results.items():
        clean[strat] = {}
        for stat, vals in strat_res.items():
            clean[strat][stat] = {k: v for k, v in vals.items() if k != "_null_dist"}

    payload = {"meta": meta, "results": clean}
    out = REPORTS_DIR / "r11_permutation_results.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return out


def _write_markdown(all_results: dict, meta: dict) -> Path:
    strategy_names = {
        "A": "A — Signal Shuffle",
        "B": "B — Circular Shift",
        "C": "C — Block Bootstrap",
    }
    stat_names = {
        "sharpe":        "Sharpe",
        "cagr":          "CAGR",
        "profit_factor": "Profit Factor",
    }

    lines = [
        "# R11 Permutation Test — Statistical Significance",
        "",
        f"**Generated:** {date.today().isoformat()}",
        f"**Asset:** BTC | **Test period:** {meta['test_start']} → {meta['test_end']}",
        f"**n_permutations:** {meta['n_permutations']} | **Random seed:** {meta['random_seed']}",
        f"**TC:** {meta['TC_bps']:.0f} bps round-trip | **Mean block size (C):** {meta['mean_block_size']}d",
        "",
    ]

    for stat_key, stat_label in stat_names.items():
        lines += [
            f"## {stat_label}",
            "",
            "| Strategy | Stat Real | Perm Mean | Perm Std | p-value | z-score | Significant? |",
            "|----------|-----------|-----------|----------|---------|---------|--------------|",
        ]
        for strat_code, strat_name in strategy_names.items():
            r = all_results[strat_code][stat_key]
            sig = "Yes ✓" if r["significant"] else "No ✗"
            real_fmt = f"{r['stat_real']:+.4f}" if stat_key != "cagr" else f"{r['stat_real']:+.2%}"
            mean_fmt = f"{r['perm_mean']:+.4f}" if stat_key != "cagr" else f"{r['perm_mean']:+.2%}"
            std_fmt  = f"{r['perm_std']:.4f}"
            lines.append(
                f"| {strat_name} | {real_fmt} | {mean_fmt} | {std_fmt} "
                f"| {r['p_value']:.4f} | {r['z_score']:+.2f} | {sig} |"
            )
        lines.append("")

    # Interpretation
    lines += ["## Interpretation", ""]

    # Per-strategy overall significance
    for strat_code, strat_name in strategy_names.items():
        sharpe_r = all_results[strat_code]["sharpe"]
        sig_count = sum(
            1 for s in ["sharpe", "cagr", "profit_factor"]
            if all_results[strat_code][s]["significant"]
        )
        conclusion = "significant across all stats" if sig_count == 3 \
            else f"significant in {sig_count}/3 stats"
        lines.append(f"- **{strat_name}**: {conclusion}  "
                     f"(Sharpe p={sharpe_r['p_value']:.4f}, z={sharpe_r['z_score']:+.2f})")
    lines.append("")

    # Most conservative test
    sharpe_pvals = {k: all_results[k]["sharpe"]["p_value"] for k in strategy_names}
    hardest = max(sharpe_pvals, key=sharpe_pvals.get)
    lines.append(
        f"**Most conservative test (Sharpe):** Strategy {hardest} — {strategy_names[hardest]} "
        f"(p={sharpe_pvals[hardest]:.4f})"
    )
    lines.append("")

    # Any p > 0.05?
    any_non_sig = [
        f"Strategy {k} / {s}"
        for k in strategy_names
        for s in ["sharpe", "cagr", "profit_factor"]
        if not all_results[k][s]["significant"]
    ]
    if any_non_sig:
        lines.append("**Non-significant results (p ≥ 0.05):**")
        for ns in any_non_sig:
            p = all_results[ns.split("/")[0].strip().split()[-1]][ns.split("/")[1].strip()]["p_value"]
            lines.append(f"  - {ns} (p={p:.4f}) — performance not distinguishable from noise at α=0.05")
    else:
        lines.append("All three strategies significant at p < 0.05 for Sharpe — R11 is robust.")
    lines.append("")

    # Robustness note
    lines += ["## Robustness Notes", ""]
    for strat_code, strat_name in strategy_names.items():
        r = all_results[strat_code]["sharpe"]
        pct_beat = r["pct_beat"] * 100
        outlier_note = " ⚠ outlier permutation found (Sharpe > 1.5× real)" if r["outlier_flag"] else ""
        lines.append(
            f"- **{strat_name}**: {pct_beat:.2f}% of permutations beat real Sharpe{outlier_note}"
        )
    lines.append("")
    lines += ["## Plots", ""]
    for code in ["a", "b", "c"]:
        lines.append(f"- `plots/r11_perm_strategy_{code}.png`")

    out = REPORTS_DIR / "r11_permutation_report.md"
    out.write_text("\n".join(lines))
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main(n_permutations: int = 5000, seed: int = 42) -> None:
    # ── Config ────────────────────────────────────────────────────────────────
    walkforward_params = _load_config()

    split3       = next(s for s in walkforward_params["splits"] if s["name"] == "split_3")
    train_start  = pd.Timestamp(str(split3["train_start"]), tz="UTC")
    train_end    = pd.Timestamp(str(split3["train_end"]),   tz="UTC")
    test_start   = pd.Timestamp(str(split3["test_start"]),  tz="UTC")
    test_end_raw = split3["test_end"]

    # ── Data ──────────────────────────────────────────────────────────────────
    btc_path = DATA_INPUT / "BTCUSDT.parquet"
    if not btc_path.exists():
        print(f"ERROR: {btc_path} not found.")
        print("  Run: conda run -n crypto_market_state kedro run --pipeline primary.spot.crypto")
        sys.exit(1)

    print("Loading BTC L3 data...")
    df = pd.read_parquet(btc_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    test_end = df.index.max() if test_end_raw is None else pd.Timestamp(str(test_end_raw), tz="UTC")

    print(f"Train: {train_start.date()} → {train_end.date()}")
    print(f"Test:  {test_start.date()} → {test_end.date()}")

    # ── Fit R11 and get real signals ──────────────────────────────────────────
    print("\nFitting R11 HMM (split_3, frozen config)...")
    real_states, daily_ret, test_idx = _fit_and_predict_r11(
        df, train_start, train_end, test_start, test_end,
    )
    real_metrics = _backtest_metrics(real_states, daily_ret)

    print(f"\nR11 Real Metrics (n={len(real_states)} test days):")
    print(f"  Sharpe:        {real_metrics['sharpe']:+.4f}")
    print(f"  CAGR:          {real_metrics['cagr']:+.2%}")
    print(f"  Profit Factor: {real_metrics['profit_factor']:.4f}")
    print(f"  Features used: {R11_FEATURES}")
    print(f"  cov_type={R11_COV_TYPE}  ordering=drawdown  entry=same_day")

    # ── Permutation tests ─────────────────────────────────────────────────────
    strategies = [
        ("A", "Strategy A — Signal Shuffle",          "strategy_a",
         lambda sig, rng: _permute_signal_shuffle(sig, rng)),
        ("B", "Strategy B — Circular Block Shift",    "strategy_b",
         lambda sig, rng: _permute_circular_shift(sig, rng)),
        ("C", "Strategy C — Stationary Block Bootstrap", "strategy_c",
         lambda sig, rng: _permute_block_bootstrap(sig, rng)),
    ]

    all_results = {}
    plot_paths  = {}

    print(f"\nRunning permutation tests ({n_permutations} per strategy)...")

    for code, label, fname, fn in strategies:
        results = _run_permutation(
            real_states, daily_ret, fn, n_permutations, label, seed=seed,
        )
        all_results[code] = results
        plot_paths[code]  = _plot_null_distributions(results, label, fname, n_permutations)
        print(f"  Plot saved: {plot_paths[code].name}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PERMUTATION TEST RESULTS — PRIMARY STAT: SHARPE RATIO")
    print("=" * 80)
    hdr = f"{'Strategy':<38} {'Stat Real':>10} {'Perm Mean':>10} {'p-value':>8} {'z-score':>8} {'Sig?':>6}"
    print(hdr)
    print("-" * 80)
    for code, label, _, _ in strategies:
        r = all_results[code]["sharpe"]
        sig = "YES" if r["significant"] else "no"
        print(
            f"{label:<38} {r['stat_real']:>+10.4f} {r['perm_mean']:>+10.4f} "
            f"{r['p_value']:>8.4f} {r['z_score']:>+8.2f} {sig:>6}"
        )
    print("=" * 80)

    print("\n" + "=" * 80)
    print("SECONDARY STATS: CAGR")
    print("=" * 80)
    for code, label, _, _ in strategies:
        r = all_results[code]["cagr"]
        sig = "YES" if r["significant"] else "no"
        print(
            f"{label:<38} {r['stat_real']:>+10.2%} {r['perm_mean']:>+10.2%} "
            f"{r['p_value']:>8.4f} {r['z_score']:>+8.2f} {sig:>6}"
        )

    print("\n" + "=" * 80)
    print("SECONDARY STATS: PROFIT FACTOR")
    print("=" * 80)
    for code, label, _, _ in strategies:
        r = all_results[code]["profit_factor"]
        sig = "YES" if r["significant"] else "no"
        print(
            f"{label:<38} {r['stat_real']:>+10.4f} {r['perm_mean']:>+10.4f} "
            f"{r['p_value']:>8.4f} {r['z_score']:>+8.2f} {sig:>6}"
        )
    print("=" * 80)

    # ── Outputs ───────────────────────────────────────────────────────────────
    meta = {
        "asset":            "BTC",
        "test_start":       str(test_start.date()),
        "test_end":         str(test_end.date()),
        "n_test":           len(real_states),
        "n_permutations":   n_permutations,
        "random_seed":      seed,
        "TC_bps":           TC * 10_000,
        "mean_block_size":  MEAN_BLOCK,
        "features_used":    R11_FEATURES,
        "covariance_type":  R11_COV_TYPE,
        "state_ordering":   "drawdown_ascending",
        "entry":            "same_day",
        "data_layer":       "03_primary",
        "real_sharpe":      real_metrics["sharpe"],
        "real_cagr":        real_metrics["cagr"],
        "real_pf":          real_metrics["profit_factor"],
        "generated":        str(date.today()),
    }

    json_path = _write_json(all_results, meta)
    md_path   = _write_markdown(all_results, meta)

    print(f"\nOutputs:")
    print(f"  {json_path.relative_to(ROOT)}")
    print(f"  {md_path.relative_to(ROOT)}")
    for path in plot_paths.values():
        print(f"  {path.relative_to(ROOT)}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R11 Permutation Test")
    parser.add_argument("--n", type=int, default=5000,
                        help="Number of permutations per strategy (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    main(n_permutations=args.n, seed=args.seed)
