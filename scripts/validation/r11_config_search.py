"""
R11 Config Search — grid over covariance_type × entry_mode to find Sharpe ≈ 0.775.

Fixed: n_states=2, n_iter=1000, random_state=42, TC=12bps
       features=[log_return, vol_short, vol_ratio, drawdown, volume_z, slope_21d]
       train 2023-01-01→2024-12-31, test 2025-01-01→present
       state_ordering: log_return ascending (lowest=Bear, highest=Bull)

Output: data/06_reports/r11_config_search.json
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
CONF_FILE   = ROOT / "conf" / "base" / "parameters.yml"
DATA_INPUT  = ROOT / "data" / "04_model_input" / "spot" / "daily"
REPORTS_DIR = ROOT / "data" / "06_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Fixed params ───────────────────────────────────────────────────────────────
FEATURES     = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]
N_STATES     = 2
N_ITER       = 1000
RANDOM_STATE = 42
TC           = 0.0012
PERIODS_YR   = 252
TARGET_SHARPE = 0.775
MATCH_TOL     = 0.05   # |sharpe - 0.775| < 0.05 → "match"

COVARIANCE_TYPES = ["diag", "full", "tied", "spherical"]
ENTRY_MODES      = ["same_day", "next_day"]


def _load_split3():
    with open(CONF_FILE) as fh:
        cfg = yaml.safe_load(fh)
    split3 = next(s for s in cfg["walkforward"]["splits"] if s["name"] == "split_3")
    train_start = pd.Timestamp(str(split3["train_start"]), tz="UTC")
    train_end   = pd.Timestamp(str(split3["train_end"]),   tz="UTC")
    test_start  = pd.Timestamp(str(split3["test_start"]),  tz="UTC")
    test_end_raw = split3["test_end"]
    return train_start, train_end, test_start, test_end_raw


def _order_by_log_return(model: GaussianHMM) -> dict[int, int]:
    lr_idx  = FEATURES.index("log_return")
    vals    = {s: model.means_[s][lr_idx] for s in range(N_STATES)}
    ordered = sorted(vals, key=vals.get)
    return {orig: new for new, orig in enumerate(ordered)}


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


def _backtest(states: np.ndarray, log_ret: np.ndarray, entry: str) -> dict:
    n = len(log_ret)
    if entry == "same_day":
        position = (states == 1).astype(float)
    else:  # next_day
        position      = np.empty(n, dtype=float)
        position[0]   = 0.0
        position[1:]  = (states[:-1] == 1).astype(float)

    prev_pos  = np.concatenate([[0.0], position[:-1]])
    tc_cost   = TC * np.abs(position - prev_pos)
    net_ret   = position * log_ret - tc_cost

    equity  = (1 + net_ret).cumprod()
    n_years = n / PERIODS_YR
    cagr    = float(equity[-1] ** (1 / n_years) - 1)
    sharpe  = float((net_ret.mean() / (net_ret.std() + 1e-10)) * np.sqrt(PERIODS_YR))

    peak   = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min())

    active   = net_ret[net_ret != 0]
    hit_rate = float((active > 0).mean()) if len(active) > 0 else float("nan")
    gains    = active[active > 0].sum()
    losses   = abs(active[active < 0].sum())
    pf       = float(gains / losses) if losses > 0 else float("nan")
    n_trades = int((np.abs(position - prev_pos) > 0).sum())

    return {
        "cagr": cagr, "sharpe": sharpe, "max_drawdown": max_dd,
        "hit_rate": hit_rate, "profit_factor": pf, "n_trades": n_trades,
        "bull_days": int((states == 1).sum()),
        "bull_dur":  _run_durations(states, 1),
        "bear_dur":  _run_durations(states, 0),
    }


def main():
    train_start, train_end, test_start, test_end_raw = _load_split3()

    df = pd.read_parquet(DATA_INPUT / "BTCUSDT.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    test_end = df.index.max() if test_end_raw is None else pd.Timestamp(str(test_end_raw), tz="UTC")

    df_clean = df.sort_index().dropna(subset=FEATURES)
    train_df = df_clean.loc[train_start:train_end]
    test_df  = df_clean.loc[test_start:test_end].dropna(subset=["log_return"])
    log_ret  = test_df["log_return"].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURES])
    X_test  = scaler.transform(test_df[FEATURES])

    print(f"Train: {train_start.date()} → {train_end.date()}  N={len(train_df)}")
    print(f"Test:  {test_start.date()} → {test_end.date()}  N={len(test_df)}")
    print(f"Target Sharpe: {TARGET_SHARPE}  (match if |delta| < {MATCH_TOL})\n")

    rows    = []
    results = []

    for cov in COVARIANCE_TYPES:
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type=cov,
            n_iter=N_ITER,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train)

        mapa       = _order_by_log_return(model)
        raw_states = model.predict(X_test)
        states     = np.array([mapa[s] for s in raw_states])

        lr_idx   = FEATURES.index("log_return")
        idx_bear = next(k for k, v in mapa.items() if v == 0)
        idx_bull = next(k for k, v in mapa.items() if v == 1)

        for entry in ENTRY_MODES:
            bt    = _backtest(states, log_ret, entry)
            delta = abs(bt["sharpe"] - TARGET_SHARPE)
            match = "YES ✓" if delta < MATCH_TOL else f"no  (Δ={delta:+.3f})"

            rows.append({
                "cov":   cov,
                "entry": entry,
                "sharpe": bt["sharpe"],
                "cagr":   bt["cagr"],
                "max_dd": bt["max_drawdown"],
                "match":  match,
                "delta":  delta,
            })
            results.append({
                "covariance_type":  cov,
                "entry_mode":       entry,
                "bear_mean_logret": float(model.means_[idx_bear][lr_idx]),
                "bull_mean_logret": float(model.means_[idx_bull][lr_idx]),
                **{k: v for k, v in bt.items()},
                "match_0775":       delta < MATCH_TOL,
                "delta_to_target":  delta,
            })

    # ── Print table ────────────────────────────────────────────────────────────
    W = "─"
    h = f"┌{'─'*18}┬{'─'*11}┬{'─'*8}┬{'─'*10}┬{'─'*8}┬{'─'*16}┐"
    s = f"├{'─'*18}┼{'─'*11}┼{'─'*8}┼{'─'*10}┼{'─'*8}┼{'─'*16}┤"
    f_= f"└{'─'*18}┴{'─'*11}┴{'─'*8}┴{'─'*10}┴{'─'*8}┴{'─'*16}┘"
    hdr= f"│ {'covariance_type':<16} │ {'entry':<9} │ {'Sharpe':>6} │ {'CAGR':>8} │ {'MaxDD':>6} │ {'Match 0.775?':<14} │"

    print(h)
    print(hdr)

    best_delta = min(r["delta"] for r in rows)
    for i, r in enumerate(rows):
        marker = " ◄ BEST" if r["delta"] == best_delta else ""
        line = (
            f"│ {r['cov']:<16} │ {r['entry']:<9} │ "
            f"{r['sharpe']:>+6.3f} │ {r['cagr']:>+8.2%} │ "
            f"{r['max_dd']:>+6.2%} │ {r['match']:<14} │"
            f"{marker}"
        )
        print(s if i == 0 else s)
        print(line)

    print(f_)

    best = min(rows, key=lambda r: r["delta"])
    print(f"\n→ Closest match: covariance_type='{best['cov']}' + entry='{best['entry']}'")
    print(f"  Sharpe={best['sharpe']:+.3f}  CAGR={best['cagr']:+.2%}  MaxDD={best['max_dd']:+.2%}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    best_result = min(results, key=lambda r: r["delta_to_target"])
    payload = {
        "generated":    str(date.today()),
        "target_sharpe": TARGET_SHARPE,
        "match_tolerance": MATCH_TOL,
        "test_start":   str(test_start.date()),
        "test_end":     str(test_end.date()),
        "n_test":       len(test_df),
        "fixed_params": {
            "features":     FEATURES,
            "n_states":     N_STATES,
            "n_iter":       N_ITER,
            "random_state": RANDOM_STATE,
            "TC_bps":       TC * 10_000,
        },
        "best_config": {
            "covariance_type": best_result["covariance_type"],
            "entry_mode":      best_result["entry_mode"],
            "sharpe":          best_result["sharpe"],
            "cagr":            best_result["cagr"],
            "max_drawdown":    best_result["max_drawdown"],
            "delta_to_target": best_result["delta_to_target"],
        },
        "all_results": results,
    }

    out = REPORTS_DIR / "r11_config_search.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nSaved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
