"""
Freeze R11 HMM model — one-time script.

Trains HMM on 2023-01-01 → 2024-12-31 using L3 data and saves:
  scripts/paper_trading/state/r11_hmm_model.pkl

The pickle contains:
  model    : trained GaussianHMM
  scaler   : fitted StandardScaler
  mapa     : {original_state → ordered_label}  (0=Bear, 1=Bull)
  features : list of feature names (order-sensitive)
  meta     : training metadata

Run once before starting paper trading:
  python scripts/paper_trading/freeze_r11_model.py
"""
from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

ROOT       = Path(__file__).resolve().parents[2]
L3_PATH    = ROOT / "data" / "03_primary" / "spot" / "daily" / "BTCUSDT.parquet"
CONF_FILE  = ROOT / "conf" / "base" / "parameters.yml"
STATE_DIR  = Path(__file__).parent / "state"
OUT_PKL    = STATE_DIR / "r11_hmm_model.pkl"

R11_FEATURES     = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]
R11_N_STATES     = 2
R11_COV_TYPE     = "diag"
R11_N_ITER       = 1000
R11_RANDOM_STATE = 42
EXPECTED_SHARPE  = 0.775


def _ordenar_estados(model: GaussianHMM, features: list[str]) -> dict[int, int]:
    idx     = features.index("drawdown") if "drawdown" in features else 0
    vals    = {s: model.means_[s][idx] for s in range(model.n_components)}
    ordered = sorted(vals, key=vals.get)
    return {orig: new for new, orig in enumerate(ordered)}


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONF_FILE) as fh:
        cfg = yaml.safe_load(fh)
    split3      = next(s for s in cfg["walkforward"]["splits"] if s["name"] == "split_3")
    train_start = pd.Timestamp(str(split3["train_start"]), tz="UTC")
    train_end   = pd.Timestamp(str(split3["train_end"]),   tz="UTC")
    test_start  = pd.Timestamp(str(split3["test_start"]),  tz="UTC")

    print(f"Loading L3 data: {L3_PATH}")
    df = pd.read_parquet(L3_PATH)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    features = [f for f in R11_FEATURES if f in df.columns]
    missing  = [f for f in R11_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in L3: {missing}")

    df_c     = df.sort_index().dropna(subset=features)
    train_df = df_c.loc[train_start:train_end]
    test_df  = df_c.loc[test_start:].dropna(subset=["log_return"])

    print(f"Train: {train_start.date()} → {train_end.date()}  N={len(train_df)}")
    print(f"Test:  {test_start.date()} →                     N={len(test_df)}")

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

    # Quick Sharpe verification (same-day, TC=12bps, f=0.75)
    TC        = 0.0012
    log_ret   = test_df["log_return"].values
    position  = np.where(states == 1, 0.75, 0.0)
    prev_pos  = np.concatenate([[0.0], position[:-1]])
    net_ret   = position * log_ret - TC * np.abs(position - prev_pos)
    sharpe    = float((net_ret.mean() / (net_ret.std() + 1e-10)) * np.sqrt(252))

    print(f"\nVerification (f=0.75, TC=12bps):  Sharpe={sharpe:+.4f}  "
          f"(expected ≈ {EXPECTED_SHARPE})")
    if abs(sharpe - EXPECTED_SHARPE) > 0.05:
        print("WARNING: Sharpe deviates > 0.05 from expected baseline.")

    state = {
        "model":      model,
        "scaler":     scaler,
        "mapa":       mapa,
        "features":   features,
        "meta": {
            "covariance_type": R11_COV_TYPE,
            "n_states":        R11_N_STATES,
            "n_iter":          R11_N_ITER,
            "random_state":    R11_RANDOM_STATE,
            "train_start":     str(train_start.date()),
            "train_end":       str(train_end.date()),
            "n_train":         len(train_df),
            "verified_sharpe": round(sharpe, 4),
            "frozen_date":     str(date.today()),
        },
    }

    with open(OUT_PKL, "wb") as fh:
        pickle.dump(state, fh, protocol=5)

    print(f"\nFrozen model saved: {OUT_PKL.relative_to(ROOT)}")
    print("HMM will never be retrained during paper trading.")


if __name__ == "__main__":
    main()
