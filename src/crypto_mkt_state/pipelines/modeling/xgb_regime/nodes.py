"""
XGBoost Regime Classifier — Fase 2A Baseline

Two nodes:
  prepare_xgb_inputs  — join v2 model input + HMM states → model-ready DataFrame
  run_walkforward_xgb — 3 splits × 2 horizons walk-forward with purging

Features (12):
  BTC core (6):       log_return, vol_short, vol_ratio, drawdown, volume_z, slope_21d
  Cross-crypto (4):   eth_btc_rs_5d, alt_dispersion_5d, mean_alt_beta_30d, mean_alt_dd_rel
  HMM context (2):    hmm_state, hmm_bull_prob

Labels (3-class):
  Bear=0   fwd_ret < -bear_threshold
  Neutral=1
  Bull=2   fwd_ret >  bull_threshold
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Any

import xgboost as xgb


_XGB_FEATURES = [
    # BTC core
    "log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d",
    # Cross-crypto
    "eth_btc_rs_5d", "alt_dispersion_5d", "mean_alt_beta_30d", "mean_alt_dd_rel",
    # HMM context
    "hmm_state", "hmm_bull_prob",
]


def prepare_xgb_inputs(
    btc_df: pd.DataFrame,
    hmm_states: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join btc_spot_daily_model_input_v2 + btc_regime_states.

    btc_df     : freq="D" UTC, OHLCV + feature cols (v2 model input)
    hmm_states : freq="D" UTC from 2023-01-01, columns state + bull_prob

    Renames state→hmm_state, bull_prob→hmm_bull_prob.
    Inner join restricts to 2023+ (HMM fit range, matches walk-forward splits).
    Drops rows where any of the 12 XGB features is NaN.
    """
    if btc_df.index.tz is None:
        btc_df = btc_df.tz_localize("UTC")
    if hmm_states.index.tz is None:
        hmm_states = hmm_states.tz_localize("UTC")

    states = hmm_states.rename(
        columns={"state": "hmm_state", "bull_prob": "hmm_bull_prob"}
    )
    combined = btc_df.join(states, how="inner")

    feature_cols = [f for f in _XGB_FEATURES if f in combined.columns]
    missing = set(_XGB_FEATURES) - set(combined.columns)
    if missing:
        raise ValueError(
            f"prepare_xgb_inputs: missing features after join: {sorted(missing)}"
        )

    combined = combined.dropna(subset=feature_cols)

    if combined.empty:
        raise ValueError("prepare_xgb_inputs: no rows after join + dropna.")

    return combined


def _make_labels(
    fwd_ret: pd.Series,
    bull_thr: float,
    bear_thr: float,
) -> pd.Series:
    """
    3-class label: Bear=0, Neutral=1, Bull=2.

    fwd_ret > bull_thr   → Bull=2
    fwd_ret < -bear_thr  → Bear=0
    else                 → Neutral=1
    """
    labels = pd.Series(1, index=fwd_ret.index, dtype=int)
    labels[fwd_ret > bull_thr] = 2
    labels[fwd_ret < -bear_thr] = 0
    return labels


def run_walkforward_xgb(
    df: pd.DataFrame,
    walkforward_params: dict[str, Any],
    xgb_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward XGBoost regime classifier — 3 splits × 2 horizons.

    horizons from xgb_params.horizons:
      horizon_2d: horizon_days=2, bull_threshold=0.010, bear_threshold=0.007
      horizon_5d: horizon_days=5, bull_threshold=0.015, bear_threshold=0.010

    Purging: purged_train_end = min(train_end, test_start - timedelta(days=horizon+embargo))

    Returns:
      metrics_df    — one row per (split, horizon) with delta, E_bull, E_bear, precisions, pcts
      importance_df — one row per (feature, split, horizon) with importance score
    """
    embargo_days: int = walkforward_params["embargo_days"]
    horizons_cfg: list[dict] = xgb_params["horizons"]
    features: list[str] = xgb_params["features"]

    n_estimators: int    = xgb_params.get("n_estimators", 200)
    max_depth: int       = xgb_params.get("max_depth", 4)
    learning_rate: float = xgb_params.get("learning_rate", 0.05)
    subsample: float     = xgb_params.get("subsample", 0.8)
    colsample: float     = xgb_params.get("colsample_bytree", 0.8)
    random_state: int    = xgb_params.get("random_state", 42)

    df = df.copy().sort_index()
    features = [f for f in features if f in df.columns]

    metrics_rows: list[dict] = []
    importance_rows: list[dict] = []

    for horizon_cfg in horizons_cfg:
        horizon: int      = int(horizon_cfg["horizon_days"])
        bull_thr: float   = float(horizon_cfg["bull_threshold"])
        bear_thr: float   = float(horizon_cfg["bear_threshold"])
        horizon_name: str = horizon_cfg["name"]

        # N-day cumulative log return (prospective)
        fwd_ret = sum(df["log_return"].shift(-k) for k in range(1, horizon + 1))
        labels  = _make_labels(fwd_ret, bull_thr, bear_thr)

        # Only rows where forward return is defined and all features are valid
        valid_mask = fwd_ret.notna() & df[features].notna().all(axis=1)
        df_v   = df.loc[valid_mask]
        y_v    = labels.loc[valid_mask].astype(int)
        fwd_v  = fwd_ret.loc[valid_mask]

        purge_total = horizon + embargo_days

        for split_cfg in walkforward_params["splits"]:
            split_name: str = split_cfg["name"]

            train_start  = pd.Timestamp(str(split_cfg["train_start"]), tz="UTC")
            train_end    = pd.Timestamp(str(split_cfg["train_end"]),   tz="UTC")
            test_start   = pd.Timestamp(str(split_cfg["test_start"]),  tz="UTC")
            test_end_raw = split_cfg["test_end"]

            test_end = (
                df_v.index.max()
                if test_end_raw is None
                else pd.Timestamp(str(test_end_raw), tz="UTC")
            )

            purged_train_end = min(
                train_end,
                test_start - timedelta(days=purge_total),
            )

            train_df = df_v.loc[train_start:purged_train_end]
            test_df  = df_v.loc[test_start:test_end]
            y_train  = y_v.loc[train_df.index]
            y_test   = y_v.loc[test_df.index]
            fwd_test = fwd_v.loc[test_df.index]

            if len(train_df) < 30 or len(test_df) < 5:
                continue

            X_train = train_df[features].values
            X_test  = test_df[features].values

            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample,
                eval_metric="mlogloss",
                verbosity=0,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train.values)

            preds = model.predict(X_test)  # 0=Bear, 1=Neutral, 2=Bull

            bull_mask = preds == 2
            bear_mask = preds == 0
            neu_mask  = preds == 1

            n_test = len(test_df)

            E_bull = fwd_test[bull_mask].mean() if bull_mask.sum() > 0 else np.nan
            E_bear = fwd_test[bear_mask].mean() if bear_mask.sum() > 0 else np.nan
            delta  = (E_bull - E_bear) if not (np.isnan(E_bull) or np.isnan(E_bear)) else np.nan

            bull_prec  = float((fwd_test[bull_mask] > 0).mean()) if bull_mask.sum() > 0 else np.nan
            bear_prec  = float((fwd_test[bear_mask] < 0).mean()) if bear_mask.sum() > 0 else np.nan

            bull_pct   = bull_mask.sum() / n_test
            bear_pct   = bear_mask.sum() / n_test
            neutro_pct = neu_mask.sum() / n_test

            metrics_rows.append(
                {
                    "split":      split_name,
                    "horizon":    horizon_name,
                    "n_train":    len(train_df),
                    "n_test":     n_test,
                    "delta":      delta,
                    "E_bull":     E_bull,
                    "E_bear":     E_bear,
                    "bull_prec":  bull_prec,
                    "bear_prec":  bear_prec,
                    "bull_pct":   bull_pct,
                    "bear_pct":   bear_pct,
                    "neutro_pct": neutro_pct,
                }
            )

            imp = model.feature_importances_
            for feat, score in zip(features, imp):
                importance_rows.append(
                    {
                        "feature":    feat,
                        "importance": float(score),
                        "split":      split_name,
                        "horizon":    horizon_name,
                    }
                )

    metrics_df    = pd.DataFrame(metrics_rows)
    importance_df = pd.DataFrame(importance_rows)

    return metrics_df, importance_df
