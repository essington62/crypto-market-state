"""
XGBoost R5 — Modelo de produção (Fase 1B).

Three nodes:
  prepare_xgb_r5_input  — join model input + HMM states, compute regime_strength
  train_xgb_r5          — train XGBoost binary classifier (label_5d)
  generate_daily_signal — apply model + decision rule to full history

Config: BASE + STRUCTURE + MOMENTUM (21 features)
Label:  return_5d > 0  (sum of log_return shifted -1 to -5)
Approved: 2/3 walk-forward splits, delta médio +0.0139.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


def prepare_xgb_r5_input(
    model_input: pd.DataFrame,
    hmm_states: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join btc_spot_daily_model_input (with technical features) + HMM states.

    model_input : L4 model input — includes technical features from R5 L3 additions
    hmm_states  : btc_regime_states — columns: state, bull_prob

    Renames state→hmm_state.
    Computes regime_strength = abs(bull_prob - 0.5).
    Inner join restricts to dates where HMM has run (2023+).
    """
    if model_input.index.tz is None:
        model_input = model_input.tz_localize("UTC")
    if hmm_states.index.tz is None:
        hmm_states = hmm_states.tz_localize("UTC")

    states = hmm_states.rename(columns={"state": "hmm_state"})
    df = model_input.join(states[["hmm_state", "bull_prob"]], how="inner")

    if "hmm_state" not in df.columns or "bull_prob" not in df.columns:
        raise ValueError(
            "prepare_xgb_r5_input: hmm_state or bull_prob missing after join. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    df["regime_strength"] = (df["bull_prob"] - 0.5).abs()

    if df.empty:
        raise ValueError("prepare_xgb_r5_input: no rows after join.")

    return df


def train_xgb_r5(
    df: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[Any, pd.DataFrame]:
    """
    Train XGBoost R5 binary classifier on label_5d (return_5d > 0).

    Training period: params["train_start"] onward.
    Excludes the last 5 rows (no valid label due to shift(-5)).

    Returns:
      clf           — fitted XGBClassifier
      importance_df — feature importance as DataFrame (feature, importance)
    """
    train_start: str = params["train_start"]
    features: list[str] = params["features"]
    model_cfg: dict = params["model"]

    # Build label — 5d forward log return
    df = df.copy()
    df["return_5d"] = df["log_return"].rolling(5).sum().shift(-5)
    df["label_5d"]  = (df["return_5d"] > 0).astype(int)

    # Select features available in df (guard against missing cols)
    features = [f for f in features if f in df.columns]
    missing  = set(params["features"]) - set(features)
    if missing:
        raise ValueError(
            f"train_xgb_r5: features missing from input: {sorted(missing)}"
        )

    train = df[train_start:][features + ["label_5d"]].dropna()

    if len(train) < 50:
        raise ValueError(
            f"train_xgb_r5: insufficient training rows ({len(train)} < 50)."
        )

    clf = xgb.XGBClassifier(
        n_estimators=int(model_cfg["n_estimators"]),
        max_depth=int(model_cfg["max_depth"]),
        learning_rate=float(model_cfg["learning_rate"]),
        subsample=float(model_cfg["subsample"]),
        colsample_bytree=float(model_cfg["colsample_bytree"]),
        min_child_weight=int(model_cfg["min_child_weight"]),
        gamma=float(model_cfg["gamma"]),
        eval_metric="logloss",
        random_state=int(model_cfg["random_state"]),
        verbosity=0,
    )
    clf.fit(train[features], train["label_5d"], verbose=False)

    importance_df = pd.DataFrame(
        {"feature": features, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    n_train = len(train)
    train_end = train.index.max().date()
    top5 = importance_df.head(5)["feature"].tolist()

    print(f"XGB R5 trained: n={n_train} | train_end={train_end}")
    print(f"Features: {len(features)} | Top 5: {top5}")
    print(
        "\n".join(
            f"  {row.feature:<35} {row.importance:.4f}"
            for row in importance_df.head(10).itertuples()
        )
    )

    return clf, importance_df


def generate_daily_signal(
    df: pd.DataFrame,
    model: Any,
    params: dict[str, Any],
) -> pd.DataFrame:
    """
    Apply XGBoost R5 + HMM decision rule to produce a daily signal DataFrame.

    Decision rule (Camada 3):
      LONG_STRONG:  regime=BULL, bull_prob > strong_bull_prob, prob_up > strong_prob_threshold
      LONG:         regime=BULL, prob_up > threshold_long
      SHORT_STRONG: regime=BEAR, bull_prob < strong_bear_prob, prob_up < strong_prob_threshold
      SHORT:        regime=BEAR, prob_up < threshold_short
      NEUTRAL:      any other combination

    Output columns: hmm_state, bull_prob, regime_strength, prob_up_5d, signal
    """
    features: list[str] = params["features"]
    sig_cfg: dict = params["signal"]

    threshold_long: float  = float(sig_cfg["threshold_long"])
    threshold_short: float = float(sig_cfg["threshold_short"])
    strong_bull_prob: float = float(sig_cfg["strong_bull_prob"])
    strong_bear_prob: float = float(sig_cfg["strong_bear_prob"])
    strong_prob_thr: float  = float(sig_cfg["strong_prob_threshold"])

    features = [f for f in features if f in df.columns]

    valid     = df[features].dropna()
    proba_arr = model.predict_proba(valid[features])[:, 1]
    prob_s    = pd.Series(proba_arr, index=valid.index, name="prob_up_5d")

    out = df[["hmm_state", "bull_prob", "regime_strength"]].join(prob_s, how="left")

    # Sanity check
    n_invalid = (~out["prob_up_5d"].between(0.0, 1.0, inclusive="both")).sum()
    if n_invalid > 0:
        raise ValueError(
            f"generate_daily_signal: {n_invalid} prob_up_5d values outside [0, 1]."
        )

    def _get_signal(row: pd.Series) -> str:
        if pd.isna(row["prob_up_5d"]):
            return "NO_DATA"
        regime = row["hmm_state"]
        prob   = row["prob_up_5d"]
        bull_p = row["bull_prob"]

        if regime == 1:
            if prob > strong_prob_thr and bull_p > strong_bull_prob:
                return "LONG_STRONG"
            if prob > threshold_long:
                return "LONG"
        elif regime == 0:
            if prob < (1 - strong_prob_thr) and bull_p < strong_bear_prob:
                return "SHORT_STRONG"
            if prob < threshold_short:
                return "SHORT"
        return "NEUTRAL"

    out["signal"] = out.apply(_get_signal, axis=1)

    # Print today's output
    today = out.dropna(subset=["prob_up_5d"]).iloc[-1]
    reg_name = "BULL" if today["hmm_state"] == 1 else "BEAR"
    print("\n" + "=" * 50)
    print("OUTPUT OPERACIONAL — HOJE")
    print("=" * 50)
    print(f"  Data:              {out.dropna(subset=['prob_up_5d']).index[-1].date()}")
    print(f"  Regime:            {reg_name}")
    print(f"  Bull probability:  {today['bull_prob']:.1%}")
    print(f"  Regime strength:   {today['regime_strength']:.3f}")
    print(f"  Prob subida 5d:    {today['prob_up_5d']:.1%}")
    print(f"  SINAL:             {today['signal']}")
    print("=" * 50)

    print("\nDistribuição de sinais (histórico completo):")
    print(out["signal"].value_counts().to_string())

    return out[["hmm_state", "bull_prob", "regime_strength", "prob_up_5d", "signal"]]
