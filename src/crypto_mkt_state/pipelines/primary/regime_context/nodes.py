"""
L4 Regime Context — daily BTC regime features.

Node 1  compute_r11_regime_features
        btc_spot_crypto_model_input (L3/L4 daily)  →  r11_regime_features (Memory)

        Runs frozen R11 HMM on every row using a sliding 10-row sequence
        context window (identical to SignalGenerator._build_context_X).

        Output columns:
          r11_regime, r11_prob_bull, r11_prob_bear, r11_entropy,
          regime_age_days, regime_age_log, regime_is_new,
          regime_prob_change_3d, regime_prob_change_7d

Node 2  compute_derivatives_daily_context
        btc_coinglass_features_4h (L3 4h)  →  derivatives_daily_context (Memory)

        Aggregates 4h features to daily by grouping on UTC date.
        Output columns:
          oi_momentum_mean_24h, oi_momentum_max_24h,
          oi_volatility_mean_24h,
          funding_divergence_mean_24h, funding_divergence_max_24h,
          leverage_expansion_mean_24h, leverage_expansion_max_24h,
          oi_stress_regime, funding_stress_regime, leverage_stress_regime,
          stress_score, vol_percentile_30d, vol_trend

Node 3  join_regime_context
        r11_regime_features + derivatives_daily_context  →  btc_regime_context_daily

        Inner join on UTC date.  Raises ValueError if result is empty.

Contracts:
  - Pure functions — no Kedro internals, no catalog access, no disk IO
    beyond the model pkl load (path supplied via parameters.yml).
  - All windows strictly backward-looking.
  - NaN permitted at burn-in; never filled.
  - UTC DatetimeIndex throughout.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ── Module constants ─────────────────────────────────────────────────────────

PROBA_WINDOW = 10           # sliding sequence length; matches SignalGenerator


# ── Private helpers ──────────────────────────────────────────────────────────

def _build_mapa(model, features: list[str]) -> dict[int, int]:
    """
    Recompute {original_state → ordered_label} from model.means_.

    Rule: sort states by drawdown mean ascending (lower = less negative =
    recovering from peak).  Lowest → Bull(1),  highest → Bear(0).
    Matches SignalGenerator._build_mapa exactly.
    """
    feat_idx = features.index("drawdown") if "drawdown" in features else 0
    dd_means = {s: float(model.means_[s][feat_idx])
                for s in range(model.n_components)}
    asc_states = sorted(dd_means, key=dd_means.get)
    n = len(asc_states)
    return {s: (n - 1 - rank) for rank, s in enumerate(asc_states)}


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """
    For each position t, compute the fraction of the previous (window-1)
    values that are strictly less than series[t].

    Returns values in [0, 1].  NaN for the first (window-1) positions.
    """
    def _rank_last(arr: np.ndarray) -> float:
        if len(arr) <= 1:
            return np.nan
        return float((arr[:-1] < arr[-1]).sum() / (len(arr) - 1))

    return series.rolling(window, min_periods=2).apply(_rank_last, raw=True)


# ── Node 1 ───────────────────────────────────────────────────────────────────

def compute_r11_regime_features(
    df: pd.DataFrame,
    r11_model_path: str,
) -> pd.DataFrame:
    """
    Run the frozen R11 HMM over every row of the daily feature DataFrame,
    using a sliding PROBA_WINDOW-row sequence context per prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Daily L3 feature DataFrame (DatetimeIndex UTC).  Must contain all
        features the frozen model was trained on.
    r11_model_path : str
        Path (relative to project root) to the frozen r11_hmm_model.pkl.

    Returns
    -------
    pd.DataFrame
        Nine-column DataFrame with the same DatetimeIndex as df.
        Rows where any HMM feature is NaN produce NaN across all output cols.
    """
    model_path = Path(r11_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"[regime_context] Frozen R11 model not found: {model_path}\n"
            "  Run: python scripts/paper_trading/freeze_r11_model.py"
        )

    with open(model_path, "rb") as fh:
        state = pickle.load(fh)

    model    = state["model"]
    scaler   = state["scaler"]
    features = state["features"]

    mapa     = _build_mapa(model, features)
    inv_mapa = {v: k for k, v in mapa.items()}   # {ordered → original}

    df = df.sort_index()
    n  = len(df)

    r11_regime_arr    = np.full(n, np.nan)
    r11_prob_bull_arr = np.full(n, np.nan)
    r11_prob_bear_arr = np.full(n, np.nan)

    for i in range(n):
        row = df.iloc[i]
        if any(pd.isna(row[f]) for f in features):
            continue

        # ── Build sliding-window context (mirrors _build_context_X) ─────────
        ctx_start = max(0, i - (PROBA_WINDOW - 1))
        context   = df.iloc[ctx_start:i].dropna(subset=features)

        current_df     = pd.DataFrame([row[features].values], columns=features).astype(float)
        current_scaled = scaler.transform(current_df)

        if len(context) > 0:
            ctx_scaled = scaler.transform(
                context[features].astype(float)
            )
            X_seq = np.vstack([ctx_scaled, current_scaled])
        else:
            X_seq = current_scaled

        raw_state     = int(model.predict(X_seq)[-1])
        ordered_state = mapa[raw_state]
        raw_proba     = model.predict_proba(X_seq)[-1]
        p_bull        = float(raw_proba[inv_mapa[1]])
        p_bear        = float(raw_proba[inv_mapa[0]])

        r11_regime_arr[i]    = ordered_state
        r11_prob_bull_arr[i] = p_bull
        r11_prob_bear_arr[i] = p_bear

    out = pd.DataFrame(index=df.index)
    out["r11_regime"]    = r11_regime_arr
    out["r11_prob_bull"] = r11_prob_bull_arr
    out["r11_prob_bear"] = r11_prob_bear_arr

    # ── Entropy ──────────────────────────────────────────────────────────────
    pb = np.clip(out["r11_prob_bull"], 1e-10, 1 - 1e-10)
    pk = np.clip(out["r11_prob_bear"], 1e-10, 1 - 1e-10)
    out["r11_entropy"] = -(pb * np.log2(pb) + pk * np.log2(pk))

    # ── Regime age ───────────────────────────────────────────────────────────
    valid_mask = out["r11_regime"].notna()
    change_flag = valid_mask & (
        (out["r11_regime"] != out["r11_regime"].shift(1))
        | ~valid_mask.shift(1).fillna(False)
    )
    run_id     = change_flag.cumsum()
    age_days   = out["r11_regime"].groupby(run_id).cumcount() + 1
    age_days   = age_days.where(valid_mask, other=np.nan)

    out["regime_age_days"] = age_days
    out["regime_age_log"]  = np.log(age_days + 1)
    out["regime_is_new"]   = (age_days <= 2).astype(float).where(valid_mask, other=np.nan)

    # ── Probability momentum ─────────────────────────────────────────────────
    out["regime_prob_change_3d"] = out["r11_prob_bull"] - out["r11_prob_bull"].shift(3)
    out["regime_prob_change_7d"] = out["r11_prob_bull"] - out["r11_prob_bull"].shift(7)

    print(
        f"[regime_context] R11 features computed — {n} rows, "
        f"{valid_mask.sum()} valid ({n - valid_mask.sum()} skipped / NaN features)."
    )
    return out


# ── Node 2 ───────────────────────────────────────────────────────────────────

def compute_derivatives_daily_context(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 4h CoinGlass features to daily (6 × 4h candles per UTC day).

    Parameters
    ----------
    df_4h : pd.DataFrame
        L3 4h derivatives DataFrame (DatetimeIndex UTC) with columns:
        oi_momentum_12h, oi_volatility_24h, funding_divergence,
        leverage_expansion.

    Returns
    -------
    pd.DataFrame
        Daily aggregated features (DatetimeIndex UTC, normalized to midnight).
        NaN permitted at burn-in.  Never filled.
    """
    required = [
        "oi_momentum_12h",
        "oi_volatility_24h",
        "funding_divergence",
        "leverage_expansion",
    ]
    missing = [c for c in required if c not in df_4h.columns]
    if missing:
        raise ValueError(
            f"[regime_context] Missing 4h feature columns: {missing}"
        )

    # ── Aggregate 4h → daily ─────────────────────────────────────────────────
    daily_date = df_4h.index.normalize()   # floor each 4h timestamp to UTC midnight

    agg = df_4h.groupby(daily_date).agg(
        oi_momentum_mean_24h    = ("oi_momentum_12h",   "mean"),
        oi_momentum_max_24h     = ("oi_momentum_12h",   "max"),
        oi_volatility_mean_24h  = ("oi_volatility_24h", "mean"),
        funding_divergence_mean_24h = ("funding_divergence", "mean"),
        funding_divergence_max_24h  = ("funding_divergence", "max"),
        leverage_expansion_mean_24h = ("leverage_expansion", "mean"),
        leverage_expansion_max_24h  = ("leverage_expansion", "max"),
    )

    # Ensure UTC-aware DatetimeIndex
    if agg.index.tz is None:
        agg.index = agg.index.tz_localize("UTC")
    agg.index.name = "timestamp"

    # ── Stress flags (rolling 30-day percentile thresholds) ──────────────────
    w = 30

    oi_lo_10   = agg["oi_momentum_mean_24h"].rolling(w, min_periods=2).quantile(0.10)
    fund_hi_90 = agg["funding_divergence_mean_24h"].rolling(w, min_periods=2).quantile(0.90)
    lev_hi_90  = agg["leverage_expansion_mean_24h"].rolling(w, min_periods=2).quantile(0.90)

    agg["oi_stress_regime"]       = (agg["oi_momentum_mean_24h"] < oi_lo_10).astype(float)
    agg["funding_stress_regime"]  = (agg["funding_divergence_mean_24h"] > fund_hi_90).astype(float)
    agg["leverage_stress_regime"] = (agg["leverage_expansion_mean_24h"] > lev_hi_90).astype(float)
    agg["stress_score"]           = (
        agg["oi_stress_regime"]
        + agg["funding_stress_regime"]
        + agg["leverage_stress_regime"]
    )

    # ── Volatility regime ─────────────────────────────────────────────────────
    agg["vol_percentile_30d"] = _rolling_percentile_rank(
        agg["oi_volatility_mean_24h"], window=w
    )
    agg["vol_trend"] = agg["vol_percentile_30d"] - agg["vol_percentile_30d"].shift(5)

    print(
        f"[regime_context] Derivatives daily context — {len(agg)} days "
        f"({agg.index[0].date()} → {agg.index[-1].date()})"
    )
    return agg


# ── Node 3 ───────────────────────────────────────────────────────────────────

def join_regime_context(
    r11_features: pd.DataFrame,
    deriv_context: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join R11 regime features with derivatives daily context on date.

    Parameters
    ----------
    r11_features : pd.DataFrame
        Output of compute_r11_regime_features (daily, UTC DatetimeIndex).
    deriv_context : pd.DataFrame
        Output of compute_derivatives_daily_context (daily, UTC DatetimeIndex).

    Returns
    -------
    pd.DataFrame
        Joined DataFrame sorted by date ascending.

    Raises
    ------
    ValueError
        If the inner join produces an empty DataFrame.
    """
    # Normalise both indices to UTC midnight before joining
    if r11_features.index.tz is None:
        r11_features = r11_features.copy()
        r11_features.index = r11_features.index.tz_localize("UTC")
    r11_norm = r11_features.copy()
    r11_norm.index = r11_norm.index.normalize()

    if deriv_context.index.tz is None:
        deriv_context = deriv_context.copy()
        deriv_context.index = deriv_context.index.tz_localize("UTC")
    deriv_norm = deriv_context.copy()
    deriv_norm.index = deriv_norm.index.normalize()

    joined = r11_norm.join(deriv_norm, how="inner")

    if joined.empty:
        raise ValueError(
            "[regime_context] join_regime_context: inner join produced an empty "
            "DataFrame.  Check date coverage overlap between R11 features "
            f"({r11_norm.index.min().date()} → {r11_norm.index.max().date()}) and "
            f"derivatives context ({deriv_norm.index.min().date()} → "
            f"{deriv_norm.index.max().date()})."
        )

    joined = joined.sort_index(ascending=True)
    joined.index.name = "timestamp"

    print(
        f"[regime_context] Join complete — {len(joined)} rows "
        f"({joined.index[0].date()} → {joined.index[-1].date()}), "
        f"{joined.shape[1]} columns."
    )
    return joined
