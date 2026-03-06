import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Any

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def _ordenar_estados(model: GaussianHMM, features: list[str]) -> dict[int, int]:
    """
    Ordena estados pelo drawdown intrínseco de model.means_.
    menor drawdown (mais negativo) = Bear (0)
    maior drawdown (próximo de zero) = Bull (n_states - 1)
    Fallback: log_return das means_ (menor = Bear).
    """
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
    """Duração média de runs consecutivos do target_state."""
    durations = []
    count = 0
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


def run_walkforward_hmm(
    btc_df: pd.DataFrame,
    walkforward_params: dict[str, Any],
    modeling_params: dict[str, Any],
) -> pd.DataFrame:
    """
    Walk-forward HMM com 3 splits, purging correto e ordenação por drawdown.
    Suporta n_states=2 (Bear/Bull) e n_states=3 (Bear/Lateral/Bull).
    """
    horizon_days: int = walkforward_params["horizon_days"]
    embargo_days: int = walkforward_params["embargo_days"]
    purge_total: int = horizon_days + embargo_days

    n_states: int = modeling_params["n_states"]
    covariance_type: str = modeling_params["covariance_type"]
    n_iter: int = modeling_params["n_iter"]
    random_state: int = modeling_params["random_state"]
    features: list[str] = modeling_params["features"]

    bull_state = n_states - 1   # 1 para n=2, 2 para n=3
    bear_state = 0

    df = btc_df.copy()
    df = df.sort_index()

    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=features)

    df["_forward_ret"] = sum(df["log_return"].shift(-k) for k in range(1, horizon_days + 1))

    resultados = []

    for split_cfg in walkforward_params["splits"]:
        split_name: str = split_cfg["name"]

        train_start = pd.Timestamp(str(split_cfg["train_start"]), tz="UTC")
        train_end   = pd.Timestamp(str(split_cfg["train_end"]),   tz="UTC")
        test_start  = pd.Timestamp(str(split_cfg["test_start"]),  tz="UTC")
        test_end_raw = split_cfg["test_end"]

        test_end = (
            df.index.max()
            if test_end_raw is None
            else pd.Timestamp(str(test_end_raw), tz="UTC")
        )

        purged_train_end = min(train_end, test_start - timedelta(days=purge_total))

        train_df = df.loc[train_start:purged_train_end]
        test_df  = df.loc[test_start:test_end]

        if len(train_df) < 30 or len(test_df) < 5:
            continue

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train_df[features])
        X_test  = scaler.transform(test_df[features])

        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        model.fit(X_train)

        mapa        = _ordenar_estados(model, features)
        raw_states  = model.predict(X_test)
        test_states = np.array([mapa[s] for s in raw_states])

        retorno_5d = test_df["_forward_ret"]

        bull_mask = test_states == bull_state
        bear_mask = test_states == bear_state

        E_bull   = retorno_5d[bull_mask].mean()
        E_bear   = retorno_5d[bear_mask].mean()
        delta_5d = E_bull - E_bear

        bull_precision = float((retorno_5d[bull_mask] > 0).mean()) if bull_mask.sum() > 0 else np.nan
        bear_precision = float((retorno_5d[bear_mask] < 0).mean()) if bear_mask.sum() > 0 else np.nan

        bull_duration = _run_durations(test_states, bull_state)
        bear_duration = _run_durations(test_states, bear_state)

        # lateral_pct: 0.0 para n_states=2; proporção do estado 1 para n_states=3
        if n_states == 2:
            lateral_pct = 0.0
        else:
            lateral_pct = float((test_states == 1).sum() / len(test_states))

        resultados.append(
            {
                "split":          split_name,
                "test_start":     test_start.date(),
                "test_end":       test_end.date(),
                "n_train":        len(train_df),
                "n_test":         len(test_df),
                "delta_5d":       delta_5d,
                "E_bull":         E_bull,
                "E_bear":         E_bear,
                "bull_precision": bull_precision,
                "bear_precision": bear_precision,
                "bull_duration":  bull_duration,
                "bear_duration":  bear_duration,
                "lateral_pct":    lateral_pct,
            }
        )

    return pd.DataFrame(resultados)


def train_final_hmm_states(
    btc_df: pd.DataFrame,
    walkforward_params: dict[str, Any],
    modeling_params: dict[str, Any],
) -> pd.DataFrame:
    """
    Train final HMM on all data from the earliest walkforward train_start.

    Returns a DatetimeIndex-aligned DataFrame with:
      state     (int)   — 0=Bear, 1=Bull (ordered by drawdown from means_)
      bull_prob (float) — posterior probability of Bull state P(Bull | X_{1:T})

    This output is consumed by downstream models (e.g. XGBoost) as a
    regime feature. Index is preserved from btc_df (BDay UTC if input is BDay).
    """
    n_states: int = modeling_params["n_states"]
    covariance_type: str = modeling_params["covariance_type"]
    n_iter: int = modeling_params["n_iter"]
    random_state: int = modeling_params["random_state"]
    features: list[str] = modeling_params["features"]

    # Earliest train_start across all splits = global fitting start
    train_start = min(
        pd.Timestamp(str(split["train_start"]), tz="UTC")
        for split in walkforward_params["splits"]
    )

    df = btc_df.copy().sort_index()
    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=features)
    df = df.loc[train_start:]

    if len(df) < 30:
        raise ValueError(
            f"train_final_hmm_states: only {len(df)} rows after filtering — "
            "need at least 30."
        )

    bull_state = n_states - 1

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)

    mapa = _ordenar_estados(model, features)

    raw_states = model.predict(X)
    states = np.array([mapa[s] for s in raw_states])

    # Posterior probabilities — P(state_t | X_{1:T}) via forward-backward
    raw_proba = model.predict_proba(X)  # shape (n_obs, n_states)

    # Identify which raw column corresponds to the mapped bull_state
    inv_mapa = {new_s: orig_s for orig_s, new_s in mapa.items()}
    orig_bull = inv_mapa[bull_state]
    bull_prob = raw_proba[:, orig_bull]

    return pd.DataFrame(
        {"state": states, "bull_prob": bull_prob},
        index=df.index,
    )
