import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Any

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def _ordenar_estados_por_vol(model: GaussianHMM, features: list[str]) -> dict[int, int]:
    """
    Ordena estados pelo vol_short intrínseco de model.means_.
    0=Bear (maior vol), 1=Lateral, 2=Bull (menor vol).
    Fallback: ordena por index 0 das means_ se vol_short não estiver em features.
    """
    n_states = model.n_components

    if "vol_short" in features:
        vol_idx = features.index("vol_short")
        vol_by_state = {s: model.means_[s][vol_idx] for s in range(n_states)}
        ordered = sorted(vol_by_state, key=vol_by_state.get, reverse=True)
    else:
        ordered = sorted(range(n_states), key=lambda s: model.means_[s][0], reverse=True)

    mapping = {orig: idx for idx, orig in enumerate(ordered)}
    return mapping


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
    Walk-forward HMM com 3 splits, purging correto e ordenação estável por vol_short.
    """
    horizon_days: int = walkforward_params["horizon_days"]
    embargo_days: int = walkforward_params["embargo_days"]
    purge_total: int = horizon_days + embargo_days

    n_states: int = modeling_params["n_states"]
    covariance_type: str = modeling_params["covariance_type"]
    n_iter: int = modeling_params["n_iter"]
    random_state: int = modeling_params["random_state"]
    features: list[str] = modeling_params["features"]

    df = btc_df.copy()
    df = df.sort_index()

    # Usa apenas features disponíveis no dataframe
    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=features)

    # Retorno 5d prospectivo: soma dos próximos horizon_days log returns
    df["_forward_ret"] = sum(df["log_return"].shift(-k) for k in range(1, horizon_days + 1))

    resultados = []

    for split_cfg in walkforward_params["splits"]:
        split_name: str = split_cfg["name"]

        train_start = pd.Timestamp(str(split_cfg["train_start"]), tz="UTC")
        train_end = pd.Timestamp(str(split_cfg["train_end"]), tz="UTC")
        test_start = pd.Timestamp(str(split_cfg["test_start"]), tz="UTC")
        test_end_raw = split_cfg["test_end"]

        test_end = (
            df.index.max()
            if test_end_raw is None
            else pd.Timestamp(str(test_end_raw), tz="UTC")
        )

        purged_train_end = min(
            train_end,
            test_start - timedelta(days=purge_total),
        )

        train_df = df.loc[train_start:purged_train_end]
        test_df = df.loc[test_start:test_end]

        if len(train_df) < 30 or len(test_df) < 5:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features])
        X_test = scaler.transform(test_df[features])

        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        model.fit(X_train)

        mapa = _ordenar_estados_por_vol(model, features)

        raw_test_states = model.predict(X_test)
        test_states = np.array([mapa[s] for s in raw_test_states])

        retorno_5d = test_df["_forward_ret"]

        bull_mask = test_states == 2
        bear_mask = test_states == 0
        lateral_mask = test_states == 1

        E_bull = retorno_5d[bull_mask].mean()
        E_bear = retorno_5d[bear_mask].mean()
        delta_5d = E_bull - E_bear

        bull_precision = float((retorno_5d[bull_mask] > 0).mean()) if bull_mask.sum() > 0 else np.nan
        bear_precision = float((retorno_5d[bear_mask] < 0).mean()) if bear_mask.sum() > 0 else np.nan

        bull_duration = _run_durations(test_states, 2)
        bear_duration = _run_durations(test_states, 0)
        lateral_pct = float(lateral_mask.sum() / len(test_states))

        resultados.append(
            {
                "split": split_name,
                "test_start": test_start.date(),
                "test_end": test_end.date(),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "delta_5d": delta_5d,
                "E_bull": E_bull,
                "E_bear": E_bear,
                "bull_precision": bull_precision,
                "bear_precision": bear_precision,
                "bull_duration": bull_duration,
                "bear_duration": bear_duration,
                "lateral_pct": lateral_pct,
            }
        )

    return pd.DataFrame(resultados)
