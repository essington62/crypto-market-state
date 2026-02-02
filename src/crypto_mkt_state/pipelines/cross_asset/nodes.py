"""
L4 cross-asset regime layer.

This module does NOT compute any new statistics. It:
- Reads existing L3 columns (from FRED and YFinance primary)
- Validates that all l4.proxies exist and required signal columns are present
- Builds regime categorical/boolean columns from params:l4.regimes (thresholds only)
- Joins everything by date (inner join)

Contract:
- No rolling_*, zscore_*, return_*, volatility_* calculations
- Fail-fast if any proxy or required feature is missing
- All logic driven by params:l4 (proxies, regimes with source/signal/thresholds)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Partition loading
# ---------------------------------------------------------------------------
def _load_partition_map(
    data: Dict[str, Union[Callable[[], pd.DataFrame], pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """
    Load a PartitionedDataset-like mapping into a dict[asset_name, DataFrame].

    Keys are taken from df["asset"].iloc[0] so that FRED and YFinance
    partitions are indexed by canonical asset name (e.g. vix, sp500, cpi).
    """
    by_asset: Dict[str, pd.DataFrame] = {}

    for _, loader in data.items():
        df = loader() if callable(loader) else loader
        if df is None or df.empty:
            continue
        df_norm = (
            df.copy()
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
        )
        asset = str(df_norm["asset"].iloc[0]).strip().lower()
        by_asset[asset] = df_norm

    return by_asset


def _safe_get_asset(
    data: Dict[str, pd.DataFrame],
    asset_name: str,
    context: str = "L4",
) -> pd.DataFrame:
    """
    Return the DataFrame for the given asset. Raise ValueError if missing.
    """
    key = asset_name.strip().lower()
    if key not in data:
        available = sorted(data.keys())
        raise ValueError(
            f"{context}: required asset '{asset_name}' not found in L3 data. "
            f"Available assets: {available}"
        )
    return data[key]


def _validate_required_columns(
    df: pd.DataFrame,
    columns: List[str],
    asset_name: str,
) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"L4: asset '{asset_name}' is missing required column(s): {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


# ---------------------------------------------------------------------------
# Regime helpers (threshold-based only; no new statistics)
# ---------------------------------------------------------------------------
def _compute_ternary_regime(
    series: pd.Series,
    high_threshold: float,
    low_threshold: float,
    high_label: str = "high",
    low_label: str = "low",
    neutral_label: str = "neutral",
) -> pd.Series:
    """
    Map a numeric series to a categorical regime: high, neutral, or low.

    Above high_threshold -> high_label; below low_threshold -> low_label; else neutral.
    NaN in input remains NaN in output.
    """
    out = pd.Series(index=series.index, dtype=object)
    out.loc[series > high_threshold] = high_label
    out.loc[series < low_threshold] = low_label
    out.loc[(series >= low_threshold) & (series <= high_threshold)] = neutral_label
    out.loc[series.isna()] = pd.NA
    return out


def _compute_binary_regime(
    series: pd.Series,
    threshold: float,
    high_label: str = "high",
    neutral_label: str = "neutral",
) -> pd.Series:
    """
    Map a numeric series to a binary regime: above threshold -> high, else neutral.
    """
    out = pd.Series(index=series.index, dtype=object)
    out.loc[series > threshold] = high_label
    out.loc[series <= threshold] = neutral_label
    out.loc[series.isna()] = pd.NA
    return out


# ---------------------------------------------------------------------------
# Merge by date (inner join)
# ---------------------------------------------------------------------------
def _merge_on_date(
    base: Optional[pd.DataFrame],
    df: pd.DataFrame,
    columns: Dict[str, str],
) -> pd.DataFrame:
    """
    Merge selected columns from df into base on date (inner join).
    """
    # date entra uma única vez, sempre
    subset = df[["date", *columns.keys()]].rename(columns=columns)
    subset = subset.loc[:, ~subset.columns.duplicated()]

    if base is None:
        return subset.copy()

    base = base.loc[:, ~base.columns.duplicated()]
    return base.merge(subset, on="date", how="inner")



# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------
def build_cross_asset_features(
    fred: Dict[str, Union[Callable[[], pd.DataFrame], pd.DataFrame]],
    yfinance: Dict[str, Union[Callable[[], pd.DataFrame], pd.DataFrame]],
    l4_config: Dict[str, Any],
    l4_regime_states: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build L4 cross-asset regime features from L3 data only.

    - Validates that every asset in l4_config.proxies exists in L3 data.
    - For each regime in l4_config.regimes: loads source asset, validates signal
      column, applies thresholds, produces a categorical regime column.
    - Joins all series on date (inner). No new statistics are computed.
    - If l4_regime_states is provided and contains regime_entropy, merges
      regime_entropy on date so it appears in cross_asset_features.

    Args:
        fred: Partitioned L3 FRED data (partition key -> loader or DataFrame).
        yfinance: Partitioned L3 YFinance data (partition key -> loader or DataFrame).
        l4_config: params:l4 (proxies, regimes with source/signal/thresholds).
        l4_regime_states: Optional L4 HMM output with date and regime_entropy.

    Returns:
        One DataFrame with columns: date, volatility_regime, dollar_regime,
        rates_regime, inflation_regime, regime_entropy (if provided), etc.

    Raises:
        ValueError: If any proxy is missing or any required signal column is absent.
    """
    fred_by_asset = _load_partition_map(fred)
    yf_by_asset = _load_partition_map(yfinance)
    all_assets: Dict[str, pd.DataFrame] = {**yf_by_asset, **fred_by_asset}

    proxies = l4_config.get("proxies") or {}
    regimes = l4_config.get("regimes") or {}
    validation = l4_config.get("validation") or {}
    require_proxies = validation.get("require_all_proxies", True)

    # 1) Validate all proxies exist (fail-fast)
    if require_proxies:
        for role, asset_name in proxies.items():
            _safe_get_asset(all_assets, asset_name, context=f"L4 proxy '{role}'")

    # 2) Build base by joining regime series on date
    cross: Optional[pd.DataFrame] = None

    for regime_name, regime_cfg in regimes.items():
        source_asset = regime_cfg.get("source")
        signal_col = regime_cfg.get("signal")
        if not source_asset or not signal_col:
            raise ValueError(
                f"L4 regime '{regime_name}' must have 'source' and 'signal' in params:l4.regimes"
            )

        df = _safe_get_asset(all_assets, source_asset, context=f"L4 regime '{regime_name}'")
        _validate_required_columns(df, ["date", signal_col], source_asset)

        series = df[signal_col]

        # Ternary or binary from params only (no new statistics)
        if "rising_threshold" in regime_cfg and "falling_threshold" in regime_cfg:
            high_t = float(regime_cfg["rising_threshold"])
            low_t = float(regime_cfg["falling_threshold"])
            regime_series = _compute_ternary_regime(
                series, high_t, low_t,
                high_label="rising", low_label="falling", neutral_label="neutral",
            )
        elif "strong_threshold" in regime_cfg and "weak_threshold" in regime_cfg:
            high_t = float(regime_cfg["strong_threshold"])
            low_t = float(regime_cfg["weak_threshold"])
            regime_series = _compute_ternary_regime(
                series, high_t, low_t,
                high_label="strong", low_label="weak", neutral_label="neutral",
            )
        elif "high_threshold" in regime_cfg and "low_threshold" in regime_cfg:
            high_t = float(regime_cfg["high_threshold"])
            low_t = float(regime_cfg["low_threshold"])
            regime_series = _compute_ternary_regime(
                series, high_t, low_t,
                high_label="high", low_label="low", neutral_label="neutral",
            )
        elif "high_threshold" in regime_cfg:
            regime_series = _compute_binary_regime(
                series, float(regime_cfg["high_threshold"]),
                high_label="high", neutral_label="neutral",
            )
        else:
            raise ValueError(
                f"L4 regime '{regime_name}' has no recognized threshold keys "
                "(high_threshold/low_threshold, strong_threshold/weak_threshold, "
                "rising_threshold/falling_threshold)."
            )

        regime_df = df[["date"]].copy()
        regime_df[f"{regime_name}_regime"] = regime_series.values
        cross = _merge_on_date(
            cross,
            regime_df,
            {f"{regime_name}_regime": f"{regime_name}_regime"},
        )

    if cross is None:
        cross = pd.DataFrame(columns=["date"])

    cross = (
        cross.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )

    # Merge regime_entropy from L4 HMM so it appears in cross_asset_features
    if l4_regime_states is not None and not l4_regime_states.empty:
        if "regime_entropy" in l4_regime_states.columns and "date" in l4_regime_states.columns:
            re_df = l4_regime_states[["date", "regime_entropy"]].copy()
            re_df["date"] = pd.to_datetime(re_df["date"], utc=True)
            re_df = re_df.drop_duplicates(subset=["date"], keep="last")
            cross = cross.merge(re_df, on="date", how="left")

    return cross


# ---------------------------------------------------------------------------
# L3: Regime entropy from HMM state probabilities (Primary Metrics)
# ---------------------------------------------------------------------------
def compute_regime_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute Shannon entropy of regime probabilities per row (pure, no I/O).

    Entropy measures uncertainty of the latent regime: when the HMM assigns
    similar probabilities to several states, entropy is high (high uncertainty);
    when one state dominates (probability near 1), entropy is low (low uncertainty).
    High values: model is unsure which regime applies (e.g. transition periods).
    Low values: model is confident in a single regime (e.g. clear risk-on/risk-off).

    Formula: H = - sum_i p_i * log(p_i). Probabilities are clipped to [eps, 1]
    for numerical stability (avoid log(0)).

    Args:
        probabilities: shape (n_samples, n_states), each row sums to 1.
        eps: lower clip for probabilities (default 1e-12).

    Returns:
        1d array of length n_samples with entropy per observation (nat).
    """
    p = np.clip(probabilities, eps, 1.0)
    log_p = np.where(p > 0, np.log(p), 0.0)
    return -np.sum(p * log_p, axis=1)


def add_regime_entropy_to_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    L3 node: enrich a DataFrame with p_state_* and date by adding regime_entropy.

    Consumes HMM posterior probabilities (p_state_0, p_state_1, ...), computes
    regime_entropy = - sum p_i * log(p_i) with clip(lower=1e-12) for stability,
    and returns the same DataFrame with a new column regime_entropy. Pure
    function; no I/O; timezone UTC; does not remove existing columns.

    Entropy measures regime uncertainty: high = model unsure; low = confident.
    """
    if df is None or df.empty:
        raise ValueError("L3 add_regime_entropy_to_states: input DataFrame is empty.")

    p_cols = [c for c in df.columns if c.startswith("p_state_")]
    if not p_cols:
        raise ValueError(
            "L3 add_regime_entropy_to_states: no p_state_* columns found. "
            f"Available: {list(df.columns)}"
        )
    if "date" not in df.columns:
        raise ValueError(
            "L3 add_regime_entropy_to_states: column 'date' is required."
        )

    p = df[p_cols].values.astype(float)
    df = df.copy()
    df["regime_entropy"] = compute_regime_entropy(p, eps=1e-12)
    return df


# ---------------------------------------------------------------------------
# L4: Caracterização ex-post dos regimes HMM (interpretação / auditoria)
# ---------------------------------------------------------------------------

# Colunas obrigatórias para a caracterização (l3_regime_features)
_L3_REGIME_FEATURE_COLS = [
    "date",
    "btc_return_1d",
    "btc_momentum_21",
    "btc_rolling_std_21",
    "vix_zscore_63",
    "dxy_zscore_63",
]
_L4_REGIME_STATES_COLS = ["date", "regime_state"]


# ---------------------------------------------------------------------------
# L3: Assemble regime features for HMM (from crypto + yfinance L3)
# ---------------------------------------------------------------------------
def assemble_l3_regime_features(
    crypto_primary: Dict[str, Union[Callable[[], pd.DataFrame], pd.DataFrame]],
    yfinance_indices_primary: Dict[str, Union[Callable[[], pd.DataFrame], pd.DataFrame]],
) -> pd.DataFrame:
    """
    Assemble a single L3 table with 5 features for HMM: btc_return_1d,
    btc_momentum_21, btc_rolling_std_21, vix_zscore_63, dxy_zscore_63.

    Loads BTC partition from crypto L3 and vix/dxy from yfinance indices L3,
    renames columns to canonical names, inner-joins on date (UTC). No I/O;
    pure transformation.
    """
    from crypto_mkt_state.utils.utils_l3_semantic import normalize_asset_name

    # ---- BTC from crypto ----
    btc_df = None
    for key, loader in crypto_primary.items():
        df = loader() if callable(loader) else loader
        if df is None or df.empty:
            continue
        if normalize_asset_name(str(key)) != "btc":
            continue
        df = df.copy()
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df.index, utc=True) if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        for col in ["return_1d", "momentum_21", "rolling_std_21"]:
            if col not in df.columns:
                raise ValueError(
                    f"L3 assemble_l3_regime_features: BTC partition missing '{col}'. "
                    f"Available: {list(df.columns)}"
                )
        btc_df = df[["date", "return_1d", "momentum_21", "rolling_std_21"]].copy()
        btc_df = btc_df.rename(columns={
            "return_1d": "btc_return_1d",
            "momentum_21": "btc_momentum_21",
            "rolling_std_21": "btc_rolling_std_21",
        })
        break
    if btc_df is None:
        raise ValueError(
            "L3 assemble_l3_regime_features: no BTC partition found in crypto_primary. "
            f"Keys: {list(crypto_primary.keys())}"
        )

    # ---- vix / dxy from yfinance indices ----
    yf_by_asset = _load_partition_map(yfinance_indices_primary)
    for name in ("vix", "dxy"):
        if name not in yf_by_asset:
            raise ValueError(
                f"L3 assemble_l3_regime_features: missing index '{name}' in yfinance_indices. "
                f"Available: {sorted(yf_by_asset.keys())}"
            )
    vix_df = yf_by_asset["vix"][["date", "zscore_63"]].copy()
    vix_df["date"] = pd.to_datetime(vix_df["date"], utc=True)
    vix_df = vix_df.rename(columns={"zscore_63": "vix_zscore_63"})
    dxy_df = yf_by_asset["dxy"][["date", "zscore_63"]].copy()
    dxy_df["date"] = pd.to_datetime(dxy_df["date"], utc=True)
    dxy_df = dxy_df.rename(columns={"zscore_63": "dxy_zscore_63"})

    out = btc_df.merge(vix_df, on="date", how="inner").merge(dxy_df, on="date", how="inner")
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out


# ---------------------------------------------------------------------------
# L4: HMM latent regime (outputs p_state_*; regime_entropy added by L3 node)
# ---------------------------------------------------------------------------
def estimate_market_regime_hmm(
    l3_regime_features: pd.DataFrame,
    l4_config: Dict[str, Any],
) -> tuple[pd.DataFrame, dict]:
    """
    Fit a Gaussian HMM on L3 regime features and output states + posteriors.

    Outputs l4_regime_states with date, regime_state, p_state_0, p_state_1, ...
    (regime_entropy is added by add_regime_entropy_to_states in L3). Also
    outputs l4_regime_transition_matrix as JSON artifact.
    """
    from hmmlearn.hmm import GaussianHMM

    if l3_regime_features is None or l3_regime_features.empty:
        raise ValueError("L4 estimate_market_regime_hmm: l3_regime_features is empty.")

    missing = [c for c in _L3_REGIME_FEATURE_COLS if c not in l3_regime_features.columns]
    if missing:
        raise ValueError(
            f"L4 estimate_market_regime_hmm: l3_regime_features missing columns: {missing}. "
            f"Available: {list(l3_regime_features.columns)}"
        )

    hmm_cfg = (l4_config or {}).get("hmm") or {}
    n_components = int(hmm_cfg.get("n_components", 3))
    covariance_type = str(hmm_cfg.get("covariance_type", "full"))
    n_iter = int(hmm_cfg.get("n_iter", 500))
    random_state = int(hmm_cfg.get("random_state", 42))
    min_observations = int(hmm_cfg.get("min_observations", 50))

    X = l3_regime_features[_L3_REGIME_FEATURE_COLS].drop(columns="date")
    X = X.dropna()
    if len(X) < min_observations:
        raise ValueError(
            f"L4 estimate_market_regime_hmm: need at least {min_observations} observations "
            f"after dropna; got {len(X)}."
        )

    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)

    # Posteriors and state per row (align to full date index)
    date_col = pd.to_datetime(l3_regime_features["date"], utc=True)
    posteriors = model.predict_proba(X)
    states = model.predict(X)
    # Map row index of X back to original index
    valid_idx = l3_regime_features[_L3_REGIME_FEATURE_COLS].drop(columns="date").dropna().index
    out_df = pd.DataFrame(index=valid_idx)
    out_df["date"] = date_col.loc[valid_idx].values
    out_df["regime_state"] = states
    for i in range(n_components):
        out_df[f"p_state_{i}"] = posteriors[:, i]
    out_df = out_df.reset_index(drop=True)
    out_df = out_df[["date", "regime_state"] + [f"p_state_{i}" for i in range(n_components)]]

    trans = model.transmat_.tolist()
    artifact = {
        "transition_matrix": trans,
        "n_states": n_components,
    }
    return out_df, artifact


def summarize_hmm_regimes(
    l4_regime_states: pd.DataFrame,
    l3_regime_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Caracterização numérica ex-post dos regimes latentes do HMM (L4).

    Os estados do HMM são latentes: o modelo não atribui significado econômico
    a cada regime. Este nó calcula estatísticas descritivas (mean, median, std)
    das features L3 *condicionadas a cada regime_state*, de forma puramente
    ex-post e read-only.

    Objetivo:
    - Validação econômica do HMM (ex.: regime com VIX alto tem vix_zscore_63_mean > 0).
    - Interpretação semântica dos estados (ex.: bull/low-vol vs bear/high-vol).
    - Auditoria e explicabilidade.

    O output serve APENAS para interpretação e auditoria. NÃO deve ser usado
    para re-treinar o HMM, re-rotular observações ou criar features para
    modelagem downstream. O HMM permanece inalterado por este nó.

    Contrato:
    - Função pura; timezone UTC; não altera inputs; fail-fast se vazio ou
      colunas obrigatórias ausentes.
    """
    if l4_regime_states is None or l4_regime_states.empty:
        raise ValueError(
            "L4 summarize_hmm_regimes: l4_regime_states is empty."
        )
    if l3_regime_features is None or l3_regime_features.empty:
        raise ValueError(
            "L4 summarize_hmm_regimes: l3_regime_features is empty."
        )

    missing_states = [c for c in _L4_REGIME_STATES_COLS if c not in l4_regime_states.columns]
    if missing_states:
        raise ValueError(
            f"L4 summarize_hmm_regimes: l4_regime_states missing columns: {missing_states}. "
            f"Available: {list(l4_regime_states.columns)}"
        )
    missing_features = [c for c in _L3_REGIME_FEATURE_COLS if c not in l3_regime_features.columns]
    if missing_features:
        raise ValueError(
            f"L4 summarize_hmm_regimes: l3_regime_features missing columns: {missing_features}. "
            f"Available: {list(l3_regime_features.columns)}"
        )

    states = l4_regime_states[["date", "regime_state"]].copy()
    states["date"] = pd.to_datetime(states["date"], utc=True)
    features = l3_regime_features[_L3_REGIME_FEATURE_COLS].copy()
    features["date"] = pd.to_datetime(features["date"], utc=True)

    merged = states.merge(features, on="date", how="inner")
    merged = merged.dropna()

    if merged.empty:
        raise ValueError(
            "L4 summarize_hmm_regimes: no rows after inner merge and dropna."
        )

    feature_cols = [c for c in _L3_REGIME_FEATURE_COLS if c != "date"]
    agg_dict = {}
    for col in feature_cols:
        if col in ("vix_zscore_63", "dxy_zscore_63"):
            agg_dict[col] = ["mean", "median"]
        else:
            agg_dict[col] = ["mean", "median", "std"]

    n_obs = (
        merged.groupby("regime_state", as_index=False)
        .size()
        .rename(columns={"size": "n_obs"})
        .sort_values("regime_state")
        .reset_index(drop=True)
    )
    out = n_obs.copy()
    for col in feature_cols:
        ops = ["mean", "median"] if col in ("vix_zscore_63", "dxy_zscore_63") else ["mean", "median", "std"]
        for op in ops:
            s = (
                merged.groupby("regime_state", as_index=False)[col]
                .agg(op)
                .sort_values("regime_state")
                .reset_index(drop=True)
            )
            out[f"{col}_{op}"] = s[col].values
    out = out.sort_values("regime_state").reset_index(drop=True)

    desired = [
        "regime_state", "n_obs",
        "btc_return_1d_mean", "btc_return_1d_median", "btc_return_1d_std",
        "btc_momentum_21_mean", "btc_momentum_21_median", "btc_momentum_21_std",
        "btc_rolling_std_21_mean", "btc_rolling_std_21_median", "btc_rolling_std_21_std",
        "vix_zscore_63_mean", "vix_zscore_63_median",
        "dxy_zscore_63_mean", "dxy_zscore_63_median",
    ]
    out = out[[c for c in desired if c in out.columns]]
    return out


# ---------------------------------------------------------------------------
# L4: Consolidação do regime atual do mercado (Current Market Regime)
# ---------------------------------------------------------------------------
def infer_current_market_regime(
    l4_regime_states: pd.DataFrame,
    l4_regime_transition_matrix: dict,
) -> pd.DataFrame:
    """
    Consolida o regime atual do mercado a partir do HMM já treinado.

    O HMM define regimes latentes (estados ocultos) que representam diferentes
    contextos de mercado. Este nó consolida o estado atual inferido pelo HMM
    (última observação) e calcula a probabilidade de permanência no regime
    atual usando a matriz de transição do modelo.

    A probabilidade de permanência (prob_stay) vem diretamente da matriz de
    transição do HMM: P(regime_t+1 = regime_t | regime_t). A probabilidade
    de mudança (prob_switch) é o complemento: 1 - prob_stay.

    O output é interpretativo (L4) e serve como "estado atual do mercado"
    para leitura humana e para consumo futuro da L5 (policy layer). Este nó
    NÃO re-treina o HMM, NÃO prevê retornos e NÃO cria targets supervisionados.
    É puramente inferência ex-post + leitura da matriz de transição.

    Args:
        l4_regime_states:
            DataFrame do HMM com colunas: date (datetime UTC), regime_state (int),
            regime_entropy (float, opcional), regime_prob_* (opcional).
        l4_regime_transition_matrix:
            Dicionário JSON com: transition_matrix (lista de listas NxN),
            n_states (int), e outros metadados.

    Returns:
        DataFrame com UMA linha contendo:
        - date: última data observada (datetime UTC)
        - regime_state: estado atual (int)
        - regime_entropy: incerteza do HMM no ponto atual (float, ou NaN)
        - prob_stay: P(regime_t+1 = regime_t) (float)
        - prob_switch: 1 - prob_stay (float)
        - n_observations: número total de observações no histórico (int)
        - inference_ts: timestamp da inferência (datetime UTC)

    Raises:
        ValueError: Se l4_regime_states estiver vazio, colunas obrigatórias
            ausentes, transition_matrix mal formatada, ou regime_state fora
            do range válido.
    """
    import numpy as np

    if l4_regime_states is None or l4_regime_states.empty:
        raise ValueError(
            "L4 infer_current_market_regime: l4_regime_states is empty."
        )

    required_cols = ["date", "regime_state"]
    missing = [c for c in required_cols if c not in l4_regime_states.columns]
    if missing:
        raise ValueError(
            f"L4 infer_current_market_regime: l4_regime_states missing columns: {missing}. "
            f"Available: {list(l4_regime_states.columns)}"
        )

    if l4_regime_transition_matrix is None:
        raise ValueError(
            "L4 infer_current_market_regime: l4_regime_transition_matrix is None."
        )

    transition_matrix = l4_regime_transition_matrix.get("transition_matrix")
    n_states = l4_regime_transition_matrix.get("n_states")

    if transition_matrix is None:
        raise ValueError(
            "L4 infer_current_market_regime: transition_matrix missing in l4_regime_transition_matrix."
        )
    if n_states is None:
        raise ValueError(
            "L4 infer_current_market_regime: n_states missing in l4_regime_transition_matrix."
        )

    transition_matrix = np.array(transition_matrix)
    if transition_matrix.shape != (n_states, n_states):
        raise ValueError(
            f"L4 infer_current_market_regime: transition_matrix shape mismatch. "
            f"Expected ({n_states}, {n_states}), got {transition_matrix.shape}."
        )

    df = l4_regime_states[required_cols].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    if df.empty:
        raise ValueError(
            "L4 infer_current_market_regime: l4_regime_states is empty after normalization."
        )

    last_row = df.iloc[-1]
    current_state = int(last_row["regime_state"])
    current_date = last_row["date"]

    if current_state < 0 or current_state >= n_states:
        raise ValueError(
            f"L4 infer_current_market_regime: current regime_state {current_state} "
            f"is outside valid range [0, {n_states-1}]."
        )

    regime_entropy = None
    if "regime_entropy" in l4_regime_states.columns:
        last_full = l4_regime_states.sort_values("date").drop_duplicates(subset=["date"], keep="last").iloc[-1]
        regime_entropy = float(last_full["regime_entropy"])

    prob_stay = float(transition_matrix[current_state, current_state])
    prob_switch = 1.0 - prob_stay

    inference_ts = pd.Timestamp.now(tz="UTC")

    out = pd.DataFrame(
        {
            "date": [current_date],
            "regime_state": [current_state],
            "regime_entropy": [regime_entropy if regime_entropy is not None else np.nan],
            "prob_stay": [prob_stay],
            "prob_switch": [prob_switch],
            "n_observations": [len(df)],
            "inference_ts": [inference_ts],
        }
    )

    return out


# ---------------------------------------------------------------------------
# L4: Viterbi rolling diagnostics (diagnostic only; no HMM change)
# ---------------------------------------------------------------------------
_EPS = 1e-12


def _viterbi_decode(
    emissions: np.ndarray,
    transition: np.ndarray,
    startprob: np.ndarray,
    eps: float = _EPS,
) -> tuple[np.ndarray, float]:
    """
    Decode the most likely path of hidden states given emissions (pure NumPy).

    emissions: (T, n_states), each row = P(obs_t | state j) (e.g. p_state_*).
    transition: (n_states, n_states), A[i,j] = P(s_{t+1}=j | s_t=i).
    startprob: (n_states,), initial distribution.
    Returns: (path, log_prob) where path is (T,) int array, log_prob is scalar.
    """
    T, N = emissions.shape
    log_A = np.log(np.clip(transition, eps, 1.0))
    log_pi = np.log(np.clip(startprob, eps, 1.0))
    log_B = np.log(np.clip(emissions, eps, 1.0))

    V = np.full((T, N), -np.inf)
    backpointer = np.zeros((T, N), dtype=np.intp)
    V[0, :] = log_pi + log_B[0, :]
    for t in range(1, T):
        for j in range(N):
            trans = V[t - 1, :] + log_A[:, j]
            best_i = np.argmax(trans)
            V[t, j] = trans[best_i] + log_B[t, j]
            backpointer[t, j] = best_i
    path = np.zeros(T, dtype=np.intp)
    path[T - 1] = np.argmax(V[T - 1, :])
    for t in range(T - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]
    log_prob = float(V[T - 1, path[T - 1]])
    return path, log_prob


def compute_viterbi_diagnostics(
    l4_regime_states_enriched: pd.DataFrame,
    l4_regime_transition_matrix: dict,
    l4_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    L4 diagnostic node: Viterbi rolling (k=5) with entropy gate.

    For each day t, applies Viterbi on a rolling window of the last 5 days
    using HMM posteriors (p_state_*) as emissions and the stored transition
    matrix. When regime_entropy_t > entropy_threshold, inference is blocked
    (blocked=True) to avoid interpreting high-uncertainty periods. Output is
    diagnostic only (where/when the regime model "gets lost"); no HMM change.

    Inputs:
        l4_regime_states_enriched: Parquet with date, regime_state, p_state_*,
            regime_entropy (UTC).
        l4_regime_transition_matrix: JSON with transition_matrix, n_states.
        l4_config: params:l4; inference.entropy_threshold used for gate.

    Output:
        l4_viterbi_diagnostics.parquet with same number of rows as input.
        Columns: date, regime_state, viterbi_state, viterbi_agrees,
        regime_entropy, blocked, viterbi_logprob.

    Constraints: Pure function, UTC, no I/O, NumPy/Pandas only.
    """
    if l4_regime_states_enriched is None or l4_regime_states_enriched.empty:
        raise ValueError(
            "L4 compute_viterbi_diagnostics: l4_regime_states_enriched is empty."
        )

    p_cols = sorted([c for c in l4_regime_states_enriched.columns if c.startswith("p_state_")])
    if not p_cols:
        raise ValueError(
            "L4 compute_viterbi_diagnostics: no p_state_* columns in l4_regime_states_enriched. "
            f"Available: {list(l4_regime_states_enriched.columns)}"
        )
    for col in ["date", "regime_state", "regime_entropy"]:
        if col not in l4_regime_states_enriched.columns:
            raise ValueError(
                f"L4 compute_viterbi_diagnostics: missing column '{col}'. "
                f"Available: {list(l4_regime_states_enriched.columns)}"
            )

    trans = l4_regime_transition_matrix.get("transition_matrix")
    n_states = l4_regime_transition_matrix.get("n_states")
    if trans is None or n_states is None:
        raise ValueError(
            "L4 compute_viterbi_diagnostics: transition_matrix or n_states missing "
            "in l4_regime_transition_matrix."
        )
    A = np.array(trans)
    if A.shape != (n_states, n_states):
        raise ValueError(
            f"L4 compute_viterbi_diagnostics: transition_matrix shape {A.shape} "
            f"expected ({n_states}, {n_states})."
        )

    inference_cfg = (l4_config or {}).get("inference") or {}
    entropy_threshold = float(inference_cfg.get("entropy_threshold", 1.0))

    df = l4_regime_states_enriched.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    n = len(df)
    p_matrix = df[p_cols].values.astype(float)
    regime_state = df["regime_state"].values.astype(int)
    regime_entropy = df["regime_entropy"].values.astype(float)
    startprob = np.ones(n_states) / n_states
    k = 5

    viterbi_state = np.full(n, np.nan, dtype=float)
    viterbi_logprob = np.full(n, np.nan, dtype=float)
    blocked = np.zeros(n, dtype=bool)

    for i in range(n):
        if regime_entropy[i] > entropy_threshold:
            blocked[i] = True
            continue
        start = max(0, i - k + 1)
        end = i + 1
        window_emissions = p_matrix[start:end]
        if window_emissions.size == 0:
            blocked[i] = True
            continue
        path, log_prob = _viterbi_decode(window_emissions, A, startprob, eps=_EPS)
        viterbi_state[i] = int(path[-1])
        viterbi_logprob[i] = log_prob
        blocked[i] = False

    viterbi_agrees = np.where(
        blocked,
        False,
        (np.array(viterbi_state, dtype=float) == regime_state),
    )

    out = pd.DataFrame({
        "date": df["date"].values,
        "regime_state": regime_state,
        "viterbi_state": viterbi_state,
        "viterbi_agrees": viterbi_agrees,
        "regime_entropy": regime_entropy,
        "blocked": blocked,
        "viterbi_logprob": viterbi_logprob,
    })
    out["date"] = pd.to_datetime(out["date"], utc=True)
    if out.shape[0] != n:
        raise ValueError(
            f"L4 compute_viterbi_diagnostics: output row count {out.shape[0]} != input {n}."
        )
    return out
