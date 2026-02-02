"""
Regime baseline: assemble L3 data, train classifier, infer current regime.

- Uses ONLY L3 data (no L4). No new statistics (no rolling, zscore, etc.).
- Target: risk_environment (binary) = 1 if btc momentum_21 > 0 and vix zscore_63 < 0 and dxy zscore_63 <= 0.
- Fail-fast if any required asset or feature is missing.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Union

import pandas as pd

from crypto_mkt_state.utils.utils_l3_semantic import normalize_asset_name


# ---------------------------------------------------------------------------
# Helpers: load partitions by asset
# ---------------------------------------------------------------------------
def _load_partition_map(
    data: Dict[str, Union[Callable[[], pd.DataFrame], pd.DataFrame]],
    key_by_asset: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load PartitionedDataset-like dict into dict[key, DataFrame].
    If key_by_asset: use df['asset'].iloc[0] as key (for fred, yfinance).
    Else: keep partition keys (e.g. BTCUSDT for crypto).
    Ensures 'date' column exists (from index if missing, for crypto L3).
    """
    out: Dict[str, pd.DataFrame] = {}
    for part_key, loader in data.items():
        df = loader() if callable(loader) else loader
        if df is None or df.empty:
            continue
        df = df.copy()
        if "date" not in df.columns and hasattr(df.index, "normalize"):
            df["date"] = pd.to_datetime(df.index, utc=True).normalize()
        df = df.sort_values("date" if "date" in df.columns else df.index)
        if key_by_asset and "asset" in df.columns:
            key = str(df["asset"].iloc[0]).strip().lower()
        else:
            key = part_key if not key_by_asset else normalize_asset_name(part_key)
        out[key] = df
    return out


def _require_columns(df: pd.DataFrame, columns: List[str], asset: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Regime baseline: asset '{asset}' missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )


def _merge_on_date(
    base: pd.DataFrame | None,
    df: pd.DataFrame,
    rename: Dict[str, str],
) -> pd.DataFrame:
    """Inner join on date; rename columns from df before merge."""
    cols = [c for c in rename.keys() if c in df.columns]
    if "date" not in df.columns or not cols:
        return base if base is not None else pd.DataFrame()
    sub = df[["date", *cols]].rename(columns=rename)
    if base is None:
        return sub.drop_duplicates(subset=["date"], keep="last")
    return base.merge(sub, on="date", how="inner").drop_duplicates(subset=["date"], keep="last")


# ---------------------------------------------------------------------------
# 1) Assemble regime dataset
# ---------------------------------------------------------------------------
def assemble_regime_dataset(
    crypto_l3: Dict[str, Union[Callable[[], pd.DataFrame], pd.DataFrame]],
    yfinance_indices_l3: Dict[str, Union[Callable[[], pd.DataFrame], pd.DataFrame]],
    fred_l3: Dict[str, Union[Callable[[], pd.DataFrame], pd.DataFrame]],
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build tabular dataset with L3 features + target risk_environment.

    - Aligns crypto (BTC), vix, dxy, us_10y_yield, cpi on date (inner join).
    - Adds date column from crypto index when needed.
    - Target: risk_environment = 1 if (btc momentum_21 > 0, vix zscore_63 < 0, dxy zscore_63 <= 0).
    - Fail-fast if any required asset or feature is missing.
    """
    cfg = params.get("regime_baseline", params)
    crypto_key = cfg.get("crypto_partition_key", "BTCUSDT")
    required_indices = cfg.get("required_indices") or {}

    # Crypto: single partition BTC (partition key e.g. BTCUSDT)
    crypto_by_key = _load_partition_map(crypto_l3, key_by_asset=False)
    if crypto_key not in crypto_by_key:
        raise ValueError(
            f"Regime baseline: crypto partition '{crypto_key}' not found. "
            f"Available: {list(crypto_by_key.keys())}"
        )
    btc_df = crypto_by_key[crypto_key]

    if btc_df is None or btc_df.empty:
        raise ValueError("Regime baseline: BTC L3 partition is empty.")

    if "date" not in btc_df.columns and hasattr(btc_df.index, "normalize"):
        btc_df = btc_df.copy()
        btc_df["date"] = pd.to_datetime(btc_df.index, utc=True).normalize()

    btc_cols = [
        "return_1d", "return_5d", "return_21d",
        "rolling_std_21", "rolling_std_63",
        "zscore_21", "zscore_63",
        "momentum_21", "momentum_63",
    ]
    _require_columns(btc_df, ["date", "momentum_21"], "btc")
    for c in btc_cols:
        if c not in btc_df.columns:
            raise ValueError(f"Regime baseline: btc missing column '{c}'.")

    rename_btc = {"date": "date", **{c: f"btc_{c}" for c in btc_cols if c in btc_df.columns}}
    base = _merge_on_date(None, btc_df, rename_btc)

    # YFinance indices: vix, dxy
    yf_by_asset = _load_partition_map(yfinance_indices_l3, key_by_asset=True)
    for asset, cols in required_indices.items():
        if asset not in ("vix", "dxy"):
            continue
        if asset not in yf_by_asset:
            raise ValueError(
                f"Regime baseline: required index '{asset}' not in yfinance L3. "
                f"Available: {list(yf_by_asset.keys())}"
            )
        df = yf_by_asset[asset]
        _require_columns(df, ["date"] + list(cols), asset)
        renames = {"date": "date", **{c: f"{asset}_{c}" for c in cols}}
        base = _merge_on_date(base, df, renames)

    # FRED: us_10y_yield, cpi
    fred_by_asset = _load_partition_map(fred_l3, key_by_asset=True)
    for asset, cols in required_indices.items():
        if asset not in ("us_10y_yield", "cpi"):
            continue
        if asset not in fred_by_asset:
            raise ValueError(
                f"Regime baseline: required index '{asset}' not in fred L3. "
                f"Available: {list(fred_by_asset.keys())}"
            )
        df = fred_by_asset[asset]
        _require_columns(df, ["date"] + list(cols), asset)
        renames = {"date": "date", **{c: f"{asset}_{c}" for c in cols}}
        base = _merge_on_date(base, df, renames)

    # Target V1: risk_environment = 1 iff btc momentum_21 > 0, vix zscore_63 < 0, dxy zscore_63 <= 0
    base["risk_environment"] = (
        (base["btc_momentum_21"] > 0)
        & (base["vix_zscore_63"] < 0)
        & (base["dxy_zscore_63"] <= 0)
    ).astype(int)

    base = base.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return base


# ---------------------------------------------------------------------------
# 2) Train regime model
# ---------------------------------------------------------------------------
def train_regime_model(
    regime_dataset: pd.DataFrame,
    params: Dict[str, Any],
) -> tuple[Any, Dict[str, float]]:
    """
    Train a simple classifier (LogisticRegression or LightGBM) on regime_dataset.

    Uses temporal split (last test_size_ratio for test). Returns fitted model and metrics.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    cfg = (params.get("regime_baseline") or params).get("train", {})
    test_ratio = float(cfg.get("test_size_ratio", 0.2))
    random_state = int(cfg.get("random_state", 42))
    model_type = (cfg.get("model_type") or "logistic_regression").strip().lower()

    target_col = "risk_environment"
    if target_col not in regime_dataset.columns:
        raise ValueError(f"Regime baseline: dataset missing target column '{target_col}'.")

    feature_cols = [c for c in regime_dataset.columns if c not in ("date", target_col)]
    X = regime_dataset[feature_cols]
    y = regime_dataset[target_col]

    # Drop rows with NaN in features or target
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    if len(X) < 20:
        raise ValueError(
            f"Regime baseline: too few rows after dropping NaN ({len(X)}). Need at least 20."
        )

    n_test = max(1, int(len(X) * test_ratio))
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

    model = None
    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=random_state,
                verbosity=-1,
            )
            model.fit(X_train, y_train)
        except ImportError:
            model_type = "logistic_regression"
    if model is None:
        lr_cfg = (cfg.get("logistic_regression") or {})
        model = LogisticRegression(
            C=float(lr_cfg.get("C", 1.0)),
            max_iter=int(lr_cfg.get("max_iter", 1000)),
            random_state=int(lr_cfg.get("random_state", random_state)),
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return model, metrics


# ---------------------------------------------------------------------------
# 3) Infer current regime
# ---------------------------------------------------------------------------
def infer_current_regime(
    regime_model: Any,
    regime_dataset: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Predict regime and probability for the latest row of regime_dataset.

    Returns dict: current_regime (risk_on / risk_off), probability, timestamp.
    """
    target_col = "risk_environment"
    feature_cols = [c for c in regime_dataset.columns if c not in ("date", target_col)]
    if not feature_cols:
        raise ValueError("Regime baseline: no feature columns in regime_dataset.")

    last = regime_dataset.dropna(subset=feature_cols).tail(1)
    if last.empty:
        return {
            "current_regime": "unknown",
            "probability": None,
            "timestamp": None,
            "message": "No valid last row (all NaN in features).",
        }

    X = last[feature_cols]
    pred = regime_model.predict(X)[0]
    proba = None
    if hasattr(regime_model, "predict_proba"):
        proba = float(regime_model.predict_proba(X)[0, 1])

    ts = last["date"].iloc[0]
    if hasattr(ts, "isoformat"):
        ts = ts.isoformat()
    return {
        "current_regime": "risk_on" if pred == 1 else "risk_off",
        "probability": proba,
        "timestamp": str(ts),
    }
