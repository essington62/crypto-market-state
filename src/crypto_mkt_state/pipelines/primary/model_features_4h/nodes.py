"""
L5 Model Features — 4h BTC final dataset.

Joins four L3/L4 sources into a single training-ready DataFrame.

Node 1  join_spot_and_derivatives
        btc_spot_features_4h + btc_coinglass_features_4h
        → spot_derivatives_4h (MemoryDataset)

        Inner join on timestamp.  Raises ValueError if result is empty.

Node 2  join_regime_context
        spot_derivatives_4h + btc_regime_context_daily
        → model_features_no_target (MemoryDataset)

        Daily regime is forward-filled onto the 4h grid with a strict
        1-day lag: regime computed on day D is visible only from
        D+1 00:00 UTC onward (never same-day).

Node 3  join_orderbook
        model_features_no_target + btc_orderbook_features_4h
        → model_features_with_ob (MemoryDataset)

        Left join on timestamp.  Order book data starts later than spot/regime;
        ~2 weeks of NaN at the start of ob columns is expected burn-in.

Node 4  add_target
        model_features_with_ob → btc_model_features_4h

        Adds target_return_12h = close.shift(-3) / close - 1.

        WARNING — INTENTIONAL LOOKAHEAD.
        target_return_12h uses future candles.
        This column must NEVER be used as a feature in production inference.
        Use it only for supervised model training where the label is
        computed offline over the full historical window.

All nodes are pure functions — no IO, no Kedro internals, no side effects.
UTC DatetimeIndex throughout.
"""
from __future__ import annotations

import pandas as pd


# ── Node 1 ────────────────────────────────────────────────────────────────────

def join_spot_and_derivatives(
    spot: pd.DataFrame,
    derivatives: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join BTC spot 4h features with CoinGlass derivatives 4h features.

    Both datasets share the same 4h UTC timestamp grid; no resampling needed.

    Parameters
    ----------
    spot : pd.DataFrame
        Output of compute_spot_features_4h (DatetimeIndex UTC, 4h).
    derivatives : pd.DataFrame
        btc_coinglass_features_4h (DatetimeIndex UTC, 4h).

    Returns
    -------
    pd.DataFrame
        Joined DataFrame, sorted ascending by timestamp.

    Raises
    ------
    ValueError
        If the inner join produces an empty DataFrame.
    """
    joined = spot.join(derivatives, how="inner", rsuffix="_deriv")

    if joined.empty:
        raise ValueError(
            "[model_features_4h] join_spot_and_derivatives: inner join is empty. "
            f"Spot range: {spot.index.min()} → {spot.index.max()}. "
            f"Derivatives range: {derivatives.index.min()} → {derivatives.index.max()}."
        )

    joined = joined.sort_index(ascending=True)

    print(
        f"[model_features_4h] Spot × Derivatives join — {len(joined)} rows "
        f"({joined.index[0].date()} → {joined.index[-1].date()})"
    )
    return joined


# ── Node 2 ────────────────────────────────────────────────────────────────────

def join_regime_context(
    spot_deriv: pd.DataFrame,
    regime: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward-fill daily regime context onto the 4h timestamp grid.

    The regime dataset is daily (one row per UTC midnight).  A regime
    computed for day D reflects data available through D 23:59 UTC, so it
    may only appear in the model from D+1 00:00 UTC onward.

    Procedure
    ---------
    1. Shift the regime index forward by exactly 1 day (D → D+1 00:00 UTC).
    2. Build a combined index (all 4h timestamps ∪ shifted regime timestamps),
       sort it, and reindex the regime onto this grid.
    3. Forward-fill the regime columns to propagate each daily value to all
       subsequent 4h candles until the next regime update.
    4. Reindex back to the 4h timestamp grid (left join semantics).
    5. Left-join with spot_deriv — spot rows without a prior regime value
       remain NaN (burn-in at the start of the series).

    Parameters
    ----------
    spot_deriv : pd.DataFrame
        Output of join_spot_and_derivatives (DatetimeIndex UTC, 4h).
    regime : pd.DataFrame
        btc_regime_context_daily (DatetimeIndex UTC, daily).

    Returns
    -------
    pd.DataFrame
        4h DataFrame with regime columns forward-filled.  Sorted ascending.

    Raises
    ------
    ValueError
        If the result is empty after the join.
    """
    # Ensure UTC awareness on both sides
    if regime.index.tz is None:
        regime = regime.copy()
        regime.index = regime.index.tz_localize("UTC")

    # ── Step 1: shift regime by 1 day (strict anti-lookahead) ────────────────
    regime_lagged = regime.copy()
    regime_lagged.index = regime_lagged.index + pd.Timedelta(days=1)

    # ── Step 2: build combined index and reindex regime ───────────────────────
    combined_idx   = spot_deriv.index.union(regime_lagged.index).sort_values()
    regime_on_grid = regime_lagged.reindex(combined_idx)

    # ── Step 3: forward-fill regime values across the combined grid ───────────
    regime_on_grid = regime_on_grid.ffill()

    # ── Step 4: project back to the 4h grid ──────────────────────────────────
    regime_4h = regime_on_grid.reindex(spot_deriv.index)

    # ── Step 5: left join ─────────────────────────────────────────────────────
    result = spot_deriv.join(regime_4h, how="left")

    if result.empty:
        raise ValueError(
            "[model_features_4h] join_regime_context: result is empty after join."
        )

    result = result.sort_index(ascending=True)

    n_regime_filled = result[regime.columns[0]].notna().sum()
    print(
        f"[model_features_4h] Regime context joined — {len(result)} rows, "
        f"{n_regime_filled} rows with regime ({len(result) - n_regime_filled} NaN burn-in)."
    )
    return result


# ── Node 3 ────────────────────────────────────────────────────────────────────

_OB_FEATURE_COLS = [
    "book_imbalance_r05", "book_imbalance_r1", "book_imbalance_r2",
    "bid_ask_ratio_r05",  "bid_ask_ratio_r1",  "bid_ask_ratio_r2",
    "book_depth_r05",     "book_depth_r1",     "book_depth_r2",
    "depth_gradient_near", "depth_gradient_far", "total_depth",
    "imbalance_ma_6", "imbalance_std_6", "depth_zscore_24h",
]


def join_orderbook(
    features: pd.DataFrame,
    orderbook: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join order book depth features onto the 4h model feature grid.

    Order book data starts later than spot/regime data.  The ~2-week gap at
    the start of the series produces NaN in order book columns — this is
    expected burn-in, not an error.  Do not dropna.

    Parameters
    ----------
    features : pd.DataFrame
        Output of join_regime_context (DatetimeIndex UTC, 4h).
    orderbook : pd.DataFrame
        btc_orderbook_features_4h (DatetimeIndex UTC, 4h).

    Returns
    -------
    pd.DataFrame
        features with 15 order book columns appended.  Sorted ascending.
        NaN permitted at start for order book columns.
    """
    ob_cols = [c for c in _OB_FEATURE_COLS if c in orderbook.columns]
    ob = orderbook[ob_cols]

    print(
        f"[model_features_4h] join_orderbook — features: {features.shape}, "
        f"orderbook: {ob.shape} ({ob.index.min().date()} → {ob.index.max().date()})"
    )

    result = features.join(ob, how="left").sort_index(ascending=True)

    nan_per_col = {c: int(result[c].isna().sum()) for c in ob_cols}
    print(
        f"[model_features_4h] join_orderbook — result: {result.shape} | "
        f"ob NaN counts: {nan_per_col}"
    )
    return result


# ── Node 4 ────────────────────────────────────────────────────────────────────

def join_coinglass_indices(
    features: pd.DataFrame,
    indices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward-fill daily CoinGlass index features onto the 4h timestamp grid.

    Same anti-lookahead pattern as join_regime_context:
    index values computed on day D are visible only from D+1 00:00 UTC onward.

    Parameters
    ----------
    features : pd.DataFrame
        Output of join_orderbook (DatetimeIndex UTC, 4h).
    indices : pd.DataFrame
        btc_coinglass_indices_features (DatetimeIndex UTC, daily).

    Returns
    -------
    pd.DataFrame
        4h DataFrame with CoinGlass index feature columns appended.
        Sorted ascending.  NaN permitted at start (burn-in where index
        history is shorter than spot, or D+1 lag leaves first candle empty).
    """
    # Ensure UTC awareness
    if indices.index.tz is None:
        indices = indices.copy()
        indices.index = indices.index.tz_localize("UTC")

    # ── Step 1: shift by 1 day (strict anti-lookahead) ───────────────────────
    indices_lagged = indices.copy()
    indices_lagged.index = indices_lagged.index + pd.Timedelta(days=1)

    # ── Step 2: build combined grid and forward-fill ─────────────────────────
    combined_idx    = features.index.union(indices_lagged.index).sort_values()
    indices_on_grid = indices_lagged.reindex(combined_idx).ffill()

    # ── Step 3: project back to 4h grid ──────────────────────────────────────
    indices_4h = indices_on_grid.reindex(features.index)

    # ── Step 4: left join ────────────────────────────────────────────────────
    result = features.join(indices_4h, how="left").sort_index(ascending=True)

    first_col   = indices.columns[0]
    n_filled    = result[first_col].notna().sum()
    n_burnin    = len(result) - n_filled
    print(
        f"[model_features_4h] CoinGlass indices joined — {len(result)} rows, "
        f"{result.shape[1]} columns total, "
        f"{n_filled} rows with index data ({n_burnin} NaN burn-in)."
    )
    return result


# ── Node 5 ────────────────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the supervised learning target column.

    target_return_12h = close[t+3] / close[t] - 1
    (3-candle forward return = 12h at 4h frequency)

    WARNING — INTENTIONAL LOOKAHEAD
    --------------------------------
    This column uses future price information.  It is only valid for
    offline supervised model training.  It must NEVER be used as a feature
    in production inference or live pipelines.  The last 3 rows will contain
    NaN — expected behaviour, do not drop them.

    Parameters
    ----------
    df : pd.DataFrame
        Joined 4h DataFrame with a `close` column.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with `target_return_12h` appended.

    Raises
    ------
    ValueError
        If the `close` column is absent.
    """
    if "close" not in df.columns:
        raise ValueError(
            "[model_features_4h] add_target: 'close' column not found."
        )

    out = df.copy()
    out["target_return_12h"] = out["close"].shift(-3) / out["close"] - 1

    n_nan = out["target_return_12h"].isna().sum()
    print(
        f"[model_features_4h] Target added — {len(out)} rows, "
        f"{n_nan} trailing NaN (expected: last 3 rows)."
    )
    return out
