"""
L2 Normalization — CoinGlass derivatives 4h (BTC only).

Responsibilities:
  - Select the 5 __close columns from L1
  - Rename to semantic names
  - Enforce dtypes (float64), set UTC DatetimeIndex
  - Sort ascending, deduplicate on timestamp
  - Raise ValueError on schema violations

Forbidden: rolling windows, feature engineering, fill, resample,
           forward fill, interpolation, cross-asset operations.

Input columns expected (from coinglass_derivatives_4h_updated L1):
  open_interest_history__close
  open_interest_aggregated_history__close
  funding_rate_history__close
  funding_rate_oi_weight_history__close
  funding_rate_vol_weight_history__close

Output columns (semantic renames):
  oi_binance            ← open_interest_history__close
  oi_aggregated         ← open_interest_aggregated_history__close
  funding_binance       ← funding_rate_history__close
  funding_oi_weighted   ← funding_rate_oi_weight_history__close
  funding_vol_weighted  ← funding_rate_vol_weight_history__close
"""
from __future__ import annotations

import pandas as pd


_REQUIRED_L1_COLUMNS: list[str] = [
    "open_interest_history__close",
    "open_interest_aggregated_history__close",
    "funding_rate_history__close",
    "funding_rate_oi_weight_history__close",
    "funding_rate_vol_weight_history__close",
]

_RENAME_MAP: dict[str, str] = {
    "open_interest_history__close":            "oi_binance",
    "open_interest_aggregated_history__close": "oi_aggregated",
    "funding_rate_history__close":             "funding_binance",
    "funding_rate_oi_weight_history__close":   "funding_oi_weighted",
    "funding_rate_vol_weight_history__close":  "funding_vol_weighted",
}


def normalize_coinglass_derivatives_4h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the BTC CoinGlass 4h L1 DataFrame to L2 contract.

    Parameters
    ----------
    df : pd.DataFrame
        Raw L1 DataFrame with `timestamp` as a column (not index) and
        the 5 required __close columns.

    Returns
    -------
    pd.DataFrame
        L2 DataFrame with DatetimeIndex UTC and 5 semantically named columns.

    Raises
    ------
    ValueError
        If required columns are missing or DataFrame is empty.
    """
    if df is None or df.empty:
        raise ValueError("[L2 derivatives_4h] Input DataFrame is None or empty.")

    # ── 1. Validate timestamp column ────────────────────────────────────────
    if "timestamp" not in df.columns:
        raise ValueError(
            "[L2 derivatives_4h] Missing required column 'timestamp'."
        )

    # ── 2. Validate required __close columns ────────────────────────────────
    missing = [c for c in _REQUIRED_L1_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[L2 derivatives_4h] Missing required L1 columns: {missing}"
        )

    # ── 3. Select timestamp + 5 __close columns ─────────────────────────────
    out = df[["timestamp"] + _REQUIRED_L1_COLUMNS].copy()

    # ── 4. Rename to semantic names ──────────────────────────────────────────
    out = out.rename(columns=_RENAME_MAP)

    # ── 5. Enforce dtypes ────────────────────────────────────────────────────
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    for col in _RENAME_MAP.values():
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    # ── 6. Set DatetimeIndex, sort ascending, deduplicate ────────────────────
    out = out.set_index("timestamp")
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index(ascending=True)

    # ── 7. Final schema check ────────────────────────────────────────────────
    final_missing = [c for c in _RENAME_MAP.values() if c not in out.columns]
    if final_missing:
        raise ValueError(
            f"[L2 derivatives_4h] Final schema missing columns: {final_missing}"
        )

    date_start = out.index[0].isoformat()
    date_end   = out.index[-1].isoformat()
    print(
        f"[L2 derivatives_4h] Normalized — shape: {out.shape} "
        f"({date_start} → {date_end})"
    )

    return out
