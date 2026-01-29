from __future__ import annotations

import pandas as pd


def enforce_l1_temporal_contract(
    df: pd.DataFrame,
    start_date: str,
    interval: str,
    *,
    assert_daily: bool = True,
) -> pd.DataFrame:
    """
    Enforce the temporal contract for L1 (ingestion layer).

    Responsibilities:
    - Ensure `date` column exists and is UTC
    - Apply global start_date cut
    - Validate expected periodicity (default: daily)
    - Fail fast if the contract is violated

    This function MUST be used only in L1.
    L2+ layers must never call it.
    """

    if df is None or df.empty:
        return df.copy()

    out = df.copy()

    # --- Date normalization ---
    if "date" not in out.columns:
        raise ValueError("L1 temporal contract violated: missing `date` column")

    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.sort_values("date")

    # --- Global temporal cut ---
    start_ts = pd.to_datetime(start_date, utc=True)
    out = out[out["date"] >= start_ts]

    if out.empty:
        return out

    # --- Periodicity validation ---
    if assert_daily and interval == "1d":
        deltas = out["date"].diff().dropna()

        if not deltas.empty:
            mode_delta = deltas.mode().iloc[0]

            if mode_delta != pd.Timedelta(days=1):
                raise AssertionError(
                    "L1 contract violated: non-daily data detected "
                    f"(mode delta = {mode_delta})"
                )

    return out
