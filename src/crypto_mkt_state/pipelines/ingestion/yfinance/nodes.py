"""
Kedro node for Yahoo Finance L1 ingestion.

This node is a thin wrapper around the yfinance client.
No business logic is implemented here.
"""

from typing import Dict, List

import pandas as pd

from crypto_mkt_state.clients.yfinance_client import fetch_yfinance_batch
from crypto_mkt_state.utils_temporal import enforce_l1_temporal_contract


def load_yfinance_l1(
    assets: List[dict],
    start_date: str,
    interval: str,
) -> Dict[str, pd.DataFrame]:
    """
    Load Yahoo Finance data for multiple assets (L1 raw).

    Args:
        assets:
            List of dicts from parameters.yml (expects key 'ticker').
        start_date:
            Global observation start date (YYYY-MM-DD, UTC).
        interval:
            Global interval string (e.g. '1d'). YFinance client fetches
            daily bars; this is validated but not changed here.

    Returns:
        Dict[ticker, DataFrame] compatible with PartitionedDataset.
    """
    tickers = [asset["ticker"] for asset in assets]

    raw = fetch_yfinance_batch(
        tickers=tickers,
        start_date=start_date,
    )

    output: Dict[str, pd.DataFrame] = {}

    for ticker, df in raw.items():
        if df.empty:
            output[ticker] = df
            continue

        df = df.copy()

        # YFinance client already returns a `date` column in UTC.
        df = enforce_l1_temporal_contract(
            df=df,
            start_date=start_date,
            interval=interval,  # expected '1d'
        )

        output[ticker] = df

    return output
