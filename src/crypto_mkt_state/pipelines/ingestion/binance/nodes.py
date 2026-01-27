"""
Kedro node for Binance L1 ingestion.

Thin wrapper over the Binance client.
No business logic, no transformations.
"""

from typing import Dict, List

import pandas as pd

from crypto_mkt_state.clients.binance_client import fetch_spot_daily_klines


def load_binance_ohlcv_daily(
    assets: List[str],
    start_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Load daily spot OHLCV data from Binance (L1 raw).

    Args:
        assets:
            List of trading pairs (e.g. ['BTCUSDT', 'ETHUSDT']).
        start_date:
            Start date (YYYY-MM-DD).

    Returns:
        Dict[symbol, DataFrame] compatible with PartitionedDataset.
    """
    output: Dict[str, pd.DataFrame] = {}

    for symbol in assets:
        df = fetch_spot_daily_klines(
            symbol=symbol,
            start_date=pd.to_datetime(start_date),
        )
        output[symbol] = df

    return output
