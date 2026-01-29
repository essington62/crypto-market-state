"""
Kedro node for Binance L1 ingestion.

Thin wrapper over the Binance client.
No business logic, no transformations.
"""

from typing import Dict, List

import pandas as pd

from crypto_mkt_state.clients.binance_client import fetch_spot_daily_klines
from crypto_mkt_state.utils_temporal import enforce_l1_temporal_contract


def load_binance_ohlcv_daily(
    assets: List[str],
    start_date: str,
    interval: str,
) -> Dict[str, pd.DataFrame]:
    """
    Load daily spot OHLCV data from Binance (L1 raw).

    Args:
        assets:
            List of trading pairs (e.g. ['BTCUSDT', 'ETHUSDT']).
        start_date:
            Global observation start date (YYYY-MM-DD, UTC).
        interval:
            Global interval string (e.g. '1d'). Binance client already
            fetches 1d klines; this parameter is validated but does not
            change the underlying API interval.

    Returns:
        Dict[symbol, DataFrame] compatible with PartitionedDataset.
    """
    output: Dict[str, pd.DataFrame] = {}

    for symbol in assets:
        df = fetch_spot_daily_klines(
            symbol=symbol,
            start_date=pd.to_datetime(start_date),
        )

        # Map Binance schema to temporal contract expectations
        df = df.copy()
        df["date"] = pd.to_datetime(df["open_time"], utc=True).dt.normalize()

        df = enforce_l1_temporal_contract(
            df=df,
            start_date=start_date,
            interval=interval,  # expected to be '1d'
        )

        output[symbol] = df

    return output
