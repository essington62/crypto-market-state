"""
Binance L1 ingestion: contrato canônico mínimo para price discovery.

Output L1 contém apenas OHLCV + metadados necessários.

"""
from __future__ import annotations

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
    L1 Binance = espelho da API.

    - Não remove colunas
    - Não cria colunas
    - Apenas garante tipos e corte temporal
    """

    start_dt = pd.to_datetime(start_date, utc=True)
    output: Dict[str, pd.DataFrame] = {}

    for symbol in assets:
        df = fetch_spot_daily_klines(
            symbol=symbol,
            start_date=start_dt,
        )

        if df is None or df.empty:
            raise ValueError(f"L1 Binance: {symbol} retornou vazio.")

        # Garantia de ordenação
        df = df.sort_values("open_time").reset_index(drop=True)

        # Aplicar contrato temporal apenas como corte
        df["date"] = df["open_time"]
        df = enforce_l1_temporal_contract(
            df=df,
            start_date=start_date,
            interval=interval,
        )
        df = df.drop(columns=["date"], errors="ignore")

        if df.empty:
            raise ValueError(
                f"L1 Binance: {symbol} vazio após aplicar contrato temporal."
            )

        output[symbol] = df

    return output

