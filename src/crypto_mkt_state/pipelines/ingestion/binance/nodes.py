"""
Binance L1 ingestion: contrato canônico mínimo para price discovery.

Output L1 contém apenas OHLCV + metadados necessários. Dados de microestrutura
(quote_volume, trades, taker_buy_*) são removidos. Preparado para alinhamento
com fontes agregadas (ex.: CoinGecko) em etapas futuras.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from crypto_mkt_state.clients.binance_client import fetch_spot_daily_klines
from crypto_mkt_state.utils_temporal import enforce_l1_temporal_contract


# Colunas obrigatórias no output L1 Binance (contrato canônico)
L1_BINANCE_REQUIRED_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]
# Coluna temporal (L2 espera open_time para renomear → timestamp)
L1_BINANCE_TIME_COLUMN = "open_time"
# Metadados adicionados pelo nó
L1_BINANCE_META = ["asset", "symbol", "source", "interval", "ingestion_ts"]


def _symbol_to_asset(symbol: str) -> str:
    """BTCUSDT → btc; mantém lowercase, remove sufixo USDT se presente."""
    s = symbol.strip().upper()
    if s.endswith("USDT"):
        s = s[:-4]
    return s.lower()


def _build_canonical_l1_binance(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    ingestion_ts: pd.Timestamp,
) -> pd.DataFrame:
    """
    Reduz o DataFrame do client ao contrato L1 canônico e aplica validações.

    - Mantém apenas: open, high, low, close, volume + metadados.
    - Remove: quote_volume, trades, taker_buy_*, close_time e demais microestrutura.
    - Garante open_time como datetime UTC, ordenado, sem duplicatas (keep last).
    - Fail-fast: vazio, coluna obrigatória ausente, duplicatas após normalização.
    """
    if df is None or df.empty:
        raise ValueError(
            f"Binance L1: partition {symbol} returned empty DataFrame."
        )

    for col in L1_BINANCE_REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"Binance L1: partition {symbol} missing required column '{col}'. "
                f"Available: {list(df.columns)}"
            )

    # Coluna temporal a partir de open_time (client)
    if "open_time" not in df.columns:
        raise ValueError(
            f"Binance L1: partition {symbol} missing time column 'open_time'. "
            f"Available: {list(df.columns)}"
        )

    out = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True).dt.normalize()
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")

    n_before = len(out)
    n_after = out["open_time"].nunique()
    if n_before != n_after:
        raise ValueError(
            f"Binance L1: partition {symbol} has duplicate dates after normalization "
            f"(rows={n_before}, unique dates={n_after})."
        )

    for col in L1_BINANCE_REQUIRED_COLUMNS:
        out[col] = out[col].astype(float)

    out["asset"] = _symbol_to_asset(symbol)
    out["symbol"] = symbol.strip().upper()
    out["source"] = "binance"
    out["interval"] = interval
    out["ingestion_ts"] = ingestion_ts

    return out


def load_binance_ohlcv_daily(
    assets: List[str],
    start_date: str,
    interval: str,
) -> Dict[str, pd.DataFrame]:
    """
    Carrega OHLCV diário spot da Binance (L1 raw) no contrato canônico.

    Cada partição contém apenas: open, high, low, close, volume (referência
    local Binance) + asset, symbol, source, interval, ingestion_ts. Eixo
    temporal: open_time (datetime UTC, normalizado para 1d), ordenado, sem
    duplicatas. Dados de microestrutura (quote_volume, trades, taker_buy_*)
    são removidos e não existem na L1.

    Validações (fail-fast):
    - DataFrame vazio → ValueError
    - Coluna obrigatória ausente → ValueError
    - Datas duplicadas após normalização → ValueError

    O contrato L1 é aplicado (start_date, interval) via enforce_l1_temporal_contract.
    L2 espera coluna open_time para renomear para timestamp e definir índice.
    """
    ingestion_ts = pd.Timestamp.now(tz="UTC")
    output: Dict[str, pd.DataFrame] = {}

    for symbol in assets:
        df = fetch_spot_daily_klines(
            symbol=symbol,
            start_date=pd.to_datetime(start_date),
        )

        df = _build_canonical_l1_binance(
            df=df,
            symbol=symbol,
            interval=interval,
            ingestion_ts=ingestion_ts,
        )

        # Contrato temporal global (corte start_date, validação 1d)
        df["date"] = df["open_time"]
        df = enforce_l1_temporal_contract(
            df=df,
            start_date=start_date,
            interval=interval,
        )
        df = df.drop(columns=["date"], errors="ignore")

        if df.empty:
            raise ValueError(
                f"Binance L1: partition {symbol} is empty after applying temporal contract."
            )

        output[symbol] = df

    return output
