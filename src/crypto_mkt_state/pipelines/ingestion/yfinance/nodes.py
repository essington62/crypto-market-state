"""
Kedro node for Yahoo Finance L1 ingestion.

L1 YFinance:
- Conteúdo é espelho da API
- Expansão para diário contínuo (UTC)
- Sem criação de colunas
- Sem metadata artificial
- Apenas normalização do nome da partição (governança)
"""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

from crypto_mkt_state.clients.yfinance_client import fetch_yfinance_batch
from crypto_mkt_state.utils_temporal import enforce_l1_temporal_contract


# -----------------------------------------------------
# Governança de Naming (somente nome do arquivo)
# -----------------------------------------------------
def normalize_ticker(ticker: str) -> str:
    """
    Normaliza apenas o nome da partição.
    NÃO altera o DataFrame.
    """

    t = ticker.strip().lower()

    mapping = {
        "^gspc": "sp500",
        "^ixic": "nasdaq",
        "gc=f": "gold",
        "dx-y.nyb": "dxy",
        "^vix": "vix",
    }

    if t in mapping:
        return mapping[t]

    return (
        t.replace("^", "")
         .replace("=", "")
         .replace(".", "")
         .replace("-", "")
    )


# -----------------------------------------------------
# Expansão diário contínuo (mantida por design)
# -----------------------------------------------------
def expand_to_continuous_daily(
    df: pd.DataFrame,
    end_date: Optional[str] = None,
) -> pd.DataFrame:

    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    start = out["date"].min()
    end_ts = out["date"].max()

    if end_date is not None:
        end_param = pd.to_datetime(end_date, utc=True).normalize()
        end_ts = min(end_ts, end_param)

    end = end_ts.normalize()

    daily_index = pd.date_range(start=start, end=end, freq="D", tz="UTC")

    out = (
        out.set_index("date")
        .reindex(daily_index, method="ffill")
        .reset_index()
        .rename(columns={"index": "date"})
    )

    return out


# -----------------------------------------------------
# Implementação interna
# -----------------------------------------------------
def _load_yfinance_l1_impl(
    items: List[dict],
    start_date: str,
    interval: str,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:

    tickers = [item["ticker"] for item in items]

    raw = fetch_yfinance_batch(
        tickers=tickers,
        start_date=start_date,
    )

    output: Dict[str, pd.DataFrame] = {}

    for ticker, df in raw.items():

        if df is None or df.empty:
            output[normalize_ticker(ticker)] = (
                df if df is not None else pd.DataFrame()
            )
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

        # Expansão para diário contínuo
        df = expand_to_continuous_daily(df, end_date=end_date)

        if df.empty:
            output[normalize_ticker(ticker)] = df
            continue

        # Aplicação do contrato temporal global
        df = enforce_l1_temporal_contract(
            df=df,
            start_date=start_date,
            interval=interval,
        )

        # 🔥 Nome da partição normalizado
        normalized_name = normalize_ticker(ticker)
        output[normalized_name] = df

    return output


# -----------------------------------------------------
# APIs públicas do node
# -----------------------------------------------------
def load_yfinance_indices_l1(
    indices: List[dict],
    start_date: str,
    interval: str,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Índices → domínio macro
    """
    return _load_yfinance_l1_impl(
        items=indices,
        start_date=start_date,
        interval=interval,
        end_date=end_date,
    )


def load_yfinance_assets_l1(
    assets: List[dict],
    start_date: str,
    interval: str,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Assets negociáveis → domínio spot
    """
    return _load_yfinance_l1_impl(
        items=assets,
        start_date=start_date,
        interval=interval,
        end_date=end_date,
    )
