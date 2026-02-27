"""
Kedro node for Yahoo Finance L1 ingestion.

L1 YFinance:
- Conteúdo é espelho da API
- Sem expansão de calendário
- Sem forward-fill
- Sem criação de datas artificiais
- Apenas ordenação, deduplicação e normalização de nome de partição
"""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

from crypto_mkt_state.clients.yfinance_client import fetch_yfinance_batch


# -----------------------------------------------------
# Governança de Naming (somente nome da partição)
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
# Implementação interna (L1 puro espelho)
# -----------------------------------------------------
def _load_yfinance_l1_impl(
    items: List[dict],
    start_date: str,
) -> Dict[str, pd.DataFrame]:

    tickers = [item["ticker"] for item in items]

    raw = fetch_yfinance_batch(
        tickers=tickers,
        start_date=start_date,
    )

    output: Dict[str, pd.DataFrame] = {}

    for ticker, df in raw.items():

        normalized_name = normalize_ticker(ticker)

        if df is None or df.empty:
            output[normalized_name] = (
                df if df is not None else pd.DataFrame()
            )
            continue

        df = df.copy()

        # Garantia mínima estrutural L1
        if "date" not in df.columns:
            raise ValueError(
                f"L1 YFinance: missing 'date' column for ticker {ticker}"
            )

        df["date"] = pd.to_datetime(df["date"], utc=True)

        # Ordenação e deduplicação apenas
        df = (
            df.sort_values("date")
              .drop_duplicates(subset=["date"], keep="last")
              .reset_index(drop=True)
        )

        output[normalized_name] = df

    return output


# -----------------------------------------------------
# APIs públicas do node
# -----------------------------------------------------
def load_yfinance_indices_l1(
    indices: List[dict],
    start_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Índices → domínio macro
    """
    return _load_yfinance_l1_impl(
        items=indices,
        start_date=start_date,
    )


def load_yfinance_assets_l1(
    assets: List[dict],
    start_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Assets negociáveis → domínio spot
    """
    return _load_yfinance_l1_impl(
        items=assets,
        start_date=start_date,
    )