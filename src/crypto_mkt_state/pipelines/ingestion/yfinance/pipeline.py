"""
Kedro pipelines for Yahoo Finance L1 ingestion.

Two pipelines by design: INDICES (macro context) and ASSETS (tradable).
Indices must never receive return/momentum/vol features in L2/L3.
"""

from kedro.pipeline import Pipeline, node

from .nodes import load_yfinance_indices_l1, load_yfinance_assets_l1


def create_pipeline_indices(**kwargs) -> Pipeline:
    """
    L1 ingestion for YFinance INDICES only (VIX, DXY, GSPC, etc.).

    Output: yfinance_indices_raw -> data/01_raw/macro/yfinance/indices/
    """
    return Pipeline(
        [
            node(
                func=load_yfinance_indices_l1,
                inputs={
                    "indices": "params:yfinance.indices",
                    "start_date": "params:global.start_date",
                    "interval": "params:global.interval",
                },
                outputs="yfinance_indices_raw",
                name="load_yfinance_indices_l1",
            )
        ]
    )


def create_pipeline_assets(**kwargs) -> Pipeline:
    """
    L1 ingestion for YFinance ASSETS only (AAPL, MSFT, etc.).

    Output: yfinance_assets_raw -> data/01_raw/market/yfinance/assets/
    """
    return Pipeline(
        [
            node(
                func=load_yfinance_assets_l1,
                inputs={
                    "assets": "params:yfinance.assets",
                    "start_date": "params:global.start_date",
                    "interval": "params:global.interval",
                },
                outputs="yfinance_assets_raw",
                name="load_yfinance_assets_l1",
            )
        ]
    )
