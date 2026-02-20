"""
Kedro pipelines for Yahoo Finance L1 ingestion.

Two pipelines by design: INDICES (macro context) and ASSETS (tradable).
Indices must never receive return/momentum/vol features in L2/L3.
"""
from kedro.pipeline import Pipeline, node
from .nodes import load_yfinance_indices_l1, load_yfinance_assets_l1


def create_pipeline_indices(**kwargs) -> Pipeline:
    """
    L1 YFinance macro context.
    Vai para domínio macro (independente da fonte).
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
                outputs="macro_daily_raw",   # 🔥 domínio macro
                name="load_yfinance_indices_l1",
            )
        ]
    )


def create_pipeline_assets(**kwargs) -> Pipeline:
    """
    L1 YFinance tradable assets.
    Vai para domínio spot (source-agnostic).
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
                outputs="spot_daily_raw",   # 🔥 domínio spot
                name="load_yfinance_assets_l1",
            )
        ]
    )
