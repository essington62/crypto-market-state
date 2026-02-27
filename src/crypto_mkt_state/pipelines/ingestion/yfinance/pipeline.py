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
    Vai para domínio macro.
    """
    return Pipeline(
        [
            node(
                func=load_yfinance_indices_l1,
                inputs={
                    "indices": "params:yfinance.indices",
                    "start_date": "params:global.start_date",
                },
                outputs="macro_daily_raw",
                name="load_yfinance_indices_l1",
            )
        ]
    )


def create_pipeline_assets(**kwargs) -> Pipeline:
    """
    L1 YFinance tradable assets (business day).
    Vai para domínio spot_business_day.
    """
    return Pipeline(
        [
            node(
                func=load_yfinance_assets_l1,
                inputs={
                    "assets": "params:yfinance.assets",
                    "start_date": "params:global.start_date",
                },
                outputs="spot_business_day_raw",
                name="load_yfinance_assets_l1",
            )
        ]
    )