"""
Kedro pipeline for Yahoo Finance L1 ingestion.
"""

from kedro.pipeline import Pipeline, node

from .nodes import load_yfinance_l1


def create_pipeline() -> Pipeline:
    """
    Create Yahoo Finance ingestion pipeline (L1).
    """
    return Pipeline(
        [
            node(
                func=load_yfinance_l1,
                inputs={
                    "assets": "params:yfinance.assets",
                    "start_date": "params:yfinance.start_date",
                },
                outputs="yfinance_macro_raw",
                name="load_yfinance_l1",
            )
        ]
    )
