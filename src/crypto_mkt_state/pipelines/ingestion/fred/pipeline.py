"""
Kedro pipeline for FRED L1 ingestion.
"""

from kedro.pipeline import Pipeline, node
from .nodes import load_fred_l1


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_fred_l1,
                inputs={
                    "series": "params:fred.series",
                    "start_date": "params:global.start_date",
                    "interval": "params:global.interval",
                },
                outputs={
                    "daily": "macro_daily_raw",
                    "weekly": "macro_weekly_raw",
                    "monthly": "macro_monthly_raw",
                },
                name="load_fred_l1",
            )
        ]
    )