from kedro.pipeline import Pipeline, node
from .nodes import normalize_yfinance_indices, normalize_yfinance_assets


def create_pipeline_indices(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=normalize_yfinance_indices,
                inputs=dict(
                    data="yfinance_indices_raw",
                    meta="params:yfinance.indices",
                ),
                outputs="yfinance_indices_intermediate",
                name="normalize_yfinance_indices",
            )
        ]
    )


def create_pipeline_assets(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=normalize_yfinance_assets,
                inputs=dict(
                    data="yfinance_assets_raw",
                    meta="params:yfinance.assets",
                ),
                outputs="yfinance_assets_intermediate",
                name="normalize_yfinance_assets",
            )
        ]
    )
