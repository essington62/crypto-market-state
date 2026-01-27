"""
L2 normalization pipeline for Yahoo Finance macro data.

This pipeline normalizes Yahoo Finance macro L1 data (`yfinance_macro_raw`) to
L2 intermediate format (`yfinance_macro_intermediate`) by:
- Adding metadata columns (symbol, asset, category, source, interval, ingestion_ts)
- Cleaning the time axis (sorted dates, duplicate dates removed)

Each asset remains as a separate partition (no merging or resampling).
"""

from kedro.pipeline import Pipeline, node

from crypto_mkt_state.pipelines.normalization.yfinance.nodes import (
    normalize_yfinance_macro,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create L2 normalization pipeline for Yahoo Finance macro data.

    Returns:
        Pipeline that normalizes `yfinance_macro_raw` into
        `yfinance_macro_intermediate` without changing timezones, resampling
        or merging assets.
    """
    return Pipeline(
        [
            node(
                func=normalize_yfinance_macro,
                inputs={
                    "data": "yfinance_macro_raw",
                    "assets_meta": "params:yfinance.assets",
                },
                outputs="yfinance_macro_intermediate",
                name="normalize_yfinance_macro",
            ),
        ]
    )
