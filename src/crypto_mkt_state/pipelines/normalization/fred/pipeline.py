"""
L2 normalization pipeline for FRED macro data.

This pipeline normalizes FRED macro L1 data (`fred_macro_raw`) to
L2 intermediate format (`fred_macro_intermediate`) by:
- Adding metadata columns (series_id, category, source)
- Cleaning the time axis (sorted dates, duplicate dates removed)

Each series remains as a separate partition (no merging or resampling).
"""

from kedro.pipeline import Pipeline, node

from crypto_mkt_state.pipelines.normalization.fred.nodes import normalize_fred_macro


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create L2 normalization pipeline for FRED macro data.

    Returns:
        Pipeline that normalizes `fred_macro_raw` into
        `fred_macro_intermediate` without changing timezones, resampling
        or merging series.
    """
    return Pipeline(
        [
            node(
                func=normalize_fred_macro,
                inputs={
                    "data": "fred_macro_raw",
                    "series_meta": "params:fred.series",
                },
                outputs="fred_macro_intermediate",
                name="normalize_fred_macro",
            ),
        ]
    )

