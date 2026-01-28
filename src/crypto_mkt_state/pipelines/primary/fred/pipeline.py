"""
L3 primary features pipeline for FRED macro data.

This pipeline computes primary statistical features from L2 intermediate data:
- Variations (deltas, percentage changes)
- Rolling statistics (mean, std, z-scores)
- Relative state (deviation from mean, percentile rank)

Each series is processed independently (no merging or resampling).
"""

from kedro.pipeline import Pipeline, node

from crypto_mkt_state.pipelines.primary.fred.nodes import build_fred_primary_features


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create L3 primary features pipeline for FRED macro data.

    Returns:
        Pipeline that computes primary statistical features from
        `fred_macro_intermediate` to `fred_macro_primary` format.
    """
    return Pipeline(
        [
            node(
                func=build_fred_primary_features,
                inputs="fred_macro_intermediate",
                outputs="fred_macro_primary",
                name="build_fred_primary_features",
            ),
        ]
    )
