from kedro.pipeline import Pipeline, node

from crypto_mkt_state.pipelines.primary.fred.nodes import (
    build_fred_primary_features,
)


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=build_fred_primary_features,
                inputs={
                    "data": "fred_macro_intermediate",
                    "semantic_config": "params:l3_semantic",
                },
                outputs="fred_macro_primary",
                name="build_fred_primary_features",
            ),
        ]
    )
