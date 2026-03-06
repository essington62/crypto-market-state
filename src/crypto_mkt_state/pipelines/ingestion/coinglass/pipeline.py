from kedro.pipeline import Pipeline, node

from .nodes import (
    fetch_liquidations,
    fetch_fear_greed,
    fetch_bubble_index,
    fetch_puell_multiple,
    fetch_ahr999,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=fetch_liquidations,
                inputs="params:coinglass",
                outputs="coinglass_liquidations_raw",
                name="fetch_liquidations",
            ),
            node(
                func=fetch_fear_greed,
                inputs="params:coinglass",
                outputs="coinglass_fear_greed_raw",
                name="fetch_fear_greed",
            ),
            node(
                func=fetch_bubble_index,
                inputs="params:coinglass",
                outputs="coinglass_bubble_index_raw",
                name="fetch_bubble_index",
            ),
            node(
                func=fetch_puell_multiple,
                inputs="params:coinglass",
                outputs="coinglass_puell_multiple_raw",
                name="fetch_puell_multiple",
            ),
            node(
                func=fetch_ahr999,
                inputs="params:coinglass",
                outputs="coinglass_ahr999_raw",
                name="fetch_ahr999",
            ),
        ]
    )
