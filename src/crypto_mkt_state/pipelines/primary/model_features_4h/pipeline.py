"""
Kedro pipeline — L5 4h model feature dataset.

Reads:
  btc_spot_features_4h          (L3 4h spot features)
  btc_coinglass_features_4h     (L3 4h derivatives features)
  btc_regime_context_daily      (L4 daily regime context)
  btc_orderbook_features_4h     (L3 4h order book features)
  btc_coinglass_indices_features (L3 daily CoinGlass indices)

Writes:
  btc_model_features_4h         (L5 data/05_model_features/4h/BTCUSDT.parquet)

Intermediates (MemoryDataset):
  spot_derivatives_4h           Node 1 → Node 2
  model_features_no_target      Node 2 → Node 3
  model_features_with_ob        Node 3 → Node 4
  model_features_with_indices   Node 4 → Node 5

DAG:
  join_spot_and_derivatives
    → join_regime_context
      → join_orderbook
        → join_coinglass_indices
          → add_target

Run:
    kedro run --pipeline primary.model_features_4h
"""
from kedro.pipeline import Pipeline, node

from .nodes import (
    add_target,
    join_coinglass_indices,
    join_orderbook,
    join_regime_context,
    join_spot_and_derivatives,
)


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=join_spot_and_derivatives,
                inputs={
                    "spot":        "btc_spot_features_4h",
                    "derivatives": "btc_coinglass_features_4h",
                },
                outputs="spot_derivatives_4h",
                name="join_spot_and_derivatives",
            ),
            node(
                func=join_regime_context,
                inputs={
                    "spot_deriv": "spot_derivatives_4h",
                    "regime":     "btc_regime_context_daily",
                },
                outputs="model_features_no_target",
                name="join_regime_context",
            ),
            node(
                func=join_orderbook,
                inputs={
                    "features":  "model_features_no_target",
                    "orderbook": "btc_orderbook_features_4h",
                },
                outputs="model_features_with_ob",
                name="join_orderbook",
            ),
            node(
                func=join_coinglass_indices,
                inputs={
                    "features": "model_features_with_ob",
                    "indices":  "btc_coinglass_indices_features",
                },
                outputs="model_features_with_indices",
                name="join_coinglass_indices",
            ),
            node(
                func=add_target,
                inputs="model_features_with_indices",
                outputs="btc_model_features_4h",
                name="add_target",
            ),
        ]
    )
