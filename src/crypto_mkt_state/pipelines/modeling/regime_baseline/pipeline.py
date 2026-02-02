"""
Regime baseline pipeline: assemble L3 → train → infer current regime.

Inputs: L3 only (crypto, yfinance indices, fred). No L4.
Output: regime_dataset, regime_model, regime_metrics, current_regime.
"""

from kedro.pipeline import Pipeline, node

from crypto_mkt_state.pipelines.modeling.regime_baseline.nodes import (
    assemble_regime_dataset,
    train_regime_model,
    infer_current_regime,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=assemble_regime_dataset,
                inputs={
                    "crypto_l3": "crypto_ohlcv_daily_primary",
                    "yfinance_indices_l3": "yfinance_indices_primary",
                    "fred_l3": "fred_macro_primary",
                    "params": "params:modeling",
                },
                outputs="regime_dataset",
                name="assemble_regime_dataset",
            ),
            node(
                func=train_regime_model,
                inputs={
                    "regime_dataset": "regime_dataset",
                    "params": "params:modeling",
                },
                outputs=["regime_model", "regime_metrics"],
                name="train_regime_model",
            ),
            node(
                func=infer_current_regime,
                inputs={
                    "regime_model": "regime_model",
                    "regime_dataset": "regime_dataset",
                },
                outputs="current_regime",
                name="infer_current_regime",
            ),
        ]
    )
