"""
Kedro pipeline — L4 Regime Context (daily).

Reads:
  btc_spot_crypto_model_input   (L4 daily BTC features — HMM input)
  btc_coinglass_features_4h     (L3 4h CoinGlass derivatives features)

Writes:
  btc_regime_context_daily      (L4 data/04_model_input/regime_context/daily/)

Intermediates (MemoryDataset):
  r11_regime_features           Node 1 → Node 3
  derivatives_daily_context     Node 2 → Node 3

Run:
    kedro run --pipeline primary.regime_context
"""
from kedro.pipeline import Pipeline, node

from .nodes import (
    compute_r11_regime_features,
    compute_derivatives_daily_context,
    join_regime_context,
)


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=compute_r11_regime_features,
                inputs={
                    "df":             "btc_spot_crypto_model_input",
                    "r11_model_path": "params:primary.regime_context.r11_model_path",
                },
                outputs="r11_regime_features",
                name="compute_r11_regime_features",
            ),
            node(
                func=compute_derivatives_daily_context,
                inputs="btc_coinglass_features_4h",
                outputs="derivatives_daily_context",
                name="compute_derivatives_daily_context",
            ),
            node(
                func=join_regime_context,
                inputs={
                    "r11_features":  "r11_regime_features",
                    "deriv_context": "derivatives_daily_context",
                },
                outputs="btc_regime_context_daily",
                name="join_regime_context",
            ),
        ]
    )
