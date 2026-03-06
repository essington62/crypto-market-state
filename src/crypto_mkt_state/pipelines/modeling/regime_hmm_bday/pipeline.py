"""
Regime HMM — BDay pipeline (Fase 2).

Runs on btc_spot_bday_model_input (business-day calendar, UTC).
Reuses nodes from regime_hmm.nodes — no logic duplication.

Two nodes:
  1. run_walkforward_hmm       → btc_regime_walkforward_bday  (3-split metrics)
  2. train_final_hmm_states    → btc_regime_states_bday       (state + bull_prob series)

Run: kedro run --pipeline modeling.regime_hmm.bday
"""

from kedro.pipeline import Pipeline, node

from crypto_mkt_state.pipelines.modeling.regime_hmm.nodes import (
    run_walkforward_hmm,
    train_final_hmm_states,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_walkforward_hmm,
                inputs={
                    "btc_df": "btc_spot_bday_model_input_validated",
                    "walkforward_params": "params:walkforward",
                    "modeling_params": "params:modeling.regime_hmm",
                },
                outputs="btc_regime_walkforward_bday",
                name="run_walkforward_hmm_bday_node",
            ),
            node(
                func=train_final_hmm_states,
                inputs={
                    "btc_df": "btc_spot_bday_model_input_validated",
                    "walkforward_params": "params:walkforward",
                    "modeling_params": "params:modeling.regime_hmm",
                },
                outputs="btc_regime_states_bday",
                name="train_final_hmm_states_bday_node",
            ),
        ]
    )
