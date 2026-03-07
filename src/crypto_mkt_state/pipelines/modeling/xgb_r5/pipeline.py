"""
XGBoost R5 pipeline — Fase 1B production model.

Node 1: prepare_xgb_r5_input
  btc_spot_crypto_model_input (L4 with technical features)
  + btc_regime_states (HMM output)
  → btc_xgb_r5_model_input

Node 2: train_xgb_r5
  btc_xgb_r5_model_input + params:xgb_r5
  → [xgb_r5_model, xgb_r5_importance]

Node 3: generate_daily_signal
  btc_xgb_r5_model_input + xgb_r5_model + params:xgb_r5
  → btc_daily_signal
"""

from kedro.pipeline import Pipeline, node

from .nodes import generate_daily_signal, prepare_xgb_r5_input, train_xgb_r5


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_xgb_r5_input,
                inputs={
                    "model_input": "btc_spot_crypto_model_input",
                    "hmm_states":  "btc_regime_states",
                },
                outputs="btc_xgb_r5_model_input",
                name="prepare_xgb_r5_input_node",
            ),
            node(
                func=train_xgb_r5,
                inputs={
                    "df":     "btc_xgb_r5_model_input",
                    "params": "params:xgb_r5",
                },
                outputs=["xgb_r5_model", "xgb_r5_importance"],
                name="train_xgb_r5_node",
            ),
            node(
                func=generate_daily_signal,
                inputs={
                    "df":     "btc_xgb_r5_model_input",
                    "model":  "xgb_r5_model",
                    "params": "params:xgb_r5",
                },
                outputs="btc_daily_signal",
                name="generate_daily_signal_node",
            ),
        ]
    )
