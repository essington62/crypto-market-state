from kedro.pipeline import Pipeline, node
from .nodes import prepare_xgb_inputs, run_walkforward_xgb


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_xgb_inputs,
                inputs={
                    "btc_df":      "btc_spot_daily_model_input_v2",
                    "hmm_states":  "btc_regime_states",
                },
                outputs="btc_xgb_model_input",
                name="prepare_xgb_inputs_node",
            ),
            node(
                func=run_walkforward_xgb,
                inputs={
                    "df":                "btc_xgb_model_input",
                    "walkforward_params": "params:walkforward",
                    "xgb_params":        "params:xgb_regime",
                },
                outputs=["xgb_regime_walkforward_metrics", "xgb_regime_feature_importance"],
                name="run_walkforward_xgb_node",
            ),
        ]
    )
