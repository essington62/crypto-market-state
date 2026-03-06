from kedro.pipeline import Pipeline, node
from .nodes import run_walkforward_hmm, train_final_hmm_states


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_walkforward_hmm,
                inputs={
                    "btc_df": "btc_spot_crypto_model_input",
                    "walkforward_params": "params:walkforward",
                    "modeling_params": "params:modeling.regime_hmm",
                },
                outputs="hmm_walkforward_metrics_l4",
                name="run_walkforward_hmm_node",
            ),
            node(
                func=train_final_hmm_states,
                inputs={
                    "btc_df": "btc_spot_crypto_model_input",
                    "walkforward_params": "params:walkforward",
                    "modeling_params": "params:modeling.regime_hmm",
                },
                outputs="btc_regime_states",
                name="train_final_hmm_states_node",
            ),
        ]
    )
