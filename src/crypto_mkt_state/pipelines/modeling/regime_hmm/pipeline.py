from kedro.pipeline import Pipeline, node
from .nodes import run_walkforward_hmm


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
        ]
    )
