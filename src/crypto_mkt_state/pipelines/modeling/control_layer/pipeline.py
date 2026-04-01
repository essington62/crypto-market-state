from kedro.pipeline import Pipeline, node, pipeline

from .nodes import apply_control_layer_batch


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=apply_control_layer_batch,
                inputs=["btc_model_features_4h", "params:modeling.control_layer"],
                outputs="specialist_4h_controlled_signals",
                name="apply_control_layer_batch",
            ),
        ]
    )
