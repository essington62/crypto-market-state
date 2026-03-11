from kedro.pipeline import Pipeline, node
from .nodes import update_binance_spot_1h_incremental


def create_pipeline():

    return Pipeline(
        [
            node(
                func=update_binance_spot_1h_incremental,
                inputs={
                    "existing_data": "params:binance_intraday.dummy",
                    "spot_symbols": "params:binance_intraday.assets",
                    "interval": "params:binance_intraday.interval",
                    "global_start_date": "params:binance_intraday.start_date",
                },
                outputs="spot_crypto_1h_updated",
                name="update_binance_spot_1h_incremental",
            )
        ]
    )