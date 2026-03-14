"""
Kedro pipeline — Coinglass intraday derivatives 4h incremental ingestion (L1).

Configuration:
    endpoints  : conf/base/coinglass_endpoints.yml  (params:coinglass_api.endpoints)
    credentials: conf/local/secrets.yml             (params:api_keys.coinglass)
    parameters : conf/base/parameters.yml           (params:coinglass_intraday.*)

Run:
    kedro run --pipeline ingestion.coinglass.derivatives_4h

Output:
    data/01_raw/derivatives/coinglass/4h/BTCUSDT.parquet
    data/01_raw/derivatives/coinglass/4h/ETHUSDT.parquet
"""

from kedro.pipeline import Pipeline, node

from .nodes import update_coinglass_4h_incremental


def create_pipeline() -> Pipeline:
    """Create L1 Coinglass 4h incremental derivatives pipeline."""
    return Pipeline(
        [
            node(
                func=update_coinglass_4h_incremental,
                inputs={
                    "existing_data": "params:coinglass_intraday.dummy",
                    "assets": "params:coinglass_intraday.assets",
                    "coins": "params:coinglass_intraday.coins",
                    "interval": "params:coinglass_intraday.interval",
                    "start_date": "params:global.start_date",
                    "exchange": "params:coinglass_intraday.exchange",
                    "api_key": "params:api_keys.coinglass",
                    "endpoints_config": "params:coinglass_api.endpoints",
                },
                outputs="coinglass_derivatives_4h_updated",
                name="update_coinglass_4h_incremental",
            ),
        ]
    )
