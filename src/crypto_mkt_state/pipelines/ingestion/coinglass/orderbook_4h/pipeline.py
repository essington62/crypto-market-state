"""
Kedro pipeline — CoinGlass order book depth 4h incremental ingestion (L1).

Configuration:
    parameters : conf/base/parameters.yml  (params:ingestion.coinglass.orderbook_4h)
    credentials: conf/local/secrets.yml    (params:api_keys.coinglass)
    catalog    : conf/base/catalog_l1.yml  (btc_orderbook_raw_4h,
                                            btc_orderbook_raw_4h_updated)

Run:
    kedro run --pipeline ingestion.coinglass.orderbook_4h

Output partitions (one Parquet file per asset × range):
    data/01_raw/orderbook/coinglass/4h/BTCUSDT_r05.parquet
    data/01_raw/orderbook/coinglass/4h/BTCUSDT_r1.parquet
    data/01_raw/orderbook/coinglass/4h/BTCUSDT_r2.parquet
"""

from kedro.pipeline import Pipeline, node

from .nodes import update_orderbook_4h_incremental


def create_pipeline(**kwargs) -> Pipeline:
    """Create L1 CoinGlass order book 4h incremental pipeline."""
    return Pipeline(
        [
            node(
                func=update_orderbook_4h_incremental,
                inputs={
                    # PartitionedDataset raises "No partitions found" when the
                    # directory is empty or missing, so existing_data uses the
                    # dummy: null param (same pattern as intraday_1h).
                    # The node handles None → {} and merges with fresh API data.
                    "existing_data": "params:ingestion.coinglass.orderbook_4h.dummy",
                    "params": "params:ingestion.coinglass.orderbook_4h",
                    "api_key": "params:api_keys.coinglass",
                },
                outputs="btc_orderbook_raw_4h_updated",
                name="update_orderbook_4h_incremental",
            ),
        ]
    )
