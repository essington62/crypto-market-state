from kedro.pipeline import Pipeline, node
from .nodes import fetch_futures_funding_hist


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=fetch_futures_funding_hist,
                inputs={
                    "symbols": "params:binance_futures.symbols",
                    "start_date": "params:global.start_date",
                    "base_url": "params:binance_futures.base_url",
                    "limit": "params:binance_futures.modules.funding.limit",  # ✅ caminho correto
                },
                outputs="crypto_funding_raw",
                name="fetch_binance_futures_funding_l1",
            )
        ]
    )