from kedro.pipeline import Pipeline, node
from .nodes import fetch_binance_futures_perpetual_klines


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=fetch_binance_futures_perpetual_klines,
                inputs={
                    "symbols": "params:binance_futures.symbols",
                    "interval": "params:binance_futures.modules.klines.interval",
                    "start_date": "params:global.start_date",  # ✅ vem do global
                    "base_url": "params:binance_futures.base_url",
                    "limit": "params:binance_futures.modules.klines.limit",  # ✅ caminho correto
                },
                outputs="crypto_perpetual_1h_raw",
                name="fetch_binance_futures_perpetual_klines_l1",
            )
        ]
    )