"""
Kedro pipeline — L3 CoinGlass derivatives 4h feature engineering.

Reads:  btc_coinglass_4h_clean       (L2 single-file ParquetDataset)
Writes: btc_coinglass_features_4h    (L3 single-file ParquetDataset)

Run:
    kedro run --pipeline primary.derivatives_4h
"""
from kedro.pipeline import Pipeline, node

from .nodes import compute_coinglass_derivatives_features


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=compute_coinglass_derivatives_features,
                inputs={
                    "df":                      "btc_coinglass_4h_clean",
                    "oi_momentum_window":      "params:primary.derivatives_4h.oi_momentum_window",
                    "oi_volatility_window":    "params:primary.derivatives_4h.oi_volatility_window",
                    "funding_pressure_window": "params:primary.derivatives_4h.funding_pressure_window",
                },
                outputs="btc_coinglass_features_4h",
                name="compute_coinglass_derivatives_features",
            )
        ]
    )
