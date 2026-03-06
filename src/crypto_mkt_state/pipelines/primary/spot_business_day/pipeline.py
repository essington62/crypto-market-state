"""
L3 Primary features for spot — multiple calendars.

Three pipelines:
  create_pipeline():        legacy per-asset (gold, nasdaq, sp500)
                            kedro run --pipeline primary.spot_business_day
  create_bday_pipeline():   BTC BDay (deprecated — kept for backward compat)
                            kedro run --pipeline primary.spot.crypto.bday
  create_daily_pipeline():  BTC 24/7 daily + cross-crypto (Fase 2 canonical)
                            kedro run --pipeline primary.spot.crypto.daily
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    build_spot_business_day_features_partitions,
    compute_btc_daily_features,
    compute_cross_crypto_features,
    consolidate_btc_model_input_daily,
)


def create_pipeline() -> Pipeline:
    """Legacy L3 spot business day pipeline (gold, nasdaq, sp500)."""
    return Pipeline(
        [
            node(
                func=build_spot_business_day_features_partitions,
                inputs={
                    "partitions": "spot_business_day_clean",
                    "params": "params:l3.crypto.spot_daily",
                },
                outputs="spot_business_day_features",
                name="build_spot_business_day_features_l3",
            )
        ]
    )


def create_bday_pipeline() -> Pipeline:
    """
    BDay pipeline — deprecated. Kept for backward compatibility.
    Use create_daily_pipeline() (primary.spot.crypto.daily) for Fase 2.
    """
    # Re-import BDay nodes lazily to avoid import errors if they're removed later.
    from .nodes import (
        compute_btc_daily_features as _btc_feat,
        compute_cross_crypto_features as _cross_feat,
        consolidate_btc_model_input_daily as _consolidate,
    )
    return Pipeline(
        [
            node(
                func=_btc_feat,
                inputs={
                    "partitions": "spot_daily_clean",
                    "params": "params:l3.crypto.spot_business_day",
                },
                outputs="btc_bday_features_raw",
                name="compute_btc_bday_features_node",
            ),
            node(
                func=_cross_feat,
                inputs={
                    "partitions": "spot_daily_clean",
                    "btc_features": "btc_bday_features_raw",
                    "params": "params:l3.crypto.spot_business_day",
                },
                outputs="btc_cross_crypto_features_raw",
                name="compute_cross_crypto_bday_features_node",
            ),
            node(
                func=_consolidate,
                inputs={
                    "btc_features": "btc_bday_features_raw",
                    "cross_features": "btc_cross_crypto_features_raw",
                },
                outputs="btc_spot_bday_model_input",
                name="consolidate_btc_model_input_bday_node",
            ),
        ]
    )


def create_daily_pipeline() -> Pipeline:
    """
    BTC 24/7 daily + cross-crypto feature pipeline (Fase 2 canonical).

    Reads spot_daily_clean (24/7 L2), normalises to midnight UTC,
    preserves weekends. Index: DatetimeIndex freq="D" UTC.

    Node order:
      1. compute_btc_daily_features       → btc_daily_features_raw
      2. compute_cross_crypto_features    → btc_cross_crypto_daily_raw
      3. consolidate_btc_model_input_daily → btc_spot_daily_model_input_v2
    """
    return Pipeline(
        [
            node(
                func=compute_btc_daily_features,
                inputs={
                    "partitions": "spot_daily_clean",
                    "params": "params:l3.crypto.spot_business_day",
                },
                outputs="btc_daily_features_raw",
                name="compute_btc_daily_features_node",
            ),
            node(
                func=compute_cross_crypto_features,
                inputs={
                    "partitions": "spot_daily_clean",
                    "btc_features": "btc_daily_features_raw",
                    "params": "params:l3.crypto.spot_business_day",
                },
                outputs="btc_cross_crypto_daily_raw",
                name="compute_cross_crypto_daily_features_node",
            ),
            node(
                func=consolidate_btc_model_input_daily,
                inputs={
                    "btc_features": "btc_daily_features_raw",
                    "cross_features": "btc_cross_crypto_daily_raw",
                },
                outputs="btc_spot_daily_model_input_v2",
                name="consolidate_btc_model_input_daily_node",
            ),
        ]
    )
