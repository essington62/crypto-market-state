from kedro.pipeline import Pipeline

# ==================== L1 ====================

from crypto_mkt_state.pipelines.ingestion.binance.spot.pipeline import (
    create_pipeline as ingestion_binance_spot_pipeline,
)

from crypto_mkt_state.pipelines.ingestion.binance.spot_incremental.pipeline import (
    create_pipeline as ingestion_binance_spot_incremental_pipeline,
)

from crypto_mkt_state.pipelines.ingestion.binance.spot_4h.pipeline import (
    create_pipeline as ingestion_binance_spot_4h_pipeline,
)

from crypto_mkt_state.pipelines.ingestion.fred.pipeline import (
    create_pipeline as ingestion_fred_pipeline,
)

from crypto_mkt_state.pipelines.ingestion.fred.fred_incremental.pipeline import (
    create_pipeline as ingestion_fred_incremental_pipeline,
)

# REMOVED: old yfinance pipelines (indices and assets)
# from crypto_mkt_state.pipelines.ingestion.yfinance.pipeline import (
#     create_pipeline_indices as ingestion_yfinance_indices_pipeline,
#     create_pipeline_assets as ingestion_yfinance_assets_pipeline,
# )

# KEPT: only the incremental yfinance pipeline
from crypto_mkt_state.pipelines.ingestion.yfinance.yfinance_incremental.pipeline import (
    create_pipeline as ingestion_yfinance_incremental_pipeline,
)

from crypto_mkt_state.pipelines.ingestion.coinglass.intraday_1h.pipeline import (
    create_pipeline as ingestion_coinglass_derivatives_4h_pipeline,
)

from crypto_mkt_state.pipelines.ingestion.coinglass.orderbook_4h.pipeline import (
    create_pipeline as ingestion_coinglass_orderbook_4h_pipeline,
)


# ==================== L2 ====================

from crypto_mkt_state.pipelines.normalization.derivatives_4h.pipeline import (
    create_pipeline as normalization_derivatives_4h_pipeline,
)

from crypto_mkt_state.pipelines.normalization.spot_intraday_4h.pipeline import (
    create_pipeline as normalization_spot_intraday_4h_pipeline,
)

from crypto_mkt_state.pipelines.normalization.spot.pipeline import (
    create_pipeline as normalization_spot_pipeline,
)

from crypto_mkt_state.pipelines.normalization.spot_business_day.pipeline import (
    create_pipeline as normalization_spot_business_day_pipeline,
)

from crypto_mkt_state.pipelines.normalization.macro_daily.pipeline import (
    create_pipeline as normalization_macro_daily_pipeline,
)

from crypto_mkt_state.pipelines.normalization.macro_weekly.pipeline import (
    create_pipeline as normalization_macro_weekly_pipeline,
)

from crypto_mkt_state.pipelines.normalization.macro_monthly.pipeline import (
    create_pipeline as normalization_macro_monthly_pipeline,
)

from crypto_mkt_state.pipelines.normalization.orderbook_4h.pipeline import (
    create_pipeline as normalization_orderbook_4h_pipeline,
)

from crypto_mkt_state.pipelines.normalization.coinglass_indices.pipeline import (
    create_pipeline as normalization_coinglass_indices_pipeline,
)

# ==================== L3 ====================

from crypto_mkt_state.pipelines.primary.derivatives_4h.pipeline import (
    create_pipeline as primary_derivatives_4h_pipeline,
)

from crypto_mkt_state.pipelines.primary.orderbook_4h.pipeline import (
    create_pipeline as primary_orderbook_4h_pipeline,
)

from crypto_mkt_state.pipelines.primary.spot_4h.pipeline import (
    create_pipeline as primary_spot_4h_pipeline,
)

from crypto_mkt_state.pipelines.primary.model_features_4h.pipeline import (
    create_pipeline as primary_model_features_4h_pipeline,
)

from crypto_mkt_state.pipelines.primary.regime_context.pipeline import (
    create_pipeline as primary_regime_context_pipeline,
)

from crypto_mkt_state.pipelines.primary.spot_business_day.pipeline import (
    create_pipeline as primary_spot_business_day_pipeline,
)

from crypto_mkt_state.pipelines.primary.spot_crypto.pipeline import (
    create_pipeline as primary_spot_crypto_pipeline,
)

from crypto_mkt_state.pipelines.primary.coinglass_indices.pipeline import (
    create_pipeline as primary_coinglass_indices_pipeline,
)

# ==================== MODELING ====================

from crypto_mkt_state.pipelines.modeling.regime_hmm.pipeline import (
    create_pipeline as modeling_regime_hmm_pipeline,
)

from crypto_mkt_state.pipelines.modeling.entry_4h.pipeline import (
    create_pipeline as modeling_entry_4h_pipeline,
)

# ==================== REGISTER ====================


def register_pipelines() -> dict[str, Pipeline]:

    pipelines: dict[str, Pipeline] = {}

    # ==================== L1 ====================

    pipelines["ingestion.binance.spot"] = ingestion_binance_spot_pipeline()

    pipelines["ingestion.binance.spot_incremental"] = (
        ingestion_binance_spot_incremental_pipeline()
    )

    pipelines["ingestion.binance.spot_4h"] = ingestion_binance_spot_4h_pipeline()

    pipelines["ingestion.fred"] = ingestion_fred_pipeline()

    pipelines["ingestion.fred.incremental"] = ingestion_fred_incremental_pipeline()

    # REMOVED: old yfinance pipelines
    # pipelines["ingestion.yfinance.indices"] = ingestion_yfinance_indices_pipeline()
    # pipelines["ingestion.yfinance.assets"] = ingestion_yfinance_assets_pipeline()

    # KEPT: only the incremental yfinance pipeline
    pipelines["ingestion.yfinance.incremental"] = ingestion_yfinance_incremental_pipeline()

    pipelines["ingestion.coinglass.derivatives_4h"] = (
        ingestion_coinglass_derivatives_4h_pipeline()
    )

    pipelines["ingestion.coinglass.orderbook_4h"] = (
        ingestion_coinglass_orderbook_4h_pipeline()
    )


    # ==================== L2 ====================

    pipelines["normalization.derivatives_4h"] = normalization_derivatives_4h_pipeline()

    pipelines["normalization.orderbook_4h"] = normalization_orderbook_4h_pipeline()

    pipelines["normalization.coinglass_indices"] = normalization_coinglass_indices_pipeline()

    pipelines["normalization.spot_intraday_4h"] = normalization_spot_intraday_4h_pipeline()

    pipelines["normalization.spot"] = normalization_spot_pipeline()

    pipelines["normalization.spot_business_day"] = (
        normalization_spot_business_day_pipeline()
    )

    pipelines["normalization.macro_daily"] = normalization_macro_daily_pipeline()

    pipelines["normalization.macro_weekly"] = normalization_macro_weekly_pipeline()

    pipelines["normalization.macro_monthly"] = normalization_macro_monthly_pipeline()

    # ==================== L3 ====================

    pipelines["primary.derivatives_4h"] = primary_derivatives_4h_pipeline()

    pipelines["primary.orderbook_4h"] = primary_orderbook_4h_pipeline()

    pipelines["primary.spot_4h"] = primary_spot_4h_pipeline()

    pipelines["primary.model_features_4h"] = primary_model_features_4h_pipeline()

    pipelines["primary.regime_context"] = primary_regime_context_pipeline()

    pipelines["primary.spot_business_day"] = primary_spot_business_day_pipeline()

    pipelines["primary.spot.crypto"] = primary_spot_crypto_pipeline()

    pipelines["primary.coinglass_indices"] = primary_coinglass_indices_pipeline()

    # ==================== MODELING ====================

    pipelines["modeling.regime_hmm"] = modeling_regime_hmm_pipeline()

    pipelines["modeling.entry_4h"] = modeling_entry_4h_pipeline()

    # ==================== FULL FLOW ====================

    pipelines["modeling.regime_hmm_full"] = (
        pipelines["primary.spot.crypto"]
        + pipelines["modeling.regime_hmm"]
    )

    # ==================== DEFAULT ====================

    pipelines["__default__"] = pipelines["modeling.regime_hmm"]

    return pipelines