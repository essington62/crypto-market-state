from kedro.pipeline import Pipeline

# ==================== L1 ====================

from crypto_mkt_state.pipelines.ingestion.binance.spot.pipeline import (
    create_pipeline as ingestion_binance_spot_pipeline,
)

from crypto_mkt_state.pipelines.ingestion.binance.spot_incremental.pipeline import (
    create_pipeline as ingestion_binance_spot_incremental_pipeline,
)

from crypto_mkt_state.pipelines.ingestion.binance.spot_1h.pipeline import (
    create_pipeline as ingestion_binance_spot_1h_pipeline,
)

from crypto_mkt_state.pipelines.ingestion.fred.pipeline import (
    create_pipeline as ingestion_fred_pipeline,
)

from crypto_mkt_state.pipelines.ingestion.fred.fred_incremental.pipeline import (
    create_pipeline as ingestion_fred_incremental_pipeline,
)
from crypto_mkt_state.pipelines.ingestion.yfinance.pipeline import (
    create_pipeline_indices as ingestion_yfinance_indices_pipeline,
    create_pipeline_assets as ingestion_yfinance_assets_pipeline,
)
from crypto_mkt_state.pipelines.ingestion.yfinance.yfinance_incremental.pipeline import (
    create_pipeline as ingestion_yfinance_incremental_pipeline,
)

from crypto_mkt_state.pipelines.ingestion.coinglass.intraday_1h.pipeline import (
    create_pipeline as ingestion_coinglass_derivatives_1h_pipeline,
)

# ==================== L2 ====================

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

# ==================== L3 ====================

from crypto_mkt_state.pipelines.primary.spot_business_day.pipeline import (
    create_pipeline as primary_spot_business_day_pipeline,
)

from crypto_mkt_state.pipelines.primary.spot_crypto.pipeline import (
    create_pipeline as primary_spot_crypto_pipeline,
)

# ==================== MODELING ====================

from crypto_mkt_state.pipelines.modeling.regime_hmm.pipeline import (
    create_pipeline as modeling_regime_hmm_pipeline,
)

# ==================== REGISTER ====================


def register_pipelines() -> dict[str, Pipeline]:

    pipelines: dict[str, Pipeline] = {}

    # ==================== L1 ====================

    pipelines["ingestion.binance.spot"] = ingestion_binance_spot_pipeline()

    pipelines["ingestion.binance.spot_incremental"] = (
        ingestion_binance_spot_incremental_pipeline()
    )

    pipelines["ingestion.binance.spot_1h"] = ingestion_binance_spot_1h_pipeline()

    pipelines["ingestion.fred"] = ingestion_fred_pipeline()

    pipelines["ingestion.fred.incremental"] = ingestion_fred_incremental_pipeline()

    pipelines["ingestion.yfinance.indices"] = ingestion_yfinance_indices_pipeline()

    pipelines["ingestion.yfinance.assets"] = ingestion_yfinance_assets_pipeline()

    pipelines["ingestion.yfinance.incremental"] = ingestion_yfinance_incremental_pipeline()

    pipelines["ingestion.coinglass.derivatives_1h"] = (
        ingestion_coinglass_derivatives_1h_pipeline()
    )

    # ==================== L2 ====================

    pipelines["normalization.spot"] = normalization_spot_pipeline()

    pipelines["normalization.spot_business_day"] = (
        normalization_spot_business_day_pipeline()
    )

    pipelines["normalization.macro_daily"] = normalization_macro_daily_pipeline()

    pipelines["normalization.macro_weekly"] = normalization_macro_weekly_pipeline()

    pipelines["normalization.macro_monthly"] = normalization_macro_monthly_pipeline()

    # ==================== L3 ====================

    pipelines["primary.spot_business_day"] = primary_spot_business_day_pipeline()

    pipelines["primary.spot.crypto"] = primary_spot_crypto_pipeline()

    # ==================== MODELING ====================

    pipelines["modeling.regime_hmm"] = modeling_regime_hmm_pipeline()

    # ==================== FULL FLOW ====================

    pipelines["modeling.regime_hmm_full"] = (
        pipelines["primary.spot.crypto"]
        + pipelines["modeling.regime_hmm"]
    )

    # ==================== DEFAULT ====================

    pipelines["__default__"] = pipelines["modeling.regime_hmm"]

    return pipelines