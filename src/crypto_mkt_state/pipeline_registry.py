from kedro.pipeline import Pipeline

# =====================================================
# ==================== L1 INGESTION ===================
# =====================================================

# BINANCE SPOT
from crypto_mkt_state.pipelines.ingestion.binance.spot.pipeline import (
    create_pipeline as ingestion_binance_spot_pipeline,
)

# BINANCE FUTURES - KLINES
from crypto_mkt_state.pipelines.ingestion.binance.futures.klines.pipeline import (
    create_pipeline as ingestion_binance_futures_klines_pipeline,
)

# BINANCE FUTURES - FUNDING  ✅ (novo)
from crypto_mkt_state.pipelines.ingestion.binance.futures.funding.pipeline import (
    create_pipeline as ingestion_binance_futures_funding_pipeline,
)

# FRED
from crypto_mkt_state.pipelines.ingestion.fred.pipeline import (
    create_pipeline as ingestion_fred_pipeline,
)

# YFINANCE
from crypto_mkt_state.pipelines.ingestion.yfinance.pipeline import (
    create_pipeline_indices as ingestion_yfinance_indices_pipeline,
    create_pipeline_assets as ingestion_yfinance_assets_pipeline,
)

# =====================================================
# ================== L2 NORMALIZATION =================
# =====================================================

from crypto_mkt_state.pipelines.normalization.spot.pipeline import (
    create_pipeline as normalization_spot_pipeline,
)

from crypto_mkt_state.pipelines.normalization.spot_business_day.pipeline import (
    create_pipeline as normalization_spot_business_day_pipeline,
)

# =====================================================
# ==================== REGISTER =======================
# =====================================================

def register_pipelines() -> dict[str, Pipeline]:

    pipelines: dict[str, Pipeline] = {}

    # ---------------- L1 ----------------

    pipelines["ingestion.binance.spot"] = ingestion_binance_spot_pipeline()
    pipelines["ingestion.binance.futures.klines"] = ingestion_binance_futures_klines_pipeline()
    pipelines["ingestion.binance.futures.funding"] = ingestion_binance_futures_funding_pipeline()  # ✅ novo

    pipelines["ingestion.fred"] = ingestion_fred_pipeline()
    pipelines["ingestion.yfinance.indices"] = ingestion_yfinance_indices_pipeline()
    pipelines["ingestion.yfinance.assets"] = ingestion_yfinance_assets_pipeline()

    # Agregador Binance (spot + futures klines + funding)
    pipelines["ingestion.binance"] = (
        pipelines["ingestion.binance.spot"]
        + pipelines["ingestion.binance.futures.klines"]
        + pipelines["ingestion.binance.futures.funding"]
    )

    # ---------------- L2 ----------------

    pipelines["normalization.spot"] = normalization_spot_pipeline()
    pipelines["normalization.spot_business_day"] = normalization_spot_business_day_pipeline()

    # ---------------- DEFAULT ----------------

    pipelines["__default__"] = pipelines["ingestion.yfinance.indices"]

    return pipelines