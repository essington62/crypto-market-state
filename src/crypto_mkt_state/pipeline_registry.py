from kedro.pipeline import Pipeline

# =====================================================
# ==================== L1 INGESTION ===================
# =====================================================

# ---------- BINANCE ----------
from crypto_mkt_state.pipelines.ingestion.binance.pipeline import (
    create_pipeline as ingestion_binance_pipeline,
)

# ---------- FRED ----------
from crypto_mkt_state.pipelines.ingestion.fred.pipeline import (
    create_pipeline as ingestion_fred_pipeline,
)

# ---------- YFINANCE (INDICES + ASSETS, separate pipelines) ----------
from crypto_mkt_state.pipelines.ingestion.yfinance.pipeline import (
    create_pipeline_indices as ingestion_yfinance_indices_pipeline,
    create_pipeline_assets as ingestion_yfinance_assets_pipeline,
)

# =====================================================
# ================== L2 NORMALIZATION =================
# =====================================================

from crypto_mkt_state.pipelines.normalization.crypto.pipeline import (
    create_pipeline as normalization_crypto_pipeline,
)
from crypto_mkt_state.pipelines.normalization.fred.pipeline import (
    create_pipeline as normalization_fred_pipeline,
)
from crypto_mkt_state.pipelines.normalization.yfinance.pipeline import (
    create_pipeline as normalization_yfinance_pipeline,
)

# =====================================================
# ==================== L3 PRIMARY =====================
# =====================================================

from crypto_mkt_state.pipelines.primary.crypto.pipeline import (
    create_pipeline as primary_crypto_pipeline,
)
from crypto_mkt_state.pipelines.primary.fred.pipeline import (
    create_pipeline as primary_fred_pipeline,
)
from crypto_mkt_state.pipelines.primary.yfinance.pipeline import (
    create_pipeline as primary_yfinance_pipeline,
)
from crypto_mkt_state.pipelines.cross_asset.pipeline import (
    create_pipeline as cross_asset_pipeline,
)
from crypto_mkt_state.pipelines.modeling.regime_baseline.pipeline import (
    create_pipeline as regime_baseline_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register all pipelines for the project."""
    pipelines: dict[str, Pipeline] = {}

    # ---------- L1 ----------
    pipelines["ingestion.binance"] = ingestion_binance_pipeline()
    pipelines["ingestion.fred"] = ingestion_fred_pipeline()
    pipelines["ingestion.yfinance.indices"] = ingestion_yfinance_indices_pipeline()
    pipelines["ingestion.yfinance.assets"] = ingestion_yfinance_assets_pipeline()

    # ---------- L2 ----------
    pipelines["normalization.crypto"] = normalization_crypto_pipeline()
    pipelines["normalization.fred"] = normalization_fred_pipeline()
    pipelines["normalization.yfinance"] = normalization_yfinance_pipeline()

    # ---------- L3 ----------
    pipelines["primary.crypto"] = primary_crypto_pipeline()
    pipelines["primary.fred"] = primary_fred_pipeline()
    pipelines["primary.yfinance"] = primary_yfinance_pipeline()

    # ---------- L4 ----------
    pipelines["cross_asset"] = cross_asset_pipeline()

    # ---------- MODELING ----------
    pipelines["modeling.regime_baseline"] = regime_baseline_pipeline()

    # ---------- DEFAULT ----------
    pipelines["__default__"] = pipelines["ingestion.yfinance.indices"]

    return pipelines
