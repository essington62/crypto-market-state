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

# ---------- YFINANCE ----------
from crypto_mkt_state.pipelines.ingestion.yfinance.pipeline import (
    create_pipeline as ingestion_yfinance_pipeline,
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


def register_pipelines() -> dict[str, Pipeline]:
    """Register all pipelines for the project."""
    pipelines: dict[str, Pipeline] = {}

    # ---------- L1 ----------
    pipelines["ingestion.binance"] = ingestion_binance_pipeline()
    pipelines["ingestion.fred"] = ingestion_fred_pipeline()
    pipelines["ingestion.yfinance"] = ingestion_yfinance_pipeline()

    # ---------- L2 ----------
    pipelines["normalization.crypto"] = normalization_crypto_pipeline()
    pipelines["normalization.fred"] = normalization_fred_pipeline()
    pipelines["normalization.yfinance"] = normalization_yfinance_pipeline()

    # ---------- L3 ----------
    pipelines["primary.crypto"] = primary_crypto_pipeline()
    pipelines["primary.fred"] = primary_fred_pipeline()
    pipelines["primary.yfinance"] = primary_yfinance_pipeline()

    # ---------- DEFAULT ----------
    # Default explícito: ingestion.yfinance (ajuste se quiser outro)
    pipelines["__default__"] = pipelines["ingestion.yfinance"]

    return pipelines
