from kedro.pipeline import Pipeline

# -----------------------
# L1 INGESTION
# -----------------------
from crypto_mkt_state.pipelines.ingestion.yfinance.pipeline import (
    create_pipeline as ingestion_yfinance_pipeline,
)

# -----------------------
# L2 NORMALIZATION
# -----------------------
from crypto_mkt_state.pipelines.normalization.crypto.pipeline import (
    create_pipeline as normalization_crypto_pipeline,
)

# -----------------------
# L3 PRIMARY
# -----------------------
from crypto_mkt_state.pipelines.primary.crypto.pipeline import (
    create_pipeline as primary_crypto_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register all pipelines for the project."""
    pipelines: dict[str, Pipeline] = {}

    # ---------- L1 ----------
    pipelines["ingestion.yfinance"] = ingestion_yfinance_pipeline()

    # ---------- L2 ----------
    pipelines["normalization.crypto"] = normalization_crypto_pipeline()

    # ---------- L3 ----------
    pipelines["primary.crypto"] = primary_crypto_pipeline()

    # ---------- DEFAULT ----------
    pipelines["__default__"] = pipelines["ingestion.yfinance"]

    return pipelines
