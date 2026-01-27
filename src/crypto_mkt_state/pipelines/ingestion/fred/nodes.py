"""
Kedro node for FRED L1 ingestion.

Thin wrapper over the FRED client.
No business logic, no transformation.
"""

from typing import Dict, List, Optional

import pandas as pd

from crypto_mkt_state.clients.fred_client import fetch_fred_batch


def load_fred_l1(
    series: List[dict],
    start_date: str,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load FRED data for multiple series (L1 raw).

    Args:
        series:
            List of dicts from parameters.yml (expects key 'id').
        start_date:
            Observation start date (YYYY-MM-DD).
        end_date:
            Optional observation end date (YYYY-MM-DD).

    Returns:
        Dict[series_id, DataFrame] compatible with PartitionedDataset.
    """
    series_ids = [s["id"] for s in series]

    return fetch_fred_batch(
        series_ids=series_ids,
        start_date=start_date,
        end_date=end_date,
    )
