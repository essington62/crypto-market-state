"""
Kedro pipeline — Entry 4h LightGBM binary classifier (Modeling layer).

Reads:
  btc_model_features_4h      (L5 — 63 columns, 4h UTC)

Writes:
  btc_entry_model_metrics    (data/06_reports/entry_4h/metrics.parquet)
  btc_entry_model_artifacts  (data/05_models/entry_4h/*.pkl)

Intermediates (MemoryDataset):
  entry_dataset              Node 1 → Node 2
  entry_walkforward_results  Node 2 → Node 3

Run:
    kedro run --pipeline modeling.entry_4h
"""
from kedro.pipeline import Pipeline, node

from .nodes import prepare_entry_dataset, run_walkforward_evaluation, save_metrics_report


def create_pipeline(**kwargs) -> Pipeline:
    """Create modeling.entry_4h pipeline."""
    return Pipeline(
        [
            node(
                func=prepare_entry_dataset,
                inputs={
                    "df":     "btc_model_features_4h",
                    "params": "params:modeling.entry_4h",
                },
                outputs="entry_dataset",
                name="prepare_entry_dataset",
            ),
            node(
                func=run_walkforward_evaluation,
                inputs={
                    "entry_dataset": "entry_dataset",
                    "params":        "params:modeling.entry_4h",
                },
                outputs=["entry_walkforward_results", "btc_entry_model_artifacts"],
                name="run_walkforward_evaluation",
            ),
            node(
                func=save_metrics_report,
                inputs={"results": "entry_walkforward_results"},
                outputs="btc_entry_model_metrics",
                name="save_metrics_report",
            ),
        ]
    )
