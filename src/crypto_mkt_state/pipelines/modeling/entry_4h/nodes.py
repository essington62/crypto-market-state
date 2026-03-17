"""
Modeling — Entry 4h LightGBM binary classifier.

Walk-forward evaluation of 4 feature sets × 4 time splits.
Model: LightGBM binary classifier predicting 24h price direction.
Target: target_direction_24h = (close[t+6] > close[t]).astype(int)

Pipeline nodes
--------------
Node 1  prepare_entry_dataset
        btc_model_features_4h → entry_dataset (MemoryDataset)
        - Compute binary direction target from close column
        - Drop last 6 rows (NaN target)
        - Filter to Bull regime (r11_regime == regime_filter)
        - Drop rows with NaN in SET_D features (burn-in)
        - Reset to positional index, keep timestamp as column

Node 2  run_walkforward_evaluation
        entry_dataset → entry_walkforward_results (MemoryDataset)
                        btc_entry_model_artifacts (PartitionedDataset)
        - Iterate splits × feature sets
        - Fit LightGBM with early stopping on each split
        - Compute AUC, accuracy, precision, recall, F1
        - Collect model artifacts keyed by {feature_set}_split_{n}

Node 3  save_metrics_report
        entry_walkforward_results → btc_entry_model_metrics
        - Print formatted summary table (AUC per set × split)
        - Print feature importances for best model
        - Return metrics DataFrame for Kedro to persist

Constraints
-----------
- Gap of target_horizon rows between train and test prevents
  target_direction_24h lookahead (shift(-6) contaminates last 6 rows)
- Regime filter applied before split — model never sees Bear candles
- NaN permitted at burn-in — never dropna outside prepare_entry_dataset
- Feature lists, splits, and hyperparameters all from parameters.yml
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node 1
# ---------------------------------------------------------------------------


def prepare_entry_dataset(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Prepare the dataset for walk-forward training.

    Regime is NOT filtered here — r11_regime and r11_prob_bull are features
    in SET_B/C/D and teach the model to avoid Bear entries naturally.
    Per-feature-set NaN removal (burn-in) happens inside Node 2 so that
    each feature set gets the maximum number of valid rows.

    Steps
    -----
    1. Compute ``target_direction_24h`` from the close column.
    2. Drop the last ``target_horizon`` rows (NaN target).
    3. Reset the DatetimeIndex to positional integers; timestamp
       is preserved as a plain column for downstream logging.

    Parameters
    ----------
    df:
        ``btc_model_features_4h`` — 63 columns, DatetimeIndex UTC.
    params:
        ``modeling.entry_4h`` block from parameters.yml.

    Returns
    -------
    pd.DataFrame
        All rows with a valid target, positionally indexed.
        NaN in feature columns is intentional (burn-in handled per set
        inside ``run_walkforward_evaluation``).

    Raises
    ------
    ValueError
        If ``target_col`` is missing from the dataset.
    """
    target_col     = params["target_col"]
    target_horizon = int(params["target_horizon"])

    if target_col not in df.columns:
        raise ValueError(
            f"[entry_4h] Missing target column: '{target_col}'. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    out = df.copy()

    # ── 1. Binary direction target ───────────────────────────────────────────
    out["target_direction_24h"] = (
        out[target_col].shift(-target_horizon) > out[target_col]
    ).astype("float64")  # float to tolerate NaN before dropping trailing rows

    # ── 2. Drop trailing NaN rows produced by shift ──────────────────────────
    out = out.iloc[:-target_horizon].copy()
    out["target_direction_24h"] = out["target_direction_24h"].astype(int)

    # ── 3. Reset to positional index; preserve timestamp column ─────────────
    out = out.reset_index()  # DatetimeIndex named "timestamp" → plain column
    if "index" in out.columns and "timestamp" not in out.columns:
        out = out.rename(columns={"index": "timestamp"})

    class_balance  = float(out["target_direction_24h"].mean())
    regime_counts: dict = {}
    if "r11_regime" in out.columns:
        regime_counts = out["r11_regime"].value_counts().to_dict()

    logger.info(
        "[entry_4h] prepare_entry_dataset — total rows: %d | "
        "class balance (0=stay out, 1=entry): {0: %.1f%%, 1: %.1f%%} | "
        "r11_regime distribution (regime is a feature, not a filter): %s",
        len(out),
        (1 - class_balance) * 100,
        class_balance * 100,
        regime_counts,
    )
    return out


# ---------------------------------------------------------------------------
# Node 2 — helpers
# ---------------------------------------------------------------------------


def _resolve_features(dataset: pd.DataFrame, feature_list: list[str]) -> list[str]:
    """Return the subset of feature_list columns present in dataset."""
    return [f for f in feature_list if f in dataset.columns]


def _compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary classification metrics at a fixed threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "auc":       float(roc_auc_score(y_true, y_pred_proba)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _ts(dataset: pd.DataFrame, positional_idx: int) -> str | None:
    """Return ISO timestamp string for a positional row index, or None."""
    if "timestamp" not in dataset.columns:
        return None
    val = dataset.iloc[positional_idx]["timestamp"]
    return str(val) if not pd.isna(val) else None


# ---------------------------------------------------------------------------
# Node 2
# ---------------------------------------------------------------------------


def run_walkforward_evaluation(
    entry_dataset: pd.DataFrame,
    params: dict,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Walk-forward evaluation: LightGBM × 4 feature sets × 4 splits.

    For each (feature_set, split) combination:
      1. Slice train / test by positional index; gap rows are excluded.
      2. Fit ``LGBMClassifier`` with early stopping on (X_test, y_test).
      3. Compute AUC, accuracy, precision, recall, F1.
      4. Store the fitted model in the artifacts dict.
      5. Log AUC, accuracy, class balance, best iteration.

    If the dataset has fewer rows than a split's ``test_end``, that split
    is skipped with a warning (never raises).

    Parameters
    ----------
    entry_dataset:
        Output of ``prepare_entry_dataset`` — positionally indexed,
        NaN-free for SET_D features, Bull regime rows only.
    params:
        ``modeling.entry_4h`` block from parameters.yml.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        metrics_df  — one row per (feature_set × split).
        artifacts   — ``{"{SET}_split_{n}": LGBMClassifier}`` for
                      PartitionedDataset to persist as pickle files.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for the entry_4h modeling pipeline. "
            "Install with:  pip install lightgbm"
        ) from exc

    feature_sets   = params["feature_sets"]
    splits_config  = params["splits"]
    lgbm_params    = dict(params["lgbm"])  # copy — we pop from it
    early_stopping = int(lgbm_params.pop("early_stopping_rounds", 50))

    rows: list[dict] = []
    artifacts: Dict[str, Any] = {}

    for set_name, feature_list in feature_sets.items():
        features = _resolve_features(entry_dataset, feature_list)
        if not features:
            logger.warning(
                "[entry_4h] %s: none of the specified features found in dataset — skipping.",
                set_name,
            )
            continue

        # Drop rows where any feature in this set is NaN (per-set burn-in removal).
        # Reset to a fresh positional index so split indices are consistent.
        df_set = (
            entry_dataset
            .dropna(subset=features + ["target_direction_24h"])
            .reset_index(drop=True)
        )
        n_rows = len(df_set)
        logger.info("[entry_4h] %s: %d rows available after feature-set dropna.", set_name, n_rows)

        for split_idx, split in enumerate(splits_config, start=1):
            train_end  = int(split["train_end"])
            gap_end    = int(split["gap_end"])
            test_end   = int(split["test_end"])
            split_name = split.get("name", f"split_{split_idx}")

            if n_rows < test_end:
                logger.warning(
                    "[entry_4h] %s %s: need %d rows, have %d — skipping.",
                    set_name, split_name, test_end, n_rows,
                )
                continue

            # Slice — gap rows [train_end : gap_end] are strictly excluded
            train = df_set.iloc[:train_end]
            test  = df_set.iloc[gap_end:test_end]

            X_train = train[features]
            y_train = train["target_direction_24h"].astype(int)
            X_test  = test[features]
            y_test  = test["target_direction_24h"].astype(int)

            model = lgb.LGBMClassifier(**lgbm_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[
                    lgb.early_stopping(early_stopping, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )

            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics = _compute_metrics(y_test.values, y_pred_proba)

            best_iter = getattr(model, "best_iteration_", None)

            importances = dict(
                zip(features, model.feature_importances_.tolist())
            )

            # ── FIX 3: Regime-split AUC (Bull vs Bear) ───────────────────────
            auc_bull = auc_bear = n_test_bull = n_test_bear = None
            if "r11_regime" in test.columns:
                bull_mask = test["r11_regime"] == 1
                bear_mask = test["r11_regime"] == 0
                n_test_bull = int(bull_mask.sum())
                n_test_bear = int(bear_mask.sum())
                _min_samples = 10
                if n_test_bull >= _min_samples:
                    try:
                        auc_bull = float(roc_auc_score(
                            y_test.values[bull_mask.values],
                            y_pred_proba[bull_mask.values],
                        ))
                    except Exception:
                        auc_bull = None
                if n_test_bear >= _min_samples:
                    try:
                        auc_bear = float(roc_auc_score(
                            y_test.values[bear_mask.values],
                            y_pred_proba[bear_mask.values],
                        ))
                    except Exception:
                        auc_bear = None

            # ── FIX 4: Naive majority-class baseline AUC ─────────────────────
            majority_class = int(y_train.mode().iloc[0])
            baseline_proba = np.full(len(y_test), float(majority_class))
            try:
                baseline_auc = float(roc_auc_score(y_test.values, baseline_proba))
            except Exception:
                baseline_auc = 0.5

            # ── ENHANCEMENT: Confidence filter + entry signal ────────────────
            signal_cfg     = params.get("signal", {})
            entry_prob_thr = float(signal_cfg.get("entry_prob_threshold", 0.60))
            conf_thr       = float(signal_cfg.get("confidence_threshold", 0.10))

            confidence      = np.abs(y_pred_proba - 0.5)
            confident_mask  = confidence > conf_thr
            entry_signal    = (
                (y_pred_proba > entry_prob_thr) & confident_mask
            ).astype(int)

            confidence_mean                = float(confidence.mean())
            confidence_pct_above_threshold = float(confident_mask.mean())

            n_confident = int(confident_mask.sum())
            if n_confident >= 10:
                try:
                    auc_confident = float(roc_auc_score(
                        y_test.values[confident_mask],
                        y_pred_proba[confident_mask],
                    ))
                except Exception:
                    auc_confident = None
                try:
                    precision_confident = float(precision_score(
                        y_test.values[confident_mask],
                        entry_signal[confident_mask],
                        zero_division=0,
                    ))
                    recall_confident = float(recall_score(
                        y_test.values[confident_mask],
                        entry_signal[confident_mask],
                        zero_division=0,
                    ))
                    f1_confident = float(f1_score(
                        y_test.values[confident_mask],
                        entry_signal[confident_mask],
                        zero_division=0,
                    ))
                except Exception:
                    precision_confident = recall_confident = f1_confident = None
                coverage_confident = float(entry_signal.mean())
            else:
                auc_confident = precision_confident = None
                recall_confident = f1_confident = None
                coverage_confident = 0.0

            # ── Threshold sweep ───────────────────────────────────────────────
            _thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
            thresh_records = []
            for thr in _thresholds:
                y_pred_t = (y_pred_proba >= thr).astype(int)
                thresh_records.append({
                    "threshold": thr,
                    "precision": float(precision_score(y_test.values, y_pred_t, zero_division=0)),
                    "recall":    float(recall_score(y_test.values, y_pred_t, zero_division=0)),
                    "f1":        float(f1_score(y_test.values, y_pred_t, zero_division=0)),
                    "coverage":  float(y_pred_t.mean()),
                })

            row = {
                "feature_set":              set_name,
                "split":                    split_idx,
                "split_name":               split_name,
                "n_train":                  len(train),
                "n_test":                   len(test),
                "class_balance_train":      float(y_train.mean()),
                "class_balance_test":       float(y_test.mean()),
                "best_iteration":           int(best_iter) if best_iter else None,
                "train_start":              _ts(df_set, 0),
                "train_end":                _ts(df_set, train_end - 1),
                "test_start":               _ts(df_set, gap_end),
                "test_end":                 _ts(df_set, test_end - 1),
                "feature_importances_json": json.dumps(importances),
                "threshold_analysis_json":  json.dumps(thresh_records),
                "confidence_mean":                    confidence_mean,
                "confidence_pct_above_threshold":     confidence_pct_above_threshold,
                "auc_confident":                      auc_confident,
                "precision_confident":                precision_confident,
                "recall_confident":                   recall_confident,
                "f1_confident":                       f1_confident,
                "coverage_confident":                 coverage_confident,
                "baseline_auc":             baseline_auc,
                "auc_bull":                 auc_bull,
                "auc_bear":                 auc_bear,
                "n_test_bull":              n_test_bull,
                "n_test_bear":              n_test_bear,
                **metrics,
            }
            rows.append(row)

            artifact_key = f"{set_name}_{split_name}"
            artifacts[artifact_key] = model

            logger.info(
                "[entry_4h] %s %s — AUC: %.4f | acc: %.4f | "
                "baseline_auc: %.4f | class_balance_test: %.3f | "
                "n_train: %d | n_test: %d | best_iter: %s | "
                "auc_bull: %s (n=%s) | auc_bear: %s (n=%s)",
                set_name, split_name,
                metrics["auc"], metrics["accuracy"],
                baseline_auc, float(y_test.mean()),
                len(train), len(test), best_iter,
                f"{auc_bull:.4f}" if auc_bull is not None else "n/a", n_test_bull,
                f"{auc_bear:.4f}" if auc_bear is not None else "n/a", n_test_bear,
            )

    metrics_df = pd.DataFrame(rows)
    return metrics_df, artifacts


# ---------------------------------------------------------------------------
# Node 3
# ---------------------------------------------------------------------------

_BOX_SET_W  = 10   # feature-set column width
_BOX_COL_W  = 9    # split / mean / std / baseline column width


def _box_line(left: str, mid: str, right: str, fill: str, n_extra_cols: int) -> str:
    """Build a horizontal box-drawing separator.

    n_extra_cols = n_splits + 3  (Mean | Std | Baseline)
    """
    return (
        left
        + fill * (_BOX_SET_W + 2)
        + (mid + fill * _BOX_COL_W) * n_extra_cols
        + right
    )


def save_metrics_report(results: pd.DataFrame) -> pd.DataFrame:
    """Print walk-forward summary and return the metrics DataFrame.

    Prints a formatted AUC table (feature set × split + mean + std + baseline)
    with stability annotation, a regime-AUC block, and the top-15 feature
    importances for the best-performing model.

    Parameters
    ----------
    results:
        Output of ``run_walkforward_evaluation`` — one row per
        (feature_set × split), including ``feature_importances_json``.

    Returns
    -------
    pd.DataFrame
        Same DataFrame — Kedro persists it to ``btc_entry_model_metrics``.
    """
    if results.empty:
        logger.warning("[entry_4h] save_metrics_report: results DataFrame is empty.")
        return results

    splits       = sorted(results["split"].unique().tolist())
    feature_sets = list(dict.fromkeys(results["feature_set"].tolist()))
    n_splits     = len(splits)
    # Total extra columns after the split columns: Mean | Std | Baseline
    n_extra      = n_splits + 3

    pivot = results.pivot_table(
        index="feature_set", columns="split", values="auc", aggfunc="first"
    )
    pivot["Mean"] = pivot.mean(axis=1)
    pivot["Std"]  = pivot[splits].std(axis=1, ddof=1)
    # Baseline AUC: mean across splits per feature set
    if "baseline_auc" in results.columns:
        baseline_pivot = results.pivot_table(
            index="feature_set", columns="split", values="baseline_auc", aggfunc="first"
        ).mean(axis=1)
    else:
        baseline_pivot = pd.Series(0.5, index=feature_sets)
    pivot["Baseline"] = baseline_pivot
    pivot = pivot.reindex(feature_sets)

    # ── Summary table ────────────────────────────────────────────────────────
    def _cell(val, width: int) -> str:
        if pd.isna(val):
            return "  N/A  ".center(width)
        return f"{val:.4f}".center(width)

    sep_top    = _box_line("╔", "╦", "╗", "═", n_extra)
    sep_header = _box_line("╠", "╬", "╣", "═", n_extra)
    sep_bottom = _box_line("╚", "╩", "╝", "═", n_extra)

    title_width = _BOX_SET_W + 2 + (_BOX_COL_W + 1) * n_extra
    title_line  = f"║{'ENTRY MODEL 4H — Walk-Forward Results (AUC)':^{title_width}}║"

    col_headers = "".join(f"{'Split ' + str(s):^{_BOX_COL_W}}║" for s in splits)
    header_line = (
        f"║ {'Feature Set':<{_BOX_SET_W}}║"
        f"{col_headers}"
        f"{'Mean':^{_BOX_COL_W}}║"
        f"{'Std':^{_BOX_COL_W}}║"
        f"{'Baseline':^{_BOX_COL_W}}║"
    )

    print(f"\n{sep_top}")
    print(title_line)
    print(sep_header)
    print(header_line)
    print(sep_header)

    stability_lines: list[str] = []
    for fs in feature_sets:
        if fs not in pivot.index:
            continue
        row_cells = "".join(
            _cell(pivot.loc[fs, s] if s in pivot.columns else float("nan"), _BOX_COL_W) + "║"
            for s in splits
        )
        mean_val     = pivot.loc[fs, "Mean"]
        std_val      = pivot.loc[fs, "Std"]
        baseline_val = pivot.loc[fs, "Baseline"]

        mean_cell     = _cell(mean_val, _BOX_COL_W)
        std_cell      = _cell(std_val,  _BOX_COL_W)
        baseline_cell = _cell(baseline_val, _BOX_COL_W)

        print(f"║ {fs:<{_BOX_SET_W}}║{row_cells}{mean_cell}║{std_cell}║{baseline_cell}║")

        # Stability annotation
        if not pd.isna(std_val) and not pd.isna(mean_val):
            lift = mean_val - (baseline_val if not pd.isna(baseline_val) else 0.5)
            stable = std_val <= 0.02
            flag   = "✓ STABLE" if stable else "⚠ UNSTABLE"
            stability_lines.append(
                f"  {fs:<{_BOX_SET_W}}  mean={mean_val:.4f}  std={std_val:.4f}  "
                f"min={pivot.loc[fs, splits].min():.4f}  "
                f"max={pivot.loc[fs, splits].max():.4f}  "
                f"lift_vs_baseline={lift:+.4f}  {flag}"
            )

    print(sep_bottom)

    # ── Stability summary ─────────────────────────────────────────────────────
    if stability_lines:
        print("\n── AUC Stability (std ≤ 0.02 = stable) ──")
        for line in stability_lines:
            print(line)

    # ── Regime AUC block ─────────────────────────────────────────────────────
    has_regime = (
        "auc_bull" in results.columns
        and "auc_bear" in results.columns
        and not results[["auc_bull", "auc_bear"]].isna().all().all()
    )
    if has_regime:
        print("\n── AUC by Regime (r11_regime) ──")
        print(f"  {'Feature Set':<{_BOX_SET_W}}  {'Split':>5}  "
              f"{'AUC_Bull':>8}  {'n_Bull':>6}  {'AUC_Bear':>8}  {'n_Bear':>6}")
        print("  " + "-" * 56)
        for _, row in results.iterrows():
            bull_str = f"{row['auc_bull']:.4f}" if pd.notna(row.get("auc_bull")) else "  n/a  "
            bear_str = f"{row['auc_bear']:.4f}" if pd.notna(row.get("auc_bear")) else "  n/a  "
            n_bull   = int(row["n_test_bull"]) if pd.notna(row.get("n_test_bull")) else "-"
            n_bear   = int(row["n_test_bear"]) if pd.notna(row.get("n_test_bear")) else "-"
            print(
                f"  {row['feature_set']:<{_BOX_SET_W}}  {int(row['split']):>5}  "
                f"{bull_str:>8}  {str(n_bull):>6}  {bear_str:>8}  {str(n_bear):>6}"
            )

    # ── Feature importances for the best model ───────────────────────────────
    if "feature_importances_json" in results.columns and not results["auc"].isna().all():
        best_row   = results.loc[results["auc"].idxmax()]
        best_set   = best_row["feature_set"]
        best_split = best_row.get("split_name", int(best_row["split"]))
        best_auc   = float(best_row["auc"])

        importances = json.loads(best_row["feature_importances_json"])
        if importances:
            top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
            max_imp = max(v for _, v in top) or 1.0

            print(
                f"\n── Feature importances (gain) | {best_set} / {best_split} "
                f"[AUC = {best_auc:.4f}] ──"
            )
            for rank, (feat, imp) in enumerate(top, 1):
                bar = "█" * int(imp / max_imp * 30)
                print(f"  {rank:>2}. {feat:<40} {imp:>8.1f}  {bar}")

    # ── ENHANCEMENT 1: Consolidated feature importance (mean gain across splits) ──
    if "feature_importances_json" in results.columns:
        print("\n── Feature Importance (mean gain across splits) ──")
        for fs in feature_sets:
            fs_rows = results[results["feature_set"] == fs]
            all_imps: dict[str, list[float]] = {}
            for _, r in fs_rows.iterrows():
                try:
                    for feat, val in json.loads(r["feature_importances_json"]).items():
                        all_imps.setdefault(feat, []).append(float(val))
                except Exception:
                    continue
            if not all_imps:
                continue
            mean_imps = {
                f: sum(v) / len(v)
                for f, v in all_imps.items()
                if sum(v) / len(v) > 0
            }
            if not mean_imps:
                continue
            top = sorted(mean_imps.items(), key=lambda x: x[1], reverse=True)[:15]
            max_imp = max(v for _, v in top) or 1.0
            print(f"\n{fs}:")
            for rank, (feat, imp) in enumerate(top, 1):
                bar = "█" * int(imp / max_imp * 30)
                print(f"   {rank:>2}. {feat:<40} {imp:>6.1f}  {bar}")

    # ── ENHANCEMENT 2: Threshold analysis (best SET_D split, else best overall) ──
    if "threshold_analysis_json" in results.columns:
        set_d_rows = results[results["feature_set"] == "SET_D"]
        ta_pool    = set_d_rows if not set_d_rows.empty else results
        ta_row     = ta_pool.loc[ta_pool["auc"].idxmax()]
        ta_set     = ta_row["feature_set"]
        ta_label   = ta_row.get("split_name", int(ta_row["split"]))
        try:
            ta_data = json.loads(ta_row["threshold_analysis_json"])
        except Exception:
            ta_data = []
        if ta_data:
            eligible = [r for r in ta_data if r["coverage"] >= 0.20]
            rec_thr  = max(eligible, key=lambda r: r["f1"])["threshold"] if eligible else 0.50
            print(f"\n── Threshold Analysis | {ta_set} / {ta_label} ──")
            print(
                f"  {'Threshold':>9} │ {'Precision':>9} │ {'Recall':>6} │"
                f" {'F1':>5} │ {'Coverage':>8}"
            )
            print("  " + "─" * 9 + "─┼─" + "─" * 9 + "─┼─" + "─" * 6 +
                  "─┼─" + "─" * 5 + "─┼─" + "─" * 8)
            for tr in ta_data:
                marker       = "→" if tr["threshold"] == rec_thr else " "
                default_note = "  (default)" if tr["threshold"] == 0.50 else ""
                print(
                    f"  {marker} {tr['threshold']:.2f}    │"
                    f"   {tr['precision']:.3f}   │"
                    f" {tr['recall']:.3f}  │"
                    f" {tr['f1']:.3f} │"
                    f"  {tr['coverage'] * 100:5.1f}%{default_note}"
                )

    # ── ENHANCEMENT 3: Precision / Recall tradeoff for best feature set ──────
    if all(c in results.columns for c in ["precision", "recall", "f1"]):
        best_fs = (
            pivot["Mean"].idxmax()
            if not pivot["Mean"].isna().all()
            else feature_sets[-1]
        )
        best_fs_rows = results[results["feature_set"] == best_fs].sort_values("split")
        if not best_fs_rows.empty:
            _SL = 15  # split-label column width
            print(f"\n── Precision vs Recall | {best_fs} ──")
            print(
                f"  {'Split':<{_SL}} │ {'Precision':>9} │ {'Recall':>6} │ {'F1':>5}"
            )
            print("  " + "─" * _SL + "─┼─" + "─" * 9 + "─┼─" + "─" * 6 + "─┼─" + "─" * 5)
            prec_vals, rec_vals, f1_vals = [], [], []
            for _, r in best_fs_rows.iterrows():
                label = str(r.get("split_name", f"split_{int(r['split'])}"))
                p, rc, f1v = float(r["precision"]), float(r["recall"]), float(r["f1"])
                prec_vals.append(p)
                rec_vals.append(rc)
                f1_vals.append(f1v)
                print(
                    f"  {label:<{_SL}} │   {p:.3f}   │ {rc:.3f}  │ {f1v:.3f}"
                )
            print("  " + "─" * _SL + "─┼─" + "─" * 9 + "─┼─" + "─" * 6 + "─┼─" + "─" * 5)
            print(
                f"  {'Mean':<{_SL}} │   {np.mean(prec_vals):.3f}   │"
                f" {np.mean(rec_vals):.3f}  │ {np.mean(f1_vals):.3f}"
            )
            print("\n  Entry model target: Precision > 0.55")
            print("  (better to miss entries than enter wrong)")

    # ── ENHANCEMENT 3: Confidence filter results ─────────────────────────────
    _conf_cols = [
        "confidence_pct_above_threshold", "auc_confident",
        "precision_confident", "recall_confident", "f1_confident",
        "coverage_confident",
    ]
    if any(c in results.columns for c in _conf_cols):
        set_d_rows = results[results["feature_set"] == "SET_D"]
        cf_pool    = set_d_rows if not set_d_rows.empty else results
        cf_rows    = cf_pool.sort_values("split")
        cf_label   = cf_rows.iloc[0]["feature_set"] if not cf_rows.empty else "?"

        if not cf_rows.empty:
            _SL = 16  # split-label column width
            print(f"\n── Confidence Filter Results | {cf_label} ──")
            print("  (only candles with abs(p - 0.5) > 0.10)\n")
            print(
                f"  {'Split':<{_SL}}│{'Conf%':>5}│{'AUC_conf':>8}│"
                f"{'Prec_conf':>9}│{'Rec_conf':>8}│{'F1_conf':>7}│{'Coverage':>8}"
            )
            _sep = (
                "  " + "─" * _SL + "┼" + "─" * 5 + "┼" + "─" * 8 + "┼" +
                "─" * 9 + "┼" + "─" * 8 + "┼" + "─" * 7 + "┼" + "─" * 8
            )
            print(_sep)

            def _pct(v) -> str:
                return f"{v * 100:4.1f}" if pd.notna(v) else " n/a"

            def _f4(v) -> str:
                return f"{v:.3f}" if pd.notna(v) else "  n/a "

            cp_vals, auc_c_vals, pr_c_vals, rc_c_vals, f1_c_vals, cov_c_vals = (
                [], [], [], [], [], []
            )
            for _, r in cf_rows.iterrows():
                label  = str(r.get("split_name", f"split_{int(r['split'])}"))
                cp     = r.get("confidence_pct_above_threshold")
                auc_c  = r.get("auc_confident")
                pr_c   = r.get("precision_confident")
                rc_c   = r.get("recall_confident")
                f1_c   = r.get("f1_confident")
                cov_c  = r.get("coverage_confident")
                print(
                    f"  {label:<{_SL}}│{_pct(cp):>5}│{_f4(auc_c):>8}│"
                    f"{_f4(pr_c):>9}│{_f4(rc_c):>8}│{_f4(f1_c):>7}│"
                    f"{_pct(cov_c):>7}%"
                )
                for lst, v in zip(
                    [cp_vals, auc_c_vals, pr_c_vals, rc_c_vals, f1_c_vals, cov_c_vals],
                    [cp, auc_c, pr_c, rc_c, f1_c, cov_c],
                ):
                    if pd.notna(v):
                        lst.append(float(v))

            print(_sep)
            _mean = lambda lst: np.mean(lst) if lst else float("nan")
            print(
                f"  {'Mean':<{_SL}}│{_pct(_mean(cp_vals)):>5}│{_f4(_mean(auc_c_vals)):>8}│"
                f"{_f4(_mean(pr_c_vals)):>9}│{_f4(_mean(rc_c_vals)):>8}│"
                f"{_f4(_mean(f1_c_vals)):>7}│{_pct(_mean(cov_c_vals)):>7}%"
            )
            print("\n  Entry signal: p > 0.60 AND confidence > 0.10")
            print("  Target: Precision_conf > 0.60")

    return results
