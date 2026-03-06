"""
XGBoost Calibration — 3 variantes vs baseline ternário

Standalone script — não modifica nenhum pipeline Kedro.
Lê dados já salvos pelo pipeline modeling.xgb_regime.

Variantes:
  A — Ternário thresholds restritivos (forçar neutro_pct >= 30%)
  B — Binário Bull vs Não-Bull (threshold 5d > 1.5%)
  C — Binário com scale_pos_weight = n_neg / n_pos por split

Baseline ternário (do pipeline):
  split_1: delta=-0.0040  split_2: delta=-0.0058  split_3: delta=-0.0021

Run:
  conda run -n crypto_market_state python notebooks/xgb_calibration.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
import yaml
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
CONF_DIR  = ROOT / "conf" / "base"

MODEL_INPUT = DATA_DIR / "04_model_input" / "xgb" / "btc_xgb_model_input.parquet"
PARAMS_FILE = CONF_DIR / "parameters.yml"


# ── Load data & params ─────────────────────────────────────────────────────────

df = pd.read_parquet(MODEL_INPUT)
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
df = df.sort_index()

with open(PARAMS_FILE) as f:
    params = yaml.safe_load(f)

wf_params   = params["walkforward"]
xgb_params  = params["xgb_regime"]

FEATURES: list[str] = xgb_params["features"]
FEATURES = [f for f in FEATURES if f in df.columns]

EMBARGO_DAYS: int = wf_params["embargo_days"]
SPLITS        = wf_params["splits"]

# XGBoost shared hyperparams
XGB_BASE = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    verbosity=0,
    random_state=42,
    n_jobs=-1,
)

BASELINE_DELTA5D = {"split_1": -0.0040, "split_2": -0.0058, "split_3": -0.0021}


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_fwd_ret(horizon: int) -> pd.Series:
    return sum(df["log_return"].shift(-k) for k in range(1, horizon + 1))


def get_split_bounds(split_cfg: dict, valid_idx: pd.DatetimeIndex, purge_total: int):
    train_start = pd.Timestamp(str(split_cfg["train_start"]), tz="UTC")
    train_end   = pd.Timestamp(str(split_cfg["train_end"]),   tz="UTC")
    test_start  = pd.Timestamp(str(split_cfg["test_start"]),  tz="UTC")
    test_end_raw = split_cfg["test_end"]
    test_end = (
        valid_idx.max() if test_end_raw is None
        else pd.Timestamp(str(test_end_raw), tz="UTC")
    )
    purged_train_end = min(train_end, test_start - timedelta(days=purge_total))
    return train_start, purged_train_end, test_start, test_end


def delta_metrics(fwd_test: pd.Series, preds: np.ndarray, bull_label: int, bear_label: int):
    bull_mask = preds == bull_label
    bear_mask = preds == bear_label
    n_test = len(preds)

    E_bull = fwd_test[bull_mask].mean() if bull_mask.sum() > 0 else np.nan
    E_bear = fwd_test[bear_mask].mean() if bear_mask.sum() > 0 else np.nan
    delta  = (E_bull - E_bear) if not (np.isnan(E_bull) or np.isnan(E_bear)) else np.nan

    bull_prec = float((fwd_test[bull_mask] > 0).mean()) if bull_mask.sum() > 0 else np.nan
    bear_prec = float((fwd_test[bear_mask] < 0).mean()) if bear_mask.sum() > 0 else np.nan

    return dict(
        delta=delta, E_bull=E_bull, E_bear=E_bear,
        bull_prec=bull_prec, bear_prec=bear_prec,
        bull_pct=bull_mask.sum() / n_test,
        bear_pct=bear_mask.sum() / n_test,
        n_bull_test=int(bull_mask.sum()),
        n_bear_test=int(bear_mask.sum()),
    )


# ── VARIANTE A — Ternário thresholds restritivos ───────────────────────────────

def run_variant_a() -> pd.DataFrame:
    """
    3-class ternário com thresholds maiores para forçar neutro_pct >= 30%.
    horizon_2d: bull_thr=0.015, bear_thr=0.012
    horizon_5d: bull_thr=0.025, bear_thr=0.020
    """
    configs = [
        dict(name="horizon_2d", horizon=2, bull_thr=0.015, bear_thr=0.012),
        dict(name="horizon_5d", horizon=5, bull_thr=0.025, bear_thr=0.020),
    ]
    rows = []
    for cfg in configs:
        h, bull_thr, bear_thr, h_name = cfg["horizon"], cfg["bull_thr"], cfg["bear_thr"], cfg["name"]
        fwd_ret = make_fwd_ret(h)

        labels = pd.Series(1, index=df.index, dtype=int)
        labels[fwd_ret > bull_thr]  = 2
        labels[fwd_ret < -bear_thr] = 0

        valid_mask = fwd_ret.notna() & df[FEATURES].notna().all(axis=1)
        df_v = df[valid_mask]
        y_v  = labels[valid_mask].astype(int)
        fwd_v = fwd_ret[valid_mask]

        purge_total = h + EMBARGO_DAYS

        for sp in SPLITS:
            ts, te, ts2, te2 = get_split_bounds(sp, df_v.index, purge_total)
            tr = df_v.loc[ts:te];  yt = y_v.loc[tr.index]
            te_df = df_v.loc[ts2:te2]; yte = y_v.loc[te_df.index]
            fwd_te = fwd_v.loc[te_df.index]

            if len(tr) < 30 or len(te_df) < 5:
                continue

            neutro_pct_train = float((yt == 1).mean())

            model = xgb.XGBClassifier(**XGB_BASE)
            model.fit(tr[FEATURES].values, yt.values)
            preds = model.predict(te_df[FEATURES].values)

            m = delta_metrics(fwd_te, preds, bull_label=2, bear_label=0)
            neutro_pct = (preds == 1).sum() / len(preds)
            rows.append(dict(
                variant="A_ternary_tight",
                split=sp["name"], horizon=h_name,
                n_train=len(tr), n_test=len(te_df),
                neutro_pct_train=neutro_pct_train,
                neutro_pct=neutro_pct,
                **m,
            ))

    return pd.DataFrame(rows)


# ── VARIANTE B — Binário Bull vs Não-Bull ──────────────────────────────────────

def run_variant_b() -> pd.DataFrame:
    """
    Binary: Bull=1 if fwd_5d > 1.5%, else 0.
    horizon_2d: Bull=1 if fwd_2d > 1.0%, else 0.
    E_bear = E[fwd | prediction=0 (Não-Bull)] — inclui neutral days.
    """
    configs = [
        dict(name="horizon_2d", horizon=2, bull_thr=0.010),
        dict(name="horizon_5d", horizon=5, bull_thr=0.015),
    ]
    rows = []
    for cfg in configs:
        h, bull_thr, h_name = cfg["horizon"], cfg["bull_thr"], cfg["name"]
        fwd_ret = make_fwd_ret(h)

        labels = (fwd_ret > bull_thr).astype(int)

        valid_mask = fwd_ret.notna() & df[FEATURES].notna().all(axis=1)
        df_v = df[valid_mask]
        y_v  = labels[valid_mask]
        fwd_v = fwd_ret[valid_mask]

        purge_total = h + EMBARGO_DAYS

        for sp in SPLITS:
            ts, te, ts2, te2 = get_split_bounds(sp, df_v.index, purge_total)
            tr = df_v.loc[ts:te];  yt = y_v.loc[tr.index]
            te_df = df_v.loc[ts2:te2]; yte = y_v.loc[te_df.index]
            fwd_te = fwd_v.loc[te_df.index]

            if len(tr) < 30 or len(te_df) < 5:
                continue

            model = xgb.XGBClassifier(**XGB_BASE)
            model.fit(tr[FEATURES].values, yt.values)
            preds = model.predict(te_df[FEATURES].values)

            # delta = E[fwd | pred=Bull] - E[fwd | pred=Non-Bull]
            m = delta_metrics(fwd_te, preds, bull_label=1, bear_label=0)
            # For binary: neutro_pct = 0 by definition
            rows.append(dict(
                variant="B_binary",
                split=sp["name"], horizon=h_name,
                n_train=len(tr), n_test=len(te_df),
                neutro_pct_train=0.0,
                neutro_pct=0.0,
                **m,
            ))

    return pd.DataFrame(rows)


# ── VARIANTE C — Binário + scale_pos_weight ────────────────────────────────────

def run_variant_c() -> pd.DataFrame:
    """
    Igual B mas com scale_pos_weight = n_neg / n_pos calculado no treino.
    Reduz viés do modelo em direção à classe maioritária.
    """
    configs = [
        dict(name="horizon_2d", horizon=2, bull_thr=0.010),
        dict(name="horizon_5d", horizon=5, bull_thr=0.015),
    ]
    rows = []
    for cfg in configs:
        h, bull_thr, h_name = cfg["horizon"], cfg["bull_thr"], cfg["name"]
        fwd_ret = make_fwd_ret(h)

        labels = (fwd_ret > bull_thr).astype(int)

        valid_mask = fwd_ret.notna() & df[FEATURES].notna().all(axis=1)
        df_v = df[valid_mask]
        y_v  = labels[valid_mask]
        fwd_v = fwd_ret[valid_mask]

        purge_total = h + EMBARGO_DAYS

        for sp in SPLITS:
            ts, te, ts2, te2 = get_split_bounds(sp, df_v.index, purge_total)
            tr = df_v.loc[ts:te];  yt = y_v.loc[tr.index]
            te_df = df_v.loc[ts2:te2]; yte = y_v.loc[te_df.index]
            fwd_te = fwd_v.loc[te_df.index]

            if len(tr) < 30 or len(te_df) < 5:
                continue

            n_pos = int((yt == 1).sum())
            n_neg = int((yt == 0).sum())
            spw   = n_neg / n_pos if n_pos > 0 else 1.0

            model = xgb.XGBClassifier(**XGB_BASE, scale_pos_weight=spw)
            model.fit(tr[FEATURES].values, yt.values)
            preds = model.predict(te_df[FEATURES].values)

            m = delta_metrics(fwd_te, preds, bull_label=1, bear_label=0)
            rows.append(dict(
                variant="C_binary_spw",
                split=sp["name"], horizon=h_name,
                n_train=len(tr), n_test=len(te_df),
                neutro_pct_train=0.0,
                neutro_pct=0.0,
                scale_pos_weight=round(spw, 2),
                **m,
            ))

    return pd.DataFrame(rows)


# ── Run all variants ───────────────────────────────────────────────────────────

print("Running variant A (ternary tight)...")
df_a = run_variant_a()

print("Running variant B (binary)...")
df_b = run_variant_b()

print("Running variant C (binary + scale_pos_weight)...")
df_c = run_variant_c()

results = pd.concat([df_a, df_b, df_c], ignore_index=True)


# ── Report ─────────────────────────────────────────────────────────────────────

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

COLS_REPORT = [
    "variant", "split", "horizon", "n_train", "n_test",
    "delta", "E_bull", "E_bear",
    "bull_prec", "bear_prec",
    "bull_pct", "neutro_pct",
    "n_bull_test", "n_bear_test",
]

for horizon in ["horizon_2d", "horizon_5d"]:
    print(f"\n{'='*100}")
    print(f"  HORIZONTE: {horizon}")
    print(f"{'='*100}")
    sub = results[results["horizon"] == horizon][COLS_REPORT].reset_index(drop=True)
    print(sub.to_string(index=False))

# ── Baseline comparison (horizon_5d) ──────────────────────────────────────────

print(f"\n{'='*100}")
print("  COMPARAÇÃO delta_5d: Variantes vs Baseline Ternário")
print(f"{'='*100}")
print(f"  {'Baseline ternário (pipeline)':.<40} split_1={BASELINE_DELTA5D['split_1']:+.4f}  split_2={BASELINE_DELTA5D['split_2']:+.4f}  split_3={BASELINE_DELTA5D['split_3']:+.4f}")

for variant in results["variant"].unique():
    sub = results[(results["variant"]==variant) & (results["horizon"]=="horizon_5d")].set_index("split")
    vals = {s: sub.loc[s, "delta"] if s in sub.index else np.nan for s in ["split_1","split_2","split_3"]}
    mean_d = np.nanmean(list(vals.values()))
    positives = sum(1 for v in vals.values() if not np.isnan(v) and v > 0)
    label = f"{variant}"
    print(f"  {label:.<40} split_1={vals['split_1']:+.4f}  split_2={vals['split_2']:+.4f}  split_3={vals['split_3']:+.4f}  mean={mean_d:+.4f}  positive={positives}/3")

# ── Ranking by mean delta_5d ───────────────────────────────────────────────────

print(f"\n{'='*100}")
print("  RANKING por mean_delta_5d")
print(f"{'='*100}")

ranking = (
    results[results["horizon"] == "horizon_5d"]
    .groupby("variant")["delta"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"delta": "mean_delta_5d"})
)
ranking["baseline"] = BASELINE_DELTA5D["split_1"]  # reference
print(ranking.to_string(index=False))

# ── Recommendation ─────────────────────────────────────────────────────────────

best_var = ranking.iloc[0]["variant"]
best_delta = ranking.iloc[0]["mean_delta_5d"]

print(f"\n{'='*100}")
print("  RECOMENDAÇÃO")
print(f"{'='*100}")

# Determine number of positive splits for best variant
best_sub = results[(results["variant"]==best_var) & (results["horizon"]=="horizon_5d")]
pos_splits = int((best_sub["delta"] > 0).sum())
mean_bull_pct = best_sub["bull_pct"].mean()
mean_neutro_pct = best_sub["neutro_pct"].mean()

# Check if any variant beats baseline in majority of splits
any_positive = {}
for var in results["variant"].unique():
    s = results[(results["variant"]==var) & (results["horizon"]=="horizon_5d")]
    any_positive[var] = int((s["delta"] > 0).sum())

print(f"\n  Melhor variante por mean_delta_5d: {best_var} (mean={best_delta:+.4f}, positivos={pos_splits}/3)")
print(f"  bull_pct médio: {mean_bull_pct:.1%}  |  neutro_pct médio: {mean_neutro_pct:.1%}\n")

for var, pos in sorted(any_positive.items(), key=lambda x: -x[1]):
    sub = results[(results["variant"]==var) & (results["horizon"]=="horizon_5d")]
    m = sub["delta"].mean()
    print(f"  {var}: {pos}/3 splits positivos, mean_delta={m:+.4f}")

print()
if any(pos >= 2 for pos in any_positive.values()):
    winner = max(any_positive.items(), key=lambda x: (x[1], results[(results["variant"]==x[0]) & (results["horizon"]=="horizon_5d")]["delta"].mean()))
    print(f"  → RECOMENDADO: {winner[0]}")
    print(f"    Razão: maior número de splits com delta > 0 dentre as variantes testadas.")
else:
    print("  → Nenhuma variante alcança delta > 0 em >= 2 splits (horizon_5d).")
    print(f"    Melhor candidato: {best_var} com mean_delta={best_delta:+.4f}.")
    print("    Próximo passo: threshold sweep ou regularização adicional.")

print()
