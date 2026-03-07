"""
XGBoost Rodada 6 — modelos condicionados por regime.
XGB_BULL: treina/testa apenas em dias Bull → label = return_5d > 0
XGB_BEAR: treina/testa apenas em dias Bear → label = return_5d < 0
Feature nova: regime_strength = abs(bull_prob - 0.5)
Walk-forward 3 splits idêntico às rodadas anteriores.
Script exploratório — NAO modifica pipeline nem salva modelos.

Ajustes em relação ao spec original:
  - close/high/low: carregados do L3 (model input nao tem OHLCV)
  - hmm: coluna 'state' renomeada para 'hmm_state'
  - use_label_encoder: removido (XGBoost >= 1.6)
  - cols deduplicados em todos os filtros
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Carregar dados ────────────────────────────────────────────
btc   = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
ohlcv = pd.read_parquet(
    "data/03_primary/spot/daily/BTCUSDT.parquet"
)[["close", "high", "low"]]
hmm   = pd.read_parquet("data/05_models/regime_hmm/btc_states.parquet")

# coluna real é 'state'
hmm = hmm.rename(columns={"state": "hmm_state"})

df = btc.copy()
df = df.join(ohlcv, how="left")
df = df.join(hmm[["hmm_state", "bull_prob"]], how="left")

close = df["close"]
print(f"Shape apos join: {df.shape}")
print(f"Range: {df.index.min().date()} -> {df.index.max().date()}")

# ── Features técnicas (igual Rodada 5) ───────────────────────
def calc_rsi(series, period):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# MOMENTUM
df["rsi_14_z"]       = (calc_rsi(close, 14) - 50) / 20
df["rsi_30"]         = calc_rsi(close, 30)
ema_12               = close.ewm(span=12, adjust=False).mean()
ema_26               = close.ewm(span=26, adjust=False).mean()
macd                 = ema_12 - ema_26
df["macd_hist_norm"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
df["roc_5"]          = close.pct_change(5)
df["roc_10"]         = close.pct_change(10)
df["roc_21"]         = close.pct_change(21)

# STRUCTURE
high_30  = close.rolling(30).max()
low_30   = close.rolling(30).min()
range_30 = high_30 - low_30
bb_mid   = close.rolling(20).mean()
bb_std   = close.rolling(20).std()
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std

df["price_vs_high_30d"]  = (close - high_30) / high_30
df["price_vs_low_30d"]   = (close - low_30)  / low_30
df["range_position_30d"] = (close - low_30) / range_30.replace(0, np.nan)
df["bb_position"]        = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
df["bb_width"]           = (bb_upper - bb_lower) / bb_mid

tr = pd.concat([
    df["high"] - df["low"],
    (df["high"] - close.shift(1)).abs(),
    (df["low"]  - close.shift(1)).abs(),
], axis=1).max(axis=1)
df["atr_14_norm"] = tr.rolling(14).mean() / close

# ── Feature nova — convicção do HMM ──────────────────────────
df["regime_strength"] = (df["bull_prob"] - 0.5).abs()

# ── Feature sets ─────────────────────────────────────────────
BASE_COLS = [c for c in [
    "log_return", "vol_short", "vol_ratio", "drawdown",
    "volume_z", "slope_21d", "hmm_state", "bull_prob",
    "regime_strength",
] if c in df.columns]

FEAT_MOMENTUM = [
    "rsi_14_z", "rsi_30",
    "macd_hist_norm", "roc_5", "roc_10", "roc_21",
]
FEAT_STRUCTURE = [
    "price_vs_high_30d", "price_vs_low_30d",
    "range_position_30d", "bb_position", "bb_width", "atr_14_norm",
]

FEATURES = BASE_COLS + FEAT_STRUCTURE + FEAT_MOMENTUM

# ── Labels ───────────────────────────────────────────────────
df["return_5d"]     = df["log_return"].rolling(5).sum().shift(-5)
df["label_up_5d"]   = (df["return_5d"] > 0).astype(int)
df["label_down_5d"] = (df["return_5d"] < 0).astype(int)

# ── Splits ───────────────────────────────────────────────────
SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]

# ── Walk-forward condicionado ─────────────────────────────────
def run_conditioned(df, features, split, regime_val, label_col,
                    threshold=0.60, label=""):
    """
    Treina apenas em dias do regime especificado.
    Avalia apenas em dias do mesmo regime no teste.
    """
    test_end = split["test_end"] or df.index.max().strftime("%Y-%m-%d")

    # deduplicar cols: label_col e return_5d não estão em features
    extra = [c for c in ["label_col_placeholder", "return_5d", "hmm_state"]
             if c != "label_col_placeholder"]
    all_cols = list(dict.fromkeys(features + [label_col, "return_5d", "hmm_state"]))

    all_train = df[split["train_start"]:split["train_end"]][all_cols].dropna()
    all_test  = df[split["test_start"]:test_end][all_cols].dropna()

    # Filtrar por regime
    train = all_train[all_train["hmm_state"] == regime_val]
    test  = all_test[all_test["hmm_state"] == regime_val]

    if len(train) < 30 or len(test) < 10:
        print(f"  {label}: SKIP (train={len(train)} test={len(test)})")
        return None

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=1.0,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    clf.fit(
        train[features], train[label_col],
        eval_set=[(test[features], test[label_col])],
        verbose=False,
    )

    proba   = clf.predict_proba(test[features])[:, 1]
    ret_fwd = test["return_5d"]
    label_gt = test[label_col]

    auc = roc_auc_score(label_gt, proba) if label_gt.nunique() > 1 else float("nan")

    hc_mask  = proba >= threshold
    hc_ret   = ret_fwd[hc_mask].mean() if hc_mask.sum() > 0 else float("nan")
    hc_n     = hc_mask.sum()
    hc_pct   = hc_mask.mean()
    nc_ret   = ret_fwd[~hc_mask].mean() if (~hc_mask).sum() > 0 else float("nan")
    edge     = hc_ret - nc_ret

    print(f"  {label}: "
          f"n_train={len(train)} n_test={len(test)} "
          f"auc={auc:.3f} "
          f"hc_ret={hc_ret:+.4f}(n={hc_n},{hc_pct:.0%}) "
          f"edge={edge:+.4f}")

    return {
        "split":    label.split("split")[-1] if "split" in label else "?",
        "auc":      auc,
        "hc_ret":   hc_ret,
        "hc_n":     hc_n,
        "hc_pct":   hc_pct,
        "nc_ret":   nc_ret,
        "edge":     edge,
        "n_train":  len(train),
        "n_test":   len(test),
    }


# ── Rodar os 3 splits para Bull e Bear ───────────────────────
print("\n" + "=" * 65)
print("XGB_BULL — treino e teste apenas em dias Bull")
print("label = return_5d > 0  |  threshold = 0.60")
print("=" * 65)

bull_results = []
for i, sp in enumerate(SPLITS):
    print(f"\nSplit {i+1}:")
    r = run_conditioned(
        df, FEATURES, sp,
        regime_val=1,
        label_col="label_up_5d",
        threshold=0.60,
        label=f"BULL_split{i+1}",
    )
    if r:
        r["split_idx"] = i
        bull_results.append(r)

print("\n" + "=" * 65)
print("XGB_BEAR — treino e teste apenas em dias Bear")
print("label = return_5d < 0  |  threshold = 0.60")
print("=" * 65)

bear_results = []
for i, sp in enumerate(SPLITS):
    print(f"\nSplit {i+1}:")
    r = run_conditioned(
        df, FEATURES, sp,
        regime_val=0,
        label_col="label_down_5d",
        threshold=0.60,
        label=f"BEAR_split{i+1}",
    )
    if r:
        r["split_idx"] = i
        bear_results.append(r)

# ── Recriar R5 para comparação direta ────────────────────────
print("\n" + "=" * 65)
print("Rodando R5 baseline (modelo único) para comparação...")
print("=" * 65)

r5_results = []
for i, sp in enumerate(SPLITS):
    test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
    all_cols = list(dict.fromkeys(FEATURES + ["label_up_5d", "return_5d", "hmm_state"]))
    train = df[sp["train_start"]:sp["train_end"]][all_cols].dropna()
    test  = df[sp["test_start"]:test_end][all_cols].dropna()

    clf_r5 = xgb.XGBClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=3,
        gamma=0.5, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    clf_r5.fit(train[FEATURES], train["label_up_5d"], verbose=False)

    proba_r5 = clf_r5.predict_proba(test[FEATURES])[:, 1]
    pred_r5  = (proba_r5 >= 0.5).astype(int)
    ret_fwd  = test["return_5d"]
    regime   = test["hmm_state"]

    bull_mask = regime == 1
    bear_mask = regime == 0

    delta_bull = float("nan")
    delta_bear = float("nan")
    if bull_mask.sum() >= 5:
        delta_bull = (
            ret_fwd[bull_mask][pred_r5[bull_mask] == 1].mean() -
            ret_fwd[bull_mask][pred_r5[bull_mask] == 0].mean()
        )
    if bear_mask.sum() >= 5:
        delta_bear = (
            ret_fwd[bear_mask][pred_r5[bear_mask] == 1].mean() -
            ret_fwd[bear_mask][pred_r5[bear_mask] == 0].mean()
        )

    r5_results.append({
        "split_idx":  i,
        "delta_bull": delta_bull,
        "delta_bear": delta_bear,
    })
    print(f"  R5 split_{i+1}: delta_bull={delta_bull:+.4f}  delta_bear={delta_bear:+.4f}")

# ── Tabela comparativa ────────────────────────────────────────
print("\n" + "=" * 65)
print("COMPARAÇÃO — Modelo único (R5) vs Condicionado (R6)")
print("=" * 65)
print(f"\n{'Split':<10} {'R5 Bull':>10} {'R6 Bull':>10} "
      f"{'R5 Bear':>10} {'R6 Bear':>10} {'Edge comb':>10}")
print("-" * 58)

bull_map = {r["split_idx"]: r for r in bull_results}
bear_map = {r["split_idx"]: r for r in bear_results}

for i in range(len(SPLITS)):
    r5  = r5_results[i]
    rb  = bull_map.get(i)
    rbe = bear_map.get(i)

    r5_bull  = f"{r5['delta_bull']:>+10.4f}"
    r5_bear  = f"{r5['delta_bear']:>+10.4f}"
    r6_bull  = f"{rb['hc_ret']:>+10.4f}"  if rb  else f"{'N/A':>10}"
    r6_bear  = f"{rbe['hc_ret']:>+10.4f}" if rbe else f"{'N/A':>10}"
    edge_bull = rb["edge"]  if rb  else 0.0
    edge_bear = rbe["edge"] if rbe else 0.0
    edge_comb = (edge_bull + edge_bear) / 2
    print(f"split_{i+1:<5} {r5_bull} {r6_bull} "
          f"{r5_bear} {r6_bear} {edge_comb:>+10.4f}")

# Veredicto split_3
print("\nVeredicto split_3 (período mais recente):")
r5_s3  = r5_results[2]
rb_s3  = bull_map.get(2)
rbe_s3 = bear_map.get(2)

ok_bull = rb_s3 and rb_s3["hc_ret"] > r5_s3["delta_bull"]
ok_bear = rbe_s3 and rbe_s3["hc_ret"] > r5_s3["delta_bear"]

r6_bull_str = f"{rb_s3['hc_ret']:+.4f}"  if rb_s3  else "N/A"
r6_bear_str = f"{rbe_s3['hc_ret']:+.4f}" if rbe_s3 else "N/A"
print(f"  Bull: R5={r5_s3['delta_bull']:+.4f} → R6={r6_bull_str}  "
      f"{'✓ MELHORA' if ok_bull else '✗ NAO MELHORA'}")
print(f"  Bear: R5={r5_s3['delta_bear']:+.4f} → R6={r6_bear_str}  "
      f"{'✓ MELHORA' if ok_bear else '✗ NAO MELHORA'}")

decisao = "✓ APROVADO" if (ok_bull or ok_bear) else "✗ REJEITADO"
print(f"\n  Decisão: {decisao} "
      f"(critério: melhora split_3 em ≥ 1 regime)")

# ── Feature importance — split_3 ─────────────────────────────
print("\n" + "=" * 65)
print("FEATURE IMPORTANCE — split_3 (período mais recente)")
print("=" * 65)

groups_map = {}
for f in FEAT_MOMENTUM:  groups_map[f] = "MOMENTUM"
for f in FEAT_STRUCTURE: groups_map[f] = "STRUCTURE"
for f in BASE_COLS:      groups_map[f] = "BASE"

for regime_name, regime_val, label_col in [
    ("XGB_BULL", 1, "label_up_5d"),
    ("XGB_BEAR", 0, "label_down_5d"),
]:
    sp       = SPLITS[2]
    all_cols = list(dict.fromkeys(FEATURES + [label_col, "hmm_state"]))
    train    = df[sp["train_start"]:sp["train_end"]][all_cols].dropna()
    train    = train[train["hmm_state"] == regime_val]

    if len(train) < 30:
        print(f"\n{regime_name}: dados insuficientes ({len(train)} amostras)")
        continue

    clf_f = xgb.XGBClassifier(
        n_estimators=300, max_depth=3,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=5,
        gamma=1.0, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    clf_f.fit(train[FEATURES], train[label_col], verbose=False)

    imp = pd.Series(
        clf_f.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    print(f"\n{regime_name} (n_train={len(train)}) — top 8:")
    for feat, val in imp.head(8).items():
        g = groups_map.get(feat, "?")
        print(f"  {feat:<35} {val:.4f}  [{g}]")

# ── Output operacional hoje ───────────────────────────────────
print("\n" + "=" * 65)
print("OUTPUT OPERACIONAL — hoje")
print("=" * 65)

regime_hoje = int(df["hmm_state"].dropna().iloc[-1])
bull_p_hoje = float(df["bull_prob"].dropna().iloc[-1])
reg_str     = abs(bull_p_hoje - 0.5)
reg_name    = "BULL" if regime_hoje == 1 else "BEAR"
label_hoje  = "label_up_5d" if regime_hoje == 1 else "label_down_5d"
threshold   = 0.60

all_cols_hoje = list(dict.fromkeys(FEATURES + [label_hoje, "hmm_state"]))
train_hoje    = df["2023-01-01":"2024-12-31"][all_cols_hoje].dropna()
train_hoje    = train_hoje[train_hoje["hmm_state"] == regime_hoje]
last          = df[df[FEATURES].notna().all(axis=1)].tail(1)

print(f"\n  Data:              {last.index[0].date()}")
print(f"  Regime HMM:        {reg_name}")
print(f"  Bull probability:  {bull_p_hoje:.1%}")
print(f"  Regime strength:   {reg_str:.3f}")

if len(train_hoje) >= 30 and len(last) > 0:
    clf_hoje = xgb.XGBClassifier(
        n_estimators=300, max_depth=3,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=5,
        gamma=1.0, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    clf_hoje.fit(train_hoje[FEATURES], train_hoje[label_hoje], verbose=False)

    prob_hoje  = clf_hoje.predict_proba(last[FEATURES])[0, 1]
    prob_label = "prob_up_5d" if regime_hoje == 1 else "prob_down_5d"

    if regime_hoje == 1:
        signal = "LONG" if prob_hoje > threshold else "NEUTRAL"
    else:
        signal = "SHORT" if prob_hoje > threshold else "NEUTRAL"

    if reg_str > 0.4 and prob_hoje > threshold:
        signal += " (forte)"

    print(f"  {prob_label}:  {prob_hoje:.1%}")
    print(f"  Threshold:         {threshold:.0%}")
    print(f"  SINAL:             {signal}")
    print(f"\n  Contexto técnico:")
    ctx = ["rsi_14_z", "macd_hist_norm", "bb_position",
           "range_position_30d", "price_vs_low_30d", "regime_strength"]
    for f in ctx:
        if f in last.columns:
            print(f"    {f:<30} {last[f].iloc[0]:>+.4f}")
else:
    print(f"  AVISO: dados insuficientes para treino "
          f"(n_train={len(train_hoje)} em regime {reg_name})")
