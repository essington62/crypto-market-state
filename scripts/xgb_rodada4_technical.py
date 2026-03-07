"""
XGBoost Rodada 4 — features tecnicas, ablacao por grupo.
3 grupos (Trend, Momentum, Structure) x 2 horizontes (5d, 10d).
Script exploratório — NAO modifica pipeline nem sobrescreve modelos.

Ajustes em relacao ao spec original:
  - close/high/low: carregados do L3 features (model input nao tem OHLCV)
  - hmm: coluna 'state' renomeada para 'hmm_state'
  - use_label_encoder: removido (XGBoost >= 1.6)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# ── Carregar dados ────────────────────────────────────────────
# Model input (features): sem OHLCV
btc = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
# L3 features: tem OHLCV + features — usar apenas close/high/low
ohlcv = pd.read_parquet("data/03_primary/spot/daily/BTCUSDT.parquet")[["close", "high", "low"]]
hmm   = pd.read_parquet("data/05_models/regime_hmm/btc_states.parquet")

# coluna real é 'state'
hmm = hmm.rename(columns={"state": "hmm_state"})

df = btc.copy()
df = df.join(ohlcv, how="left")
df = df.join(hmm[["hmm_state", "bull_prob"]], how="left")

close = df["close"]
print(f"Shape apos join: {df.shape}")
print(f"Range: {df.index.min().date()} -> {df.index.max().date()}")
print(f"close/high/low presentes: {all(c in df.columns for c in ['close','high','low'])}")

# ── GRUPO T — Trend ───────────────────────────────────────────
df["ma_50"]              = close.rolling(50).mean()
df["ma_200"]             = close.rolling(200).mean()
df["price_vs_ma50"]      = (close - df["ma_50"]) / df["ma_50"]
df["price_vs_ma200"]     = (close - df["ma_200"]) / df["ma_200"]
df["ma_trend_strength"]  = (df["ma_50"] - df["ma_200"]) / df["ma_200"]
df["ma_cross_50_200"]    = (df["ma_50"] > df["ma_200"]).astype(int)
df["slope_50d"] = close.rolling(50).apply(
    lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean(),
    raw=True,
)

FEAT_TREND = [
    "price_vs_ma50", "price_vs_ma200",
    "ma_trend_strength", "ma_cross_50_200", "slope_50d",
]

# ── GRUPO M — Momentum ────────────────────────────────────────
def calc_rsi(series, period):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df["rsi_14"]        = calc_rsi(close, 14)
df["rsi_30"]        = calc_rsi(close, 30)
df["rsi_14_z"]      = (df["rsi_14"] - 50) / 20

ema_12              = close.ewm(span=12, adjust=False).mean()
ema_26              = close.ewm(span=26, adjust=False).mean()
df["macd"]          = ema_12 - ema_26
df["macd_sig"]      = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"]     = df["macd"] - df["macd_sig"]
df["macd_hist_norm"]= df["macd_hist"] / close

df["roc_5"]   = close.pct_change(5)
df["roc_10"]  = close.pct_change(10)
df["roc_21"]  = close.pct_change(21)

FEAT_MOMENTUM = [
    "rsi_14_z", "rsi_30",
    "macd_hist_norm",
    "roc_5", "roc_10", "roc_21",
]

# ── GRUPO S — Structure ───────────────────────────────────────
high_30  = close.rolling(30).max()
low_30   = close.rolling(30).min()
range_30 = high_30 - low_30

df["price_vs_high_30d"]  = (close - high_30) / high_30
df["price_vs_low_30d"]   = (close - low_30)  / low_30
df["range_position_30d"] = (close - low_30)  / range_30.replace(0, np.nan)

bb_mid   = close.rolling(20).mean()
bb_std   = close.rolling(20).std()
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
df["bb_width"]    = (bb_upper - bb_lower) / bb_mid

# ATR com high/low reais
tr = pd.concat([
    df["high"] - df["low"],
    (df["high"] - close.shift(1)).abs(),
    (df["low"]  - close.shift(1)).abs(),
], axis=1).max(axis=1)
df["atr_14_norm"] = tr.rolling(14).mean() / close

FEAT_STRUCTURE = [
    "price_vs_high_30d", "price_vs_low_30d",
    "range_position_30d", "bb_position",
    "bb_width", "atr_14_norm",
]

# ── Labels 5d e 10d ──────────────────────────────────────────
df["return_5d"]  = df["log_return"].rolling(5).sum().shift(-5)
df["return_10d"] = df["log_return"].rolling(10).sum().shift(-10)
df["label_5d"]   = (df["return_5d"]  > 0).astype(int)
df["label_10d"]  = (df["return_10d"] > 0).astype(int)

# ── Feature sets ─────────────────────────────────────────────
BASE_COLS = [c for c in [
    "log_return", "vol_short", "vol_ratio", "drawdown",
    "volume_z", "slope_21d", "hmm_state", "bull_prob",
] if c in df.columns]

print(f"\nFeatures base:      {len(BASE_COLS)}")
print(f"Features Trend:     {len(FEAT_TREND)}")
print(f"Features Momentum:  {len(FEAT_MOMENTUM)}")
print(f"Features Structure: {len(FEAT_STRUCTURE)}")

# Verificar cobertura pos-2023
df23      = df[df.index >= "2023-01-01"]
all_feats = FEAT_TREND + FEAT_MOMENTUM + FEAT_STRUCTURE
print(f"\nCobertura pos-2023 ({len(df23)} rows):")
for f in all_feats:
    valid = df23[f].notna().sum()
    pct   = valid / len(df23)
    flag  = "  BAIXA COBERTURA" if pct < 0.95 else ""
    print(f"  {f:<30} {valid}/{len(df23)} ({pct:.1%}){flag}")

# ── Walk-forward ─────────────────────────────────────────────
SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]


def run_xgb(df, features, label_col, label=""):
    results     = []
    importances = []
    for i, sp in enumerate(SPLITS):
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        cols  = features + [label_col]
        train = df[sp["train_start"]:sp["train_end"]][cols].dropna()
        test  = df[sp["test_start"]:test_end][cols].dropna()

        if len(train) < 50 or len(test) < 20:
            print(f"  {label} split_{i+1}: SKIP (train={len(train)} test={len(test)})")
            continue

        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            gamma=0.5,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        clf.fit(train[features], train[label_col],
                eval_set=[(test[features], test[label_col])],
                verbose=False)

        proba    = clf.predict_proba(test[features])[:, 1]
        pred     = (proba >= 0.5).astype(int)
        auc      = roc_auc_score(test[label_col], proba)
        acc      = (pred == test[label_col]).mean()
        ret_col  = label_col.replace("label_", "return_")
        ret_fwd  = df[ret_col].reindex(test.index)
        bull_ret = ret_fwd[pred == 1].mean()
        bear_ret = ret_fwd[pred == 0].mean()
        delta    = bull_ret - bear_ret
        pct_conv = ((proba > 0.60) | (proba < 0.40)).mean()

        results.append({
            "split":    f"split_{i+1}",
            "n_train":  len(train),
            "n_test":   len(test),
            "auc":      auc,
            "acc":      acc,
            "delta":    delta,
            "pct_conv": pct_conv,
        })
        importances.append(pd.Series(
            clf.feature_importances_, index=features, name=f"split_{i+1}"
        ))
        print(f"  {label} split_{i+1}: "
              f"n={len(train)}/{len(test)}  "
              f"auc={auc:.3f}  acc={acc:.1%}  "
              f"delta={delta:+.4f}  conv={pct_conv:.0%}")

    return pd.DataFrame(results), pd.DataFrame(importances) if importances else pd.DataFrame()


# ── Rodar todos os grupos x horizontes ───────────────────────
configs = [
    ("BASE",      BASE_COLS),
    ("TREND",     BASE_COLS + FEAT_TREND),
    ("MOMENTUM",  BASE_COLS + FEAT_MOMENTUM),
    ("STRUCTURE", BASE_COLS + FEAT_STRUCTURE),
    ("ALL_TECH",  BASE_COLS + FEAT_TREND + FEAT_MOMENTUM + FEAT_STRUCTURE),
]

results_5d  = {}
results_10d = {}
imps_5d     = {}
imps_10d    = {}

for name, feats in configs:
    print(f"\n{'=' * 60}")
    print(f"{name} — 5d")
    print(f"{'=' * 60}")
    results_5d[name], imps_5d[name] = run_xgb(df, feats, "label_5d",  f"{name[:6]}_5d")

    print(f"\n{'=' * 60}")
    print(f"{name} — 10d")
    print(f"{'=' * 60}")
    results_10d[name], imps_10d[name] = run_xgb(df, feats, "label_10d", f"{name[:6]}_10d")

# ── Tabelas comparativas ──────────────────────────────────────
names = [c[0] for c in configs]
groups_map = {}
for f in FEAT_TREND:     groups_map[f] = "TREND"
for f in FEAT_MOMENTUM:  groups_map[f] = "MOMENTUM"
for f in FEAT_STRUCTURE: groups_map[f] = "STRUCTURE"
for f in BASE_COLS:      groups_map[f] = "BASE"

for horizon, all_res in [("5d", results_5d), ("10d", results_10d)]:
    print(f"\n{'=' * 70}")
    print(f"COMPARACAO — AUC | horizonte {horizon}")
    print(f"{'=' * 70}")
    print(f"{'Split':<10}" + "".join(f"{n:>13}" for n in names))
    print("-" * 70)
    for i in range(3):
        row = f"split_{i+1:<5}"
        for n in names:
            if i < len(all_res[n]):
                row += f"{all_res[n]['auc'].iloc[i]:>13.3f}"
            else:
                row += f"{'N/A':>13}"
        print(row)
    print("-" * 70)
    row = f"{'Media':<10}"
    for n in names:
        row += f"{all_res[n]['auc'].mean():>13.3f}"
    print(row)

    print(f"\n{'=' * 70}")
    print(f"COMPARACAO — Delta {horizon} | horizonte {horizon}")
    print(f"{'=' * 70}")
    print(f"{'Split':<10}" + "".join(f"{n:>13}" for n in names))
    print("-" * 70)
    base_deltas = all_res["BASE"]["delta"].values
    for i in range(3):
        row = f"split_{i+1:<5}"
        for n in names:
            if i < len(all_res[n]):
                row += f"{all_res[n]['delta'].iloc[i]:>+13.4f}"
            else:
                row += f"{'N/A':>13}"
        print(row)
    print("-" * 70)
    means = {}
    row = f"{'Media':<10}"
    for n in names:
        m = all_res[n]["delta"].mean()
        means[n] = m
        row += f"{m:>+13.4f}"
    print(row)

    print(f"\n  VEREDICTO {horizon}:")
    for n in names[1:]:
        diff      = means[n] - means["BASE"]
        splits_ok = (all_res[n]["delta"].values > base_deltas).sum()
        status    = "APROVADO" if splits_ok >= 2 and diff > 0 else "REJEITADO"
        print(f"  {n:<12}: diff={diff:>+.4f}  splits={splits_ok}/3  {status}")

# ── Feature importance — ALL_TECH ────────────────────────────
print(f"\n{'=' * 70}")
print("FEATURE IMPORTANCE — ALL_TECH (media 3 splits)")
print(f"{'=' * 70}")
for horizon, all_imp in [("5d", imps_5d), ("10d", imps_10d)]:
    print(f"\nHorizonte {horizon}:")
    imp_df = all_imp.get("ALL_TECH")
    if imp_df is not None and len(imp_df) > 0:
        imp_mean = imp_df.mean().sort_values(ascending=False)
        for feat, val in imp_mean.head(12).items():
            g = groups_map.get(feat, "?")
            print(f"  {feat:<35} {val:.4f}  [{g}]")

# ── Resumo final ─────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("RESUMO FINAL")
print(f"{'=' * 70}")
for horizon, all_res in [("5d", results_5d), ("10d", results_10d)]:
    means_h = {n: all_res[n]["delta"].mean() for n in names}
    best    = max(means_h, key=means_h.get)
    print(f"\n  Horizonte {horizon}:")
    print(f"    Melhor grupo: {best}")
    print(f"    Delta medio:  {means_h[best]:>+.4f}")
    print(f"    vs baseline:  {means_h[best] - means_h['BASE']:>+.4f}")
