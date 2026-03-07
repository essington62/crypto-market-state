"""
Diagnóstico final — performance XGBoost 5d por regime HMM.
Usa split_3 (treino 2023-2024, teste 2025+).
Script exploratório — NAO modifica pipeline nem salva modelos.

Ajustes em relação ao spec original:
  - close/high/low: carregados do L3 (model input nao tem OHLCV)
  - hmm: coluna 'state' renomeada para 'hmm_state'
  - use_label_encoder: removido (XGBoost >= 1.6)
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

# ── Recalcular features (igual Rodada 5) ─────────────────────
# MOMENTUM
def calc_rsi(series, period):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

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

# ── Feature set final (Rodada 5) ─────────────────────────────
BASE_COLS = [c for c in [
    "log_return", "vol_short", "vol_ratio", "drawdown",
    "volume_z", "slope_21d", "hmm_state", "bull_prob",
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

# Labels
df["return_5d"] = df["log_return"].rolling(5).sum().shift(-5)
df["label_5d"]  = (df["return_5d"] > 0).astype(int)

# ── Treinar com split_3 train (2023-2024) ────────────────────
# deduplicar: hmm_state já está em BASE_COLS / FEATURES
cols  = list(dict.fromkeys(FEATURES + ["label_5d", "return_5d", "hmm_state"]))
train = df["2023-01-01":"2024-12-31"][cols].dropna()
test  = df["2025-01-01":][cols].dropna()

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
clf.fit(train[FEATURES], train["label_5d"], verbose=False)

proba   = clf.predict_proba(test[FEATURES])[:, 1]
pred    = (proba >= 0.5).astype(int)
ret_fwd = test["return_5d"]
regime  = test["hmm_state"]

print(f"Período teste: {test.index.min().date()} -> {test.index.max().date()}")
print(f"N total: {len(test)}")
print(f"Bull days: {(regime==1).sum()} ({(regime==1).mean():.1%})")
print(f"Bear days: {(regime==0).sum()} ({(regime==0).mean():.1%})")

# ── Diagnóstico por regime ────────────────────────────────────
print("\n" + "=" * 65)
print("PERFORMANCE XGB5d POR REGIME HMM")
print("=" * 65)

for reg_name, reg_val in [("BULL", 1), ("BEAR", 0)]:
    mask = regime == reg_val
    if mask.sum() < 10:
        print(f"\n{reg_name}: poucos dados ({mask.sum()}), skip")
        continue

    r_proba  = proba[mask]
    r_pred   = pred[mask]
    r_ret    = ret_fwd[mask]
    r_label  = test["label_5d"][mask]

    auc = roc_auc_score(r_label, r_proba) if r_label.nunique() > 1 else float("nan")
    acc = (r_pred == r_label).mean()
    bull_ret = r_ret[r_pred == 1].mean()
    bear_ret = r_ret[r_pred == 0].mean()
    delta    = bull_ret - bear_ret

    # Convicção dentro do regime
    long_mask  = r_proba > 0.60
    short_mask = r_proba < 0.40
    conv_long  = r_ret[long_mask].mean()
    conv_short = r_ret[short_mask].mean()
    conv_edge  = conv_long - conv_short
    pct_neutral = (~(long_mask | short_mask)).mean()

    # Distribuição de probabilidades
    p25, p50, p75 = (
        np.percentile(r_proba, 25),
        np.percentile(r_proba, 50),
        np.percentile(r_proba, 75),
    )

    print(f"\n── Regime {reg_name} ({mask.sum()} dias) ──────────────────")
    print(f"  AUC:              {auc:.3f}")
    print(f"  Accuracy:         {acc:.1%}")
    print(f"  Delta 5d:         {delta:+.4f}")
    print(f"  Bull pred ret:    {bull_ret:+.4f}")
    print(f"  Bear pred ret:    {bear_ret:+.4f}")
    print(f"  Conv edge:        {conv_edge:+.4f}")
    print(f"  Conv long ret:    {conv_long:+.4f}  (n={long_mask.sum()})")
    print(f"  Conv short ret:   {conv_short:+.4f}  (n={short_mask.sum()})")
    print(f"  Neutro:           {pct_neutral:.1%}")
    print(f"  Prob p25/p50/p75: {p25:.2f}/{p50:.2f}/{p75:.2f}")

# ── Matriz de decisão regime x convicção ─────────────────────
print("\n" + "=" * 65)
print("MATRIZ — Regime x Convicção → retorno médio 5d")
print("=" * 65)
print(f"\n{'':20} {'prob<0.40':>14} {'0.40-0.60':>14} {'prob>0.60':>14} {'N':>6}")
print("-" * 72)

for reg_name, reg_val in [("BULL", 1), ("BEAR", 0)]:
    mask  = regime == reg_val
    row   = f"{reg_name:<20}"
    total = mask.sum()
    for lo, hi in [(0.0, 0.40), (0.40, 0.60), (0.60, 1.01)]:
        cell_mask = mask & (proba >= lo) & (proba < hi)
        if cell_mask.sum() >= 3:
            ret_mean = ret_fwd[cell_mask].mean()
            n        = cell_mask.sum()
            row     += f"  {ret_mean:>+8.4f}({n:2d})"
        else:
            row     += f"  {'---':>10}    "
    print(row + f"  {total:>4}")

# ── Conclusão diagnóstico ─────────────────────────────────────
print("\n" + "=" * 65)
print("DIAGNÓSTICO — vale modelo condicionado ao regime?")
print("=" * 65)

bull_mask  = regime == 1
bear_mask  = regime == 0

delta_bull = float("nan")
delta_bear = float("nan")

if bull_mask.sum() >= 10:
    delta_bull = (
        ret_fwd[bull_mask][pred[bull_mask] == 1].mean() -
        ret_fwd[bull_mask][pred[bull_mask] == 0].mean()
    )
if bear_mask.sum() >= 10:
    delta_bear = (
        ret_fwd[bear_mask][pred[bear_mask] == 1].mean() -
        ret_fwd[bear_mask][pred[bear_mask] == 0].mean()
    )

ratio = abs(delta_bull) / (abs(delta_bear) + 1e-8)

print(f"\n  Delta em Bull: {delta_bull:+.4f}")
print(f"  Delta em Bear: {delta_bear:+.4f}")
print(f"  Razão Bull/Bear: {ratio:.2f}x")

if ratio > 2:
    assimetria = "FORTE (>2x) — modelo condicionado recomendado"
elif ratio > 1:
    assimetria = "MODERADA (1-2x) — pode beneficiar"
else:
    assimetria = "FRACA (<1x) — modelo único suficiente"

print(f"\n  Assimetria: {assimetria}")
print(f"\n  Recomendação:")
if ratio > 2:
    print("  → Treinar dois modelos separados:")
    print("    XGB_bull: treino apenas em dias Bull")
    print("    XGB_bear: treino apenas em dias Bear")
    print("    Edge esperado: significativamente maior")
elif ratio > 1:
    print("  → Adicionar hmm_state como feature de interação")
    print("    hmm_state * rsi_14_z")
    print("    hmm_state * range_position_30d")
    print("  → Modelo único com contexto de regime explícito")
else:
    print("  → Modelo único é suficiente")
    print("  → Congelar BASE+STRUCT+MOM como definitivo")
