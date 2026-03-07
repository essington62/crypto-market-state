"""
XGBoost Rodada 5 — combinação final.
BASE + STRUCTURE + MOMENTUM (5d)
BASE + ALL_TECH (10d)
Walk-forward 3 splits idêntico às rodadas anteriores.
Script exploratório — NAO modifica pipeline nem sobrescreve modelos.

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
btc  = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
ohlcv = pd.read_parquet(
    "data/03_primary/spot/daily/BTCUSDT.parquet"
)[["close", "high", "low"]]
hmm  = pd.read_parquet("data/05_models/regime_hmm/btc_states.parquet")

# coluna real é 'state'
hmm = hmm.rename(columns={"state": "hmm_state"})

df = btc.copy()
df = df.join(ohlcv, how="left")
df = df.join(hmm[["hmm_state", "bull_prob"]], how="left")

close = df["close"]
print(f"Shape apos join: {df.shape}")
print(f"Range: {df.index.min().date()} -> {df.index.max().date()}")
print(f"close/high/low presentes: {all(c in df.columns for c in ['close','high','low'])}")

# ── Recalcular todas as features técnicas ─────────────────────
# TREND
df["ma_50"]             = close.rolling(50).mean()
df["ma_200"]            = close.rolling(200).mean()
df["price_vs_ma50"]     = (close - df["ma_50"]) / df["ma_50"]
df["price_vs_ma200"]    = (close - df["ma_200"]) / df["ma_200"]
df["ma_trend_strength"] = (df["ma_50"] - df["ma_200"]) / df["ma_200"]
df["ma_cross_50_200"]   = (df["ma_50"] > df["ma_200"]).astype(int)
df["slope_50d"] = close.rolling(50).apply(
    lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean(),
    raw=True,
)

# MOMENTUM
def calc_rsi(series, period):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df["rsi_14"]         = calc_rsi(close, 14)
df["rsi_30"]         = calc_rsi(close, 30)
df["rsi_14_z"]       = (df["rsi_14"] - 50) / 20
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

# ── Feature sets ─────────────────────────────────────────────
BASE_COLS = [c for c in [
    "log_return", "vol_short", "vol_ratio", "drawdown",
    "volume_z", "slope_21d", "hmm_state", "bull_prob",
] if c in df.columns]

FEAT_TREND = [
    "price_vs_ma50", "price_vs_ma200",
    "ma_trend_strength", "ma_cross_50_200", "slope_50d",
]
FEAT_MOMENTUM = [
    "rsi_14_z", "rsi_30",
    "macd_hist_norm", "roc_5", "roc_10", "roc_21",
]
FEAT_STRUCTURE = [
    "price_vs_high_30d", "price_vs_low_30d",
    "range_position_30d", "bb_position", "bb_width", "atr_14_norm",
]
FEAT_ALL_TECH = FEAT_TREND + FEAT_MOMENTUM + FEAT_STRUCTURE

FEAT_5D  = BASE_COLS + FEAT_STRUCTURE + FEAT_MOMENTUM
FEAT_10D = BASE_COLS + FEAT_ALL_TECH

# Labels
df["return_5d"]  = df["log_return"].rolling(5).sum().shift(-5)
df["return_10d"] = df["log_return"].rolling(10).sum().shift(-10)
df["label_5d"]   = (df["return_5d"]  > 0).astype(int)
df["label_10d"]  = (df["return_10d"] > 0).astype(int)

print(f"\nFeatures 5d config:  {len(FEAT_5D)}")
print(f"Features 10d config: {len(FEAT_10D)}")

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]


def run_xgb(df, features, label_col, ret_col, label=""):
    results     = []
    importances = []
    for i, sp in enumerate(SPLITS):
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        cols  = features + [label_col, ret_col]
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
        clf.fit(
            train[features], train[label_col],
            eval_set=[(test[features], test[label_col])],
            verbose=False,
        )

        proba    = clf.predict_proba(test[features])[:, 1]
        pred     = (proba >= 0.5).astype(int)
        auc      = roc_auc_score(test[label_col], proba)
        acc      = (pred == test[label_col]).mean()
        ret_fwd  = df[ret_col].reindex(test.index)
        bull_ret = ret_fwd[pred == 1].mean()
        bear_ret = ret_fwd[pred == 0].mean()
        delta    = bull_ret - bear_ret

        # Camada 3 — convicção
        long_mask  = proba > 0.60
        short_mask = proba < 0.40
        neutral    = ~(long_mask | short_mask)
        conv_long  = ret_fwd[long_mask].mean()
        conv_short = ret_fwd[short_mask].mean()
        pct_neutral = neutral.mean()
        conv_edge  = conv_long - conv_short

        results.append({
            "split":       f"split_{i+1}",
            "n_train":     len(train),
            "n_test":      len(test),
            "auc":         auc,
            "acc":         acc,
            "delta":       delta,
            "conv_long":   conv_long,
            "conv_short":  conv_short,
            "conv_edge":   conv_edge,
            "pct_neutral": pct_neutral,
        })
        importances.append(pd.Series(
            clf.feature_importances_, index=features, name=f"split_{i+1}"
        ))
        print(f"  {label} split_{i+1}: "
              f"auc={auc:.3f} acc={acc:.1%} "
              f"delta={delta:+.4f} "
              f"edge={conv_edge:+.4f} "
              f"neutral={pct_neutral:.0%}")

    return pd.DataFrame(results), pd.DataFrame(importances) if importances else pd.DataFrame()


# ── Rodar configurações finais ────────────────────────────────
print("\n" + "=" * 65)
print("BASELINE 5d — BASE only")
print("=" * 65)
base5, _ = run_xgb(df, BASE_COLS, "label_5d", "return_5d", "BASE_5d ")

print("\n" + "=" * 65)
print("RODADA 5 — BASE + STRUCTURE + MOMENTUM (5d)")
print("=" * 65)
r5_5d, imp_5d = run_xgb(df, FEAT_5D, "label_5d", "return_5d", "R5_5d  ")

print("\n" + "=" * 65)
print("BASELINE 10d — BASE only")
print("=" * 65)
base10, _ = run_xgb(df, BASE_COLS, "label_10d", "return_10d", "BASE_10d")

print("\n" + "=" * 65)
print("RODADA 5 — BASE + ALL_TECH (10d)")
print("=" * 65)
r5_10d, imp_10d = run_xgb(df, FEAT_10D, "label_10d", "return_10d", "R5_10d ")

# ── Tabela comparativa principal ─────────────────────────────
for horizon, base_r, new_r, new_label in [
    ("5d",  base5,  r5_5d,  "BASE+STRUCT+MOM"),
    ("10d", base10, r5_10d, "BASE+ALL_TECH  "),
]:
    print(f"\n{'=' * 65}")
    print(f"COMPARAÇÃO — horizonte {horizon}")
    print(f"{'=' * 65}")
    print(f"\n{'Split':<10} {'Base AUC':>10} {'New AUC':>10} "
          f"{'Base Δ':>10} {'New Δ':>10} "
          f"{'Diff':>8}  Res")
    print("-" * 65)

    diffs = []
    for rb, rn in zip(base_r.itertuples(), new_r.itertuples()):
        diff = rn.delta - rb.delta
        diffs.append(diff)
        ok = "✓" if diff > 0 else "✗"
        print(f"{rb.split:<10} {rb.auc:>10.3f} {rn.auc:>10.3f} "
              f"{rb.delta:>+10.4f} {rn.delta:>+10.4f} "
              f"{diff:>+8.4f}  {ok}")

    bm = base_r["delta"].mean()
    nm = new_r["delta"].mean()
    ok = sum(d > 0 for d in diffs)
    print(f"\n{'Média':<10} {base_r['auc'].mean():>10.3f} "
          f"{new_r['auc'].mean():>10.3f} "
          f"{bm:>+10.4f} {nm:>+10.4f} "
          f"{nm - bm:>+8.4f}")
    print(f"\nSplits melhorados: {ok}/3")
    print(f"Decisão: {'✓ APROVADO' if ok >= 2 and nm > bm else '✗ REJEITADO'}")

# ── Camada 3 — edge de convicção ─────────────────────────────
print(f"\n{'=' * 65}")
print("CAMADA 3 — Edge de convicção (prob>0.60 / <0.40)")
print(f"{'=' * 65}")
for horizon, new_r in [("5d", r5_5d), ("10d", r5_10d)]:
    print(f"\nHorizonte {horizon}:")
    print(f"  {'Split':<10} {'Long ret':>12} {'Short ret':>12} {'Edge':>10} {'Neutro%':>10}")
    print("  " + "-" * 50)
    for rn in new_r.itertuples():
        print(f"  {rn.split:<10} {rn.conv_long:>+12.4f} "
              f"{rn.conv_short:>+12.4f} "
              f"{rn.conv_edge:>+10.4f} "
              f"{rn.pct_neutral:>10.1%}")
    print(f"  {'Média':<10} "
          f"{new_r['conv_long'].mean():>+12.4f} "
          f"{new_r['conv_short'].mean():>+12.4f} "
          f"{new_r['conv_edge'].mean():>+10.4f} "
          f"{new_r['pct_neutral'].mean():>10.1%}")

# ── Feature importance final ──────────────────────────────────
print(f"\n{'=' * 65}")
print("FEATURE IMPORTANCE FINAL")
print(f"{'=' * 65}")
groups_map = {}
for f in FEAT_TREND:     groups_map[f] = "TREND"
for f in FEAT_MOMENTUM:  groups_map[f] = "MOMENTUM"
for f in FEAT_STRUCTURE: groups_map[f] = "STRUCTURE"
for f in BASE_COLS:      groups_map[f] = "BASE"

for horizon, imp_df in [("5d", imp_5d), ("10d", imp_10d)]:
    print(f"\nTop 10 — horizonte {horizon}:")
    imp_mean = imp_df.mean().sort_values(ascending=False)
    for feat, val in imp_mean.head(10).items():
        g = groups_map.get(feat, "?")
        print(f"  {feat:<35} {val:.4f}  [{g}]")

# ── Output operacional hoje ───────────────────────────────────
print(f"\n{'=' * 65}")
print("OUTPUT OPERACIONAL — hoje (modelo 10d, treino split_3)")
print(f"{'=' * 65}")

sp      = SPLITS[2]
cols10  = FEAT_10D + ["label_10d", "return_10d"]
train10 = df[sp["train_start"]:sp["train_end"]][cols10].dropna()
last    = df[df[FEAT_10D].notna().all(axis=1)].tail(1)

if len(last) > 0 and len(train10) > 50:
    clf_f = xgb.XGBClassifier(
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
    clf_f.fit(train10[FEAT_10D], train10["label_10d"], verbose=False)

    prob_up = clf_f.predict_proba(last[FEAT_10D])[0, 1]
    regime  = "BULL" if last["hmm_state"].iloc[0] == 1 else "BEAR"
    bull_p  = last["bull_prob"].iloc[0]

    if regime == "BULL" and prob_up > 0.60:
        signal = "LONG"
    elif regime == "BEAR" and prob_up < 0.40:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    if bull_p > 0.65 and prob_up > 0.60:
        signal += " forte"
    elif bull_p < 0.35 and prob_up < 0.40:
        signal += " forte"

    print(f"\n  Data:               {last.index[0].date()}")
    print(f"  Regime HMM:         {regime}")
    print(f"  Bull probability:   {bull_p:.1%}")
    print(f"  Prob subida 10d:    {prob_up:.1%}")
    print(f"  SINAL:              {signal}")
    print(f"\n  Contexto técnico:")
    ctx = ["rsi_14_z", "macd_hist_norm", "bb_position",
           "price_vs_ma200", "range_position_30d", "price_vs_low_30d"]
    for f in ctx:
        if f in last.columns:
            print(f"    {f:<30} {last[f].iloc[0]:>+.4f}")
else:
    print("\n  AVISO: dados insuficientes para output operacional.")
