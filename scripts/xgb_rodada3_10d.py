"""
XGBoost Rodada 3 — horizonte 10d, classificacao + regressao paralela.
Script exploratório — NAO modifica pipeline nem sobrescreve modelos.

Ajustes em relacao ao spec original:
  - hmm: coluna 'state' renomeada para 'hmm_state'
  - top_position_ratio: ja presente em btc (Config E) — ls join skippado
  - use_label_encoder: removido (XGBoost >= 1.6)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

BASE = Path("data/01_raw/derivatives/coinglass")

# ── Carregar dados ────────────────────────────────────────────
# btc ja contem top_position_ratio (Config E ativo no pipeline)
btc   = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
hmm   = pd.read_parquet("data/05_models/regime_hmm/btc_states.parquet")
oi    = pd.read_parquet(BASE / "open_interest/BTCUSDT_aggregated.parquet")
fr    = pd.read_parquet(BASE / "funding/BTCUSDT_oi_weighted.parquet")
fg    = pd.read_parquet(BASE / "indices/fear_greed.parquet")
ahr   = pd.read_parquet(BASE / "indices/ahr999.parquet")
stbl  = pd.read_parquet(BASE / "indices/stablecoin_mcap.parquet")
cdri  = pd.read_parquet(BASE / "indices/cdri_index.parquet")
etf_f = pd.read_parquet(BASE / "etf/BTC_flows_total.parquet")
etf_h = pd.read_parquet(BASE / "etf/BTC_holdings_consolidated.parquet")

# coluna real é 'state'
hmm = hmm.rename(columns={"state": "hmm_state"})

# ── Features derivadas ────────────────────────────────────────
oi["oi_change_1d_lag1"] = oi["open_interest_usd"].pct_change(fill_method=None).shift(1)
oi["oi_zscore_30d"] = (
    (oi["open_interest_usd"] - oi["open_interest_usd"].rolling(30).mean()) /
     oi["open_interest_usd"].rolling(30).std()
)
fr["funding_zscore_30d"] = (
    (fr["funding_rate_oi_weighted"] - fr["funding_rate_oi_weighted"].rolling(30).mean()) /
     fr["funding_rate_oi_weighted"].rolling(30).std()
)
stbl["stablecoin_mcap_zscore"] = (
    (stbl["stablecoin_mcap_usd"] - stbl["stablecoin_mcap_usd"].rolling(30).mean()) /
     stbl["stablecoin_mcap_usd"].rolling(30).std()
)
etf_f["etf_flow_7d"] = etf_f["flow_usd"].rolling(7).sum()
etf_h_col = [c for c in etf_h.columns if "total_btc" in c.lower()][0]
etf_h = etf_h.rename(columns={etf_h_col: "etf_btc_holdings_total"})

# ── Montar DataFrame único ────────────────────────────────────
df = btc.copy()
df = df.join(hmm[["hmm_state", "bull_prob"]], how="left")
df = df.join(oi[["oi_change_1d_lag1", "oi_zscore_30d"]], how="left")
df = df.join(fr[["funding_zscore_30d"]], how="left")
# top_position_ratio ja esta em df (de btc Config E) — sem re-join
df = df.join(fg[["fear_greed"]], how="left")
df = df.join(ahr[["ahr999"]], how="left")
df = df.join(stbl[["stablecoin_mcap_zscore"]], how="left")
df = df.join(cdri[["cdri_index"]], how="left")
df = df.join(etf_f[["flow_usd", "etf_flow_7d"]], how="left")
df = df.join(etf_h[["etf_btc_holdings_total"]], how="left")

df["etf_era_flag"]           = (df.index >= "2024-01-26").astype(int)
df["etf_flow_usd"]           = df["flow_usd"].fillna(0)
df["etf_flow_7d"]            = df["etf_flow_7d"].fillna(0)
df["etf_btc_holdings_total"] = df["etf_btc_holdings_total"].ffill().fillna(0)
df["cdri_index"]             = df["cdri_index"].ffill()
df = df.drop(columns=["flow_usd"], errors="ignore")

# ── Labels 10d ───────────────────────────────────────────────
df["return_10d"] = df["log_return"].rolling(10).sum().shift(-10)
df["label_10d"]  = (df["return_10d"] > 0).astype(int)

print(f"Shape total: {df.shape}")
print(f"Range: {df.index.min().date()} -> {df.index.max().date()}")
print(f"\nDistribuicao label_10d (pos-2023):")
df23 = df[df.index >= "2023-01-01"].dropna(subset=["label_10d"])
print(df23["label_10d"].value_counts(normalize=True).round(3))

# ── Feature sets ─────────────────────────────────────────────
btc_cols = [c for c in [
    "log_return", "vol_short", "vol_ratio", "drawdown",
    "volume_z", "slope_21d", "hmm_state", "bull_prob",
] if c in df.columns]

FEAT_INSTITUTIONAL = ["etf_flow_usd", "etf_flow_7d", "etf_btc_holdings_total", "etf_era_flag"]
FEAT_SENTIMENT     = ["fear_greed", "ahr999"]
FEAT_DERIVATIVES   = ["funding_zscore_30d", "oi_change_1d_lag1", "oi_zscore_30d", "top_position_ratio"]
FEAT_LIQUIDITY     = ["stablecoin_mcap_zscore", "cdri_index"]

FEATURES_BASE = btc_cols
FEATURES_ALL  = btc_cols + FEAT_INSTITUTIONAL + FEAT_SENTIMENT + FEAT_DERIVATIVES + FEAT_LIQUIDITY

print(f"\nFeatures baseline:  {len(FEATURES_BASE)}")
print(f"Features completo:  {len(FEATURES_ALL)}")
print(f"  Institutional: {FEAT_INSTITUTIONAL}")
print(f"  Sentiment:     {FEAT_SENTIMENT}")
print(f"  Derivatives:   {FEAT_DERIVATIVES}")
print(f"  Liquidity:     {FEAT_LIQUIDITY}")

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]


def run_xgb_10d(df, features, label=""):
    results        = []
    all_importance = []

    for i, sp in enumerate(SPLITS):
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        cols  = features + ["label_10d", "return_10d"]
        train = df[sp["train_start"]:sp["train_end"]][cols].dropna()
        test  = df[sp["test_start"]:test_end][cols].dropna()

        # ── Classificador ─────────────────────────────────────
        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=1,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        clf.fit(train[features], train["label_10d"],
                eval_set=[(test[features], test["label_10d"])],
                verbose=False)

        proba = clf.predict_proba(test[features])[:, 1]
        pred  = (proba >= 0.5).astype(int)
        auc   = roc_auc_score(test["label_10d"], proba)
        acc   = (pred == test["label_10d"]).mean()

        # ── Regressor ─────────────────────────────────────────
        reg = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=1,
            random_state=42,
            verbosity=0,
        )
        reg.fit(train[features], train["return_10d"],
                eval_set=[(test[features], test["return_10d"])],
                verbose=False)

        pred_ret          = reg.predict(test[features])
        spear_r, spear_p  = spearmanr(pred_ret, test["return_10d"])

        # ── Metricas de trading ───────────────────────────────
        ret_10d  = test["return_10d"]
        bull_ret = ret_10d[pred == 1].mean()
        bear_ret = ret_10d[pred == 0].mean()
        delta    = bull_ret - bear_ret

        high_conv_long  = proba > 0.60
        high_conv_short = proba < 0.40
        neutral         = ~(high_conv_long | high_conv_short)

        conv_long_ret  = ret_10d[high_conv_long].mean()
        conv_short_ret = ret_10d[high_conv_short].mean()
        pct_neutral    = neutral.mean()

        results.append({
            "split":          f"split_{i+1}",
            "n_train":        len(train),
            "n_test":         len(test),
            "auc":            auc,
            "acc":            acc,
            "delta_10d":      delta,
            "bull_ret":       bull_ret,
            "bear_ret":       bear_ret,
            "spearman_r":     spear_r,
            "spearman_p":     spear_p,
            "conv_long_ret":  conv_long_ret,
            "conv_short_ret": conv_short_ret,
            "pct_neutral":    pct_neutral,
        })
        all_importance.append(pd.Series(
            clf.feature_importances_, index=features, name=f"split_{i+1}"
        ))
        print(f"  {label} split_{i+1}: "
              f"auc={auc:.3f}  acc={acc:.1%}  delta_10d={delta:+.4f}  "
              f"spearman={spear_r:+.3f}(p={spear_p:.3f})  neutral={pct_neutral:.1%}")

    return pd.DataFrame(results), pd.DataFrame(all_importance)


print("\n" + "=" * 65)
print("BASELINE — features originais, horizonte 10d")
print("=" * 65)
res_base, _ = run_xgb_10d(df, FEATURES_BASE, "BASE")

print("\n" + "=" * 65)
print("RODADA 3 — todas features, horizonte 10d")
print("=" * 65)
res_new, imp_new = run_xgb_10d(df, FEATURES_ALL, "R3  ")

# ── Comparação principal ──────────────────────────────────────
print("\n" + "=" * 65)
print("COMPARACAO BASELINE vs RODADA 3")
print("=" * 65)
print(f"\n{'Split':<10} {'Base AUC':>10} {'R3 AUC':>10} "
      f"{'Base d10d':>11} {'R3 d10d':>11} {'Spearman':>10}  OK")
print("-" * 68)

diffs = []
for rb, rn in zip(res_base.itertuples(), res_new.itertuples()):
    diff = rn.delta_10d - rb.delta_10d
    diffs.append(diff)
    print(f"{rb.split:<10} {rb.auc:>10.3f} {rn.auc:>10.3f} "
          f"{rb.delta_10d:>+11.4f} {rn.delta_10d:>+11.4f} "
          f"{rn.spearman_r:>+10.3f}  {'✓' if diff > 0 else '✗'}")

base_mean = res_base["delta_10d"].mean()
new_mean  = res_new["delta_10d"].mean()
splits_ok = sum(d > 0 for d in diffs)
print(f"\n{'Media':<10} {res_base['auc'].mean():>10.3f} "
      f"{res_new['auc'].mean():>10.3f} "
      f"{base_mean:>+11.4f} {new_mean:>+11.4f} "
      f"{res_new['spearman_r'].mean():>+10.3f}")

print(f"\nSplits melhorados: {splits_ok}/3")
print(f"Decisao: {'APROVADO' if splits_ok >= 2 and new_mean > base_mean else 'REJEITADO'}")

# ── Regra de convicção ────────────────────────────────────────
print("\n" + "=" * 65)
print("CAMADA 3 — Regra de conviccao (prob > 0.60 / < 0.40)")
print("=" * 65)
print(f"\n{'Split':<10} {'Long ret':>12} {'Short ret':>12} {'Neutro%':>10} {'Edge':>10}")
print("-" * 55)
for rn in res_new.itertuples():
    edge = rn.conv_long_ret - rn.conv_short_ret
    print(f"{rn.split:<10} {rn.conv_long_ret:>+12.4f} "
          f"{rn.conv_short_ret:>+12.4f} "
          f"{rn.pct_neutral:>10.1%} {edge:>+10.4f}")

# ── Feature importance ────────────────────────────────────────
print("\n" + "=" * 65)
print("FEATURE IMPORTANCE — media 3 splits (Rodada 3)")
print("=" * 65)
imp_mean = imp_new.mean().sort_values(ascending=False)
groups = {
    "INSTITUTIONAL": set(FEAT_INSTITUTIONAL),
    "SENTIMENT":     set(FEAT_SENTIMENT),
    "DERIVATIVES":   set(FEAT_DERIVATIVES),
    "LIQUIDITY":     set(FEAT_LIQUIDITY),
}
for feat, val in imp_mean.items():
    group = next((g for g, s in groups.items() if feat in s), "BASE")
    print(f"  {feat:<35} {val:.4f}  [{group}]")

# ── Output operacional simulado ───────────────────────────────
print("\n" + "=" * 65)
print("OUTPUT OPERACIONAL — simulacao com dados mais recentes")
print("=" * 65)

sp    = SPLITS[2]
cols  = FEATURES_ALL + ["label_10d", "return_10d"]
train = df[sp["train_start"]:sp["train_end"]][cols].dropna()
last  = df[df[FEATURES_ALL].notna().all(axis=1)].tail(1)

if len(last) > 0:
    clf_final = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5, gamma=1,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    clf_final.fit(train[FEATURES_ALL], train["label_10d"], verbose=False)

    reg_final = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5, gamma=1, random_state=42, verbosity=0,
    )
    reg_final.fit(train[FEATURES_ALL], train["return_10d"], verbose=False)

    prob_up  = clf_final.predict_proba(last[FEATURES_ALL])[0, 1]
    exp_ret  = reg_final.predict(last[FEATURES_ALL])[0]
    regime   = "BULL" if last["hmm_state"].iloc[0] == 1 else "BEAR"
    bull_p   = last["bull_prob"].iloc[0]

    if regime == "BULL" and prob_up > 0.60:
        signal = "LONG"
    elif regime == "BEAR" and prob_up < 0.40:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    if bull_p > 0.65 and prob_up > 0.60:
        signal += " (forte)"
    elif bull_p < 0.35 and prob_up < 0.40:
        signal += " (forte)"

    print(f"\n  Data:                  {last.index[0].date()}")
    print(f"  Regime:                {regime}")
    print(f"  Bull probability:      {bull_p:.1%}")
    print(f"  Prob subida 10d:       {prob_up:.1%}")
    print(f"  Retorno esperado 10d:  {exp_ret:+.2%}")
    print(f"  SINAL:                 {signal}")
    print(f"\n  Contexto das features:")
    ctx_features = [
        "fear_greed", "ahr999", "funding_zscore_30d",
        "etf_flow_7d", "stablecoin_mcap_zscore", "top_position_ratio",
    ]
    for f in ctx_features:
        if f in last.columns:
            print(f"    {f:<30} {last[f].iloc[0]:.4f}")
else:
    print("  Nenhuma linha com todas as features disponivel.")
