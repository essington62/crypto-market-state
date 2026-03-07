"""
XGBoost Rodada 1 — baseline + 13 features Coinglass.
Walk-forward 3 splits identico ao HMM.
Script exploratório — NAO modifica pipeline nem sobrescreve modelos.

Ajustes em relacao ao spec original:
  - hmm: coluna 'state' renomeada para 'hmm_state'
  - top_position_ratio: já presente em btc (pipeline atualizado) — join skippado
  - use_label_encoder: removido (XGBoost >= 1.6)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

BASE = Path("data/01_raw/derivatives/coinglass")

# ── Carregar dados L1 Coinglass ───────────────────────────────
oi    = pd.read_parquet(BASE / "open_interest/BTCUSDT_aggregated.parquet")
fr    = pd.read_parquet(BASE / "funding/BTCUSDT_oi_weighted.parquet")
ls    = pd.read_parquet(BASE / "long_short_ratio/BTCUSDT_top_positions.parquet")
fg    = pd.read_parquet(BASE / "indices/fear_greed.parquet")
ahr   = pd.read_parquet(BASE / "indices/ahr999.parquet")
stbl  = pd.read_parquet(BASE / "indices/stablecoin_mcap.parquet")
cdri  = pd.read_parquet(BASE / "indices/cdri_index.parquet")
etf_f = pd.read_parquet(BASE / "etf/BTC_flows_total.parquet")
etf_h = pd.read_parquet(BASE / "etf/BTC_holdings_consolidated.parquet")

# ── Carregar model input e estados HMM ───────────────────────
# btc já contém top_position_ratio (pipeline atualizado — Config E)
btc = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
hmm = pd.read_parquet("data/05_models/regime_hmm/btc_states.parquet")

# coluna real é 'state', nao 'hmm_state'
hmm = hmm.rename(columns={"state": "hmm_state"})

# ── Calcular features derivadas ──────────────────────────────
# Open Interest — defasado 1 dia para evitar lookahead
oi["oi_change_1d_lag1"] = oi["open_interest_usd"].pct_change(fill_method=None).shift(1)

# Funding zscore 30d
fr["funding_zscore_30d"] = (
    (fr["funding_rate_oi_weighted"] -
     fr["funding_rate_oi_weighted"].rolling(30).mean()) /
     fr["funding_rate_oi_weighted"].rolling(30).std()
)

# top_position_ratio: já está em btc — preparar ls apenas para as colunas extras
ratio_col = [c for c in ls.columns if "ratio" in c.lower()][0]
ls = ls.rename(columns={ratio_col: "top_position_ratio"})

# Stablecoin mcap zscore
stbl["stablecoin_mcap_zscore"] = (
    (stbl["stablecoin_mcap_usd"] -
     stbl["stablecoin_mcap_usd"].rolling(30).mean()) /
     stbl["stablecoin_mcap_usd"].rolling(30).std()
)

# ETF flows
etf_f["etf_flow_7d"] = etf_f["flow_usd"].rolling(7).sum()

# ETF holdings — coluna total_btc_holdings
etf_h_col = [c for c in etf_h.columns if "total_btc" in c.lower()][0]
etf_h = etf_h.rename(columns={etf_h_col: "etf_btc_holdings_total"})

# ── Montar DataFrame único ────────────────────────────────────
df = btc.copy()
df = df.join(hmm[["hmm_state", "bull_prob"]], how="left")
df = df.join(oi[["oi_change_1d_lag1"]], how="left")
df = df.join(fr[["funding_zscore_30d"]], how="left")
# top_position_ratio já está em df (de btc) — nao re-join ls
df = df.join(fg[["fear_greed"]], how="left")
df = df.join(ahr[["ahr999"]], how="left")
df = df.join(stbl[["stablecoin_mcap_zscore"]], how="left")
df = df.join(cdri[["cdri_index"]], how="left")
df = df.join(etf_f[["flow_usd", "etf_flow_7d"]], how="left")
df = df.join(etf_h[["etf_btc_holdings_total"]], how="left")

# ETF era flag — 0 antes de Jan/2024 (primeiro dia com dados de holdings)
df["etf_era_flag"] = (df.index >= "2024-01-26").astype(int)

# ETF flows: NaN antes do ETF spot → 0 (sem fluxo)
df["etf_flow_usd"] = df["flow_usd"].fillna(0)
df["etf_flow_7d"]  = df["etf_flow_7d"].fillna(0)
df = df.drop(columns=["flow_usd"], errors="ignore")

# ETF holdings — ffill para fins de semana/gaps; 0 antes do ETF
df["etf_btc_holdings_total"] = df["etf_btc_holdings_total"].ffill().fillna(0)

# Label: retorno 2d forward positivo
df["label"] = (df["log_return"].shift(-2) + df["log_return"].shift(-1) > 0).astype(int)

print(f"Shape total: {df.shape}")
print(f"Range: {df.index.min().date()} -> {df.index.max().date()}")

print(f"\nCobertura por feature (pos-2023):")
df23 = df[df.index >= "2023-01-01"]
new_features = [
    "oi_change_1d_lag1", "funding_zscore_30d", "top_position_ratio",
    "fear_greed", "ahr999", "stablecoin_mcap_zscore", "cdri_index",
    "etf_flow_usd", "etf_flow_7d", "etf_era_flag", "etf_btc_holdings_total",
]
for f in new_features:
    valid = df23[f].notna().sum()
    total = len(df23)
    print(f"  {f:<32} {valid}/{total} ({valid/total:.1%})")

# ── Definir feature sets ──────────────────────────────────────
FEATURES_BASE = [c for c in [
    "log_return", "vol_short", "vol_ratio", "drawdown",
    "volume_z", "slope_21d", "hmm_state", "bull_prob",
] if c in df.columns]

FEATURES_NEW = FEATURES_BASE + new_features

print(f"\nFeatures baseline:  {len(FEATURES_BASE)} → {FEATURES_BASE}")
print(f"Features expandido: {len(FEATURES_NEW)}")

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]


def run_xgb_walkforward(df, features, label=""):
    results = []
    last_model = None
    for i, sp in enumerate(SPLITS):
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")

        cols  = features + ["label"]
        train = df[sp["train_start"]:sp["train_end"]][cols].dropna()
        test  = df[sp["test_start"]:test_end][cols].dropna()

        X_train = train[features]
        y_train = train["label"]
        X_test  = test[features]
        y_test  = test["label"]

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        proba = model.predict_proba(X_test)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        auc   = roc_auc_score(y_test, proba)
        acc   = (pred == y_test).mean()

        # Delta: retorno 2d quando modelo diz Bull vs Bear
        ret_2d = df["log_return"].shift(-1) + df["log_return"].shift(-2)
        ret_2d = ret_2d.reindex(test.index)
        bull_ret = ret_2d[pred == 1].mean()
        bear_ret = ret_2d[pred == 0].mean()
        delta    = bull_ret - bear_ret

        results.append({
            "split":    f"split_{i+1}",
            "n_train":  len(train),
            "n_test":   len(test),
            "auc":      auc,
            "acc":      acc,
            "bull_ret": bull_ret,
            "bear_ret": bear_ret,
            "delta_2d": delta,
        })
        print(f"  {label} split_{i+1}: "
              f"n_train={len(train):3d}  n_test={len(test):3d}  "
              f"auc={auc:.3f}  acc={acc:.1%}  delta_2d={delta:+.4f}")
        last_model = model

    return pd.DataFrame(results), last_model


print("\n" + "=" * 65)
print("BASELINE XGBoost — features originais")
print("=" * 65)
res_base, _ = run_xgb_walkforward(df, FEATURES_BASE, "BASE")

print("\n" + "=" * 65)
print("RODADA 1 — baseline + features Coinglass")
print("=" * 65)
res_new, model_new = run_xgb_walkforward(df, FEATURES_NEW, "NEW ")

# ── Comparação ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("COMPARACAO BASELINE vs RODADA 1")
print("=" * 65)
print(f"\n{'Split':<10} {'Base AUC':>10} {'New AUC':>10} "
      f"{'Base d2d':>10} {'New d2d':>10} {'Diff':>10}  OK")
print("-" * 68)

diffs = []
for rb, rn in zip(res_base.itertuples(), res_new.itertuples()):
    diff = rn.delta_2d - rb.delta_2d
    diffs.append(diff)
    print(f"{rb.split:<10} {rb.auc:>10.3f} {rn.auc:>10.3f} "
          f"{rb.delta_2d:>+10.4f} {rn.delta_2d:>+10.4f} "
          f"{diff:>+10.4f}  {'✓' if diff > 0 else '✗'}")

base_mean = res_base["delta_2d"].mean()
new_mean  = res_new["delta_2d"].mean()
splits_ok = sum(d > 0 for d in diffs)

print(f"\n{'Media':<10} {res_base['auc'].mean():>10.3f} "
      f"{res_new['auc'].mean():>10.3f} "
      f"{base_mean:>+10.4f} {new_mean:>+10.4f} "
      f"{new_mean - base_mean:>+10.4f}  "
      f"{'✓' if new_mean > base_mean else '✗'}")

print(f"\nSplits melhorados: {splits_ok}/3")
print(f"Decisao: {'APROVADO' if splits_ok >= 2 and new_mean > base_mean else 'REJEITADO'}")

# ── Feature importance ────────────────────────────────────────
print("\n" + "=" * 65)
print("FEATURE IMPORTANCE — Rodada 1 (split_3, modelo mais recente)")
print("=" * 65)
importance = pd.Series(
    model_new.feature_importances_,
    index=FEATURES_NEW,
).sort_values(ascending=False)
print(importance.round(4).to_string())

print("\nTop 5 features Coinglass:")
coinglass_imp = importance[importance.index.isin(new_features)]
print(coinglass_imp.head(5).round(4).to_string())
