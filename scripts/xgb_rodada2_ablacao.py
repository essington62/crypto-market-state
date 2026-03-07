"""
XGBoost Rodada 2 — ablacao por grupo de features.
Walk-forward 3 splits identico a Rodada 1.
Script exploratório — NAO modifica pipeline nem sobrescreve modelos.

Ajustes em relacao ao spec original:
  - hmm: coluna 'state' renomeada para 'hmm_state'
  - top_position_ratio: ja presente em btc (pipeline Config E) — ls join skippado
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

# ── Carregar dados ────────────────────────────────────────────
# btc ja contem top_position_ratio (Config E ativo no pipeline)
btc   = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
hmm   = pd.read_parquet("data/05_models/regime_hmm/btc_states.parquet")
oi    = pd.read_parquet(BASE / "open_interest/BTCUSDT_aggregated.parquet")
fr    = pd.read_parquet(BASE / "funding/BTCUSDT_oi_weighted.parquet")
fg    = pd.read_parquet(BASE / "indices/fear_greed.parquet")
ahr   = pd.read_parquet(BASE / "indices/ahr999.parquet")
etf_f = pd.read_parquet(BASE / "etf/BTC_flows_total.parquet")
etf_h = pd.read_parquet(BASE / "etf/BTC_holdings_consolidated.parquet")

# coluna real é 'state'
hmm = hmm.rename(columns={"state": "hmm_state"})

# ── Features derivadas ────────────────────────────────────────
oi["oi_change_1d_lag1"] = oi["open_interest_usd"].pct_change(fill_method=None).shift(1)
fr["funding_zscore_30d"] = (
    (fr["funding_rate_oi_weighted"] -
     fr["funding_rate_oi_weighted"].rolling(30).mean()) /
     fr["funding_rate_oi_weighted"].rolling(30).std()
)
etf_f["etf_flow_7d"] = etf_f["flow_usd"].rolling(7).sum()
etf_h_col = [c for c in etf_h.columns if "total_btc" in c.lower()][0]
etf_h = etf_h.rename(columns={etf_h_col: "etf_btc_holdings_total"})

# ── Montar DataFrame único ────────────────────────────────────
df = btc.copy()
df = df.join(hmm[["hmm_state", "bull_prob"]], how="left")
df = df.join(oi[["oi_change_1d_lag1"]], how="left")
df = df.join(fr[["funding_zscore_30d"]], how="left")
# top_position_ratio ja esta em df (btc Config E) — sem re-join
df = df.join(fg[["fear_greed"]], how="left")
df = df.join(ahr[["ahr999"]], how="left")
df = df.join(etf_f[["flow_usd", "etf_flow_7d"]], how="left")
df = df.join(etf_h[["etf_btc_holdings_total"]], how="left")

df["etf_era_flag"]           = (df.index >= "2024-01-26").astype(int)
df["etf_flow_usd"]           = df["flow_usd"].fillna(0)
df["etf_flow_7d"]            = df["etf_flow_7d"].fillna(0)
df["etf_btc_holdings_total"] = df["etf_btc_holdings_total"].ffill().fillna(0)
df = df.drop(columns=["flow_usd"], errors="ignore")

df["label"] = (
    df["log_return"].shift(-2) + df["log_return"].shift(-1) > 0
).astype(int)

# ── Feature sets ──────────────────────────────────────────────
btc_cols = [c for c in [
    "log_return", "vol_short", "vol_ratio", "drawdown",
    "volume_z", "slope_21d", "hmm_state", "bull_prob",
] if c in df.columns]

FEATURES_BASE  = btc_cols
FEATURES_ETF   = btc_cols + [
    "etf_flow_usd", "etf_flow_7d",
    "etf_btc_holdings_total", "etf_era_flag",
]
FEATURES_SENT  = btc_cols + ["fear_greed", "ahr999"]
FEATURES_DERIV = btc_cols + [
    "funding_zscore_30d", "oi_change_1d_lag1", "top_position_ratio",
]

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]


def run_xgb(df, features, label=""):
    results     = []
    importances = []
    for i, sp in enumerate(SPLITS):
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        cols  = features + ["label"]
        train = df[sp["train_start"]:sp["train_end"]][cols].dropna()
        test  = df[sp["test_start"]:test_end][cols].dropna()

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
        model.fit(train[features], train["label"],
                  eval_set=[(test[features], test["label"])],
                  verbose=False)

        proba    = model.predict_proba(test[features])[:, 1]
        pred     = (proba >= 0.5).astype(int)
        auc      = roc_auc_score(test["label"], proba)
        ret_2d   = df["log_return"].shift(-1) + df["log_return"].shift(-2)
        ret_2d   = ret_2d.reindex(test.index)
        bull_ret = ret_2d[pred == 1].mean()
        bear_ret = ret_2d[pred == 0].mean()
        delta    = bull_ret - bear_ret

        results.append({
            "split":    f"split_{i+1}",
            "auc":      auc,
            "delta_2d": delta,
            "bull_ret": bull_ret,
            "bear_ret": bear_ret,
            "n_train":  len(train),
            "n_test":   len(test),
        })
        importances.append(pd.Series(
            model.feature_importances_, index=features, name=f"split_{i+1}"
        ))
        print(f"  {label} split_{i+1}: "
              f"auc={auc:.3f}  delta_2d={delta:+.4f}  "
              f"(bull={bull_ret:+.4f}  bear={bear_ret:+.4f})")

    return pd.DataFrame(results), pd.DataFrame(importances)


# ── Rodar 4 configurações ─────────────────────────────────────
configs = [
    (FEATURES_BASE,  "BASE  — price action only     "),
    (FEATURES_ETF,   "XGB_A — baseline + ETF        "),
    (FEATURES_SENT,  "XGB_B — baseline + sentimento  "),
    (FEATURES_DERIV, "XGB_C — baseline + derivativos "),
]

all_res = {}
all_imp = {}
for features, label in configs:
    print(f"\n{'=' * 60}")
    print(label)
    print(f"{'=' * 60}")
    all_res[label.strip()], all_imp[label.strip()] = run_xgb(
        df, features, label[:5]
    )

# ── Tabela comparativa — AUC ──────────────────────────────────
labels = [c[1].strip() for c in configs]

print("\n" + "=" * 75)
print("COMPARACAO — AUC por split")
print("=" * 75)
print(f"{'Split':<10}" + "".join(f"{l[:18]:>20}" for l in labels))
print("-" * 75)
for i in range(3):
    row = f"split_{i+1:<5}"
    for l in labels:
        row += f"{all_res[l]['auc'].iloc[i]:>20.3f}"
    print(row)
print("-" * 75)
row = f"{'Media':<10}"
for l in labels:
    row += f"{all_res[l]['auc'].mean():>20.3f}"
print(row)

# ── Tabela comparativa — Delta 2d ─────────────────────────────
print("\n" + "=" * 75)
print("COMPARACAO — Delta 2d por split")
print("=" * 75)
print(f"{'Split':<10}" + "".join(f"{l[:18]:>20}" for l in labels))
print("-" * 75)
for i in range(3):
    row = f"split_{i+1:<5}"
    for l in labels:
        row += f"{all_res[l]['delta_2d'].iloc[i]:>+20.4f}"
    print(row)
print("-" * 75)
row = f"{'Media':<10}"
for l in labels:
    row += f"{all_res[l]['delta_2d'].mean():>+20.4f}"
print(row)

# ── Veredicto por grupo ───────────────────────────────────────
print("\n" + "=" * 75)
print("VEREDICTO POR GRUPO")
print("=" * 75)
base_deltas = all_res[labels[0]]["delta_2d"].values
base_mean   = all_res[labels[0]]["delta_2d"].mean()

for l in labels[1:]:
    deltas     = all_res[l]["delta_2d"].values
    diff       = deltas.mean() - base_mean
    splits_ok  = (deltas > base_deltas).sum()
    split3_ok  = deltas[2] > base_deltas[2]
    status     = "APROVADO" if splits_ok >= 2 and diff > 0 else "REJEITADO"
    split3_tag = "resolve split_3" if split3_ok else "nao resolve split_3"
    print(f"  {l[:35]}: diff={diff:+.4f}  splits={splits_ok}/3  "
          f"{split3_tag}  {status}")

# ── Feature importance por grupo ─────────────────────────────
base_set = set(FEATURES_BASE)
print("\n" + "=" * 75)
print("FEATURE IMPORTANCE — media dos 3 splits por grupo")
print("=" * 75)
for l in labels[1:]:
    print(f"\n{l}:")
    imp_mean = all_imp[l].mean().sort_values(ascending=False)
    for feat, val in imp_mean.head(8).items():
        tag = "  <-- COINGLASS" if feat not in base_set else ""
        print(f"  {feat:<35} {val:.4f}{tag}")

# ── Qual grupo resolve split_3 ────────────────────────────────
print("\n" + "=" * 75)
print("QUAL GRUPO RESOLVE O SPLIT_3?")
print("=" * 75)
for l in labels[1:]:
    d3 = all_res[l]["delta_2d"].iloc[2]
    b3 = base_deltas[2]
    print(f"  {l[:35]}: split_3 delta={d3:+.4f}  "
          f"({'MELHOR' if d3 > b3 else 'PIOR'} que baseline {b3:+.4f})")
