"""
XGBoost Rodada 6 — ablação incremental Coinglass sobre R5.
4 sub-rodadas sequenciais (A→D). Horizonte 5d.
Script exploratório — NAO modifica pipeline nem sobrescreve modelos.

Ajustes em relação ao spec original:
  - Model input já tem features técnicas R5 (L3 atualizado) — sem recalcular
  - bb_width_20d (model input) equivale ao bb_width dos scripts
  - hmm: coluna 'state' renomeada para 'hmm_state'
  - top_position_ratio já está no model input (Config E) — skip join ls
  - use_label_encoder: removido (XGBoost >= 1.6)
  - pct_change com fill_method=None
"""
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

BASE_PATH = Path("data/01_raw/derivatives/coinglass")

# ── Carregar dados ────────────────────────────────────────────
# Model input já tem features técnicas do R5 (L3 pipeline atualizado)
btc  = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
hmm  = pd.read_parquet("data/05_models/regime_hmm/btc_states.parquet")
fr   = pd.read_parquet(BASE_PATH / "funding/BTCUSDT_oi_weighted.parquet")
oi   = pd.read_parquet(BASE_PATH / "open_interest/BTCUSDT_aggregated.parquet")
stbl = pd.read_parquet(BASE_PATH / "indices/stablecoin_mcap.parquet")
cdri = pd.read_parquet(BASE_PATH / "indices/cdri_index.parquet")
etf_f = pd.read_parquet(BASE_PATH / "etf/BTC_flows_total.parquet")
etf_h = pd.read_parquet(BASE_PATH / "etf/BTC_holdings_consolidated.parquet")
fg   = pd.read_parquet(BASE_PATH / "indices/fear_greed.parquet")
ahr  = pd.read_parquet(BASE_PATH / "indices/ahr999.parquet")

# coluna real é 'state'
hmm = hmm.rename(columns={"state": "hmm_state"})

# ── Features Coinglass derivadas ─────────────────────────────
fr["funding_zscore_30d"] = (
    (fr["funding_rate_oi_weighted"] -
     fr["funding_rate_oi_weighted"].rolling(30).mean()) /
     fr["funding_rate_oi_weighted"].rolling(30).std()
)
oi["oi_change_1d_lag1"] = oi["open_interest_usd"].pct_change(fill_method=None).shift(1)

stbl["stablecoin_mcap_zscore"] = (
    (stbl["stablecoin_mcap_usd"] -
     stbl["stablecoin_mcap_usd"].rolling(30).mean()) /
     stbl["stablecoin_mcap_usd"].rolling(30).std()
)

etf_f["etf_flow_7d"] = etf_f["flow_usd"].rolling(7).sum()
etf_h_col = [c for c in etf_h.columns if "total_btc" in c.lower()][0]
etf_h = etf_h.rename(columns={etf_h_col: "etf_btc_holdings_total"})

# ── Montar DataFrame único ────────────────────────────────────
df = btc.copy()
df = df.join(hmm[["hmm_state", "bull_prob"]], how="left")
df["regime_strength"] = (df["bull_prob"] - 0.5).abs()

# Coinglass joins
df = df.join(fr[["funding_zscore_30d"]], how="left")
df = df.join(oi[["oi_change_1d_lag1"]], how="left")
# top_position_ratio já está no model input (Config E) — skip join ls
df = df.join(stbl[["stablecoin_mcap_zscore"]], how="left")
df = df.join(cdri[["cdri_index"]], how="left")
df = df.join(etf_f[["etf_flow_7d"]], how="left")
df = df.join(etf_h[["etf_btc_holdings_total"]], how="left")
df = df.join(fg[["fear_greed"]], how="left")
df = df.join(ahr[["ahr999"]], how="left")

# Tratamento especial
df["etf_era_flag"]           = (df.index >= "2024-01-26").astype(int)
df["etf_flow_7d"]            = df["etf_flow_7d"].fillna(0)
df["etf_btc_holdings_total"] = df["etf_btc_holdings_total"].ffill().fillna(0)
df["cdri_index"]             = df["cdri_index"].ffill()

# Labels
df["return_5d"] = df["log_return"].rolling(5).sum().shift(-5)
df["label_5d"]  = (df["return_5d"] > 0).astype(int)

print(f"Shape apos join: {df.shape}")
print(f"Range: {df.index.min().date()} -> {df.index.max().date()}")
print(f"top_position_ratio no model input: {'top_position_ratio' in df.columns}")

# ── Feature sets ─────────────────────────────────────────────
BASE_COLS = [c for c in [
    "log_return", "vol_short", "vol_ratio", "drawdown",
    "volume_z", "slope_21d", "hmm_state", "bull_prob", "regime_strength",
] if c in df.columns]

FEAT_MOMENTUM = [c for c in [
    "rsi_14_z", "rsi_30", "macd_hist_norm",
    "roc_5", "roc_10", "roc_21",
] if c in df.columns]

# bb_width_20d (model input) é equivalente ao bb_width dos scripts
FEAT_STRUCTURE = [c for c in [
    "price_vs_high_30d", "price_vs_low_30d",
    "range_position_30d", "bb_position",
    "bb_width_20d", "atr_14_norm",
] if c in df.columns]

# R5 baseline
FEAT_R5 = BASE_COLS + FEAT_STRUCTURE + FEAT_MOMENTUM

# Grupos Coinglass incrementais
# top_position_ratio já em df via model input — incluir direto
FEAT_DERIV = [c for c in [
    "funding_zscore_30d", "oi_change_1d_lag1", "top_position_ratio",
] if c in df.columns]

FEAT_LIQ  = [c for c in ["stablecoin_mcap_zscore", "cdri_index"] if c in df.columns]
FEAT_ETF  = [c for c in ["etf_flow_7d", "etf_btc_holdings_total", "etf_era_flag"] if c in df.columns]
FEAT_SENT = [c for c in ["fear_greed", "ahr999"] if c in df.columns]

print(f"\nR5 features:    {len(FEAT_R5)}")
print(f"DERIV features: {len(FEAT_DERIV)} → {FEAT_DERIV}")
print(f"LIQ features:   {len(FEAT_LIQ)} → {FEAT_LIQ}")
print(f"ETF features:   {len(FEAT_ETF)} → {FEAT_ETF}")
print(f"SENT features:  {len(FEAT_SENT)} → {FEAT_SENT}")

# Verificar cobertura pós-2023
df23 = df[df.index >= "2023-01-01"]
all_cg = FEAT_DERIV + FEAT_LIQ + FEAT_ETF + FEAT_SENT
print("\nCobertura Coinglass pós-2023:")
for f in all_cg:
    valid = df23[f].notna().sum()
    total = len(df23)
    flag  = "  ⚠ BAIXA" if valid / total < 0.95 else ""
    print(f"  {f:<35} {valid}/{total} ({valid/total:.1%}){flag}")

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]


# ── Walk-forward ──────────────────────────────────────────────
def run_xgb(df, features, label=""):
    results     = []
    importances = []
    for i, sp in enumerate(SPLITS):
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        # deduplicar cols
        cols  = list(dict.fromkeys(features + ["label_5d", "return_5d"]))
        train = df[sp["train_start"]:sp["train_end"]][cols].dropna()
        test  = df[sp["test_start"]:test_end][cols].dropna()

        if len(train) < 50 or len(test) < 20:
            print(f"  {label} split_{i+1}: SKIP (train={len(train)} test={len(test)})")
            continue

        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4,
            learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.7, min_child_weight=3,
            gamma=0.5, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        clf.fit(
            train[features], train["label_5d"],
            eval_set=[(test[features], test["label_5d"])],
            verbose=False,
        )

        proba    = clf.predict_proba(test[features])[:, 1]
        pred     = (proba >= 0.5).astype(int)
        auc      = roc_auc_score(test["label_5d"], proba)
        ret_fwd  = test["return_5d"]
        bull_ret = ret_fwd[pred == 1].mean()
        bear_ret = ret_fwd[pred == 0].mean()
        delta    = bull_ret - bear_ret

        long_mask  = proba > 0.60
        short_mask = proba < 0.40
        conv_long  = ret_fwd[long_mask].mean() if long_mask.sum() > 0 else float("nan")
        conv_short = ret_fwd[short_mask].mean() if short_mask.sum() > 0 else float("nan")
        conv_edge  = conv_long - conv_short
        pct_neutral = (~(long_mask | short_mask)).mean()

        results.append({
            "split":       f"split_{i+1}",
            "n_train":     len(train),
            "n_test":      len(test),
            "auc":         auc,
            "delta":       delta,
            "conv_edge":   conv_edge,
            "pct_neutral": pct_neutral,
        })
        importances.append(pd.Series(
            clf.feature_importances_, index=features, name=f"split_{i+1}"
        ))
        print(f"  {label} split_{i+1}: "
              f"n={len(train)}/{len(test)} "
              f"auc={auc:.3f} delta={delta:+.4f} "
              f"edge={conv_edge:+.4f} "
              f"neutral={pct_neutral:.0%}")

    return pd.DataFrame(results), pd.DataFrame(importances) if importances else pd.DataFrame()


def veredicto(res_new, res_prev, label):
    new_mean    = res_new["delta"].mean()
    prev_mean   = res_prev["delta"].mean()
    prev_deltas = res_prev["delta"].values
    diff        = new_mean - prev_mean
    splits_ok   = (res_new["delta"].values > prev_deltas).sum()
    aprovado    = splits_ok >= 2 and diff > 0
    print(f"\n  {'=' * 55}")
    print(f"  VEREDICTO {label}")
    print(f"  {'=' * 55}")
    print(f"  Delta médio anterior: {prev_mean:>+.4f}")
    print(f"  Delta médio novo:     {new_mean:>+.4f}")
    print(f"  Diferença:            {diff:>+.4f}")
    print(f"  Splits melhorados:    {splits_ok}/3")
    decisao = "✓ APROVADO — continuar" if aprovado else "✗ REJEITADO — parar ablação"
    print(f"  Decisão: {decisao}")
    return aprovado, new_mean


# ── R5 baseline ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("R5 BASELINE — BASE + STRUCTURE + MOMENTUM (5d)")
print("=" * 65)
res_r5, _ = run_xgb(df, FEAT_R5, "R5  ")
r5_mean   = res_r5["delta"].mean()
r5_deltas = res_r5["delta"].values

# ── R6-A: R5 + Derivativos ───────────────────────────────────
print("\n" + "=" * 65)
print("R6-A — R5 + DERIVATIVOS")
print(f"  +{FEAT_DERIV}")
print("=" * 65)
feat_a       = list(dict.fromkeys(FEAT_R5 + FEAT_DERIV))
res_a, imp_a = run_xgb(df, feat_a, "R6-A")
ok_a, mean_a = veredicto(res_a, res_r5, "R6-A")

all_results  = [("R5 baseline", res_r5)]
all_results.append(("R6-A +derivativos", res_a))

best_feat = feat_a
best_imp  = imp_a
best_name = "R6-A"
best_res  = res_a

if not ok_a:
    print("\n  ABLAÇÃO ENCERRADA em R6-A")
    print("  Derivativos não adicionam sinal ao R5")
    best_feat = FEAT_R5
    best_imp  = pd.DataFrame()
    best_name = "R5"
    best_res  = res_r5
else:
    # ── R6-B: R6-A + Liquidez ────────────────────────────────
    print("\n" + "=" * 65)
    print("R6-B — R6-A + LIQUIDEZ")
    print(f"  +{FEAT_LIQ}")
    print("=" * 65)
    feat_b       = list(dict.fromkeys(feat_a + FEAT_LIQ))
    res_b, imp_b = run_xgb(df, feat_b, "R6-B")
    ok_b, mean_b = veredicto(res_b, res_a, "R6-B")
    all_results.append(("R6-B +liquidez", res_b))

    if not ok_b:
        print("\n  ABLAÇÃO ENCERRADA em R6-B")
        print(f"  Melhor config: R6-A (delta={mean_a:+.4f})")
    else:
        best_feat = feat_b
        best_imp  = imp_b
        best_name = "R6-B"
        best_res  = res_b

        # ── R6-C: R6-B + ETF ─────────────────────────────────
        print("\n" + "=" * 65)
        print("R6-C — R6-B + ETF")
        print(f"  +{FEAT_ETF}")
        print("=" * 65)
        feat_c       = list(dict.fromkeys(feat_b + FEAT_ETF))
        res_c, imp_c = run_xgb(df, feat_c, "R6-C")
        ok_c, mean_c = veredicto(res_c, res_b, "R6-C")
        all_results.append(("R6-C +ETF", res_c))

        if not ok_c:
            print(f"\n  ABLAÇÃO ENCERRADA em R6-C")
            print(f"  Melhor config: {best_name} (delta={mean_b:+.4f})")
        else:
            best_feat = feat_c
            best_imp  = imp_c
            best_name = "R6-C"
            best_res  = res_c

            # ── R6-D: R6-C + Sentimento ──────────────────────
            print("\n" + "=" * 65)
            print("R6-D — R6-C + SENTIMENTO")
            print(f"  +{FEAT_SENT}")
            print("=" * 65)
            feat_d       = list(dict.fromkeys(feat_c + FEAT_SENT))
            res_d, imp_d = run_xgb(df, feat_d, "R6-D")
            ok_d, mean_d = veredicto(res_d, res_c, "R6-D")
            all_results.append(("R6-D +sentimento", res_d))

            if ok_d:
                best_feat = feat_d
                best_imp  = imp_d
                best_name = "R6-D"
                best_res  = res_d

# ── Tabela comparativa final ──────────────────────────────────
print("\n" + "=" * 65)
print("TABELA COMPARATIVA FINAL")
print("=" * 65)
print(f"\n{'Config':<22}" +
      "".join(f"{'split_'+str(i+1):>12}" for i in range(3)) +
      f"{'Média':>12} {'vs R5':>10}")
print("-" * 70)
for name, res in all_results:
    row = f"{name:<22}"
    for d in res["delta"].values:
        row += f"{d:>+12.4f}"
    mean = res["delta"].mean()
    diff = mean - r5_mean
    row += f"{mean:>+12.4f} {diff:>+10.4f}"
    print(row)

# ── Feature importance da melhor config ──────────────────────
if not best_imp.empty:
    print(f"\n{'=' * 65}")
    print(f"FEATURE IMPORTANCE — {best_name} (média 3 splits)")
    print(f"{'=' * 65}")

    groups_map = {}
    for f in FEAT_MOMENTUM:  groups_map[f] = "MOMENTUM"
    for f in FEAT_STRUCTURE: groups_map[f] = "STRUCTURE"
    for f in BASE_COLS:      groups_map[f] = "BASE"
    for f in FEAT_DERIV:     groups_map[f] = "CG-DERIV"
    for f in FEAT_LIQ:       groups_map[f] = "CG-LIQ"
    for f in FEAT_ETF:       groups_map[f] = "CG-ETF"
    for f in FEAT_SENT:      groups_map[f] = "CG-SENT"

    imp_mean = best_imp.mean().sort_values(ascending=False)
    print(f"\nTop 12 features — {best_name}:")
    for feat, val in imp_mean.head(12).items():
        g = groups_map.get(feat, "?")
        print(f"  {feat:<35} {val:.4f}  [{g}]")

# ── Output operacional hoje ───────────────────────────────────
print(f"\n{'=' * 65}")
print(f"OUTPUT OPERACIONAL HOJE — {best_name}")
print(f"{'=' * 65}")

sp      = SPLITS[2]
cols_f  = list(dict.fromkeys(best_feat + ["label_5d"]))
train_f = df[sp["train_start"]:sp["train_end"]][cols_f].dropna()
last    = df[df[best_feat].notna().all(axis=1)].tail(1)

if len(train_f) > 50 and len(last) > 0:
    clf_f = xgb.XGBClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=3,
        gamma=0.5, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    clf_f.fit(train_f[best_feat], train_f["label_5d"], verbose=False)
    prob_up = clf_f.predict_proba(last[best_feat])[0, 1]

    regime  = "BULL" if last["hmm_state"].iloc[0] == 1 else "BEAR"
    bull_p  = float(last["bull_prob"].iloc[0])
    reg_str = float(last["regime_strength"].iloc[0])

    if regime == "BULL" and prob_up > 0.60:
        signal = "LONG" + (" FORTE" if bull_p > 0.65 and prob_up > 0.65 else "")
    elif regime == "BEAR" and prob_up < 0.40:
        signal = "SHORT" + (" FORTE" if bull_p < 0.35 and prob_up < 0.35 else "")
    else:
        signal = "NEUTRAL"

    print(f"\n  Modelo:             {best_name}")
    print(f"  Data:               {last.index[0].date()}")
    print(f"  Regime HMM:         {regime}")
    print(f"  Bull probability:   {bull_p:.1%}")
    print(f"  Regime strength:    {reg_str:.3f}")
    print(f"  Prob subida 5d:     {prob_up:.1%}")
    print(f"  SINAL:              {signal}")

    # Comparação com R5 puro
    cols_r5  = list(dict.fromkeys(FEAT_R5 + ["label_5d"]))
    train_r5 = df[sp["train_start"]:sp["train_end"]][cols_r5].dropna()
    last_r5  = df[df[FEAT_R5].notna().all(axis=1)].tail(1)
    if len(train_r5) > 50 and len(last_r5) > 0:
        clf_r5 = xgb.XGBClassifier(
            n_estimators=300, max_depth=4,
            learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.7, min_child_weight=3,
            gamma=0.5, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        clf_r5.fit(train_r5[FEAT_R5], train_r5["label_5d"], verbose=False)
        prob_r5 = clf_r5.predict_proba(last_r5[FEAT_R5])[0, 1]
        print(f"\n  Comparação R5:      prob_up={prob_r5:.1%}")
        print(f"  Diferença:          {prob_up - prob_r5:>+.1%}")

    # Contexto Coinglass hoje
    cg_feats = [f for f in best_feat
                if f in (FEAT_DERIV + FEAT_LIQ + FEAT_ETF + FEAT_SENT)
                and f in last.columns]
    if cg_feats:
        print(f"\n  Contexto Coinglass:")
        for f in cg_feats:
            val = last[f].iloc[0]
            print(f"    {f:<35} {val:>+.4f}" if pd.notna(val) else f"    {f:<35} NaN")
else:
    print("\n  AVISO: dados insuficientes para output operacional.")
    print(f"  n_train={len(train_f)} | n_last={len(last)}")
