"""
XGBoost R6-B — Coinglass Liquidity / Sentiment Ablation.

Features novas sobre R5:
  taker_buy_ratio       = buy_vol / (buy_vol + sell_vol)  — taker spot
  stablecoin_mcap_zscore = zscore 30d da soma USDT+USDC+DAI+FDUSD
  fear_greed            — direto (ffill)

Modelo: XGBRegressor (mesmo config do R7b — gamma=0, min_child_weight=1)
Target: return_5d / vol_short  (retorno normalizado, mesmo do R7b)
Sinal:  percentil 70/30 calibrado no treino
Script exploratório — NAO modifica pipeline nem sobrescreve modelos.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

BASE_CG = Path("data/01_raw/derivatives/coinglass")

# ── Carregar dados ─────────────────────────────────────────────
btc    = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
hmm    = pd.read_parquet("data/05_models/regime_hmm/btc_states.parquet")
taker  = pd.read_parquet(BASE_CG / "taker/spot_BTC_aggregated.parquet")
stbl   = pd.read_parquet(BASE_CG / "indices/stablecoin_mcap.parquet")
fg     = pd.read_parquet(BASE_CG / "indices/fear_greed.parquet")

hmm = hmm.rename(columns={"state": "hmm_state"})

# ── Derivar features Coinglass ────────────────────────────────
# taker_buy_ratio
buy  = taker["aggregated_buy_volume_usd"]
sell = taker["aggregated_sell_volume_usd"]
taker["taker_buy_ratio"] = buy / (buy + sell)

# stablecoin_mcap_total — corrigir bug R6: somar colunas reais
mcap_cols = [c for c in stbl.columns if c.startswith("mcap_")]
stbl["stablecoin_mcap_total"] = stbl[mcap_cols].sum(axis=1)
stbl["stablecoin_mcap_zscore"] = (
    (stbl["stablecoin_mcap_total"] -
     stbl["stablecoin_mcap_total"].rolling(30).mean()) /
     stbl["stablecoin_mcap_total"].rolling(30).std()
)

# ── Montar DataFrame ───────────────────────────────────────────
df = btc.copy()
df = df.join(hmm[["hmm_state", "bull_prob"]], how="left")
df["regime_strength"] = (df["bull_prob"] - 0.5).abs()

df = df.join(taker[["taker_buy_ratio"]], how="left")
df = df.join(stbl[["stablecoin_mcap_zscore"]], how="left")
df = df.join(fg[["fear_greed"]], how="left")
df["fear_greed"] = df["fear_greed"].ffill()

# Labels
df["return_5d"] = df["log_return"].rolling(5).sum().shift(-5)
# Target normalizado (vol_short = 21d rolling std = vol_21 no domínio)
df["target_norm"] = df["return_5d"] / df["vol_short"].replace(0, np.nan)

print(f"Shape: {df.shape}")
print(f"Range: {df.index.min().date()} -> {df.index.max().date()}")

# Cobertura das novas features pós-2023
df23 = df["2023-01-01":]
print("\nCobertura pós-2023:")
for feat in ["taker_buy_ratio", "stablecoin_mcap_zscore", "fear_greed"]:
    v = df23[feat].notna().sum()
    t = len(df23)
    print(f"  {feat:<30} {v}/{t} ({v/t:.1%})")

# ── Feature sets ──────────────────────────────────────────────
BASE_COLS = [c for c in [
    "log_return", "vol_short", "vol_ratio", "drawdown",
    "volume_z", "slope_21d", "hmm_state", "bull_prob", "regime_strength",
] if c in df.columns]

FEAT_MOMENTUM = [c for c in [
    "rsi_14_z", "rsi_30", "macd_hist_norm",
    "roc_5", "roc_10", "roc_21",
] if c in df.columns]

FEAT_STRUCTURE = [c for c in [
    "price_vs_high_30d", "price_vs_low_30d",
    "range_position_30d", "bb_position",
    "bb_width_20d", "atr_14_norm",
] if c in df.columns]

FEAT_R5 = BASE_COLS + FEAT_STRUCTURE + FEAT_MOMENTUM

FEAT_NEW = [c for c in [
    "taker_buy_ratio", "stablecoin_mcap_zscore", "fear_greed",
] if c in df.columns]

FEAT_R6B = FEAT_R5 + FEAT_NEW

print(f"\nFeatures R5:   {len(FEAT_R5)}")
print(f"Features novas:{len(FEAT_NEW)} → {FEAT_NEW}")
print(f"Features R6-B: {len(FEAT_R6B)}")

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]


# ── Funções ───────────────────────────────────────────────────
def to_signal_percentile(pred, train_pred, long_pct=70, short_pct=30):
    p_long  = np.percentile(train_pred, long_pct)
    p_short = np.percentile(train_pred, short_pct)
    sigs    = np.where(pred > p_long, 1, np.where(pred < p_short, -1, 0))
    return sigs, p_long, p_short


def evaluate_signals(signals, actual_ret):
    long_mask  = signals == 1
    short_mask = signals == -1

    long_ret  = actual_ret[long_mask].mean()  if long_mask.sum()  > 0 else float("nan")
    short_ret = actual_ret[short_mask].mean() if short_mask.sum() > 0 else float("nan")
    delta     = (long_ret - short_ret
                 if not (np.isnan(long_ret) or np.isnan(short_ret))
                 else float("nan"))

    strat_ret  = np.where(long_mask, actual_ret, np.where(short_mask, -actual_ret, 0))
    sharpe     = (strat_ret.mean() / (strat_ret.std() + 1e-8)) * np.sqrt(252 / 5)

    return {
        "delta":     delta,
        "sharpe":    sharpe,
        "long_ret":  long_ret,
        "short_ret": short_ret,
        "pct_long":  long_mask.mean(),
        "pct_short": short_mask.mean(),
        "pct_neut":  (signals == 0).mean(),
        "n_long":    int(long_mask.sum()),
        "n_short":   int(short_mask.sum()),
    }


def fit_xgb_reg(X, y):
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=1,
        gamma=0, random_state=42, verbosity=0,
    )
    model.fit(X, y, verbose=False)
    return model


def run_xgb_reg(df, features, label=""):
    results = []
    for i, sp in enumerate(SPLITS):
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        cols  = list(dict.fromkeys(features + ["target_norm", "return_5d"]))
        train = df[sp["train_start"]:sp["train_end"]][cols].dropna()
        test  = df[sp["test_start"]:test_end][cols].dropna()

        if len(train) < 50 or len(test) < 20:
            print(f"  {label} split_{i+1}: SKIP (train={len(train)}, test={len(test)})")
            continue

        model      = fit_xgb_reg(train[features], train["target_norm"])
        pred       = model.predict(test[features])
        train_pred = model.predict(train[features])

        sigs, p_long, p_short = to_signal_percentile(pred, train_pred)
        ev = evaluate_signals(sigs, test["return_5d"].values)

        results.append({**ev, "split": f"split_{i+1}",
                        "n_train": len(train), "n_test": len(test),
                        "model": model})
        delta_str  = f"{ev['delta']:+.4f}" if not np.isnan(ev["delta"]) else "NaN"
        sharpe_str = f"{ev['sharpe']:+.3f}"
        print(f"  {label} split_{i+1}: "
              f"n={len(train)}/{len(test)} "
              f"delta={delta_str} sharpe={sharpe_str} "
              f"long={ev['pct_long']:.0%} short={ev['pct_short']:.0%} "
              f"neut={ev['pct_neut']:.0%} "
              f"| p_long={p_long:+.4f} p_short={p_short:+.4f}")

    return results


# ── R7b baseline (R5 com XGBRegressor + target normalizado) ──
print("\n" + "=" * 65)
print("R7b BASELINE — R5 + XGBRegressor + target normalizado")
print("=" * 65)

r7b_results = run_xgb_reg(df, FEAT_R5, "R7b")
r7b_delta_mean  = np.nanmean([r["delta"]  for r in r7b_results])
r7b_sharpe_mean = np.nanmean([r["sharpe"] for r in r7b_results])
print(f"  Média: delta={r7b_delta_mean:+.4f} sharpe={r7b_sharpe_mean:+.3f}")

# ── R6-B: R5 + Liquidity + Sentiment ─────────────────────────
print("\n" + "=" * 65)
print("R6-B — R5 + taker_buy_ratio + stablecoin_mcap_zscore + fear_greed")
print(f"  novas: {FEAT_NEW}")
print("=" * 65)

r6b_results = run_xgb_reg(df, FEAT_R6B, "R6B")
r6b_delta_mean  = np.nanmean([r["delta"]  for r in r6b_results])
r6b_sharpe_mean = np.nanmean([r["sharpe"] for r in r6b_results])
print(f"  Média: delta={r6b_delta_mean:+.4f} sharpe={r6b_sharpe_mean:+.3f}")


# ── Veredicto ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("VEREDICTO R6-B")
print("=" * 65)

diff_delta  = r6b_delta_mean  - r7b_delta_mean
diff_sharpe = r6b_sharpe_mean - r7b_sharpe_mean

r7b_deltas  = np.array([r["delta"]  for r in r7b_results])
r6b_deltas  = np.array([r["delta"]  for r in r6b_results])
splits_ok   = int((r6b_deltas > r7b_deltas).sum())
aprovado    = splits_ok >= 2 and diff_delta > 0

print(f"  Delta R7b baseline: {r7b_delta_mean:+.4f}")
print(f"  Delta R6-B:         {r6b_delta_mean:+.4f}  (diff {diff_delta:+.4f})")
print(f"  Sharpe R7b:         {r7b_sharpe_mean:+.3f}")
print(f"  Sharpe R6-B:        {r6b_sharpe_mean:+.3f}  (diff {diff_sharpe:+.3f})")
print(f"  Splits melhorados:  {splits_ok}/3")
print(f"  Decisão: {'✓ APROVADO' if aprovado else '✗ REJEITADO'}")


# ── Tabela comparativa ────────────────────────────────────────
print("\n" + "=" * 65)
print("TABELA COMPARATIVA — R5-class vs R7b-XR vs R6-B")
print("=" * 65)

# R5 referência (valores da sessão anterior)
R5_REF_DELTA  = [+0.0256, +0.0262, +0.0103]
R5_REF_SHARPE = [+1.504,  +1.541,  +0.028]

configs = [
    ("R5 classificação",  R5_REF_DELTA,                         R5_REF_SHARPE),
    ("R7b-XR baseline",   [r["delta"]  for r in r7b_results],   [r["sharpe"] for r in r7b_results]),
    ("R6-B +liq+sent",    [r["delta"]  for r in r6b_results],   [r["sharpe"] for r in r6b_results]),
]

for metric_name, metric_idx in [("DELTA", 1), ("SHARPE", 2)]:
    print(f"\n--- {metric_name}")
    print("{:<22}".format("Config") +
          "".join("{:>12}".format("split_" + str(i+1)) for i in range(3)) +
          "{:>12}".format("Média"))
    print("-" * 58)
    for name, deltas, sharpes in configs:
        vals = deltas if metric_idx == 1 else sharpes
        row  = "{:<22}".format(name)
        for v in vals:
            s = "{:>+12.4f}".format(v) if metric_idx == 1 else "{:>+12.3f}".format(v)
            row += s
        mean = np.nanmean(vals)
        s = "{:>+12.4f}".format(mean) if metric_idx == 1 else "{:>+12.3f}".format(mean)
        row += s
        print(row)

# Distribuição split_3
print(f"\n--- DISTRIBUIÇÃO DE SINAIS (split_3)")
print("{:<22} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
    "Config", "%LONG", "%SHORT", "%NEUT", "n_long", "n_short"))
print("-" * 62)
r7b_s3 = r7b_results[2] if len(r7b_results) >= 3 else {}
r6b_s3 = r6b_results[2] if len(r6b_results) >= 3 else {}
for name, r in [("R7b-XR baseline", r7b_s3), ("R6-B +liq+sent", r6b_s3)]:
    if r:
        print("{:<22} {:>8.0%} {:>8.0%} {:>8.0%} {:>8} {:>8}".format(
            name,
            r["pct_long"], r["pct_short"], r["pct_neut"],
            r["n_long"], r["n_short"],
        ))


# ── Feature importance R6-B split_3 ──────────────────────────
print(f"\n{'=' * 65}")
print("FEATURE IMPORTANCE — R6-B split_3")
print("=" * 65)

sp3    = SPLITS[2]
cols3  = list(dict.fromkeys(FEAT_R6B + ["target_norm", "return_5d"]))
train3 = df[sp3["train_start"]:sp3["train_end"]][cols3].dropna()
xr3    = fit_xgb_reg(train3[FEAT_R6B], train3["target_norm"])

imp = pd.Series(xr3.feature_importances_, index=FEAT_R6B).sort_values(ascending=False)
groups_map = {}
for f in BASE_COLS:      groups_map[f] = "BASE"
for f in FEAT_STRUCTURE: groups_map[f] = "STRUCTURE"
for f in FEAT_MOMENTUM:  groups_map[f] = "MOMENTUM"
for f in FEAT_NEW:       groups_map[f] = "NEW-R6B"

print("\nTop 12 features:")
for feat, val in imp.head(12).items():
    g = groups_map.get(feat, "?")
    print("  {:<35} {:.4f}  [{}]".format(feat, val, g))

print("\nNovas features (posição no ranking):")
for feat in FEAT_NEW:
    rank = list(imp.index).index(feat) + 1
    val  = imp[feat]
    print("  {:<35} {:.4f}  rank={}/{}".format(feat, val, rank, len(FEAT_R6B)))


# ── Output operacional hoje ───────────────────────────────────
print(f"\n{'=' * 65}")
print("OUTPUT OPERACIONAL HOJE")
print("=" * 65)

last = df[df[FEAT_R6B].notna().all(axis=1)].tail(1)
if len(last) > 0:
    train_op_cols = list(dict.fromkeys(FEAT_R6B + ["target_norm"]))
    train_op      = df["2023-01-01":"2024-12-31"][train_op_cols].dropna()

    xr_op      = fit_xgb_reg(train_op[FEAT_R6B], train_op["target_norm"])
    pred_hoje  = xr_op.predict(last[FEAT_R6B])[0]
    train_pred = xr_op.predict(train_op[FEAT_R6B])
    p_long_h   = np.percentile(train_pred, 70)
    p_short_h  = np.percentile(train_pred, 30)

    sig = ("LONG"  if pred_hoje > p_long_h
           else "SHORT"  if pred_hoje < p_short_h
           else "NEUTRAL")

    regime = "BULL" if last["hmm_state"].iloc[0] == 1 else "BEAR"
    bull_p = float(last["bull_prob"].iloc[0])

    print(f"\n  Data:                    {last.index[0].date()}")
    print(f"  Regime HMM:              {regime}")
    print(f"  Bull probability:        {bull_p:.1%}")
    print(f"\n  expected_norm_ret:       {pred_hoje:+.4f}")
    print(f"  Threshold LONG  (p70):   {p_long_h:+.4f}")
    print(f"  Threshold SHORT (p30):   {p_short_h:+.4f}")
    print(f"  SINAL:                   {sig}")

    print(f"\n  Contexto Coinglass hoje:")
    for feat in FEAT_NEW:
        val = last[feat].iloc[0]
        if pd.notna(val):
            print("    {:<35} {:+.4f}".format(feat, val))
        else:
            print("    {:<35} NaN".format(feat))
else:
    print("\n  AVISO: sem dados válidos para output operacional.")
    avail = df[FEAT_R6B].notna().all(axis=1).sum()
    print(f"  Rows com todas features válidas: {avail}")
