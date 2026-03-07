"""
R9 — XGBoost Feature Pruning.

Hipótese: apenas 5–8 features carregam a maior parte do sinal.
Testar TOP5, TOP10, TOP15 por importância (do R7b split_3).

Modelo: XGBRegressor (config R7b)
Target: return_5d / vol_short
Conversão: percentil 70/30 calibrado no treino
Baseline: R7b delta=+0.0267 Sharpe=+1.118

Script exploratório — NAO modifica pipeline nem sobrescreve modelos.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ── Dados ─────────────────────────────────────────────────────
btc = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
hmm = pd.read_parquet("data/05_models/regime_hmm/btc_states.parquet")
hmm = hmm.rename(columns={"state": "hmm_state"})

df = btc.copy()
df = df.join(hmm[["hmm_state", "bull_prob"]], how="left")
df["regime_strength"] = (df["bull_prob"] - 0.5).abs()
df["return_5d"]   = df["log_return"].rolling(5).sum().shift(-5)
df["target_norm"] = df["return_5d"] / df["vol_short"].replace(0, np.nan)

# ── Features R5 (21 total) ─────────────────────────────────────
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

FEATURES = BASE_COLS + FEAT_STRUCTURE + FEAT_MOMENTUM

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]

SHARPE_ANN = np.sqrt(252 / 5)


def fit_xgb_reg(X, y):
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=1,
        gamma=0, random_state=42, verbosity=0,
    )
    model.fit(X, y, verbose=False)
    return model


def evaluate_signals(signals, actual_ret):
    long_mask  = signals == 1
    short_mask = signals == -1
    long_ret  = actual_ret[long_mask].mean()  if long_mask.sum()  > 0 else float("nan")
    short_ret = actual_ret[short_mask].mean() if short_mask.sum() > 0 else float("nan")
    delta     = (long_ret - short_ret
                 if not (np.isnan(long_ret) or np.isnan(short_ret)) else float("nan"))
    strat_ret = np.where(long_mask, actual_ret, np.where(short_mask, -actual_ret, 0))
    sharpe    = (strat_ret.mean() / (strat_ret.std() + 1e-8)) * SHARPE_ANN
    return {
        "delta": delta, "sharpe": sharpe,
        "pct_long": long_mask.mean(), "pct_short": short_mask.mean(),
        "pct_neut": (signals == 0).mean(),
        "n_long": int(long_mask.sum()), "n_short": int(short_mask.sum()),
    }


def run_walk_forward(df, features, label=""):
    results = []
    for i, sp in enumerate(SPLITS):
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        cols  = list(dict.fromkeys(features + ["target_norm", "return_5d"]))
        train = df[sp["train_start"]:sp["train_end"]][cols].dropna()
        test  = df[sp["test_start"]:test_end][cols].dropna()

        if len(train) < 50 or len(test) < 20:
            print(f"  {label} split_{i+1}: SKIP")
            continue

        model      = fit_xgb_reg(train[features], train["target_norm"])
        pred       = model.predict(test[features])
        train_pred = model.predict(train[features])

        p_long = np.percentile(train_pred, 70)
        p_sht  = np.percentile(train_pred, 30)
        sigs   = np.where(pred > p_long, 1, np.where(pred < p_sht, -1, 0))
        ev     = evaluate_signals(sigs, test["return_5d"].values)
        results.append(ev)

        d_str = f"{ev['delta']:+.4f}" if not np.isnan(ev["delta"]) else "NaN"
        print(f"  {label} split_{i+1}: "
              f"n={len(train)}/{len(test)} "
              f"delta={d_str} sharpe={ev['sharpe']:+.3f} "
              f"long={ev['pct_long']:.0%} short={ev['pct_short']:.0%} "
              f"neut={ev['pct_neut']:.0%}")
    return results


# ── Passo 1: feature importance R7b split_3 ───────────────────
print("=" * 65)
print("PASSO 1 — Feature importance R7b (split_3 train: 2023-2024)")
print("=" * 65)

sp3    = SPLITS[2]
cols3  = list(dict.fromkeys(FEATURES + ["target_norm", "return_5d"]))
train3 = df[sp3["train_start"]:sp3["train_end"]][cols3].dropna()

model3 = fit_xgb_reg(train3[FEATURES], train3["target_norm"])
imp    = pd.Series(model3.feature_importances_, index=FEATURES).sort_values(ascending=False)

groups_map = {}
for f in BASE_COLS:      groups_map[f] = "BASE"
for f in FEAT_STRUCTURE: groups_map[f] = "STRUCTURE"
for f in FEAT_MOMENTUM:  groups_map[f] = "MOMENTUM"

print(f"\nRanking completo ({len(FEATURES)} features):")
for rank, (feat, val) in enumerate(imp.items(), 1):
    g   = groups_map.get(feat, "?")
    bar = "█" * int(val * 200)
    print(f"  {rank:>2}. {feat:<35} {val:.4f}  [{g}]  {bar}")

# Definir subsets
top5_feats  = imp.index[:5].tolist()
top10_feats = imp.index[:10].tolist()
top15_feats = imp.index[:15].tolist()

print(f"\nTOP5:  {top5_feats}")
print(f"TOP10: {top10_feats}")
print(f"TOP15: {top15_feats}")


# ── Passo 2: R7b baseline ─────────────────────────────────────
print("\n" + "=" * 65)
print("R7b BASELINE — 21 features")
print("=" * 65)
r7b_results = run_walk_forward(df, FEATURES, "R7b")
r7b_dm = np.nanmean([r["delta"]  for r in r7b_results])
r7b_sm = np.nanmean([r["sharpe"] for r in r7b_results])
print(f"  Média: delta={r7b_dm:+.4f} sharpe={r7b_sm:+.3f}")


# ── Passo 3: versões podadas ──────────────────────────────────
configs_pruned = [
    ("TOP5",  top5_feats),
    ("TOP10", top10_feats),
    ("TOP15", top15_feats),
]

all_results = [("R7b (21 feats)", r7b_results, r7b_dm, r7b_sm)]

for name, feats in configs_pruned:
    print(f"\n{'=' * 65}")
    print(f"R9-{name} — {len(feats)} features")
    print(f"  {feats}")
    print("=" * 65)
    res = run_walk_forward(df, feats, name)
    dm  = np.nanmean([r["delta"]  for r in res])
    sm  = np.nanmean([r["sharpe"] for r in res])
    print(f"  Média: delta={dm:+.4f} sharpe={sm:+.3f}")
    all_results.append((f"R9-{name}", res, dm, sm))


# ── Tabela comparativa ────────────────────────────────────────
print("\n" + "=" * 65)
print("TABELA COMPARATIVA — R9 Feature Pruning")
print("=" * 65)

for metric_name, key in [("DELTA", "delta"), ("SHARPE", "sharpe")]:
    print(f"\n--- {metric_name}")
    print("{:<18}".format("Config") +
          "".join("{:>12}".format("split_" + str(i + 1)) for i in range(3)) +
          "{:>12}".format("Média") + "{:>10}".format("vs R7b"))
    print("-" * 64)
    for name, res, dm, sm in all_results:
        vals = [r[key] for r in res]
        mean = dm if key == "delta" else sm
        ref  = r7b_dm if key == "delta" else r7b_sm
        diff = mean - ref
        row  = "{:<18}".format(name)
        for v in vals:
            fmt = "{:>+12.4f}" if key == "delta" else "{:>+12.3f}"
            row += fmt.format(v)
        fmt  = "{:>+12.4f} {:>+10.4f}" if key == "delta" else "{:>+12.3f} {:>+10.3f}"
        row += fmt.format(mean, diff)
        print(row)

# Distribuição split_3
print(f"\n--- DISTRIBUIÇÃO SINAIS (split_3)")
print("{:<18} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
    "Config", "%LONG", "%SHORT", "%NEUT", "n_long", "n_short"))
print("-" * 58)
for name, res, _, __ in all_results:
    r = res[2] if len(res) >= 3 else {}
    if r:
        print("{:<18} {:>8.0%} {:>8.0%} {:>8.0%} {:>8} {:>8}".format(
            name, r["pct_long"], r["pct_short"], r["pct_neut"],
            r["n_long"], r["n_short"]))

# Veredicto
print("\n--- VEREDICTO")
for name, res, dm, sm in all_results[1:]:
    splits_ok = int(sum(r["delta"] > r7b_results[j]["delta"]
                        for j, r in enumerate(res)
                        if j < len(r7b_results)))
    diff = dm - r7b_dm
    aprovado = splits_ok >= 2 and diff > 0
    print(f"  {name:<12} delta={dm:+.4f} (diff {diff:+.4f}) "
          f"splits_ok={splits_ok}/3 → {'✓ APROVADO' if aprovado else '✗ REJEITADO'}")
