"""
R11 — Validação de robustez com split_4.

Adiciona split_4 (test 2025-04-01 → now) aos 3 splits existentes.
Lógica idêntica ao xgb_r11_ensemble_pruning.py — sem modificações.

Configs comparadas:
  R7b  — XGBRegressor, 21 features, sem ensemble
  R9   — XGBRegressor, TOP10 features, sem ensemble
  R11  — XGBRegressor, TOP10 features, ensemble R8-A

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
df["label_5d"]    = (df["return_5d"] > 0).astype(int)
df["target_norm"] = df["return_5d"] / df["vol_short"].replace(0, np.nan)

print(f"Dataset: {df.shape} | {df.index.min().date()} → {df.index.max().date()}")


# ── Feature sets ───────────────────────────────────────────────
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

R7B_FEATURES = BASE_COLS + FEAT_STRUCTURE + FEAT_MOMENTUM  # 21 features

TOP10 = [c for c in [
    "atr_14_norm", "bb_width_20d", "vol_ratio", "roc_21",
    "slope_21d", "bb_position", "price_vs_high_30d",
    "range_position_30d", "drawdown", "macd_hist_norm",
] if c in df.columns]

SPLITS = [
    {"name": "split_1", "train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start": "2024-01-01", "test_end": "2024-06-30"},
    {"name": "split_2", "train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start": "2024-07-01", "test_end": "2024-12-31"},
    {"name": "split_3", "train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start": "2025-01-01", "test_end": "2025-03-31"},
    {"name": "split_4", "train_start": "2023-01-01", "train_end": "2025-03-31",
     "test_start": "2025-04-01", "test_end": None},
]

SHARPE_ANN = np.sqrt(252 / 5)

print(f"R7B_FEATURES: {len(R7B_FEATURES)} | TOP10: {len(TOP10)}")
print(f"Splits: {[s['name'] for s in SPLITS]}")


# ── Model factories ────────────────────────────────────────────
def fit_r5_clf(X, y):
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=3,
        gamma=0.5, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    clf.fit(X, y, verbose=False)
    return clf


def fit_r7b_reg(X, y):
    reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=1,
        gamma=0, random_state=42, verbosity=0,
    )
    reg.fit(X, y, verbose=False)
    return reg


# ── Evaluation ────────────────────────────────────────────────
def evaluate_signals(signals, actual_ret):
    long_mask  = signals == 1
    short_mask = signals == -1
    long_ret   = actual_ret[long_mask].mean()  if long_mask.sum()  > 0 else float("nan")
    short_ret  = actual_ret[short_mask].mean() if short_mask.sum() > 0 else float("nan")
    delta      = (long_ret - short_ret
                  if not (np.isnan(long_ret) or np.isnan(short_ret)) else float("nan"))
    strat_ret  = np.where(long_mask, actual_ret, np.where(short_mask, -actual_ret, 0.0))
    sharpe     = (strat_ret.mean() / (strat_ret.std() + 1e-8)) * SHARPE_ANN
    return {
        "delta":     delta,
        "sharpe":    sharpe,
        "pct_long":  float(long_mask.mean()),
        "pct_short": float(short_mask.mean()),
        "pct_neut":  float((signals == 0).mean()),
        "n_long":    int(long_mask.sum()),
        "n_short":   int(short_mask.sum()),
    }


# ── Walk-forward: regressão simples ───────────────────────────
def run_regressor(features, label=""):
    results = []
    for sp in SPLITS:
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        req      = features + ["target_norm", "return_5d"]
        cols     = list(dict.fromkeys(features + ["target_norm", "return_5d"]))

        train = df[sp["train_start"]:sp["train_end"]][cols].dropna(subset=req)
        test  = df[sp["test_start"]:test_end][cols].dropna(subset=features + ["return_5d"])

        if len(train) < 50 or len(test) < 10:
            print(f"  {label} {sp['name']}: SKIP (n_train={len(train)}, n_test={len(test)})")
            results.append(None)
            continue

        reg        = fit_r7b_reg(train[features], train["target_norm"])
        pred       = reg.predict(test[features])
        train_pred = reg.predict(train[features])

        p_long = np.percentile(train_pred, 70)
        p_sht  = np.percentile(train_pred, 30)
        sigs   = np.where(pred > p_long, 1, np.where(pred < p_sht, -1, 0))
        actual = test["return_5d"].values
        ev     = evaluate_signals(sigs, actual)
        results.append(ev)

        print(f"  {label} {sp['name']}: n={len(train)}/{len(test)} "
              f"delta={ev['delta']:+.4f} sharpe={ev['sharpe']:+.3f} "
              f"long={ev['pct_long']:.0%} short={ev['pct_short']:.0%} "
              f"neut={ev['pct_neut']:.0%}")
    return results


# ── Walk-forward: R11 ensemble ────────────────────────────────
def run_r11():
    results = []
    for sp in SPLITS:
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        aux  = ["label_5d", "target_norm", "return_5d"]
        cols = list(dict.fromkeys(TOP10 + aux))

        train = df[sp["train_start"]:sp["train_end"]][cols].dropna(
            subset=TOP10 + ["label_5d", "target_norm", "return_5d"])
        test  = df[sp["test_start"]:test_end][cols].dropna(
            subset=TOP10 + ["return_5d"])

        if len(train) < 50 or len(test) < 10:
            print(f"  R11 {sp['name']}: SKIP (n_train={len(train)}, n_test={len(test)})")
            results.append(None)
            continue

        # R5 classificador (TOP10)
        clf    = fit_r5_clf(train[TOP10], train["label_5d"])
        prob   = clf.predict_proba(test[TOP10])[:, 1]
        sig_r5 = np.where(prob > 0.60, 1, np.where(prob < 0.40, -1, 0))

        # R7b regressor (TOP10)
        reg        = fit_r7b_reg(train[TOP10], train["target_norm"])
        pred       = reg.predict(test[TOP10])
        train_pred = reg.predict(train[TOP10])

        p_long  = np.percentile(train_pred, 70)
        p_sht   = np.percentile(train_pred, 30)
        sig_r7b = np.where(pred > p_long, 1, np.where(pred < p_sht, -1, 0))

        # Regra R8-A
        sigs                                    = np.zeros(len(sig_r5), dtype=int)
        sigs[(sig_r5 == 1)  & (sig_r7b == 1)]  = 1
        sigs[(sig_r5 == -1) & (sig_r7b == -1)] = -1

        actual = test["return_5d"].values
        ev     = evaluate_signals(sigs, actual)
        results.append(ev)

        print(f"  R11 {sp['name']}: n={len(train)}/{len(test)} "
              f"delta={ev['delta']:+.4f} sharpe={ev['sharpe']:+.3f} "
              f"long={ev['pct_long']:.0%} short={ev['pct_short']:.0%} "
              f"neut={ev['pct_neut']:.0%}")
    return results


# ══════════════════════════════════════════════════════════════
# EXECUÇÃO
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("R7b BASELINE — 21 features")
print("=" * 70)
res_r7b = run_regressor(R7B_FEATURES, "R7b")
r7b_dm  = np.nanmean([r["delta"]  for r in res_r7b if r])
r7b_sm  = np.nanmean([r["sharpe"] for r in res_r7b if r])
print(f"  Média: delta={r7b_dm:+.4f} sharpe={r7b_sm:+.3f}")

print("\n" + "=" * 70)
print("R9 TOP10 — sem ensemble")
print("=" * 70)
res_r9 = run_regressor(TOP10, "R9 ")
r9_dm  = np.nanmean([r["delta"]  for r in res_r9 if r])
r9_sm  = np.nanmean([r["sharpe"] for r in res_r9 if r])
print(f"  Média: delta={r9_dm:+.4f} sharpe={r9_sm:+.3f}")

print("\n" + "=" * 70)
print("R11 — Ensemble R8-A + TOP10")
print("=" * 70)
res_r11 = run_r11()
r11_dm  = np.nanmean([r["delta"]  for r in res_r11 if r])
r11_sm  = np.nanmean([r["sharpe"] for r in res_r11 if r])
print(f"  Média: delta={r11_dm:+.4f} sharpe={r11_sm:+.3f}")


# ── Tabela comparativa ────────────────────────────────────────
n_splits = len(SPLITS)
all_configs = [
    ("R7b (21 feats)",     res_r7b, r7b_dm, r7b_sm),
    ("R9 TOP10",           res_r9,  r9_dm,  r9_sm),
    ("R11 Ensemble+TOP10", res_r11, r11_dm, r11_sm),
]

print("\n" + "=" * 70)
print("TABELA COMPARATIVA")
print("=" * 70)

for metric_name, key in [("DELTA", "delta"), ("SHARPE (ann. sqrt(252/5))", "sharpe")]:
    print(f"\n--- {metric_name}")
    hdr = f"{'Config':<22}"
    for sp in SPLITS:
        hdr += f"{sp['name']:>12}"
    hdr += f"{'Média':>12}{'vs R7b':>10}"
    print(hdr)
    print("-" * (22 + 12 * n_splits + 22))
    for name, res, dm, sm in all_configs:
        row = f"{name:<22}"
        for r in res:
            v   = r[key] if r else float("nan")
            fmt = f"{v:>+12.4f}" if key == "delta" else f"{v:>+12.3f}"
            row += fmt
        mean = dm if key == "delta" else sm
        ref  = r7b_dm if key == "delta" else r7b_sm
        diff = mean - ref
        if key == "delta":
            row += f"{mean:>+12.4f}{diff:>+10.4f}"
        else:
            row += f"{mean:>+12.3f}{diff:>+10.3f}"
        print(row)


# ── Diagnóstico split_4 ───────────────────────────────────────
print(f"\n--- DIAGNÓSTICO DE SINAIS (split_4)")
print(f"{'Config':<22} {'%LONG':>8} {'%SHORT':>8} {'%NEUT':>8} {'n_long':>8} {'n_short':>8}")
print("-" * 66)
for name, res, _, __ in all_configs:
    r4 = res[3] if len(res) >= 4 else None
    if not r4:
        print(f"{name:<22}  sem dados split_4")
        continue
    print(f"{name:<22} "
          f"{r4['pct_long']:>8.0%} {r4['pct_short']:>8.0%} {r4['pct_neut']:>8.0%} "
          f"{r4['n_long']:>8} {r4['n_short']:>8}")


# ── Veredicto ─────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("VEREDICTO (vs R7b, critério: média_delta > 0 e 2+/4 splits batidos)")
print("=" * 70)
for name, res, dm, sm in all_configs[1:]:
    valid_r = [r for r in res     if r]
    valid_b = [r for r in res_r7b if r]
    n_beats = sum(
        r["delta"] > rb["delta"]
        for r, rb in zip(valid_r, valid_b)
    )
    d_diff = dm - r7b_dm
    s_diff = sm - r7b_sm
    ok     = n_beats >= 2 and d_diff > 0
    print(f"  {name:<22} delta={dm:+.4f} ({d_diff:+.4f} vs R7b) "
          f"sharpe={sm:+.3f} ({s_diff:+.3f}) "
          f"splits_beat={n_beats}/{len(valid_r)} → {'✓ APROVADO' if ok else '✗ REJEITADO'}")
