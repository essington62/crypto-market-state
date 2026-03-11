"""
R11 — Ensemble R8-A + Feature Pruning R9 (TOP10).

Hipótese: ensemble (R5 classificação + R7b regressão, TOP10 features)
melhora delta e Sharpe vs R7b baseline e R9 TOP10.

Ensemble rule (R8-A):
  LONG    se R5 == LONG  AND R7b_pred > p70
  SHORT   se R5 == SHORT AND R7b_pred < p30
  NEUTRAL caso contrário

Baselines:
  1. R7b   — XGBRegressor, 21 features, sem ensemble
  2. R9    — XGBRegressor, TOP10 features, sem ensemble
  3. R11   — XGBRegressor, TOP10 features, ensemble R8-A

Modelo: XGBRegressor (config R7b)
Target: return_5d / vol_short (vol_21)
Threshold: p70/p30 calibrado no treino de cada split
Walk-forward: 3 splits (split_1 H1-2024, split_2 H2-2024, split_3 2025→now)

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

# TOP10 fixo da R9 (feature importance ordenada por split_3 do R7b)
TOP10 = [c for c in [
    "atr_14_norm", "bb_width_20d", "vol_ratio", "roc_21",
    "slope_21d", "bb_position", "price_vs_high_30d",
    "range_position_30d", "drawdown", "macd_hist_norm",
] if c in df.columns]

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]

SHARPE_ANN = np.sqrt(252 / 5)

print(f"R7B_FEATURES: {len(R7B_FEATURES)}")
print(f"TOP10:        {len(TOP10)} → {TOP10}")


# ── Model factories ────────────────────────────────────────────
def fit_r5_clf(X, y):
    """XGBClassifier configuração R5."""
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
    """XGBRegressor configuração R7b."""
    reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=1,
        gamma=0, random_state=42, verbosity=0,
    )
    reg.fit(X, y, verbose=False)
    return reg


# ── Evaluation helpers ─────────────────────────────────────────
def evaluate_signals(signals, actual_ret):
    """Delta e Sharpe a partir de array de sinais (+1/0/-1) e retornos brutos."""
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


def evaluate_regime(signals, actual_ret, regimes):
    """Delta e Sharpe separados por bull (1) e bear (0)."""
    out = {}
    for label, val in [("bull", 1), ("bear", 0)]:
        mask = regimes == val
        if mask.sum() < 5:
            out[label] = {"delta": float("nan"), "sharpe": float("nan"), "n": int(mask.sum())}
            continue
        ev = evaluate_signals(signals[mask], actual_ret[mask])
        out[label] = {"delta": ev["delta"], "sharpe": ev["sharpe"], "n": int(mask.sum())}
    return out


# ── Walk-forward: regressão simples (R7b ou R9 TOP10) ─────────
def run_regressor(features, label=""):
    """Retorna lista de dicts com ev + sigs/actual/regimes por split."""
    results = []
    for i, sp in enumerate(SPLITS):
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        req_feats = features + ["target_norm", "return_5d"]
        cols      = list(dict.fromkeys(features + ["target_norm", "return_5d", "hmm_state"]))

        train = df[sp["train_start"]:sp["train_end"]][cols].dropna(subset=req_feats)
        test  = df[sp["test_start"]:test_end][cols].dropna(subset=features + ["return_5d"])

        if len(train) < 50 or len(test) < 20:
            print(f"  {label} split_{i+1}: SKIP (n_train={len(train)}, n_test={len(test)})")
            results.append(None)
            continue

        reg        = fit_r7b_reg(train[features], train["target_norm"])
        pred       = reg.predict(test[features])
        train_pred = reg.predict(train[features])

        p_long  = np.percentile(train_pred, 70)
        p_sht   = np.percentile(train_pred, 30)
        sigs    = np.where(pred > p_long, 1, np.where(pred < p_sht, -1, 0))
        actual  = test["return_5d"].values
        regimes = test["hmm_state"].fillna(-1).astype(int).values
        ev      = evaluate_signals(sigs, actual)
        results.append({"ev": ev, "sigs": sigs, "actual": actual, "regimes": regimes})

        print(f"  {label} split_{i+1}: n={len(train)}/{len(test)} "
              f"delta={ev['delta']:+.4f} sharpe={ev['sharpe']:+.3f} "
              f"long={ev['pct_long']:.0%} short={ev['pct_short']:.0%} "
              f"neut={ev['pct_neut']:.0%}")
    return results


# ── Walk-forward: R11 ensemble (R8-A rule + TOP10) ────────────
def run_r11():
    """R5 classificador (TOP10) + R7b regressor (TOP10), regra R8-A."""
    results = []
    for i, sp in enumerate(SPLITS):
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        aux      = ["label_5d", "target_norm", "return_5d", "hmm_state"]
        cols     = list(dict.fromkeys(TOP10 + aux))

        train = df[sp["train_start"]:sp["train_end"]][cols].dropna(
            subset=TOP10 + ["label_5d", "target_norm", "return_5d"])
        test  = df[sp["test_start"]:test_end][cols].dropna(
            subset=TOP10 + ["return_5d"])

        if len(train) < 50 or len(test) < 20:
            print(f"  R11 split_{i+1}: SKIP (n_train={len(train)}, n_test={len(test)})")
            results.append(None)
            continue

        # R5 — classificador
        clf  = fit_r5_clf(train[TOP10], train["label_5d"])
        prob = clf.predict_proba(test[TOP10])[:, 1]
        sig_r5 = np.where(prob > 0.60, 1, np.where(prob < 0.40, -1, 0))

        # R7b — regressor com TOP10
        reg        = fit_r7b_reg(train[TOP10], train["target_norm"])
        pred       = reg.predict(test[TOP10])
        train_pred = reg.predict(train[TOP10])

        p_long  = np.percentile(train_pred, 70)
        p_sht   = np.percentile(train_pred, 30)
        sig_r7b = np.where(pred > p_long, 1, np.where(pred < p_sht, -1, 0))

        # Regra R8-A: ambos concordam na direção forte
        sigs                                   = np.zeros(len(sig_r5), dtype=int)
        sigs[(sig_r5 == 1)  & (sig_r7b == 1)] = 1
        sigs[(sig_r5 == -1) & (sig_r7b == -1)] = -1

        actual  = test["return_5d"].values
        regimes = test["hmm_state"].fillna(-1).astype(int).values
        ev      = evaluate_signals(sigs, actual)
        results.append({"ev": ev, "sigs": sigs, "actual": actual, "regimes": regimes})

        print(f"  R11 split_{i+1}: n={len(train)}/{len(test)} "
              f"delta={ev['delta']:+.4f} sharpe={ev['sharpe']:+.3f} "
              f"long={ev['pct_long']:.0%} short={ev['pct_short']:.0%} "
              f"neut={ev['pct_neut']:.0%}")
    return results


# ══════════════════════════════════════════════════════════════
# EXECUÇÃO
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 68)
print("R7b BASELINE — 21 features, sem ensemble")
print("=" * 68)
res_r7b = run_regressor(R7B_FEATURES, "R7b")
r7b_dm  = np.nanmean([r["ev"]["delta"]  for r in res_r7b if r])
r7b_sm  = np.nanmean([r["ev"]["sharpe"] for r in res_r7b if r])
print(f"  Média: delta={r7b_dm:+.4f} sharpe={r7b_sm:+.3f}")

print("\n" + "=" * 68)
print("R9 TOP10 — 10 features, sem ensemble")
print("=" * 68)
res_r9  = run_regressor(TOP10, "R9 ")
r9_dm   = np.nanmean([r["ev"]["delta"]  for r in res_r9 if r])
r9_sm   = np.nanmean([r["ev"]["sharpe"] for r in res_r9 if r])
print(f"  Média: delta={r9_dm:+.4f} sharpe={r9_sm:+.3f}")

print("\n" + "=" * 68)
print("R11 — Ensemble R8-A + TOP10")
print("=" * 68)
res_r11 = run_r11()
r11_dm  = np.nanmean([r["ev"]["delta"]  for r in res_r11 if r])
r11_sm  = np.nanmean([r["ev"]["sharpe"] for r in res_r11 if r])
print(f"  Média: delta={r11_dm:+.4f} sharpe={r11_sm:+.3f}")


# ── Tabela comparativa ────────────────────────────────────────
all_configs = [
    ("R7b (21 feats)",       res_r7b, r7b_dm, r7b_sm),
    ("R9 TOP10",             res_r9,  r9_dm,  r9_sm),
    ("R11 Ensemble+TOP10",   res_r11, r11_dm, r11_sm),
]

print("\n" + "=" * 68)
print("TABELA COMPARATIVA")
print("=" * 68)

for metric_name, key in [("DELTA", "delta"), ("SHARPE (ann. sqrt(252/5))", "sharpe")]:
    print(f"\n--- {metric_name}")
    hdr = f"{'Config':<22}" + "".join(f"{'split_'+str(i+1):>12}" for i in range(3))
    hdr += f"{'Média':>12}{'vs R7b':>10}"
    print(hdr)
    print("-" * 68)
    for name, res, dm, sm in all_configs:
        row = f"{name:<22}"
        for r in res:
            v = r["ev"][key] if r else float("nan")
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


# ── Diagnóstico de sinais — split_3 ──────────────────────────
print(f"\n--- DIAGNÓSTICO DE SINAIS (split_3)")
print(f"{'Config':<22} {'%LONG':>8} {'%SHORT':>8} {'%NEUT':>8} {'n_long':>8} {'n_short':>8}")
print("-" * 68)
for name, res, _, __ in all_configs:
    r3 = res[2] if len(res) >= 3 else None
    if not r3:
        print(f"{name:<22}  sem dados split_3")
        continue
    ev = r3["ev"]
    print(f"{name:<22} "
          f"{ev['pct_long']:>8.0%} {ev['pct_short']:>8.0%} {ev['pct_neut']:>8.0%} "
          f"{ev['n_long']:>8} {ev['n_short']:>8}")


# ── Stress test por regime ────────────────────────────────────
print(f"\n{'=' * 68}")
print("STRESS TEST — BULL vs BEAR (pooled splits 1-3)")
print("=" * 68)


def pool_results(res_list):
    """Concatena sigs, actual, regimes de todos os splits válidos."""
    all_sigs, all_actual, all_regimes = [], [], []
    for r in res_list:
        if r is None:
            continue
        all_sigs.append(r["sigs"])
        all_actual.append(r["actual"])
        all_regimes.append(r["regimes"])
    if not all_sigs:
        return None, None, None
    return (np.concatenate(all_sigs),
            np.concatenate(all_actual),
            np.concatenate(all_regimes))


print(f"\n{'Config':<22} {'Regime':>8} {'delta':>10} {'sharpe':>10} {'n':>6}")
print("-" * 60)
for name, res, _, __ in all_configs:
    sigs, actual, regimes = pool_results(res)
    if sigs is None:
        continue
    reg_ev = evaluate_regime(sigs, actual, regimes)
    for regime_label in ("bull", "bear"):
        rv = reg_ev[regime_label]
        d_str = f"{rv['delta']:>+10.4f}" if not np.isnan(rv["delta"]) else f"{'NaN':>10}"
        s_str = f"{rv['sharpe']:>+10.3f}" if not np.isnan(rv["sharpe"]) else f"{'NaN':>10}"
        print(f"{name:<22} {regime_label:>8} {d_str} {s_str} {rv['n']:>6}")
    print()


# ── Veredicto final ───────────────────────────────────────────
print("=" * 68)
print("VEREDICTO")
print("=" * 68)

for name, res, dm, sm in all_configs[1:]:
    valid = [r for r in res if r]
    r7b_valid = [r for r in res_r7b if r]
    n_beats = sum(
        r["ev"]["delta"] > rb["ev"]["delta"]
        for r, rb in zip(valid, r7b_valid)
    )
    delta_diff  = dm - r7b_dm
    sharpe_diff = sm - r7b_sm
    aprovado = n_beats >= 2 and delta_diff > 0
    print(f"  {name:<22} delta={dm:+.4f} ({delta_diff:+.4f} vs R7b) "
          f"sharpe={sm:+.3f} ({sharpe_diff:+.3f}) "
          f"splits_beat={n_beats}/3 → {'✓ APROVADO' if aprovado else '✗ REJEITADO'}")
