"""
R8 — Ensemble R5 (classificação) + R7b (regressão normalizada).

Três regras de combinação:
  Regra A — Confirmação:  ambos concordam na direção forte
  Regra B — Filtro:       R5 define direção, R7b veta contradição
  Regra C — Score:        score = 0.6*prob_R5 + 0.4*rank_norm(R7b_pred)

Baseline:
  R5  Classificação: delta=+0.0207  Sharpe=+1.02
  R7b XGBRegressor:  delta=+0.0267  Sharpe=+1.12

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
df["return_5d"]  = df["log_return"].rolling(5).sum().shift(-5)
df["label_5d"]   = (df["return_5d"] > 0).astype(int)
df["target_norm"] = df["return_5d"] / df["vol_short"].replace(0, np.nan)

# ── Features R5 ───────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────
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


def rank_norm(test_vals, train_vals):
    """Percentile rank of test_vals w.r.t. training distribution → [0, 1]."""
    sorted_train = np.sort(train_vals)
    ranks = np.searchsorted(sorted_train, test_vals, side="right") / len(sorted_train)
    return np.clip(ranks, 0.0, 1.0)


def fit_r5(X, y):
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=3,
        gamma=0.5, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    clf.fit(X, y, verbose=False)
    return clf


def fit_r7b(X, y):
    reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=1,
        gamma=0, random_state=42, verbosity=0,
    )
    reg.fit(X, y, verbose=False)
    return reg


# ── Walk-forward ──────────────────────────────────────────────
print(f"Features R5: {len(FEATURES)}")

# Baselines individuais
print("\n" + "=" * 65)
print("BASELINES INDIVIDUAIS")
print("=" * 65)

r5_results   = []
r7b_results  = []

for i, sp in enumerate(SPLITS):
    test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
    cols_cls = list(dict.fromkeys(FEATURES + ["label_5d", "return_5d"]))
    cols_reg = list(dict.fromkeys(FEATURES + ["target_norm", "return_5d"]))

    train_cls = df[sp["train_start"]:sp["train_end"]][cols_cls].dropna()
    test_cls  = df[sp["test_start"]:test_end][cols_cls].dropna()
    train_reg = df[sp["train_start"]:sp["train_end"]][cols_reg].dropna()
    test_reg  = df[sp["test_start"]:test_end][cols_reg].dropna()

    # Alinhar índices para comparação
    common_idx = test_cls.index.intersection(test_reg.index)
    test_cls   = test_cls.loc[common_idx]
    test_reg   = test_reg.loc[common_idx]

    clf    = fit_r5(train_cls[FEATURES], train_cls["label_5d"])
    reg    = fit_r7b(train_reg[FEATURES], train_reg["target_norm"])

    prob   = clf.predict_proba(test_cls[FEATURES])[:, 1]
    pred   = reg.predict(test_reg[FEATURES])
    tp     = reg.predict(train_reg[FEATURES])
    p_long = np.percentile(tp, 70)
    p_sht  = np.percentile(tp, 30)

    sig_r5  = np.where(prob > 0.60, 1, np.where(prob < 0.40, -1, 0))
    sig_r7b = np.where(pred > p_long, 1, np.where(pred < p_sht, -1, 0))

    actual = test_cls["return_5d"].values
    ev_r5  = evaluate_signals(sig_r5,  actual)
    ev_r7b = evaluate_signals(sig_r7b, actual)
    r5_results.append(ev_r5)
    r7b_results.append(ev_r7b)

    print(f"  split_{i+1}: "
          f"R5 delta={ev_r5['delta']:+.4f} sh={ev_r5['sharpe']:+.3f} "
          f"| R7b delta={ev_r7b['delta']:+.4f} sh={ev_r7b['sharpe']:+.3f}")

r5_dm  = np.nanmean([r["delta"]  for r in r5_results])
r5_sm  = np.nanmean([r["sharpe"] for r in r5_results])
r7b_dm = np.nanmean([r["delta"]  for r in r7b_results])
r7b_sm = np.nanmean([r["sharpe"] for r in r7b_results])
print(f"  Média R5:  delta={r5_dm:+.4f} sharpe={r5_sm:+.3f}")
print(f"  Média R7b: delta={r7b_dm:+.4f} sharpe={r7b_sm:+.3f}")


# ── Ensemble rules ────────────────────────────────────────────
def run_ensemble(rule_fn, rule_name):
    results = []
    for i, sp in enumerate(SPLITS):
        test_end  = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        cols_cls  = list(dict.fromkeys(FEATURES + ["label_5d", "return_5d"]))
        cols_reg  = list(dict.fromkeys(FEATURES + ["target_norm", "return_5d"]))

        train_cls = df[sp["train_start"]:sp["train_end"]][cols_cls].dropna()
        test_cls  = df[sp["test_start"]:test_end][cols_cls].dropna()
        train_reg = df[sp["train_start"]:sp["train_end"]][cols_reg].dropna()
        test_reg  = df[sp["test_start"]:test_end][cols_reg].dropna()

        common_idx = test_cls.index.intersection(test_reg.index)
        test_cls   = test_cls.loc[common_idx]
        test_reg   = test_reg.loc[common_idx]

        clf = fit_r5(train_cls[FEATURES], train_cls["label_5d"])
        reg = fit_r7b(train_reg[FEATURES], train_reg["target_norm"])

        prob       = clf.predict_proba(test_cls[FEATURES])[:, 1]
        pred       = reg.predict(test_reg[FEATURES])
        train_pred = reg.predict(train_reg[FEATURES])

        p_long = np.percentile(train_pred, 70)
        p_sht  = np.percentile(train_pred, 30)

        sig_r5  = np.where(prob > 0.60, 1, np.where(prob < 0.40, -1, 0))
        sig_r7b = np.where(pred > p_long, 1, np.where(pred < p_sht, -1, 0))

        signals = rule_fn(prob, pred, sig_r5, sig_r7b, train_pred)
        actual  = test_cls["return_5d"].values
        ev = evaluate_signals(signals, actual)
        results.append(ev)

        d_str = f"{ev['delta']:+.4f}" if not np.isnan(ev["delta"]) else "NaN"
        print(f"  {rule_name} split_{i+1}: "
              f"delta={d_str} sharpe={ev['sharpe']:+.3f} "
              f"long={ev['pct_long']:.0%} short={ev['pct_short']:.0%} "
              f"neut={ev['pct_neut']:.0%}")

    dm = np.nanmean([r["delta"]  for r in results])
    sm = np.nanmean([r["sharpe"] for r in results])
    print(f"  Média: delta={dm:+.4f} sharpe={sm:+.3f}")
    return results, dm, sm


# Regra A — Confirmação
def rule_a(prob, pred, sig_r5, sig_r7b, train_pred):
    p_long = np.percentile(train_pred, 70)
    p_sht  = np.percentile(train_pred, 30)
    sig    = np.zeros(len(prob), dtype=int)
    sig[(prob > 0.60) & (pred > p_long)] = 1
    sig[(prob < 0.40) & (pred < p_sht)]  = -1
    return sig


# Regra B — Filtro
def rule_b(prob, pred, sig_r5, sig_r7b, train_pred):
    sig  = np.zeros(len(prob), dtype=int)
    # R5 LONG mas R7b não é SHORT → LONG
    sig[(sig_r5 == 1)  & (sig_r7b != -1)] = 1
    # R5 SHORT mas R7b não é LONG → SHORT
    sig[(sig_r5 == -1) & (sig_r7b != 1)]  = -1
    return sig


# Regra C — Score combinado
def rule_c(prob, pred, sig_r5, sig_r7b, train_pred):
    rk    = rank_norm(pred, train_pred)
    score = 0.6 * prob + 0.4 * rk
    # Percentile thresholds calibrados no treino
    train_prob_approx = 0.5  # não disponível para calibração sem rerun; usar 70/30 fixo
    p_long_s = 0.70
    p_sht_s  = 0.30
    sig = np.where(score > p_long_s, 1, np.where(score < p_sht_s, -1, 0))
    return sig


print("\n" + "=" * 65)
print("REGRA A — Confirmação (R5=LONG AND R7b>p70)")
print("=" * 65)
res_a, dm_a, sm_a = run_ensemble(rule_a, "A")

print("\n" + "=" * 65)
print("REGRA B — Filtro (R5 decide, R7b veta contradição)")
print("=" * 65)
res_b, dm_b, sm_b = run_ensemble(rule_b, "B")

print("\n" + "=" * 65)
print("REGRA C — Score combinado (0.6*prob_R5 + 0.4*rank_R7b)")
print("=" * 65)
res_c, dm_c, sm_c = run_ensemble(rule_c, "C")


# ── Tabela comparativa ────────────────────────────────────────
print("\n" + "=" * 65)
print("TABELA COMPARATIVA — R8 Ensemble")
print("=" * 65)

all_configs = [
    ("R5 classificação",  r5_results,  r5_dm,  r5_sm),
    ("R7b-XR baseline",   r7b_results, r7b_dm, r7b_sm),
    ("R8-A Confirmação",  res_a,       dm_a,   sm_a),
    ("R8-B Filtro",       res_b,       dm_b,   sm_b),
    ("R8-C Score",        res_c,       dm_c,   sm_c),
]

for metric_name, key in [("DELTA", "delta"), ("SHARPE", "sharpe")]:
    print(f"\n--- {metric_name}")
    print("{:<22}".format("Config") +
          "".join("{:>12}".format("split_" + str(i + 1)) for i in range(3)) +
          "{:>12}".format("Média") + "{:>10}".format("vs R7b"))
    print("-" * 68)
    for name, res, dm, sm in all_configs:
        vals = [r[key] for r in res]
        mean = dm if key == "delta" else sm
        ref  = r7b_dm if key == "delta" else r7b_sm
        diff = mean - ref
        row  = "{:<22}".format(name)
        for v in vals:
            fmt = "{:>+12.4f}" if key == "delta" else "{:>+12.3f}"
            row += fmt.format(v)
        fmt  = "{:>+12.4f} {:>+10.4f}" if key == "delta" else "{:>+12.3f} {:>+10.3f}"
        row += fmt.format(mean, diff)
        print(row)

# Distribuição split_3
print(f"\n--- DISTRIBUIÇÃO SINAIS (split_3)")
print("{:<22} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
    "Config", "%LONG", "%SHORT", "%NEUT", "n_long", "n_short"))
print("-" * 64)
for name, res, _, __ in all_configs:
    r = res[2]
    print("{:<22} {:>8.0%} {:>8.0%} {:>8.0%} {:>8} {:>8}".format(
        name, r["pct_long"], r["pct_short"], r["pct_neut"],
        r["n_long"], r["n_short"]))

# Melhor ensemble
best_name = max(
    [("R8-A", dm_a, sm_a), ("R8-B", dm_b, sm_b), ("R8-C", dm_c, sm_c)],
    key=lambda x: x[1]
)
print(f"\n  Melhor ensemble: {best_name[0]} "
      f"(delta={best_name[1]:+.4f} sharpe={best_name[2]:+.3f})")
print(f"  vs R7b: delta {best_name[1]-r7b_dm:+.4f} | "
      f"sharpe {best_name[2]-r7b_sm:+.3f}")
aprovado = best_name[1] > r7b_dm
print(f"  Decisão: {'✓ APROVADO' if aprovado else '✗ REJEITADO'}")
