"""
R10 — Horizon Experiment.

Testar se horizonte 3d ou 7d supera o baseline 5d.
Target normalizado: return_Nd / vol_short
Conversão: percentil 70/30 calibrado no treino.
Sharpe anualizado: sqrt(252 / horizon_days).

Baseline: R7b-XR 5d → delta=+0.0267 Sharpe=+1.118

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

# Retornos e targets para os três horizontes
for h in [3, 5, 7]:
    df[f"return_{h}d"] = df["log_return"].rolling(h).sum().shift(-h)
    df[f"target_{h}d"] = df[f"return_{h}d"] / df["vol_short"].replace(0, np.nan)

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


def fit_xgb_reg(X, y):
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=1,
        gamma=0, random_state=42, verbosity=0,
    )
    model.fit(X, y, verbose=False)
    return model


def evaluate_signals(signals, actual_ret, horizon):
    long_mask  = signals == 1
    short_mask = signals == -1
    long_ret  = actual_ret[long_mask].mean()  if long_mask.sum()  > 0 else float("nan")
    short_ret = actual_ret[short_mask].mean() if short_mask.sum() > 0 else float("nan")
    delta     = (long_ret - short_ret
                 if not (np.isnan(long_ret) or np.isnan(short_ret)) else float("nan"))
    ann_factor = np.sqrt(252 / horizon)
    strat_ret  = np.where(long_mask, actual_ret, np.where(short_mask, -actual_ret, 0))
    sharpe     = (strat_ret.mean() / (strat_ret.std() + 1e-8)) * ann_factor
    return {
        "delta": delta, "sharpe": sharpe,
        "pct_long": long_mask.mean(), "pct_short": short_mask.mean(),
        "pct_neut": (signals == 0).mean(),
        "n_long": int(long_mask.sum()), "n_short": int(short_mask.sum()),
    }


def run_horizon(df, horizon, label=""):
    target_col = f"target_{horizon}d"
    return_col = f"return_{horizon}d"
    results    = []

    for i, sp in enumerate(SPLITS):
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        cols  = list(dict.fromkeys(FEATURES + [target_col, return_col]))
        train = df[sp["train_start"]:sp["train_end"]][cols].dropna()
        test  = df[sp["test_start"]:test_end][cols].dropna()

        if len(train) < 50 or len(test) < 20:
            print(f"  {label} split_{i+1}: SKIP")
            continue

        model      = fit_xgb_reg(train[FEATURES], train[target_col])
        pred       = model.predict(test[FEATURES])
        train_pred = model.predict(train[FEATURES])

        p_long = np.percentile(train_pred, 70)
        p_sht  = np.percentile(train_pred, 30)
        sigs   = np.where(pred > p_long, 1, np.where(pred < p_sht, -1, 0))
        ev     = evaluate_signals(sigs, test[return_col].values, horizon)
        results.append(ev)

        d_str = f"{ev['delta']:+.4f}" if not np.isnan(ev["delta"]) else "NaN"
        print(f"  {label} split_{i+1}: "
              f"n={len(train)}/{len(test)} "
              f"delta={d_str} sharpe={ev['sharpe']:+.3f} "
              f"long={ev['pct_long']:.0%} short={ev['pct_short']:.0%} "
              f"neut={ev['pct_neut']:.0%} "
              f"| p_long={p_long:+.4f} p_short={p_sht:+.4f}")

    dm = np.nanmean([r["delta"]  for r in results])
    sm = np.nanmean([r["sharpe"] for r in results])
    print(f"  Média: delta={dm:+.4f} sharpe={sm:+.3f}")
    return results, dm, sm


# ── Rodar os três horizontes ──────────────────────────────────
print(f"Features R5: {len(FEATURES)}")

horizon_results = {}
for h, name, label in [(3, "3d", "H3"), (5, "5d (R7b)", "H5"), (7, "7d", "H7")]:
    print(f"\n{'=' * 65}")
    print(f"HORIZONTE {h}d — target = return_{h}d / vol_short | Sharpe ann=sqrt(252/{h})")
    print("=" * 65)
    res, dm, sm = run_horizon(df, h, label)
    horizon_results[h] = {"name": name, "results": res, "dm": dm, "sm": sm}


# ── Tabela comparativa ────────────────────────────────────────
print("\n" + "=" * 65)
print("TABELA COMPARATIVA — R10 Horizon Experiment")
print("=" * 65)

all_configs = [
    (f"{h}d ({horizon_results[h]['name']})",
     horizon_results[h]["results"],
     horizon_results[h]["dm"],
     horizon_results[h]["sm"])
    for h in [3, 5, 7]
]

# Referência = 5d
ref_dm = horizon_results[5]["dm"]
ref_sm = horizon_results[5]["sm"]

for metric_name, key in [("DELTA", "delta"), ("SHARPE (ann. por horizonte)", "sharpe")]:
    print(f"\n--- {metric_name}")
    print("{:<16}".format("Horizonte") +
          "".join("{:>12}".format("split_" + str(i + 1)) for i in range(3)) +
          "{:>12}".format("Média") + "{:>10}".format("vs 5d"))
    print("-" * 64)
    for name, res, dm, sm in all_configs:
        vals = [r[key] for r in res]
        mean = dm if key == "delta" else sm
        ref  = ref_dm if key == "delta" else ref_sm
        diff = mean - ref
        row  = "{:<16}".format(name)
        for v in vals:
            fmt = "{:>+12.4f}" if key == "delta" else "{:>+12.3f}"
            row += fmt.format(v)
        fmt  = "{:>+12.4f} {:>+10.4f}" if key == "delta" else "{:>+12.3f} {:>+10.3f}"
        row += fmt.format(mean, diff)
        print(row)

# Distribuição split_3
print(f"\n--- DISTRIBUIÇÃO SINAIS (split_3)")
print("{:<16} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
    "Horizonte", "%LONG", "%SHORT", "%NEUT", "n_long", "n_short"))
print("-" * 56)
for name, res, _, __ in all_configs:
    r = res[2] if len(res) >= 3 else {}
    if r:
        print("{:<16} {:>8.0%} {:>8.0%} {:>8.0%} {:>8} {:>8}".format(
            name, r["pct_long"], r["pct_short"], r["pct_neut"],
            r["n_long"], r["n_short"]))

# Veredicto por horizonte
print("\n--- VEREDICTO")
r5d = horizon_results[5]
for h in [3, 7]:
    rh = horizon_results[h]
    splits_ok = int(sum(rh["results"][j]["delta"] > r5d["results"][j]["delta"]
                        for j in range(min(len(rh["results"]), len(r5d["results"])))))
    diff = rh["dm"] - r5d["dm"]
    aprovado = splits_ok >= 2 and diff > 0
    print(f"  {h}d vs 5d: delta={rh['dm']:+.4f} (diff {diff:+.4f}) "
          f"splits_ok={splits_ok}/3 → {'✓ APROVADO' if aprovado else '✗ REJEITADO'}")

# Output operacional — todos os horizontes
print(f"\n{'=' * 65}")
print("OUTPUT OPERACIONAL HOJE — comparação por horizonte")
print("=" * 65)

sp3 = SPLITS[2]
for h in [3, 5, 7]:
    target_col = f"target_{h}d"
    return_col = f"return_{h}d"
    cols = list(dict.fromkeys(FEATURES + [target_col, return_col]))
    train_op = df["2023-01-01":"2024-12-31"][cols].dropna()
    last     = df[df[FEATURES].notna().all(axis=1)].tail(1)

    if len(last) == 0 or len(train_op) < 50:
        print(f"  {h}d: dados insuficientes")
        continue

    model_op      = fit_xgb_reg(train_op[FEATURES], train_op[target_col])
    pred_hoje     = model_op.predict(last[FEATURES])[0]
    train_pred_op = model_op.predict(train_op[FEATURES])
    p_long_h      = np.percentile(train_pred_op, 70)
    p_sht_h       = np.percentile(train_pred_op, 30)
    sig = ("LONG"  if pred_hoje > p_long_h
           else "SHORT"  if pred_hoje < p_sht_h
           else "NEUTRAL")
    print(f"  {h}d: expected_norm_ret={pred_hoje:+.4f}  "
          f"p70={p_long_h:+.4f}  p30={p_sht_h:+.4f}  SINAL={sig}")
