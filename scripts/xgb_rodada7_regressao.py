"""
XGBoost Rodada 7b — regressão de retorno normalizado.
ElasticNet + XGBoost regression sobre features R5.
Target: return_5d / vol_short  (vol_short = 21d rolling std = vol_21 no dataset)
Threshold de sinal via percentil (70/30) calibrado no treino.
Script exploratório — NAO modifica pipeline nem sobrescreve modelos.

Diferenças vs R7:
  - Target normalizado: return_5d / vol_short (Sharpe-like target)
  - XGBRegressor: min_child_weight=1, gamma=0 (corrige underfitting severo da R7)
  - ElasticNet: sem alteração
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Carregar dados ─────────────────────────────────────────────
# Model input tem features técnicas do R5 (L3 pipeline atualizado)
btc = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
hmm = pd.read_parquet("data/05_models/regime_hmm/btc_states.parquet")

hmm = hmm.rename(columns={"state": "hmm_state"})

df = btc.copy()
df = df.join(hmm[["hmm_state", "bull_prob"]], how="left")
df["regime_strength"] = (df["bull_prob"] - 0.5).abs()

# Labels — retorno bruto para avaliação (delta/sharpe), normalizado para treino
df["return_5d"] = df["log_return"].rolling(5).sum().shift(-5)
df["label_5d"]  = (df["return_5d"] > 0).astype(int)

# vol_short = 21d rolling std de log_return (vol_21 no domínio do negócio)
# Usado para normalizar o target de regressão
if "vol_short" not in df.columns:
    raise ValueError("vol_short (vol_21) não encontrado no model input")
df["target_norm"] = df["return_5d"] / df["vol_short"].replace(0, np.nan)

print(f"Shape: {df.shape}")
print(f"Range: {df.index.min().date()} -> {df.index.max().date()}")
print(f"Target normalizado (target_norm) — estatísticas treino 2023-2024:")
t_sample = df["2023-01-01":"2024-12-31"]["target_norm"].dropna()
print(f"  n={len(t_sample)} | mean={t_sample.mean():+.4f} | "
      f"std={t_sample.std():.4f} | "
      f"p10={np.percentile(t_sample, 10):+.4f} p90={np.percentile(t_sample, 90):+.4f}")

# ── Feature set R5 (exato) ─────────────────────────────────────
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

print(f"\nFeatures R5: {len(FEATURES)}")
print(f"  BASE:      {len(BASE_COLS)}")
print(f"  STRUCTURE: {len(FEAT_STRUCTURE)}")
print(f"  MOMENTUM:  {len(FEAT_MOMENTUM)}")

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]


# ── Funções de avaliação ───────────────────────────────────────
def to_signal_percentile(pred_ret, train_pred, long_pct=70, short_pct=30):
    """Converte previsão em sinal via percentil calibrado no treino."""
    p_long  = np.percentile(train_pred, long_pct)
    p_short = np.percentile(train_pred, short_pct)
    signals = np.where(pred_ret > p_long, 1, np.where(pred_ret < p_short, -1, 0))
    return signals, p_long, p_short


def evaluate_signals(signals, actual_ret):
    """Delta e Sharpe ratio (long=+ret, short=-ret, neutral=0)."""
    long_mask  = signals == 1
    short_mask = signals == -1
    neut_mask  = signals == 0

    long_ret  = actual_ret[long_mask].mean()  if long_mask.sum()  > 0 else float("nan")
    short_ret = actual_ret[short_mask].mean() if short_mask.sum() > 0 else float("nan")
    neut_ret  = actual_ret[neut_mask].mean()  if neut_mask.sum()  > 0 else float("nan")
    delta     = (long_ret - short_ret
                 if not (np.isnan(long_ret) or np.isnan(short_ret))
                 else float("nan"))

    strat_ret  = np.where(long_mask, actual_ret, np.where(short_mask, -actual_ret, 0))
    sharpe_raw = (strat_ret.mean() / (strat_ret.std() + 1e-8)) * np.sqrt(252 / 5)

    return {
        "delta":     delta,
        "sharpe":    sharpe_raw,
        "long_ret":  long_ret,
        "short_ret": short_ret,
        "neut_ret":  neut_ret,
        "pct_long":  long_mask.mean(),
        "pct_short": short_mask.mean(),
        "pct_neut":  neut_mask.mean(),
        "n_long":    int(long_mask.sum()),
        "n_short":   int(short_mask.sum()),
    }


# ── R5 classificação — baseline ────────────────────────────────
print("\n" + "=" * 65)
print("R5 CLASSIFICAÇÃO — baseline (delta + sharpe)")
print("=" * 65)

r5_results = []
for i, sp in enumerate(SPLITS):
    test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
    cols  = list(dict.fromkeys(FEATURES + ["label_5d", "return_5d"]))
    train = df[sp["train_start"]:sp["train_end"]][cols].dropna()
    test  = df[sp["test_start"]:test_end][cols].dropna()

    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=3,
        gamma=0.5, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    clf.fit(train[FEATURES], train["label_5d"], verbose=False)
    proba = clf.predict_proba(test[FEATURES])[:, 1]

    sig_r5 = np.where(proba > 0.60, 1, np.where(proba < 0.40, -1, 0))
    ev     = evaluate_signals(sig_r5, test["return_5d"].values)
    r5_results.append(ev)

    print(f"  split_{i+1}: delta={ev['delta']:+.4f} "
          f"sharpe={ev['sharpe']:+.3f} "
          f"long={ev['pct_long']:.0%} "
          f"short={ev['pct_short']:.0%} "
          f"neut={ev['pct_neut']:.0%}")

r5_delta_mean  = np.nanmean([r["delta"]  for r in r5_results])
r5_sharpe_mean = np.nanmean([r["sharpe"] for r in r5_results])
print(f"  Média: delta={r5_delta_mean:+.4f} sharpe={r5_sharpe_mean:+.3f}")


# ── Walk-forward regressão (target normalizado) ────────────────
def run_regression(df, features, model_fn, label=""):
    """
    Walk-forward com target_norm no treino.
    Avaliação usa return_5d bruto (sem leakage).
    """
    results = []
    for i, sp in enumerate(SPLITS):
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        cols  = list(dict.fromkeys(features + ["target_norm", "return_5d"]))
        train = df[sp["train_start"]:sp["train_end"]][cols].dropna()
        test  = df[sp["test_start"]:test_end][cols].dropna()

        if len(train) < 50 or len(test) < 20:
            print(f"  {label} split_{i+1}: SKIP (n_train={len(train)}, n_test={len(test)})")
            continue

        model      = model_fn(train[features], train["target_norm"])
        pred       = model.predict(test[features])
        train_pred = model.predict(train[features])

        r2 = r2_score(test["target_norm"], pred)

        signals, p_long, p_short = to_signal_percentile(pred, train_pred)
        ev = evaluate_signals(signals, test["return_5d"].values)

        results.append({**ev, "r2": r2, "split": f"split_{i+1}"})
        r2_diag = f"r2={r2:+.4f}" + (" [NEG-esperado]" if r2 < 0 else "")
        print(f"  {label} split_{i+1}: "
              f"{r2_diag} "
              f"delta={ev['delta']:+.4f} "
              f"sharpe={ev['sharpe']:+.3f} "
              f"long={ev['pct_long']:.0%} "
              f"short={ev['pct_short']:.0%} "
              f"neut={ev['pct_neut']:.0%} "
              f"| p_long={p_long:+.4f} p_short={p_short:+.4f}")

    return results


# ── ElasticNet ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ELASTICNET REGRESSION — R7b-EN (target normalizado)")
print("=" * 65)


class ScaledModel:
    """ElasticNet wrapper com StandardScaler embutido."""
    def __init__(self, model, scaler):
        self.model     = model
        self.scaler    = scaler
        self.coef_     = model.coef_
        self.alpha_    = model.alpha_
        self.l1_ratio_ = model.l1_ratio_

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


def fit_elasticnet(X, y):
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    model  = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
        alphas=np.logspace(-4, 1, 30),
        cv=5, max_iter=5000, random_state=42,
    )
    model.fit(Xs, y)
    return ScaledModel(model, scaler)


en_results = run_regression(df, FEATURES, fit_elasticnet, "EN ")

en_delta_mean  = np.nanmean([r["delta"]  for r in en_results])
en_sharpe_mean = np.nanmean([r["sharpe"] for r in en_results])
print(f"  Média: delta={en_delta_mean:+.4f} sharpe={en_sharpe_mean:+.3f}")

# Coeficientes ElasticNet split_3
print(f"\n  ElasticNet split_3 — features ativas:")
sp3    = SPLITS[2]
cols3  = list(dict.fromkeys(FEATURES + ["target_norm", "return_5d"]))
train3 = df[sp3["train_start"]:sp3["train_end"]][cols3].dropna()
en3    = fit_elasticnet(train3[FEATURES], train3["target_norm"])
coef_s = pd.Series(en3.coef_, index=FEATURES)
active = coef_s[coef_s.abs() > 0].sort_values(key=lambda x: x.abs(), ascending=False)
print(f"  alpha={en3.alpha_:.6f} l1_ratio={en3.l1_ratio_:.2f} "
      f"n_active={len(active)}/{len(FEATURES)}")
for feat, coef in active.head(10).items():
    grp = ("BASE" if feat in BASE_COLS
           else "MOMENTUM" if feat in FEAT_MOMENTUM
           else "STRUCTURE")
    print(f"    {feat:<35} {coef:>+.6f}  [{grp}]")


# ── XGBoost Regression ─────────────────────────────────────────
print("\n" + "=" * 65)
print("XGBOOST REGRESSION — R7b-XR (target normalizado, gamma=0)")
print("=" * 65)


def fit_xgb_reg(X, y):
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=1,
        gamma=0, random_state=42, verbosity=0,
    )
    model.fit(X, y, verbose=False)
    return model


xr_results = run_regression(df, FEATURES, fit_xgb_reg, "XR ")

xr_delta_mean  = np.nanmean([r["delta"]  for r in xr_results])
xr_sharpe_mean = np.nanmean([r["sharpe"] for r in xr_results])
print(f"  Média: delta={xr_delta_mean:+.4f} sharpe={xr_sharpe_mean:+.3f}")

# Feature importance XGBRegressor split_3
xr3 = fit_xgb_reg(train3[FEATURES], train3["target_norm"])
imp = pd.Series(xr3.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n  XGBRegressor split_3 — top 8 features:")
for feat, val in imp.head(8).items():
    grp = ("BASE" if feat in BASE_COLS
           else "MOMENTUM" if feat in FEAT_MOMENTUM
           else "STRUCTURE")
    print(f"    {feat:<35} {val:.4f}  [{grp}]")


# ── Tabela comparativa final ───────────────────────────────────
print("\n" + "=" * 65)
print("TABELA COMPARATIVA — R5 vs R7b-EN vs R7b-XR")
print("=" * 65)

configs = [
    ("R5 classificação",  r5_results, r5_delta_mean,  r5_sharpe_mean),
    ("R7b-EN ElasticNet", en_results, en_delta_mean,  en_sharpe_mean),
    ("R7b-XR XGB Reg",   xr_results, xr_delta_mean,  xr_sharpe_mean),
]

for metric_name, metric_key in [("DELTA", "delta"), ("SHARPE (anualizado)", "sharpe")]:
    print(f"\n--- {metric_name}")
    print(f"\n{'Config':<24}" +
          "".join(f"{'split_'+str(i+1):>12}" for i in range(3)) +
          f"{'Média':>12} {'vs R5':>10}")
    print("-" * 72)
    r5_ref = r5_delta_mean if metric_key == "delta" else r5_sharpe_mean
    for name, res, dmean, smean in configs:
        row  = f"{name:<24}"
        vals = [r[metric_key] for r in res]
        for v in vals:
            fmt = f"{v:>+12.4f}" if metric_key == "delta" else f"{v:>+12.3f}"
            row += fmt
        mean = dmean if metric_key == "delta" else smean
        diff = mean - r5_ref
        row += (f"{mean:>+12.4f} {diff:>+10.4f}"
                if metric_key == "delta"
                else f"{mean:>+12.3f} {diff:>+10.3f}")
        print(row)

# Distribuição split_3
print(f"\n--- DISTRIBUIÇÃO DE SINAIS (split_3)")
print(f"\n{'Config':<24} {'%LONG':>8} {'%SHORT':>8} {'%NEUT':>8} {'n_long':>8} {'n_short':>8}")
print("-" * 64)
for name, res, _, __ in configs:
    if len(res) < 3:
        print(f"{name:<24} sem dados split_3")
        continue
    r = res[2]
    print(f"{name:<24} "
          f"{r['pct_long']:>8.0%} {r['pct_short']:>8.0%} {r['pct_neut']:>8.0%} "
          f"{r['n_long']:>8} {r['n_short']:>8}")


# ── Diagnóstico — sensibilidade ao threshold (XGBRegressor split_3) ──
print(f"\n{'=' * 65}")
print("DIAGNÓSTICO — sensibilidade ao threshold (XGBRegressor split_3)")
print("=" * 65)

test3_end      = df.index.max().strftime("%Y-%m-%d")
test3          = df[sp3["test_start"]:test3_end][cols3].dropna()
xr3_pred       = xr3.predict(test3[FEATURES])
xr3_train_pred = xr3.predict(train3[FEATURES])

print(f"  Estatísticas das previsões XR treino: "
      f"mean={xr3_train_pred.mean():+.4f} std={xr3_train_pred.std():.4f} "
      f"p30={np.percentile(xr3_train_pred, 30):+.4f} "
      f"p70={np.percentile(xr3_train_pred, 70):+.4f}")
print(f"  Previsões distintas no treino: {len(np.unique(xr3_train_pred))}")

print(f"\n{'Threshold':<12} {'Delta':>10} {'Sharpe':>10} {'%Long':>8} {'%Short':>8} {'%Neut':>8}")
print("-" * 56)
for long_p, short_p in [(60, 40), (70, 30), (80, 20)]:
    sigs, pl, ps = to_signal_percentile(xr3_pred, xr3_train_pred, long_p, short_p)
    ev = evaluate_signals(sigs, test3["return_5d"].values)
    delta_str  = f"{ev['delta']:>+10.4f}" if not np.isnan(ev["delta"]) else f"{'NaN':>10}"
    sharpe_str = f"{ev['sharpe']:>+10.3f}" if not np.isnan(ev["sharpe"]) else f"{'NaN':>10}"
    print(f"  p{long_p}/{short_p}      "
          f"{delta_str} {sharpe_str} "
          f"{ev['pct_long']:>8.0%} {ev['pct_short']:>8.0%} {ev['pct_neut']:>8.0%}")


# ── Output operacional hoje ────────────────────────────────────
print(f"\n{'=' * 65}")
print("OUTPUT OPERACIONAL HOJE — R7b-XR vs R5")
print("=" * 65)

last = df[df[FEATURES].notna().all(axis=1)].tail(1)
if len(last) > 0:
    train_all_cols = list(dict.fromkeys(FEATURES + ["target_norm", "label_5d"]))
    train_all      = df["2023-01-01":"2024-12-31"][train_all_cols].dropna()

    # XGBRegressor operacional
    xr_final      = fit_xgb_reg(train_all[FEATURES], train_all["target_norm"])
    pred_hoje     = xr_final.predict(last[FEATURES])[0]
    train_preds   = xr_final.predict(train_all[FEATURES])
    p_long_h      = np.percentile(train_preds, 70)
    p_short_h     = np.percentile(train_preds, 30)
    sig_xr        = ("LONG"  if pred_hoje > p_long_h
                     else "SHORT"  if pred_hoje < p_short_h
                     else "NEUTRAL")

    # R5 classificação operacional
    clf_r5 = xgb.XGBClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=3,
        gamma=0.5, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    clf_r5.fit(train_all[FEATURES], train_all["label_5d"], verbose=False)
    prob_r5  = clf_r5.predict_proba(last[FEATURES])[0, 1]
    sig_r5_h = ("LONG"  if prob_r5 > 0.60
                else "SHORT" if prob_r5 < 0.40
                else "NEUTRAL")

    regime = "BULL" if last["hmm_state"].iloc[0] == 1 else "BEAR"
    bull_p = float(last["bull_prob"].iloc[0])

    print(f"\n  Data:              {last.index[0].date()}")
    print(f"  Regime HMM:        {regime}")
    print(f"  Bull probability:  {bull_p:.1%}")

    print(f"\n  R5 classificação:")
    print(f"    prob_up_5d:       {prob_r5:.1%}")
    print(f"    Sinal:            {sig_r5_h}")

    print(f"\n  R7b-XR regressão (target normalizado):")
    print(f"    expected_norm_ret: {pred_hoje:+.4f}")
    print(f"    Threshold LONG:    p70 = {p_long_h:+.4f}")
    print(f"    Threshold SHORT:   p30 = {p_short_h:+.4f}")
    print(f"    Sinal:             {sig_xr}")

    concordancia = sig_xr == sig_r5_h
    print(f"\n  Concordância:      {'SIM' if concordancia else 'NAO — divergência'}")
    if not concordancia:
        print(f"    R5 diz {sig_r5_h}, R7b-XR diz {sig_xr}")
        print(f"    pred={pred_hoje:+.4f} | p30={p_short_h:+.4f} p70={p_long_h:+.4f}")
else:
    print("\n  AVISO: sem dados válidos para output operacional.")
