"""
R11 — Análise de performance por regime de mercado (Bull / Bear).

Mesmos sinais e thresholds do xgb_r11_backtest.py.
Apenas análise — sem modificação de modelo, sinais ou thresholds.

NAO modifica pipeline nem sobrescreve modelos.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

TC         = 0.0010
PERIODS_YR = 252 / 5


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
     "test_start":  "2025-01-01", "test_end":  "2025-03-31"},
    {"train_start": "2023-01-01", "train_end": "2025-03-31",
     "test_start":  "2025-04-01", "test_end":  None},
]


# ── Model factories (idêntico ao backtest) ─────────────────────
def fit_r5(X, y):
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        gamma=0.5, eval_metric="logloss", random_state=42, verbosity=0,
    )
    clf.fit(X, y, verbose=False)
    return clf


def fit_r7b(X, y):
    reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=1,
        gamma=0, random_state=42, verbosity=0,
    )
    reg.fit(X, y, verbose=False)
    return reg


# ── Geração de sinais (idêntico ao backtest) ───────────────────
print("Gerando sinais R11...")
records = []
for sp in SPLITS:
    test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
    aux      = ["label_5d", "target_norm", "return_5d", "hmm_state"]
    cols     = list(dict.fromkeys(TOP10 + aux))

    train = df[sp["train_start"]:sp["train_end"]][cols].dropna(
        subset=TOP10 + ["label_5d", "target_norm", "return_5d"])
    test  = df[sp["test_start"]:test_end][cols].dropna(
        subset=TOP10 + ["return_5d"])

    if len(train) < 50 or len(test) < 10:
        continue

    clf     = fit_r5(train[TOP10], train["label_5d"])
    prob    = clf.predict_proba(test[TOP10])[:, 1]
    sig_r5  = np.where(prob > 0.60, 1, np.where(prob < 0.40, -1, 0))

    reg        = fit_r7b(train[TOP10], train["target_norm"])
    pred       = reg.predict(test[TOP10])
    train_pred = reg.predict(train[TOP10])
    p_long     = np.percentile(train_pred, 70)
    p_sht      = np.percentile(train_pred, 30)
    sig_r7b    = np.where(pred > p_long, 1, np.where(pred < p_sht, -1, 0))

    sigs                                    = np.zeros(len(sig_r5), dtype=int)
    sigs[(sig_r5 == 1)  & (sig_r7b == 1)]  = 1
    sigs[(sig_r5 == -1) & (sig_r7b == -1)] = -1

    strat = np.where(sigs == 1, test["return_5d"].values,
                     np.where(sigs == -1, -test["return_5d"].values, 0.0))
    prev  = np.concatenate([[0], sigs[:-1]])
    tc    = TC * (sigs != prev).astype(float)

    for date, ret, sig, sr, tc_v, hs in zip(
        test.index,
        test["return_5d"].values,
        sigs,
        strat - tc,
        tc,
        test["hmm_state"].fillna(-1).astype(int).values,
    ):
        records.append({
            "date":              date,
            "signal_r11":        int(sig),
            "strategy_return":   float(sr),
            "btc_return":        float(ret),
            "hmm_state":         int(hs),
        })

bt = (pd.DataFrame(records)
        .set_index("date")
        .sort_index())
bt = bt[~bt.index.duplicated(keep="last")]

print(f"Backtest: {len(bt)} obs | {bt.index.min().date()} → {bt.index.max().date()}")
print(f"HMM: bull={( bt['hmm_state']==1).sum()} obs | bear={(bt['hmm_state']==0).sum()} obs\n")


# ── Métricas ──────────────────────────────────────────────────
def compute_metrics(subset: pd.DataFrame, label: str) -> dict:
    net  = subset["strategy_return"].values
    n    = len(net)
    if n == 0:
        return {"label": label, **{k: float("nan") for k in
                ["CAGR", "Sharpe", "MaxDD", "HitRate", "ProfitFactor", "n_trades", "n_obs"]}}

    equity      = (1 + net).cumprod()
    peak        = np.maximum.accumulate(equity)
    dd          = (equity - peak) / peak
    n_years     = n / PERIODS_YR
    cagr        = equity[-1] ** (1 / n_years) - 1
    sharpe      = (net.mean() / (net.std() + 1e-10)) * np.sqrt(PERIODS_YR)
    max_dd      = float(dd.min())
    active      = net[net != 0]
    hit_rate    = float((active > 0).mean()) if len(active) > 0 else float("nan")
    gains       = active[active > 0].sum()
    losses      = abs(active[active < 0].sum())
    prof_factor = gains / losses if losses > 0 else float("nan")
    sigs        = subset["signal_r11"].values
    prev_sigs   = np.concatenate([[0], sigs[:-1]])
    n_trades    = int((sigs != prev_sigs).sum())

    return {
        "label":        label,
        "CAGR":         cagr,
        "Sharpe":       sharpe,
        "MaxDD":        max_dd,
        "HitRate":      hit_rate,
        "ProfitFactor": prof_factor,
        "n_trades":     n_trades,
        "n_obs":        n,
    }


# ── Separar por regime ────────────────────────────────────────
subsets = {
    "Bull (hmm=1)": bt[bt["hmm_state"] == 1],
    "Bear (hmm=0)": bt[bt["hmm_state"] == 0],
    "All":          bt,
}

metrics = [compute_metrics(sub, label) for label, sub in subsets.items()]


# ── Tabela de métricas ────────────────────────────────────────
print("=" * 80)
print("MÉTRICAS POR REGIME")
print("=" * 80)
hdr = (f"{'Regime':<16} {'CAGR':>9} {'Sharpe':>8} {'MaxDD':>9} "
       f"{'HitRate':>9} {'ProfFactor':>11} {'Trades':>8} {'Obs':>6}")
print(hdr)
print("-" * 80)
for m in metrics:
    pf = f"{m['ProfitFactor']:>11.3f}" if not np.isnan(m["ProfitFactor"]) else f"{'NaN':>11}"
    print(
        f"{m['label']:<16} "
        f"{m['CAGR']:>+9.2%} "
        f"{m['Sharpe']:>+8.3f} "
        f"{m['MaxDD']:>+9.2%} "
        f"{m['HitRate']:>+9.2%} "
        f"{pf} "
        f"{m['n_trades']:>8} "
        f"{m['n_obs']:>6}"
    )


# ── Distribuição de sinais por regime ─────────────────────────
print(f"\n{'=' * 60}")
print("DISTRIBUIÇÃO DE SINAIS POR REGIME")
print("=" * 60)
print(f"{'Regime':<16} {'%LONG':>8} {'%SHORT':>8} {'%NEUTRAL':>10} "
      f"{'n_long':>8} {'n_short':>8} {'n_neut':>8}")
print("-" * 60)
for label, sub in subsets.items():
    if len(sub) == 0:
        continue
    sigs = sub["signal_r11"]
    n    = len(sigs)
    nl   = int((sigs == 1).sum())
    ns   = int((sigs == -1).sum())
    nn   = int((sigs == 0).sum())
    print(f"{label:<16} {nl/n:>8.1%} {ns/n:>8.1%} {nn/n:>10.1%} "
          f"{nl:>8} {ns:>8} {nn:>8}")


# ── Retorno médio por regime e sinal ─────────────────────────
print(f"\n{'=' * 55}")
print("RETORNO MÉDIO POR REGIME × SINAL (retorno BTC bruto)")
print("=" * 55)
print(f"{'Regime':<14} {'Sinal':<10} {'mean_ret':>10} {'n':>6}")
print("-" * 44)
for label, sub in [("Bull", bt[bt["hmm_state"] == 1]),
                   ("Bear", bt[bt["hmm_state"] == 0])]:
    for sig_val, sig_name in [(1, "LONG"), (-1, "SHORT"), (0, "NEUTRAL")]:
        mask = sub["signal_r11"] == sig_val
        n    = int(mask.sum())
        mean = float(sub.loc[mask, "btc_return"].mean()) if n > 0 else float("nan")
        flag = " ✓" if ((sig_val == 1 and mean > 0) or (sig_val == -1 and mean < 0)) else ""
        mean_str = f"{mean:>+10.4f}" if not np.isnan(mean) else f"{'NaN':>10}"
        print(f"{label:<14} {sig_name:<10} {mean_str}{flag:>2}  {n:>6}")
    print()
