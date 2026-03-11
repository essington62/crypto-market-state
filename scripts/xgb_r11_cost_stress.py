"""
R11 — Stress test de custos de transação.

Testa 10bps, 25bps e 50bps sobre os mesmos sinais do backtest R11.
Sinais e modelo não são alterados.

NAO modifica pipeline nem sobrescreve modelos.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

PERIODS_YR = 252 / 5
COST_SCENARIOS = [0.0010, 0.0025, 0.0050]   # 10, 25, 50 bps


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


# ── Geração de sinais (uma só vez) ────────────────────────────
print("Gerando sinais R11 (uma vez)...")
raw_records = []   # (date, strat_ret_bruto, sinal)
for sp in SPLITS:
    test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
    aux      = ["label_5d", "target_norm", "return_5d"]
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

    strat_gross = np.where(sigs == 1, test["return_5d"].values,
                           np.where(sigs == -1, -test["return_5d"].values, 0.0))

    for date, sg, sig in zip(test.index, strat_gross, sigs):
        raw_records.append({"date": date, "strat_gross": float(sg), "signal": int(sig)})

raw = (pd.DataFrame(raw_records)
         .set_index("date")
         .sort_index())
raw = raw[~raw.index.duplicated(keep="last")]

sigs_arr  = raw["signal"].values
gross_arr = raw["strat_gross"].values
prev_sigs = np.concatenate([[0], sigs_arr[:-1]])
trade_flag = (sigs_arr != prev_sigs).astype(float)
n_trades  = int(trade_flag.sum())
n_obs     = len(raw)

print(f"Sinais gerados: {n_obs} obs | {n_trades} trades\n")


# ── Métricas ──────────────────────────────────────────────────
def metrics(net_ret):
    n           = len(net_ret)
    equity      = (1 + net_ret).cumprod()
    peak        = np.maximum.accumulate(equity)
    max_dd      = float(((equity - peak) / peak).min())
    n_years     = n / PERIODS_YR
    cagr        = equity[-1] ** (1 / n_years) - 1
    sharpe      = (net_ret.mean() / (net_ret.std() + 1e-10)) * np.sqrt(PERIODS_YR)
    active      = net_ret[net_ret != 0]
    gains       = active[active > 0].sum()
    losses      = abs(active[active < 0].sum())
    pf          = gains / losses if losses > 0 else float("nan")
    return cagr, sharpe, max_dd, pf


# ── Stress test ───────────────────────────────────────────────
print("=" * 62)
print("STRESS TEST — CUSTO DE TRANSAÇÃO")
print("=" * 62)
print(f"{'Cost':>8} {'CAGR':>10} {'Sharpe':>8} {'MaxDD':>10} {'ProfitFactor':>13}  {'TC total':>9}")
print("-" * 62)

for tc in COST_SCENARIOS:
    net       = gross_arr - tc * trade_flag
    cagr, sh, mdd, pf = metrics(net)
    tc_total  = tc * trade_flag.sum()
    pf_str    = f"{pf:>13.3f}" if not np.isnan(pf) else f"{'NaN':>13}"
    print(f"{tc*10000:>6.0f}bps "
          f"{cagr:>+10.2%} "
          f"{sh:>+8.3f} "
          f"{mdd:>+10.2%} "
          f"{pf_str}  "
          f"{tc_total*10000:>7.0f}bps")

print("-" * 62)
print(f"{'(sem TC)':>8}", end="")
net0 = gross_arr.copy()
cagr0, sh0, mdd0, pf0 = metrics(net0)
print(f" {cagr0:>+10.2%} {sh0:>+8.3f} {mdd0:>+10.2%} {pf0:>13.3f}  {'0bps':>9}")
