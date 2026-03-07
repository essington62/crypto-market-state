"""
Ablation: HMM baseline (6 features) vs Rodada 1 (+ oi_zscore_30d).
Script exploratorio — NAO modifica pipeline nem sobrescreve btc_states.parquet.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ── Carregar dados ────────────────────────────────────────
btc = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
oi  = pd.read_parquet(
    "data/01_raw/derivatives/coinglass/open_interest/BTCUSDT_aggregated.parquet"
)

# ── ACAO 1 — Calcular oi_zscore_30d e join ────────────────
oi["oi_zscore_30d"] = (
    (oi["open_interest_usd"] - oi["open_interest_usd"].rolling(30).mean()) /
     oi["open_interest_usd"].rolling(30).std()
)

df = btc.join(oi[["oi_zscore_30d"]], how="left")

print(f"Shape apos join: {df.shape}")
print(f"Range: {df.index.min().date()} -> {df.index.max().date()}")

nan_count = df["oi_zscore_30d"].isna().sum()
nan_pct   = nan_count / len(df)
print(f"NaN oi_zscore_30d: {nan_count} ({nan_pct:.1%})")

if nan_pct > 0.10:
    raise RuntimeError(
        f"NaN em oi_zscore_30d ({nan_pct:.1%}) excede limite de 10%. "
        "Verificar cobertura do arquivo open_interest."
    )

first_valid = df["oi_zscore_30d"].first_valid_index()
print(f"Primeira data valida: {first_valid.date()}")

print("\nCorrelacao com features existentes:")
feat_orig = ["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]
for f in feat_orig:
    if f in df.columns:
        r = df[["oi_zscore_30d", f]].dropna().corr().iloc[0, 1]
        print(f"  oi_zscore_30d vs {f}: {r:+.3f}")

# ── ACAO 2 — Walk-forward HMM ────────────────────────────
FEATURES_BASELINE = ["log_return", "vol_short", "vol_ratio",
                     "drawdown", "volume_z", "slope_21d"]
FEATURES_NEW      = FEATURES_BASELINE + ["oi_zscore_30d"]

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]


def run_hmm_walkforward(df, features, n_states=2, label=""):
    results = []
    for i, sp in enumerate(SPLITS):
        train    = df[sp["train_start"]:sp["train_end"]][features].dropna()
        test_end = sp["test_end"] if sp["test_end"] else df.index.max().strftime("%Y-%m-%d")
        test     = df[sp["test_start"]:test_end][features].dropna()

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train)
        X_test  = scaler.transform(test)

        model = GaussianHMM(n_components=n_states, covariance_type="full",
                            n_iter=1000, random_state=42)
        model.fit(X_train)

        # Ordenar por drawdown ascendente: menor = Bear (0), maior = Bull (1)
        feat_idx = features.index("drawdown")
        order    = np.argsort(model.means_[:, feat_idx])
        state_map = {order[j]: j for j in range(n_states)}

        states_raw = model.predict(X_test)
        states     = np.array([state_map[s] for s in states_raw])

        test_ret = df[sp["test_start"]:test_end]["log_return"].reindex(test.index)
        ret_5d   = test_ret.rolling(5).sum().shift(-5)

        bull_ret = ret_5d[states == 1].mean()
        bear_ret = ret_5d[states == 0].mean()
        delta    = bull_ret - bear_ret

        results.append({
            "split":     f"split_{i+1}",
            "n_train":   len(train),
            "n_test":    len(test),
            "bull_ret":  bull_ret,
            "bear_ret":  bear_ret,
            "delta_5d":  delta,
            "bull_days": (states == 1).sum(),
            "bear_days": (states == 0).sum(),
        })
        print(f"  {label} split_{i+1}: n_train={len(train):3d}  n_test={len(test):3d}  "
              f"delta={delta:+.4f}  (bull={bull_ret:+.4f}  bear={bear_ret:+.4f})")
    return pd.DataFrame(results)


print("\n" + "=" * 60)
print("BASELINE — 6 features originais")
print("=" * 60)
res_base = run_hmm_walkforward(df, FEATURES_BASELINE, label="BASE")

print("\n" + "=" * 60)
print("RODADA 1 — 6 features + oi_zscore_30d")
print("=" * 60)
res_new = run_hmm_walkforward(df, FEATURES_NEW, label="OI  ")

# ── ACAO 3 — Comparar resultados ────────────────────────
print("\n" + "=" * 60)
print("COMPARACAO BASELINE vs RODADA 1")
print("=" * 60)
print(f"\n{'Split':<10} {'Base delta':>12} {'OI delta':>12} {'Diff':>10}  Resultado")
print("-" * 58)
for rb, rn in zip(res_base.itertuples(), res_new.itertuples()):
    diff   = rn.delta_5d - rb.delta_5d
    result = "MELHOROU" if diff > 0 else "PIOROU"
    print(f"{rb.split:<10} {rb.delta_5d:>+12.4f} {rn.delta_5d:>+12.4f} "
          f"{diff:>+10.4f}  {result}")

base_mean = res_base["delta_5d"].mean()
new_mean  = res_new["delta_5d"].mean()
diff_mean = new_mean - base_mean
print(f"\n{'MEDIA':<10} {base_mean:>+12.4f} {new_mean:>+12.4f} "
      f"{diff_mean:>+10.4f}  {'MELHOROU' if new_mean > base_mean else 'PIOROU'}")

splits_improved = (res_new["delta_5d"] > res_base["delta_5d"]).sum()
mean_improved   = new_mean > base_mean
approved        = splits_improved >= 2 and mean_improved

print(f"\nSplits melhorados: {splits_improved}/3")
print(f"Delta medio superior: {'sim' if mean_improved else 'nao'}")
print(f"Decisao: {'oi_zscore_30d APROVADO' if approved else 'oi_zscore_30d REJEITADO'}")
