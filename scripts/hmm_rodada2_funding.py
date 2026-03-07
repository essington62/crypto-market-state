"""
HMM Rodada 2: baseline + funding_zscore_30d.
Script exploratório — NAO modifica pipeline nem sobrescreve btc_states.parquet.
"""
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

BASE = Path("data/01_raw/derivatives/coinglass")

# Carregar dados
btc = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
fr  = pd.read_parquet(BASE / "funding/BTCUSDT_oi_weighted.parquet")

# Calcular funding_zscore_30d
fr["funding_zscore_30d"] = (
    (fr["funding_rate_oi_weighted"] -
     fr["funding_rate_oi_weighted"].rolling(30).mean()) /
     fr["funding_rate_oi_weighted"].rolling(30).std()
)

# Join
df = btc.join(fr[["funding_zscore_30d"]], how="left")
print(f"Shape: {df.shape}")
print(f"NaN funding_zscore_30d: {df['funding_zscore_30d'].isna().sum()}")

FEATURES_BASELINE = ["log_return", "vol_short", "vol_ratio",
                     "drawdown", "volume_z", "slope_21d"]
FEATURES_NEW      = FEATURES_BASELINE + ["funding_zscore_30d"]

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]


def run_walkforward(df, features, label=""):
    results = []
    for i, sp in enumerate(SPLITS):
        train    = df[sp["train_start"]:sp["train_end"]][features].dropna()
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        test     = df[sp["test_start"]:test_end][features].dropna()

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train)
        X_test  = scaler.transform(test)

        model = GaussianHMM(n_components=2, covariance_type="full",
                            n_iter=1000, random_state=42)
        model.fit(X_train)

        feat_idx  = features.index("drawdown")
        order     = np.argsort(model.means_[:, feat_idx])
        state_map = {order[j]: j for j in range(2)}

        states = np.array([state_map[s] for s in model.predict(X_test)])

        test_ret = df[sp["test_start"]:test_end]["log_return"].reindex(test.index)
        ret_5d   = test_ret.rolling(5).sum().shift(-5)

        bull_ret = ret_5d[states == 1].mean()
        bear_ret = ret_5d[states == 0].mean()
        delta    = bull_ret - bear_ret

        results.append({
            "split":    f"split_{i+1}",
            "n_train":  len(train),
            "n_test":   len(test),
            "bull_ret": bull_ret,
            "bear_ret": bear_ret,
            "delta_5d": delta,
        })
        print(f"  {label} split_{i+1}: "
              f"n_train={len(train):3d}  n_test={len(test):3d}  "
              f"delta={delta:+.4f}  "
              f"(bull={bull_ret:+.4f}  bear={bear_ret:+.4f})")
    return pd.DataFrame(results)


print("\n" + "=" * 60)
print("BASELINE — 6 features originais")
print("=" * 60)
res_base = run_walkforward(df, FEATURES_BASELINE, label="BASE")

print("\n" + "=" * 60)
print("RODADA 2 — baseline + funding_zscore_30d")
print("=" * 60)
res_new = run_walkforward(df, FEATURES_NEW, label="FUND")

# ── Comparação ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARACAO BASELINE vs RODADA 2")
print("=" * 60)
print(f"\n{'Split':<10} {'Base':>10} {'Rodada2':>10} {'Diff':>10}  Resultado")
print("-" * 55)

diffs = []
for rb, rn in zip(res_base.itertuples(), res_new.itertuples()):
    diff = rn.delta_5d - rb.delta_5d
    diffs.append(diff)
    result = "MELHOROU" if diff > 0 else "PIOROU"
    print(f"{rb.split:<10} {rb.delta_5d:>+10.4f} "
          f"{rn.delta_5d:>+10.4f} {diff:>+10.4f}  {result}")

base_mean = res_base["delta_5d"].mean()
new_mean  = res_new["delta_5d"].mean()
diff_mean = new_mean - base_mean
splits_ok = sum(d > 0 for d in diffs)

print(f"\n{'MEDIA':<10} {base_mean:>+10.4f} "
      f"{new_mean:>+10.4f} {diff_mean:>+10.4f}  "
      f"{'MELHOROU' if diff_mean > 0 else 'PIOROU'}")
print(f"\nSplits melhorados: {splits_ok}/3")
print(f"\nDecisao: "
      f"{'funding_zscore_30d APROVADO' if splits_ok >= 2 and diff_mean > 0 else 'funding_zscore_30d REJEITADO'}")
