"""
HMM Rodada 4: diag + top_position_ratio (sem funding).
Comparar A (full 6f) vs D (diag 8f) vs E (diag 7f).
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
ls  = pd.read_parquet(BASE / "long_short_ratio/BTCUSDT_top_positions.parquet")

fr["funding_zscore_30d"] = (
    (fr["funding_rate_oi_weighted"] -
     fr["funding_rate_oi_weighted"].rolling(30).mean()) /
     fr["funding_rate_oi_weighted"].rolling(30).std()
)
ratio_col = [c for c in ls.columns if "ratio" in c.lower()][0]
ls = ls.rename(columns={ratio_col: "top_position_ratio"})

df = btc.join(fr[["funding_zscore_30d"]], how="left")
df = df.join(ls[["top_position_ratio"]], how="left")

FEATURES_BASE = ["log_return", "vol_short", "vol_ratio",
                 "drawdown", "volume_z", "slope_21d"]
FEATURES_D    = FEATURES_BASE + ["funding_zscore_30d", "top_position_ratio"]
FEATURES_E    = FEATURES_BASE + ["top_position_ratio"]

SPLITS = [
    {"train_start": "2023-01-01", "train_end": "2023-12-31",
     "test_start":  "2024-01-01", "test_end":  "2024-06-30"},
    {"train_start": "2023-01-01", "train_end": "2024-06-30",
     "test_start":  "2024-07-01", "test_end":  "2024-12-31"},
    {"train_start": "2023-01-01", "train_end": "2024-12-31",
     "test_start":  "2025-01-01", "test_end":  None},
]


def run_walkforward(df, features, cov_type="full", label=""):
    results = []
    for i, sp in enumerate(SPLITS):
        train    = df[sp["train_start"]:sp["train_end"]][features].dropna()
        test_end = sp["test_end"] or df.index.max().strftime("%Y-%m-%d")
        test     = df[sp["test_start"]:test_end][features].dropna()

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train)
        X_test  = scaler.transform(test)

        model = GaussianHMM(n_components=2, covariance_type=cov_type,
                            n_iter=1000, random_state=42)
        model.fit(X_train)

        feat_idx  = features.index("drawdown")
        order     = np.argsort(model.means_[:, feat_idx])
        state_map = {order[j]: j for j in range(2)}
        states    = np.array([state_map[s] for s in model.predict(X_test)])

        test_ret = df[sp["test_start"]:test_end]["log_return"].reindex(test.index)
        ret_5d   = test_ret.rolling(5).sum().shift(-5)

        bull_ret = ret_5d[states == 1].mean()
        bear_ret = ret_5d[states == 0].mean()
        delta    = bull_ret - bear_ret

        results.append({
            "split":    f"split_{i+1}",
            "n_train":  len(train),
            "n_test":   len(test),
            "delta_5d": delta,
        })
        print(f"  {label} split_{i+1}: "
              f"n_train={len(train):3d}  n_test={len(test):3d}  "
              f"delta={delta:+.4f}")
    return pd.DataFrame(results)


configs = [
    (FEATURES_BASE, "full", "A — baseline full (6f)          "),
    (FEATURES_D,    "diag", "D — diag + funding + top (8f)   "),
    (FEATURES_E,    "diag", "E — diag + top_position (7f)    "),
]

all_results = {}
for features, cov, label in configs:
    print(f"\n{'=' * 60}")
    print(label)
    print(f"{'=' * 60}")
    all_results[label.strip()] = run_walkforward(
        df, features, cov_type=cov, label=label[:4]
    )

# ── Tabela comparativa ────────────────────────────────────────
print("\n" + "=" * 70)
print("TABELA COMPARATIVA — A vs D vs E")
print("=" * 70)

labels = [c[2].strip() for c in configs]
print(f"{'Split':<10}" + "".join(f"{l[:24]:>26}" for l in labels))
print("-" * 70)

for i in range(3):
    row = f"split_{i+1:<5}"
    for label in labels:
        d = all_results[label]["delta_5d"].iloc[i]
        row += f"{d:>+26.4f}"
    print(row)

print("-" * 70)
row = f"{'Media':<10}"
means = {}
for label in labels:
    m = all_results[label]["delta_5d"].mean()
    means[label] = m
    row += f"{m:>+26.4f}"
print(row)

# ── Veredicto ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VEREDICTO")
print("=" * 70)
base_mean   = means[labels[0]]
base_deltas = all_results[labels[0]]["delta_5d"].values

for label in labels[1:]:
    diff      = means[label] - base_mean
    splits_ok = (all_results[label]["delta_5d"].values > base_deltas).sum()
    status    = "APROVADO" if splits_ok >= 2 and diff > 0 else "REJEITADO"
    print(f"  {label[:40]}: diff={diff:+.4f}  splits={splits_ok}/3  {status}")

best = max(means, key=means.get)
print(f"\n  Melhor delta medio: {best}")
print(f"  Delta medio:        {means[best]:+.4f}")

# Comparação E vs D
mean_e = means[labels[2]]
mean_d = means[labels[1]]
print(f"\n  E vs D: {mean_e - mean_d:+.4f}  "
      f"{'E e o candidato final' if mean_e > mean_d else 'D permanece como melhor candidato'}")

# ── Diagnóstico adicional ─────────────────────────────────────
print("\n" + "=" * 70)
print("DIAGNOSTICO — top_position_ratio no treino")
print("=" * 70)
df_post23 = df[df.index >= "2023-01-01"].dropna(subset=FEATURES_E)
r = df_post23[["top_position_ratio", "slope_21d",
               "drawdown", "log_return"]].corr().round(3)
print(r.to_string())
