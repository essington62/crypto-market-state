"""
Diagnóstico de correlação — escolher melhor feature para HMM Rodada 2.
Script exploratório — NAO modifica dados, NAO roda HMM.
Criterio de ortogonalidade: max_corr < 0.40 com qualquer feature original.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("data/01_raw/derivatives/coinglass")

# Carregar features originais do HMM
btc = pd.read_parquet("data/04_model_input/spot/daily/BTCUSDT.parquet")
FEATURES_ORIG = ["log_return", "vol_short", "vol_ratio",
                 "drawdown", "volume_z", "slope_21d"]

# Carregar candidatas
oi  = pd.read_parquet(BASE / "open_interest/BTCUSDT_aggregated.parquet")
fr  = pd.read_parquet(BASE / "funding/BTCUSDT_oi_weighted.parquet")
ls  = pd.read_parquet(BASE / "long_short_ratio/BTCUSDT_top_positions.parquet")
fg  = pd.read_parquet(BASE / "indices/fear_greed.parquet")

# Calcular features derivadas
oi["oi_zscore_30d"] = (
    (oi["open_interest_usd"] - oi["open_interest_usd"].rolling(30).mean()) /
     oi["open_interest_usd"].rolling(30).std()
)
fr["funding_zscore_30d"] = (
    (fr["funding_rate_oi_weighted"] - fr["funding_rate_oi_weighted"].rolling(30).mean()) /
     fr["funding_rate_oi_weighted"].rolling(30).std()
)

# Coluna ratio do top positions
ratio_col = [c for c in ls.columns if "ratio" in c.lower()][0]
ls = ls.rename(columns={ratio_col: "top_position_ratio"})

# Montar DataFrame único
df = btc[FEATURES_ORIG].copy()
df = df.join(oi[["oi_zscore_30d"]], how="left")
df = df.join(fr[["funding_zscore_30d"]], how="left")
df = df.join(ls[["top_position_ratio"]], how="left")
df = df.join(fg[["fear_greed"]], how="left")

# Filtrar pos-2023 (período relevante)
df = df[df.index >= "2023-01-01"].dropna()
print(f"Shape pos-2023: {df.shape}")

CANDIDATES = ["oi_zscore_30d", "funding_zscore_30d",
              "top_position_ratio", "fear_greed"]

print("\n" + "=" * 65)
print("CORRELACAO CANDIDATAS vs FEATURES ORIGINAIS HMM")
print("=" * 65)
print(f"\n{'Feature':<25} " + " ".join(f"{f:>12}" for f in FEATURES_ORIG))
print("-" * 65)

max_corr = {}
for cand in CANDIDATES:
    corrs = []
    for feat in FEATURES_ORIG:
        r = df[[cand, feat]].dropna().corr().iloc[0, 1]
        corrs.append(r)
    max_abs = max(abs(r) for r in corrs)
    max_corr[cand] = max_abs
    flag = "  REDUNDANTE" if max_abs > 0.5 else ("  LIMIAR" if max_abs > 0.4 else "  ORTOGONAL")
    print(f"{cand:<25} " +
          " ".join(f"{r:>+12.3f}" for r in corrs) +
          f"  max_abs={max_abs:.3f}{flag}")

print("\n" + "=" * 65)
print("CORRELACAO ENTRE AS PROPRIAS CANDIDATAS")
print("=" * 65)
corr_matrix = df[CANDIDATES].corr().round(3)
print(corr_matrix.to_string())

print("\n" + "=" * 65)
print("RANKING — menor correlacao maxima = mais ortogonal")
print("=" * 65)
ranking = sorted(max_corr.items(), key=lambda x: x[1])
for i, (feat, max_c) in enumerate(ranking):
    status = "FORTE  <-- TESTAR PRIMEIRO" if max_c < 0.40 else \
             ("LIMIAR" if max_c < 0.50 else "REDUNDANTE")
    print(f"  {i+1}. {feat:<25} max_corr={max_c:.3f}  {status}")

print("\n" + "=" * 65)
print("COBERTURA — % de rows validos pos-2023")
print("=" * 65)
df_full = btc[FEATURES_ORIG].copy()
df_full = df_full.join(oi[["oi_zscore_30d"]], how="left")
df_full = df_full.join(fr[["funding_zscore_30d"]], how="left")
df_full = df_full.join(ls[["top_position_ratio"]], how="left")
df_full = df_full.join(fg[["fear_greed"]], how="left")
df_full = df_full[df_full.index >= "2023-01-01"]

for cand in CANDIDATES:
    valid = df_full[cand].notna().sum()
    total = len(df_full)
    print(f"  {cand:<25} {valid}/{total} ({valid/total:.1%})")
