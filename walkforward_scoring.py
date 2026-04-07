"""
Walk-Forward Validation: Scoring System nos períodos Sideways históricos do R5C.
Objetivo: validar se BB < 0.30 tem edge real com 200+ sinais.
Usa dados daily + 1h para calcular BB, RSI e simular o scoring.
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path("/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state")
XGB_PROJECT = Path("/Users/brown/Documents/MLGeral/xgboost_R5c_1h")

# ══════════════════════════════════════════════════════════
# 1. Carregar R5C e classificar todos os dias
# ══════════════════════════════════════════════════════════
with open(XGB_PROJECT / "data/03_models/r5c_hmm.pkl", "rb") as f:
    hmm = pickle.load(f)

means = hmm.means_[:, 0]
bull_st, bear_st = int(means.argmax()), int(means.argmin())
side_st = [i for i in range(3) if i != bull_st and i != bear_st][0]
state_names = {bull_st: "Bull", bear_st: "Bear", side_st: "Sideways"}

daily = pd.read_parquet(PROJECT / "data/02_intermediate/spot/daily/BTCUSDT.parquet")
daily.index = pd.to_datetime(daily.index, utc=True)
close_d = daily["close"]

feat = pd.DataFrame(index=daily.index)
feat["log_return"] = np.log(close_d / close_d.shift(1))
feat["vol_short"] = feat["log_return"].rolling(7).std()
feat["vol_ratio"] = feat["vol_short"] / feat["log_return"].rolling(30).std()
feat["drawdown"] = close_d / close_d.rolling(63).max() - 1
feat["volume_z"] = (daily["volume"] - daily["volume"].rolling(30).mean()) / daily["volume"].rolling(30).std()
feat["slope_21d"] = close_d.pct_change(21)
feat = feat.dropna()

X_daily = feat[["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]].values
states = hmm.predict(X_daily)
feat["regime"] = [state_names[s] for s in states]

print(f"Total dias: {len(feat)}")
print(f"Sideways: {(feat['regime']=='Sideways').sum()} dias")
print(f"Bull: {(feat['regime']=='Bull').sum()} dias")
print(f"Bear: {(feat['regime']=='Bear').sum()} dias")

# ══════════════════════════════════════════════════════════
# 2. Carregar candles diários e calcular BB e RSI em daily
# ══════════════════════════════════════════════════════════
close = daily["close"]

# RSI 14 daily
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
rsi_daily = 100 - (100 / (1 + rs))

# BB %B (20 períodos daily)
sma20 = close.rolling(20).mean()
std20 = close.rolling(20).std()
upper = sma20 + 2 * std20
lower = sma20 - 2 * std20
bb_daily = (close - lower) / (upper - lower)

feat["rsi"] = rsi_daily
feat["bb"] = bb_daily
feat["close"] = close

# Retornos futuros
feat["ret_1d"] = close.shift(-1) / close - 1
feat["ret_3d"] = close.shift(-3) / close - 1
feat["ret_5d"] = close.shift(-5) / close - 1

# Max DD 3d (calculado sobre close alinhado ao índice de feat)
close_feat = feat["close"]
max_dd_3d = []
for i in range(len(close_feat)):
    if i + 3 < len(close_feat):
        window = close_feat.iloc[i+1:i+4]
        dd = (window.min() / close_feat.iloc[i]) - 1
        max_dd_3d.append(dd)
    else:
        max_dd_3d.append(np.nan)
feat["max_dd_3d"] = max_dd_3d

feat = feat.dropna()

# ══════════════════════════════════════════════════════════
# 3. Filtrar apenas Sideways
# ══════════════════════════════════════════════════════════
sideways = feat[feat["regime"] == "Sideways"].copy()
print(f"\nSideways com dados completos: {len(sideways)} dias")

# ══════════════════════════════════════════════════════════
# 4. Aplicar scoring system
# ══════════════════════════════════════════════════════════
def compute_score(row):
    score = 0
    bb = row["bb"]
    rsi = row["rsi"]
    
    # BB score
    if bb < 0.20:
        score += 2.0
    elif bb < 0.30:
        score += 1.5
    elif bb < 0.40:
        score += 0.5
    
    # RSI score
    if rsi < 35:
        score += 1.0
    elif rsi < 45:
        score += 0.5
    elif rsi > 60:
        score -= 1.0
    
    return score

sideways["score"] = sideways.apply(compute_score, axis=1)

# Nota: alloc e news nao disponiveis no historico, so BB + RSI
print(f"\nScore distribution (BB + RSI only, sem alloc/news):")
print(sideways["score"].describe().round(2))

# ══════════════════════════════════════════════════════════
# 5. Analisar por faixa de score
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("WALK-FORWARD: SCORING SYSTEM EM SIDEWAYS HISTORICOS")
print(f"Periodo: {sideways.index.min().date()} a {sideways.index.max().date()}")
print(f"Total sinais Sideways: {len(sideways)}")
print(f"{'=' * 70}")

# Por faixa de score
bins = [(-10, 0), (0, 0.5), (0.5, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 5)]
print(f"\n{'Score Range':>15} | {'N':>5} | {'ret_1d':>8} | {'ret_3d':>8} | {'ret_5d':>8} | {'win_1d':>7} | {'win_3d':>7} | {'MaxDD3d':>8}")
print("-" * 85)

for lo, hi in bins:
    mask = (sideways["score"] >= lo) & (sideways["score"] < hi)
    subset = sideways[mask]
    if len(subset) == 0:
        continue
    r1 = subset["ret_1d"].mean() * 100
    r3 = subset["ret_3d"].mean() * 100
    r5 = subset["ret_5d"].mean() * 100
    w1 = (subset["ret_1d"] > 0).mean() * 100
    w3 = (subset["ret_3d"] > 0).mean() * 100
    dd3 = subset["max_dd_3d"].mean() * 100
    print(f"  [{lo:+.1f}, {hi:+.1f}) | {len(subset):5d} | {r1:+.3f}% | {r3:+.3f}% | {r5:+.3f}% | {w1:5.0f}% | {w3:5.0f}% | {dd3:+.3f}%")

# ══════════════════════════════════════════════════════════
# 6. BB isolado (confirmar se BB < 0.30 tem edge em 200+ sinais)
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("BB ISOLADO EM SIDEWAYS")
print(f"{'=' * 70}")

bb_bins = [(0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
           (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.1)]

print(f"\n{'BB Range':>15} | {'N':>5} | {'ret_1d':>8} | {'ret_3d':>8} | {'ret_5d':>8} | {'win_1d':>7} | {'win_3d':>7}")
print("-" * 75)

for lo, hi in bb_bins:
    mask = (sideways["bb"] >= lo) & (sideways["bb"] < hi)
    subset = sideways[mask]
    if len(subset) == 0:
        continue
    r1 = subset["ret_1d"].mean() * 100
    r3 = subset["ret_3d"].mean() * 100
    r5 = subset["ret_5d"].mean() * 100
    w1 = (subset["ret_1d"] > 0).mean() * 100
    w3 = (subset["ret_3d"] > 0).mean() * 100
    print(f"  [{lo:.2f}, {hi:.2f}) | {len(subset):5d} | {r1:+.3f}% | {r3:+.3f}% | {r5:+.3f}% | {w1:5.0f}% | {w3:5.0f}%")

# ══════════════════════════════════════════════════════════
# 7. RSI isolado em Sideways
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("RSI ISOLADO EM SIDEWAYS")
print(f"{'=' * 70}")

rsi_bins = [(0, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50),
            (50, 55), (55, 60), (60, 65), (65, 70), (70, 100)]

print(f"\n{'RSI Range':>15} | {'N':>5} | {'ret_1d':>8} | {'ret_3d':>8} | {'ret_5d':>8} | {'win_1d':>7} | {'win_3d':>7}")
print("-" * 75)

for lo, hi in rsi_bins:
    mask = (sideways["rsi"] >= lo) & (sideways["rsi"] < hi)
    subset = sideways[mask]
    if len(subset) == 0:
        continue
    r1 = subset["ret_1d"].mean() * 100
    r3 = subset["ret_3d"].mean() * 100
    r5 = subset["ret_5d"].mean() * 100
    w1 = (subset["ret_1d"] > 0).mean() * 100
    w3 = (subset["ret_3d"] > 0).mean() * 100
    print(f"  [{lo:3d}, {hi:3d}) | {len(subset):5d} | {r1:+.3f}% | {r3:+.3f}% | {r5:+.3f}% | {w1:5.0f}% | {w3:5.0f}%")

# ══════════════════════════════════════════════════════════
# 8. BB + RSI combinado (top combos)
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("BB + RSI COMBINADO EM SIDEWAYS")
print(f"{'=' * 70}")

combos = [
    ("BB<0.20", sideways[sideways["bb"] < 0.20]),
    ("BB<0.30", sideways[sideways["bb"] < 0.30]),
    ("BB<0.20 + RSI<35", sideways[(sideways["bb"] < 0.20) & (sideways["rsi"] < 35)]),
    ("BB<0.30 + RSI<45", sideways[(sideways["bb"] < 0.30) & (sideways["rsi"] < 45)]),
    ("BB<0.40 + RSI<40", sideways[(sideways["bb"] < 0.40) & (sideways["rsi"] < 40)]),
    ("RSI<35", sideways[sideways["rsi"] < 35]),
    ("RSI<40", sideways[sideways["rsi"] < 40]),
    ("Score >= 2.0", sideways[sideways["score"] >= 2.0]),
    ("Score >= 2.5", sideways[sideways["score"] >= 2.5]),
    ("Score >= 3.0", sideways[sideways["score"] >= 3.0]),
    ("BB>0.70 (topo)", sideways[sideways["bb"] > 0.70]),
    ("RSI>60 (overbought)", sideways[sideways["rsi"] > 60]),
]

print(f"\n{'Combo':>25} | {'N':>5} | {'ret_1d':>8} | {'ret_3d':>8} | {'ret_5d':>8} | {'win_3d':>7} | {'MaxDD3d':>8}")
print("-" * 85)

for name, subset in combos:
    if len(subset) == 0:
        print(f"  {name:>23} | {0:5d} | {'N/A':>8} |")
        continue
    r1 = subset["ret_1d"].mean() * 100
    r3 = subset["ret_3d"].mean() * 100
    r5 = subset["ret_5d"].mean() * 100
    w3 = (subset["ret_3d"] > 0).mean() * 100
    dd3 = subset["max_dd_3d"].mean() * 100
    print(f"  {name:>23} | {len(subset):5d} | {r1:+.3f}% | {r3:+.3f}% | {r5:+.3f}% | {w3:5.0f}% | {dd3:+.3f}%")

# ══════════════════════════════════════════════════════════
# 9. Walk-forward por período (estabilidade temporal)
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("ESTABILIDADE TEMPORAL: Score >= 2.0 por semestre")
print(f"{'=' * 70}")

sideways["half"] = sideways.index.to_period("2Q")
for period, group in sideways.groupby("half"):
    high_score = group[group["score"] >= 2.0]
    if len(high_score) == 0:
        print(f"  {period}: 0 sinais")
        continue
    r3 = high_score["ret_3d"].mean() * 100
    w3 = (high_score["ret_3d"] > 0).mean() * 100
    print(f"  {period}: {len(high_score):3d} sinais | ret_3d={r3:+.3f}% | win_3d={w3:.0f}%")

# Salvar
output = PROJECT / "data/06_reports/walkforward_sideways_scoring.csv"
sideways.to_csv(output)
print(f"\nSalvo em: {output}")
