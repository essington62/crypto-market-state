"""
Análise Cruzada: L1 (R5C) / L2 (Specialist) / L3 (Timing) vs Mercado Real
Período: 2026-03-24 a 2026-04-06

Pergunta central: onde cada layer acertou/errou?
- R11 bloqueou e BTC subiu? (oportunidade perdida)
- R5C teria permitido entrada? (Sideways vs Bear)
- Specialist queria entrar? (alloc_raw > threshold)
- O timing teria confirmado? (RSI + BB oversold)
- Resultado real: BTC subiu ou caiu nas 4h/12h/24h seguintes?
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path("/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state")
XGB_PROJECT = Path("/Users/brown/Documents/MLGeral/xgboost_R5c_1h")

# ══════════════════════════════════════════════════════════
# 1. Carregar sinais do specialist
# ══════════════════════════════════════════════════════════
signals = pd.read_csv(PROJECT / "scripts/paper_trading/state/specialist_4h/signals.csv")
print(f"Sinais specialist: {len(signals)} rows")
print(f"Colunas: {list(signals.columns)[:10]}...")

# Identificar colunas de timestamp e preço
# A primeira coluna é timestamp, terceira é preço
cols = signals.columns.tolist()
signals.columns = [c.strip() for c in cols]

# Renomear para facilitar (ajustar conforme os nomes reais)
ts_col = signals.columns[0]
price_col = signals.columns[2]  # btc_close
alloc_raw_col = signals.columns[3]  # allocation_raw
alloc_final_col = signals.columns[4]  # allocation_final

signals["timestamp"] = pd.to_datetime(signals[ts_col], utc=True)
signals["btc_price"] = pd.to_numeric(signals[price_col], errors="coerce")
signals["alloc_raw"] = pd.to_numeric(signals[alloc_raw_col], errors="coerce")
signals["alloc_final"] = pd.to_numeric(signals[alloc_final_col], errors="coerce")

# Tentar pegar prob_bull e action
try:
    signals["prob_bull"] = pd.to_numeric(signals[signals.columns[5]], errors="coerce")
except:
    signals["prob_bull"] = np.nan

action_col = [c for c in signals.columns if "action" in c.lower() or "hold" in str(signals[c].iloc[0]).upper() or c == signals.columns[17]]
if action_col:
    signals["action"] = signals[action_col[0]]
else:
    signals["action"] = signals.iloc[:, 17] if len(signals.columns) > 17 else "UNKNOWN"

signals = signals.set_index("timestamp").sort_index()

# ══════════════════════════════════════════════════════════
# 2. Carregar candles 1h para calcular retornos futuros
# ══════════════════════════════════════════════════════════
candles = pd.read_parquet(PROJECT / "data/01_raw/spot/crypto/1h/BTCUSDT_1h.parquet")
candles.index = pd.to_datetime(candles.index, utc=True)
candles = candles.sort_index()
close_1h = candles["close"]

# ══════════════════════════════════════════════════════════
# 3. Calcular R5C para o período
# ══════════════════════════════════════════════════════════
with open(XGB_PROJECT / "data/03_models/r5c_hmm.pkl", "rb") as f:
    hmm = pickle.load(f)

means = hmm.means_[:, 0]
bull_st, bear_st = int(means.argmax()), int(means.argmin())
side_st = [i for i in range(3) if i != bull_st and i != bear_st][0]
state_names = {bull_st: "Bull", bear_st: "Bear", side_st: "Sideways"}

daily = pd.read_parquet(PROJECT / "data/02_intermediate/spot/daily/BTCUSDT.parquet")
daily.index = pd.to_datetime(daily.index, utc=True)
close_daily = daily["close"]

df_feat = pd.DataFrame(index=daily.index)
df_feat["log_return"] = np.log(close_daily / close_daily.shift(1))
df_feat["vol_short"] = df_feat["log_return"].rolling(7).std()
df_feat["vol_ratio"] = df_feat["vol_short"] / df_feat["log_return"].rolling(30).std()
df_feat["drawdown"] = close_daily / close_daily.rolling(63).max() - 1
df_feat["volume_z"] = (daily["volume"] - daily["volume"].rolling(30).mean()) / daily["volume"].rolling(30).std()
df_feat["slope_21d"] = close_daily.pct_change(21)
df_feat = df_feat.dropna()

feats = df_feat[["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]].values
r5c_probs = hmm.predict_proba(feats)
r5c_states = hmm.predict(feats)

r5c_df = pd.DataFrame({
    "r5c_regime": [state_names[s] for s in r5c_states],
    "r5c_prob_bull": r5c_probs[:, bull_st],
    "r5c_prob_bear": r5c_probs[:, bear_st],
    "r5c_prob_side": r5c_probs[:, side_st],
}, index=df_feat.index)

# ══════════════════════════════════════════════════════════
# 4. Calcular RSI e BB para cada sinal (do candle 1h)
# ══════════════════════════════════════════════════════════
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_bb_pctb(series, period=20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (series - lower) / (upper - lower)

rsi_1h = calc_rsi(close_1h)
bb_1h = calc_bb_pctb(close_1h)

# ══════════════════════════════════════════════════════════
# 5. Montar tabela cruzada
# ══════════════════════════════════════════════════════════
results = []

for ts, row in signals.iterrows():
    price = row["btc_price"]
    alloc_raw = row["alloc_raw"]
    alloc_final = row["alloc_final"]
    action = row.get("action", "UNKNOWN")

    # R5C regime para este dia
    day = ts.normalize()
    r5c_row = r5c_df.loc[:day].iloc[-1] if day in r5c_df.index or len(r5c_df.loc[:day]) > 0 else None
    r5c_regime = r5c_row["r5c_regime"] if r5c_row is not None else "N/A"

    # RSI e BB no momento do sinal (1h mais próximo)
    nearest_1h = close_1h.index.asof(ts)
    rsi_val = rsi_1h.loc[nearest_1h] if nearest_1h is not pd.NaT else np.nan
    bb_val = bb_1h.loc[nearest_1h] if nearest_1h is not pd.NaT else np.nan

    # Retornos futuros (4h, 12h, 24h)
    future_prices = {}
    for hours, label in [(4, "ret_4h"), (12, "ret_12h"), (24, "ret_24h")]:
        future_ts = ts + pd.Timedelta(hours=hours)
        future_idx = close_1h.index.asof(future_ts)
        if future_idx is not pd.NaT and future_idx in close_1h.index:
            future_prices[label] = (close_1h.loc[future_idx] / price - 1) * 100
        else:
            future_prices[label] = np.nan

    # Max drawdown nas próximas 4h
    window_start = close_1h.index.searchsorted(ts)
    window_end = close_1h.index.searchsorted(ts + pd.Timedelta(hours=4))
    if window_end > window_start:
        window = close_1h.iloc[window_start:window_end]
        max_dd_4h = ((window.min() - price) / price) * 100
    else:
        max_dd_4h = np.nan

    # Decisão hipotética com R5C
    threshold = 0.38
    would_enter_r5c = (alloc_raw > threshold) and (r5c_regime in ["Bull", "Sideways"])
    would_enter_r11 = alloc_final > 0  # R11 permitiu?

    results.append({
        "timestamp": ts,
        "date": ts.strftime("%m/%d"),
        "hour": ts.strftime("%H:%M"),
        "btc_price": price,
        "alloc_raw": alloc_raw,
        "alloc_final": alloc_final,
        "action": action,
        "r5c_regime": r5c_regime,
        "rsi_1h": round(rsi_val, 1) if not np.isnan(rsi_val) else np.nan,
        "bb_1h": round(bb_val, 2) if not np.isnan(bb_val) else np.nan,
        "ret_4h": round(future_prices["ret_4h"], 2) if not np.isnan(future_prices["ret_4h"]) else np.nan,
        "ret_12h": round(future_prices["ret_12h"], 2) if not np.isnan(future_prices["ret_12h"]) else np.nan,
        "ret_24h": round(future_prices["ret_24h"], 2) if not np.isnan(future_prices["ret_24h"]) else np.nan,
        "max_dd_4h": round(max_dd_4h, 2) if not np.isnan(max_dd_4h) else np.nan,
        "would_enter_r5c": would_enter_r5c,
        "would_enter_r11": would_enter_r11,
    })

analysis = pd.DataFrame(results)

# ══════════════════════════════════════════════════════════
# 6. Relatório
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANÁLISE CRUZADA: L1/L2/L3 vs MERCADO REAL")
print(f"Período: {analysis['date'].iloc[0]} a {analysis['date'].iloc[-1]}")
print(f"Sinais analisados: {len(analysis)}")
print("=" * 80)

# 6a. Resumo geral
print("\n── RESUMO GERAL ──")
print(f"BTC início: ${analysis['btc_price'].iloc[0]:,.0f}")
print(f"BTC fim:    ${analysis['btc_price'].iloc[-1]:,.0f}")
print(f"Variação:   {(analysis['btc_price'].iloc[-1]/analysis['btc_price'].iloc[0]-1)*100:+.1f}%")

# 6b. O que o specialist queria fazer
wanted_entry = analysis[analysis["alloc_raw"] > 0.38]
print(f"\n── SPECIALIST (Layer 2) ──")
print(f"Sinais onde queria entrar (alloc_raw > 0.38): {len(wanted_entry)} de {len(analysis)} ({len(wanted_entry)/len(analysis)*100:.0f}%)")
print(f"Sinais onde NÃO queria entrar: {len(analysis) - len(wanted_entry)}")
if len(wanted_entry) > 0:
    print(f"Retorno médio 4h quando queria entrar: {wanted_entry['ret_4h'].mean():+.2f}%")
    print(f"Retorno médio 12h quando queria entrar: {wanted_entry['ret_12h'].mean():+.2f}%")
    print(f"Retorno médio 24h quando queria entrar: {wanted_entry['ret_24h'].mean():+.2f}%")
    print(f"Max DD médio 4h: {wanted_entry['max_dd_4h'].mean():+.2f}%")

# 6c. R11 vs R5C
print(f"\n── REGIME GATE: R11 vs R5C ──")
r11_blocked = analysis[analysis["alloc_final"] == 0]
r5c_would_allow = analysis[analysis["would_enter_r5c"] == True]
r11_blocked_r5c_would = analysis[(analysis["alloc_final"] == 0) & (analysis["would_enter_r5c"] == True)]

print(f"R11 bloqueou: {len(r11_blocked)} sinais")
print(f"R5C teria permitido: {len(r5c_would_allow)} sinais")
print(f"R11 bloqueou MAS R5C teria permitido: {len(r11_blocked_r5c_would)} sinais")

if len(r11_blocked_r5c_would) > 0:
    print(f"\n  Esses {len(r11_blocked_r5c_would)} sinais perdidos:")
    print(f"  Retorno médio 4h:  {r11_blocked_r5c_would['ret_4h'].mean():+.2f}%")
    print(f"  Retorno médio 12h: {r11_blocked_r5c_would['ret_12h'].mean():+.2f}%")
    print(f"  Retorno médio 24h: {r11_blocked_r5c_would['ret_24h'].mean():+.2f}%")
    print(f"  Max DD médio 4h:   {r11_blocked_r5c_would['max_dd_4h'].mean():+.2f}%")

    # Quantos teriam sido lucrativos
    profitable_4h = (r11_blocked_r5c_would["ret_4h"] > 0).sum()
    profitable_12h = (r11_blocked_r5c_would["ret_12h"] > 0).sum()
    total = len(r11_blocked_r5c_would)
    print(f"  Win rate 4h:  {profitable_4h}/{total} ({profitable_4h/total*100:.0f}%)")
    print(f"  Win rate 12h: {profitable_12h}/{total} ({profitable_12h/total*100:.0f}%)")

# 6d. Gate Técnico (RSI + BB)
print(f"\n── GATE TÉCNICO (RSI + BB) ──")
oversold = analysis[(analysis["rsi_1h"] < 35) & (analysis["bb_1h"] < 0.25)]
overbought = analysis[(analysis["rsi_1h"] > 65) | (analysis["bb_1h"] > 0.75)]
neutral = analysis[~analysis.index.isin(oversold.index) & ~analysis.index.isin(overbought.index)]

print(f"Oversold (RSI<35 + BB<0.25): {len(oversold)} sinais")
if len(oversold) > 0:
    print(f"  Ret 4h:  {oversold['ret_4h'].mean():+.2f}%  (win: {(oversold['ret_4h']>0).sum()}/{len(oversold)})")
    print(f"  Ret 12h: {oversold['ret_12h'].mean():+.2f}%")
    print(f"  Ret 24h: {oversold['ret_24h'].mean():+.2f}%")

print(f"\nOverbought (RSI>65 ou BB>0.75): {len(overbought)} sinais")
if len(overbought) > 0:
    print(f"  Ret 4h:  {overbought['ret_4h'].mean():+.2f}%  (win: {(overbought['ret_4h']>0).sum()}/{len(overbought)})")
    print(f"  Ret 12h: {overbought['ret_12h'].mean():+.2f}%")

print(f"\nNeutral: {len(neutral)} sinais")
if len(neutral) > 0:
    print(f"  Ret 4h:  {neutral['ret_4h'].mean():+.2f}%")

# 6e. Melhores entradas perdidas
print(f"\n── TOP 10 MELHORES ENTRADAS PERDIDAS ──")
print("(R11 bloqueou, R5C teria permitido, ordenado por ret_12h)")
if len(r11_blocked_r5c_would) > 0:
    top = r11_blocked_r5c_would.nlargest(10, "ret_12h")
    for _, r in top.iterrows():
        print(f"  {r['date']} {r['hour']} ${r['btc_price']:,.0f} | alloc={r['alloc_raw']:.2f} RSI={r['rsi_1h']} BB={r['bb_1h']} | ret4h={r['ret_4h']:+.1f}% ret12h={r['ret_12h']:+.1f}% ret24h={r['ret_24h']:+.1f}%")

# 6f. Piores entradas que R5C teria permitido
print(f"\n── TOP 5 PIORES ENTRADAS (R5C teria permitido) ──")
if len(r5c_would_allow) > 0:
    worst = r5c_would_allow.nsmallest(5, "ret_12h")
    for _, r in worst.iterrows():
        print(f"  {r['date']} {r['hour']} ${r['btc_price']:,.0f} | alloc={r['alloc_raw']:.2f} RSI={r['rsi_1h']} BB={r['bb_1h']} | ret4h={r['ret_4h']:+.1f}% ret12h={r['ret_12h']:+.1f}%")

# 6g. Tabela diária resumida
print(f"\n── RESUMO DIÁRIO ──")
analysis["day"] = analysis["timestamp"].dt.date
daily_summary = analysis.groupby("day").agg({
    "btc_price": ["first", "last"],
    "alloc_raw": "mean",
    "r5c_regime": "first",
    "rsi_1h": "mean",
    "ret_4h": "mean",
    "would_enter_r5c": "sum",
}).round(2)

print(daily_summary.to_string())

# Salvar para referência
output_path = PROJECT / "data/06_reports/cross_analysis_l1_l2_l3.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
analysis.to_csv(output_path, index=False)
print(f"\nSalvo em: {output_path}")
