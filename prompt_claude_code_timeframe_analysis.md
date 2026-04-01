# TASK: Análise Estatística de Timeframe Ótimo para Day Trade

## Contexto
Specialist 4h AB aprovado como decisor estratégico (Sharpe s4=+3.872).
Queremos adicionar camada de execução em timeframe mais curto para:
- Timing de entrada preciso (pegar fundo local)
- Stop loss/gain monitorado com alta frequência
- Alvo: movimentos de 1-2% em BTC

Precisamos definir com dados (não por intuição) qual timeframe é ótimo:
15min, 30min ou 1h para execução e contexto.

## Parte 1 — Download de dados Binance Vision

### Passo 1 — Baixar candles 15min de BTCUSDT
Fonte: https://data.binance.vision/
Path: data/klines/BTCUSDT/15m/

Baixar dados mensais de set/2025 até mar/2026 (alinhado com período de CoinGlass/treino).

```python
import urllib.request
import zipfile
import os
import pandas as pd

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m"
OUTPUT_DIR = "data/01_raw/spot/crypto/15m"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Meses para baixar
months = [
    "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03"
]

for month in months:
    filename = f"BTCUSDT-15m-{month}.zip"
    url = f"{BASE_URL}/{filename}"
    zip_path = f"/tmp/{filename}"
    
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(OUTPUT_DIR)
        print(f"  ✓ {filename}")
    except Exception as e:
        print(f"  ✗ {filename}: {e}")
        # Mês atual pode não ter arquivo mensal ainda
        # Tentar daily para o mês corrente
```

NOTA: para março/2026 (mês corrente), pode precisar baixar arquivos diários:
```
BASE_URL_DAILY = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/15m"
# BTCUSDT-15m-2026-03-01.zip, ..., BTCUSDT-15m-2026-03-25.zip
```

### Passo 2 — Converter para parquet
```python
# Colunas do CSV da Binance Vision (sem header)
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore"
]

# Ler todos os CSVs, concatenar, converter
all_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')])
dfs = []
for f in all_files:
    df = pd.read_csv(f"{OUTPUT_DIR}/{f}", header=None, names=COLUMNS)
    dfs.append(df)

df = pd.concat(dfs).drop_duplicates(subset="open_time").sort_values("open_time")
df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
df = df.set_index("timestamp")
for col in ["open", "high", "low", "close", "volume", "quote_volume", 
            "taker_buy_volume", "taker_buy_quote_volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

df.to_parquet(f"{OUTPUT_DIR}/BTCUSDT_15m.parquet")
print(f"Parquet salvo: {len(df)} candles, {df.index.min()} → {df.index.max()}")
```

### Passo 3 — Agregar para 30min e 1h
```python
def resample_ohlcv(df, freq):
    return df.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "taker_buy_volume": "sum",
        "trades": "sum",
    }).dropna(subset=["open"])

df_30m = resample_ohlcv(df, "30min")
df_1h = resample_ohlcv(df, "1h")

df_30m.to_parquet(f"data/01_raw/spot/crypto/30m/BTCUSDT_30m.parquet")
df_1h.to_parquet(f"data/01_raw/spot/crypto/1h/BTCUSDT_1h.parquet")

print(f"30m: {len(df_30m)} candles")
print(f"1h:  {len(df_1h)} candles")
```

## Parte 2 — Análise Estatística

Criar script: `scripts/analysis/timeframe_analysis.py`

### Análise 1 — Time-to-Target
**Pergunta:** Quando BTC sobe X% a partir de um ponto, em quanto tempo atinge?

```python
def time_to_target(df_15m, targets=[0.01, 0.015, 0.02]):
    """
    Para cada candle, medir quanto tempo até o preço subir X%.
    Retornar distribuição de tempos por alvo.
    """
    results = []
    close = df_15m['close'].values
    timestamps = df_15m.index
    
    for i in range(len(close) - 1):
        entry = close[i]
        for target in targets:
            target_price = entry * (1 + target)
            # Procurar primeiro candle que atinge o alvo
            for j in range(i + 1, min(i + 96, len(close))):  # max 24h lookforward
                if df_15m['high'].values[j] >= target_price:
                    time_minutes = (timestamps[j] - timestamps[i]).total_seconds() / 60
                    results.append({
                        "entry_time": timestamps[i],
                        "target_pct": target,
                        "time_to_hit_minutes": time_minutes,
                        "hit": True,
                    })
                    break
            else:
                results.append({
                    "entry_time": timestamps[i],
                    "target_pct": target,
                    "time_to_hit_minutes": None,
                    "hit": False,
                })
    
    return pd.DataFrame(results)

# Rodar e reportar
ttg = time_to_target(df_15m)
for target in [0.01, 0.015, 0.02]:
    subset = ttg[(ttg["target_pct"] == target) & (ttg["hit"] == True)]
    print(f"\nTarget {target*100:.1f}%:")
    print(f"  Hit rate (24h): {len(subset)/len(ttg[ttg['target_pct']==target])*100:.1f}%")
    print(f"  Mediana: {subset['time_to_hit_minutes'].median():.0f} min")
    print(f"  P25:     {subset['time_to_hit_minutes'].quantile(0.25):.0f} min")
    print(f"  P75:     {subset['time_to_hit_minutes'].quantile(0.75):.0f} min")
```

### Análise 2 — Timeframe ótimo para RSI de entrada
**Pergunta:** Em qual timeframe o RSI oversold melhor prevê subida de 1.5%?

```python
def rsi_entry_analysis(df, timeframe_label, target_pct=0.015):
    """
    Quando RSI < 30 (oversold), qual % das vezes o preço sobe target_pct em 24h?
    Comparar por timeframe.
    """
    close = df['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    
    # Forward return máximo em 24h
    periods_in_24h = int(24 * 60 / {"15m": 15, "30m": 30, "1h": 60, "4h": 240}[timeframe_label])
    df["max_forward_return"] = df["high"].rolling(periods_in_24h).max().shift(-periods_in_24h) / close - 1
    df["rsi"] = rsi
    
    # Filtrar RSI oversold
    oversold = df[df["rsi"] < 30].copy()
    hit_rate = (oversold["max_forward_return"] >= target_pct).mean()
    
    # Também testar RSI < 35, < 40
    for thresh in [30, 35, 40]:
        subset = df[df["rsi"] < thresh]
        hr = (subset["max_forward_return"] >= target_pct).mean()
        n = len(subset)
        print(f"  {timeframe_label} RSI<{thresh}: hit_rate={hr*100:.1f}%, n={n}")
    
    return hit_rate

# Comparar timeframes
for label, data in [("15m", df_15m), ("30m", df_30m), ("1h", df_1h), ("4h", df_4h)]:
    print(f"\n{label}:")
    rsi_entry_analysis(data, label)
```

### Análise 3 — Bollinger %B como sinal de entrada
**Pergunta:** Em qual timeframe o Bollinger %B < 0.2 melhor prevê subida?

```python
def bollinger_entry_analysis(df, timeframe_label, target_pct=0.015):
    """
    Quando Bollinger %B < 0.2 (perto do lower), qual % sobe target_pct em 24h?
    """
    close = df['close']
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower)
    
    periods_in_24h = int(24 * 60 / {"15m": 15, "30m": 30, "1h": 60, "4h": 240}[timeframe_label])
    df["max_forward_return"] = df["high"].rolling(periods_in_24h).max().shift(-periods_in_24h) / close - 1
    df["bb_pct_b"] = bb_pct_b
    
    for thresh in [0.1, 0.2, 0.3]:
        subset = df[df["bb_pct_b"] < thresh]
        hr = (subset["max_forward_return"] >= target_pct).mean()
        n = len(subset)
        print(f"  {timeframe_label} BB<{thresh}: hit_rate={hr*100:.1f}%, n={n}")

for label, data in [("15m", df_15m), ("30m", df_30m), ("1h", df_1h), ("4h", df_4h)]:
    print(f"\n{label}:")
    bollinger_entry_analysis(data, label)
```

### Análise 4 — Time-to-Stop-Loss
**Pergunta:** Quando preço cai 2% (stop loss), em quanto tempo atinge?

```python
def time_to_stop(df_15m, stop_pct=0.02):
    """
    Para cada candle, medir quanto tempo até cair stop_pct%.
    Define frequência mínima de monitoramento.
    """
    results = []
    close = df_15m['close'].values
    timestamps = df_15m.index
    
    for i in range(len(close) - 1):
        entry = close[i]
        stop_price = entry * (1 - stop_pct)
        for j in range(i + 1, min(i + 96, len(close))):  # 24h
            if df_15m['low'].values[j] <= stop_price:
                time_minutes = (timestamps[j] - timestamps[i]).total_seconds() / 60
                results.append({
                    "time_to_stop_minutes": time_minutes,
                })
                break
    
    r = pd.DataFrame(results)
    print(f"Stop {stop_pct*100:.1f}% analysis:")
    print(f"  Mediana: {r['time_to_stop_minutes'].median():.0f} min")
    print(f"  P10:     {r['time_to_stop_minutes'].quantile(0.10):.0f} min (10% mais rápidos)")
    print(f"  P25:     {r['time_to_stop_minutes'].quantile(0.25):.0f} min")
    return r

time_to_stop(df_15m, 0.02)
time_to_stop(df_15m, 0.03)
```

### Análise 5 — Combinação ótima (RSI + BB + Volume + Regime)
**Pergunta:** Qual combinação de timeframe + indicadores gera melhor hit rate com melhor risk/reward?

```python
def combined_signal_analysis(df, timeframe_label, target_pct=0.015, stop_pct=0.02):
    """
    Quando RSI<35 AND BB<0.3 AND volume_z>0, qual o resultado?
    Medir: hit rate, avg profit, avg loss, profit factor.
    """
    close = df['close']
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss_s)
    
    # Bollinger
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_pct_b = (close - (bb_mid - 2*bb_std)) / (4*bb_std)
    
    # Volume z-score
    vol_z = (df['volume'] - df['volume'].rolling(50).mean()) / df['volume'].rolling(50).std()
    
    # Forward returns
    periods_24h = int(24 * 60 / {"15m": 15, "30m": 30, "1h": 60, "4h": 240}[timeframe_label])
    max_up = df['high'].rolling(periods_24h).max().shift(-periods_24h) / close - 1
    max_down = df['low'].rolling(periods_24h).min().shift(-periods_24h) / close - 1
    
    # Signal: RSI<35 AND BB<0.3 AND vol_z>0
    signal = (rsi < 35) & (bb_pct_b < 0.3) & (vol_z > 0)
    
    entries = df[signal].copy()
    entries["max_up"] = max_up[signal]
    entries["max_down"] = max_down[signal]
    entries["hit_target"] = entries["max_up"] >= target_pct
    entries["hit_stop"] = entries["max_down"] <= -stop_pct
    
    n = len(entries)
    if n == 0:
        print(f"  {timeframe_label}: 0 signals")
        return
    
    hit_rate = entries["hit_target"].mean()
    stop_rate = entries["hit_stop"].mean()
    
    print(f"  {timeframe_label}: n={n}, hit_target={hit_rate*100:.1f}%, hit_stop={stop_rate*100:.1f}%, ratio={hit_rate/max(stop_rate,0.01):.2f}")

for label, data in [("15m", df_15m.copy()), ("30m", df_30m.copy()), ("1h", df_1h.copy())]:
    print(f"\n{label}:")
    combined_signal_analysis(data, label)
```

## Parte 3 — Relatório final
Gerar summary com recomendação:

```python
print("""
═══════════════════════════════════════════
TIMEFRAME ANALYSIS — SUMMARY
═══════════════════════════════════════════

Pergunta 1: Time-to-target (1.5%)
  → Mediana: XX min → Candle recomendado: YYmin

Pergunta 2: RSI oversold hit rate
  → Melhor timeframe: ZZ (hit_rate=XX%)

Pergunta 3: Bollinger %B hit rate
  → Melhor timeframe: ZZ (hit_rate=XX%)

Pergunta 4: Time-to-stop-loss (2%)
  → P10=XX min → Monitoramento mínimo: cada YYmin

Pergunta 5: Combinação ótima
  → Melhor: ZZmin (hit=XX%, stop=YY%, ratio=ZZ)

RECOMENDAÇÃO:
  Execução:  cada XXmin
  Contexto:  indicadores em YYmin
  Modelo AB: mantém 4h (decisor estratégico)
═══════════════════════════════════════════
""")
```

## Onde salvar
- Script: `scripts/analysis/timeframe_analysis.py`
- Dados 15m: `data/01_raw/spot/crypto/15m/BTCUSDT_15m.parquet`
- Dados 30m: `data/01_raw/spot/crypto/30m/BTCUSDT_30m.parquet`
- Dados 1h: `data/01_raw/spot/crypto/1h/BTCUSDT_1h.parquet`
- Report: `data/07_reports/timeframe_analysis_report.txt`

## Restrições
- Download do Binance Vision requer internet
- Candles 15min de set/2025 a mar/2026 = ~17.000 candles (leve)
- Usar SOMENTE período set/2025 → mar/2026 (alinhado com dados CoinGlass)
- Forward-looking nas análises é intencional (estamos analisando, não treinando)
- NÃO registrar como pipeline Kedro (é análise exploratória)

## Validação
1. Parquets 15m, 30m, 1h criados com range correto
2. Script roda sem erros e gera report
3. Todas as 5 análises produzem resultados numéricos
4. Recomendação final com timeframe sugerido
