# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Arquitetura Atual (v4 — Scoring System + R5C, desde 2026-04-06)

### Produção: Pipeline Integrado (1 portfolio)
```
Layer 1: HMM R5C (daily) → 3 estados: Bull / Sideways / Bear
    ↓ Bear bloqueia, Sideways permite (sizing x0.5), Bull full
Layer 2: Specialist LightGBM (4h) → alloc_raw (informativo, nao decisorio)
    ↓
Scoring System: BB + RSI + alloc + news → score >= 2.5
    ↓ score >= 2.5 → gera INTENCAO
Layer 3: Timing Layer (1h) → execucao condicionada (RSI + BB + volume)
    ↓ timing score >= 0.30
ENTRADA com position sizing (Sideways x0.5, score multiplier x0.8/1.0/1.2)

Monitor 1h: stops (SL 2%, SG 1.5%, trailing 1%) + timing execution
```

### Scoring System (validado com walk-forward 208 sinais Sideways 2025+)
```python
# BB (dominante)
if bb > 0.80:   bb_score = -2.0   # kill switch topo (win 43%)
elif bb < 0.20: bb_score = +3.0   # sinal forte (win 88%, ret_3d +1.75%)
elif bb < 0.30: bb_score = +2.0   # sinal bom (win 77%, ret_3d +1.65%)
elif bb < 0.40: bb_score = +0.5   # zona favoravel
else:           bb_score = +0.0

# RSI (complementar)
if rsi < 35:    rsi_score = +1.0
elif rsi < 45:  rsi_score = +0.5
elif rsi > 60:  rsi_score = -1.0
else:           rsi_score = +0.0

# Alloc (fraco)
if 0.50 <= alloc <= 0.54: alloc_score = +0.5
elif 0.48 <= alloc < 0.50: alloc_score = +0.25
else:                       alloc_score = +0.0

# News
if news == "BULL" and score > 3:   news_score = +1.0
elif news == "BULL":                news_score = +0.5
elif news == "BEAR" and score < -3: news_score = -1.5
elif news == "BEAR":                news_score = -0.5
else:                               news_score = +0.0

# Threshold: score >= 2.5 → ENTER
```

Walk-forward evidence (2025+, Sideways):
- BB < 0.30: 51 sinais, win_3d=75%, ret_3d=+1.37%, MaxDD=-0.12%
- Score >= 2.5: 32 sinais, win_3d=72%, ret_3d=+1.18%
- BB > 0.30: retorno NEGATIVO (win 41-42%)

### Decisoes de design
- alloc_raw correlacao = 0 com retorno (82 sinais) — nao eh decisorio
- ML supervisionado nao funciona em Sideways (XGBoost AUC=0.484, LR AUC=0.352)
- Regras estatisticas (BB + RSI) > ML complexo em Sideways
- BB eh o indicador dominante para mean reversion em Sideways
- News BEAR forte cancela sinal tecnico bom

### R5C HMM (Layer 1) — substituiu R11 em 2026-04-06
- 3 estados: Bull / Sideways / Bear (covariance_type=full)
- Features: log_return, vol_short, vol_ratio, drawdown, volume_z, slope_21d
- Modelo: /data/05_models/r5c_hmm.pkl
- Distribuicao historica: Sideways 41%, Bull 31%, Bear 28%
- R11 (2 estados) DESATIVADO — classificava Sideways como Bear

### Modulo compartilhado: scripts/paper_trading/shared/execution.py
- Fonte UNICA para: execute_buy(), execute_sell(), portfolio, pending signal
- atomic_write_json(), parse_utc(), idempotencia, stale-write protection

## Control Layer — Gates

1. Stop Gain (+1.5%)
2. Stop Loss (-2%, cooldown 1 candle)
3. Trailing Stop (1% do maximo, ativa apos 0.5% lucro)
4. Regime Gate (R5C): Bear bloqueia, Sideways sizing x0.5, Bull full
5. News Gate: regime das noticias (Bull/Sideways/Bear), crypto 40% + macro 60%
6. Technical Gate: RSI + BB (bloqueia se RSI>65 ou BB>0.75)
7. Entry Scoring: BB+RSI+alloc+news, threshold 2.5

## News Intelligence

### Classificador DeepSeek — 3 regimes
- Bull / Sideways / Bear (simplificado de 9 event_types)
- Score -10 a +10, source weighting (Reuters=1.0, Yahoo=0.5)
- Custo: ~$2/mes

### Fontes
- CryptoCompare: 50 artigos/hora (categories=BTC)
- Google News RSS: 5 grupos macro (energy, fed, geopolitical, inflation, global_risk)

## Crontab (8 entries)

```
50 * * * *              update_1h_candles.py
55 * * * *              cryptocompare_news_ingest.py
56 * * * *              macro_news_ingest.py
57 * * * *              classify_news.py
0 7 * * *               daily_update.sh
2 * * * *               specialist_4h_monitor_1h.py
5 21,1,5,9,13,17 * * *  specialist_4h_paper_trader.py
8 * * * *               xgb_1h_trader.py (projeto xgboost_R5c_1h)
```

## Analise Cruzada — Resultados (2026-04-06)

82 sinais, 13 dias: alloc_raw sem edge, BB < 0.30 = sinal dominante.
Walk-forward 208 sinais 2025+: BB < 0.30 win=75% confirmado.
ML (XGBoost, LR) AUC < 0.5 em Sideways — nao funciona.

## Contexto de Mercado (abril 2026)

- BTC $65k-$70k lateralizando (R5C Sideways 100%)
- NUPL 10% (realizacao), MicroStrategy no prejuizo
- Guerra Iran, oil $104, Basileia III
- F&G 12 (Fear) vs Binance 7.5 (Forte Positivo) — desalinhado

## Proximos Passos

### Alta prioridade
- [ ] Acumular dados scoring (30+ trades)
- [ ] Migrar CoinGlass → Binance API
- [ ] Bayesian optimization pesos scoring (quando 200+ trades)

### Media prioridade
- [ ] Orquestrador: R5C → Hold/DayTrade/FundingArb/Cash
- [ ] Funding rate arbitrage (projeto separado)
- [ ] NUPL no dashboard

### Completado
- [x] R5C substitui R11
- [x] Classificador news Bull/Sideways/Bear
- [x] Gate tecnico RSI + BB
- [x] Scoring system validado walk-forward
- [x] Analise cruzada L1/L2/L3
- [x] XGBoost regime-aware testado
- [x] Macro news Google RSS
- [x] shared/execution.py
- [x] Dashboard com scoring visual

## Contracts

### Day-Shift (CRITICAL)
R5C regime of day D applied to candles from D+1 00:00 UTC onward.

### Shared Execution
- atomic_write_json() for all state files
- parse_utc() for all timestamps
- executed flag for idempotency

### Code Rules
- No hardcoded dates/paths/features
- DatetimeIndex UTC always
- Pure functions, no side effects
- L1→L2→L3→L4→Modeling (downward only)
