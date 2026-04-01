# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Arquitetura Atual (v3 — Pipeline Integrado + News Intelligence, desde 2026-03-31)

### Produção: Pipeline Integrado (1 portfolio, $99,135)
```
Layer 1: HMM R11 (daily) → regime gate (Bull/Bear)
    ↓ só se Bull
Layer 2: Specialist LightGBM (4h) → decisão + convicção → gera INTENÇÃO
    ↓ intenção pendente (pending_signal.json)
Layer 3: Timing Layer (1h) → execução condicionada (RSI + BB + volume)
    ↓ timing score ≥ 0.30
ENTRADA com position sizing

Monitor 1h: stops (SL 2%, SG 1.5%, trailing 1%) + timing execution
```

**Decisões de design:**
- Specialist gera INTENÇÃO, não executa. Timing layer (monitor 1h) executa quando timing alinha.
- Timing gate REMOVIDO do specialist — vive exclusivamente no monitor (Layer 3).
- Intenção expira em 4h (próximo candle specialist pode mudar).
- Saídas executadas direto pelo specialist (não passam pelo timing layer).

**Módulo compartilhado:** `scripts/paper_trading/shared/execution.py`
- Fonte ÚNICA de verdade para: execute_buy(), execute_sell(), portfolio read/write, pending signal
- Paths derivados do PROJECT root (nunca hardcoded)
- atomic_write_json() para container/NFS safety
- parse_utc() para timezone-safe parsing
- Idempotência: flag `executed` no pending_signal.json
- Portfolio stale-write protection: compara last_update antes de escrever
- Validação forte: _REQUIRED_TYPES com isinstance check

### Benchmark: XGBoost Solo (congelado em 2026-03-31)
- Portfolio: $99,995 (-0.00%)
- Não executa trades
- Dados históricos preservados para comparação futura

### Desativado: R11 Paper Trader Solo (2026-03-31)
- Incorporado como Layer 1 do Pipeline
- Portfolio final: $79,294 (-1.26%)
- Dados preservados em state/portfolio.json e state/equity_curve.csv

## Control Layer — 5 Gates Ativos

```
1. Stop Gain (+1.5%) — se em posição, take profit
2. Stop Loss (-2%, cooldown 1 candle) — corta perda
3. Trailing Stop (1% do máximo, ativa após 0.5% lucro)
4. Regime Gate — bloqueia se r11_prob_bull < 0.30 ou entropy > 0.85
5. News Gate — bloqueia se combined_score < -0.30 ou escalation cluster ≥ 2
   - Lê sentiment_metrics.json (crypto + macro combinado)
   - Deescalation boost: +15% allocation se score > 3
   - Fail-open: se dados stale > 2h, não bloqueia
```

Stubs prontos (não ativados):
- Macro gate (MOVE-based)
- Derivatives gate (funding_z + oi_price_div)

## News Intelligence Pipeline

### Fluxo de dados
```
:55  CryptoCompare crypto news (categories=BTC, 50 artigos)
     → data/01_raw/news/cryptocompare/btc_news.parquet (L1)

:56  Google News RSS macro (5 grupos × 10 artigos)
     → data/01_raw/news/macro/google_news.parquet (L1)

:57  DeepSeek Classifier (L3 + L4):
     1. Filtro de relevância (keywords) → reduz custo
     2. DeepSeek classifica: topic, event_type, impact, score (-10 a +10)
     3. Source weighting (Reuters=1.0, Yahoo=0.5)
     4. Agrega métricas crypto + macro + combinado
     → sentiment_metrics.json
```

### Classificador DeepSeek
- Modelo: deepseek-chat (V3.2)
- Custo: ~$0.003/hora, ~$2/mês
- Prompt classifica pelo RESULTADO FINAL no preço do BTC
- 8 event_types: ESCALATION, DEESCALATION, POLICY_SHIFT, CAPITAL_FLOW, MARKET_SHOCK, DATA_RELEASE, INSTITUTIONAL_MOVE, NOISE
- Direction derivada do score (≥+2 BULLISH, ≤-2 BEARISH)
- Sem contaminação: NÃO envia sentiment do CryptoCompare ao DeepSeek
- Batch de até 30 notícias por request

### Fontes macro (Google News RSS)
Configurável: `conf/macro_keywords.json`
- energy_oil: oil crisis, crude oil, hormuz, OPEC
- fed_monetary: federal reserve rate, FOMC, Powell, inflation
- geopolitical: iran war, trump iran, middle east conflict
- inflation: CPI, consumer prices, cost of living
- global_risk: recession, market crash, banking crisis

### Métricas (sentiment_metrics.json)
```json
{
  "1h/4h/24h": { "crypto metrics..." },
  "impact": { "crypto_score, escalation_count, top_stories..." },
  "macro": { "macro_score, by_group, dominant_group, top_stories..." },
  "combined": {
    "4h": {
      "crypto_score": -0.14,
      "macro_score": -2.98,
      "combined_score": -1.84,
      "dominant_driver": "macro"
    }
  }
}
```
Peso: macro 60%, crypto 40% (macro tem mais impacto em regime).

## Dashboard (projeto separado: crypto-trading-dashboard)

Path: `/Users/brown/Documents/MLGeral/crypto-trading-dashboard/`

### Layout (7 linhas, estilo CoinGlass)
1. System status (1 linha inline)
2. Análise AI (DeepSeek, botão manual)
3. Modelos (2 cards: Pipeline Integrado + XGB Benchmark)
4. Análise Gráfica (price performance 1h + indicadores daily + S/R)
5. Sentimento & Notícias (crypto + macro feeds com scroll + timeline)
6. Contexto de Mercado (cards: OI, Funding, F&G, VIX, MOVE, DXY, SP500, Oil, Yield, Fed + CoinGlass indices)
7. Performance & Trades (expander)
8. Data Quality (expander final, 12+ fontes monitoradas)

### Análise AI (DeepSeek)
- API: deepseek-chat via openai SDK (base_url=api.deepseek.com)
- Key em .streamlit/secrets.toml (DEEPSEEK_API_KEY)
- _gather_market_context() coleta: preço, 3 layers, gates, técnicos, performance, sentimento, headlines crypto + macro
- Custo: ~$0.003/análise

### Price Performance
- Fonte principal: candles 1h (delay máximo 1h)
- Fallback: candles daily
- Indicadores (RSI, BB, MAs): continuam em daily (timeframe correto)
- High/Low 7d/30d: calculados em 1h (mais preciso)

## Crontab (6 entries)

```
50 * * * *              update_1h_candles.py        # L1: candles 1h Binance API
55 * * * *              cryptocompare_news_ingest.py # L1: crypto news
56 * * * *              macro_news_ingest.py         # L1: macro news Google RSS
57 * * * *              classify_news.py             # L3+L4: DeepSeek classify + aggregate
0 7 * * *               daily_update.sh              # Daily pipeline completo
2 * * * *               specialist_4h_monitor_1h.py  # Stops + timing layer execution
5 21,1,5,9,13,17 * * *  specialist_4h_paper_trader.py # Layer 2: decisão + intenção
```

## Project Overview

A **Kedro**-based ML pipeline for crypto market state modeling. The system ingests raw market data (Binance, Yahoo Finance, FRED), normalizes it through a layered architecture, engineers features, and trains Hidden Markov Models (HMMs) to detect market regimes (bull, bear, transition).

## Environment Setup

```bash
conda env create -f environment.yml
conda activate crypto_market_state
pip install -e .
```

Key dependencies: `kedro==1.2.0`, `kedro-datasets==9.1.1`, `hmmlearn`, `fredapi`, `python-binance`, `yfinance`, `hmmlearn`, `optuna`.

## Common Commands

```bash
kedro run --pipeline ingestion.binance.spot
kedro run --pipeline normalization.spot
kedro run --pipeline primary.spot.crypto
kedro run --pipeline modeling.regime_hmm
kedro run --pipeline modeling.regime_hmm_full
kedro registry list
```

## Layered Architecture

All data flows strictly downward through layers.

| Layer | Kedro Stage | Data Path | Role |
|-------|------------|-----------|------|
| **L1** | `ingestion.*` | `data/01_raw/` | Raw API mirror — no transformations |
| **L2** | `normalization.*` | `data/02_intermediate/` | Schema normalization only |
| **L3** | `primary.*` | `data/03_primary/` | Per-asset feature engineering |
| **L4** | `primary.regime_context` | `data/04_model_input/` | Regime context — daily consolidation |
| **L5** | `primary.model_features_4h` | `data/05_model_features/` | Final model input — 4h supervised dataset |
| **Modeling** | `modeling.*` | `data/05_models/`, `data/06_reports/` | Model training & validation |

## Multi-Horizon Model Stack

### R11 — Daily Regime Model (frozen, Layer 1)
- HMM, 2 states (Bear/Bull), `covariance_type=diag`
- Features: `log_return`, `vol_short`, `vol_ratio`, `drawdown`, `volume_z`, `slope_21d`
- Train: 2023-01-01 → 2024-12-31 (frozen)
- Role: regime gate only — não faz trades
- Frozen model: `scripts/paper_trading/state/r11_hmm_model.pkl`

### Specialist AB — 4h Entry Model (Layer 2)
- LightGBM binary classifier (SET_B_split_4_recent.pkl)
- 9 features: Group A (6 technical 4h) + Group B (3 R5C regime)
- Sharpe s4: +3.872
- Gera INTENÇÃO de entrada (pending_signal.json), não executa

### Timing Layer — 1h Execution (Layer 3)
- Regras (não ML): RSI < 40, BB < 0.30, volume > 0
- Score = 0.4 × RSI_score + 0.3 × BB_score + 0.3 × volume_score
- Executa entrada quando score ≥ 0.30 E intenção pendente existe
- Roda no monitor_1h a cada hora

### Ablation 4h — Complete Results (2026-03-24)

| Model | Label | Sharpe s4 | Verdict |
|-------|-------|-----------|---------|
| A | 4h Technical | +3.323 | baseline |
| AB | + R5C regime | **+3.872** | ✓ APPROVED |
| ABC | + CoinGlass deriv | +2.783 | ✗ EXCLUDE |
| ABCD | + OrderBook | +1.599 | ✗ EXCLUDE |
| ABCDE | + MacroExt | +2.116 | ✗ EXCLUDE |

**Key insight:** Simplicity wins under concept drift — every addition degrades s4.

## Paper Trading Results (as of 2026-03-31)

### Pipeline Integrado
- Portfolio: $99,135 (-0.86%)
- Trades: BUY $71,261 → STOP_LOSS $69,338 (-2.7%) | BUY $66,644 → TRAILING_STOP $66,192 (-0.7%) | BUY $66,431 → TRAILING_STOP $66,192 (-0.4%)
- Status: CASH (regime Bear, p_bull=0.033)
- Gates ativos: regime gate bloqueando, news gate OK

### R11 Solo (desativado)
- Portfolio final: $79,294 (-1.26%)
- 4 trades, último: SELL $66,381 (regime Bear)

### XGBoost Solo (congelado)
- Portfolio: $99,995 (-0.00%)
- 0 trades (nunca entrou — modelo ultra-conservador)

## Análise Estrutural de Mercado (contexto para próximos passos)

### Vetores de pressão convergentes (março 2026)
1. **NUPL caindo** de 60% para ~10% — zona de realização de lucros, holders vendendo
2. **MicroStrategy risk**: 700k BTC a preço médio $74,972 (no prejuízo), dívida de $8.2B vencendo 2027-2032. Forçamento de venda possível se preço não recuperar
3. **Deleveraging**: OI explodiu em 2025 e está desalavancando (liquidações massivas jan-fev 2026)
4. **Basileia III**: reclassificou BTC em jan/2026, bancos vendendo desde out/2025 (1:1 capital requirement)
5. **Guerra Iran + Oil crisis**: Hormuz fechado, oil acima de $94, pressão inflacionária
6. **Ciclo histórico**: quedas de 70-87% são normais pós-ATH. BTC caiu de $126k ATH para $66k (-47%)

### Métricas on-chain para monitorar
- **NUPL**: Net Unrealized Profit/Loss — indicador de ciclo (capitulação < 0%, euforia > 75%)
- **Exchange balance**: bancos/instituições vendendo → pressão vendedora
- **OI vs preço**: divergência indica deleveraging iminente

## Próximos Passos

### Curto prazo (abril 2026)
1. **Calibrar gates** com dados reais acumulados do paper trading
2. **NUPL no dashboard** como indicador informativo (CoinGlass)
3. **Contexto estrutural no prompt DeepSeek** (MicroStrategy risk, NUPL, ciclo)
4. **Correlação BTC vs iShares Software ETF** (correlação 0.73 reportada)
5. **Stop loss/gain no R11** — histórico mostra -6.5% drawdown sem proteção

### Médio prazo (maio-junho 2026)
6. **News Impact Model** — correlacionar notícias históricas com movimentos de preço BTC
7. **NUPL como feature do HMM** — retreinar R11 com NUPL para melhor detecção de regime
8. **Derivatives gate** — funding_z + oi_price_div como gate pós-modelo
9. **Macro gate** — MOVE-based continuous gate
10. **Deploy AWS Fargate** — containerizar pipeline + dashboard

### Longo prazo
11. **Multi-Horizon Stack V2** — HMM com features on-chain (NUPL, exchange flow, whale activity)
12. **News sentiment model treinado** — classificador próprio com dados históricos acumulados
13. **Keyword scoring dinâmico** — pesos calibrados por correlação com retorno BTC
14. **Live trading** — Binance Testnet → produção real

## Critical Contracts

### L1 Contract
- Mirror of API — forbidden: feature engineering, aggregations, rolling windows
- All timestamps: `datetime64[ns, UTC]`
- `drop_duplicates` mandatory
- Cast numerics: `pd.to_numeric(..., errors="coerce").astype("float64")`

### L2 Contract
- Only: column renaming, index setting, dtype enforcement
- Forbidden: rolling, z-score, returns, fill, resample
- `timestamp` (UTC DatetimeIndex) is universal canonical name

### L3 Contract
- Per-asset features only; no cross-asset
- All windows strictly backward-looking
- L2 columns preserved unchanged
- NaN permitted at start (burn-in)

### Day-Shift Contract (CRITICAL)
R11 regime of day D applied to candles from D+1 00:00 UTC onward.
Never use same-day regime for same-day candles.

### Shared Execution Contract
- `shared/execution.py` is the ONLY source for trade execution
- atomic_write_json() for all state files
- parse_utc() for all timestamp parsing
- File locks (fcntl) for all reads/writes
- Idempotência: `executed` flag in pending_signal.json

## Code Constraints
- Sem hardcode de datas, paths ou features
- Index sempre DatetimeIndex UTC
- Código completo nos arquivos afetados
- Funções puras, sem side effects ocultos
- Log tudo — se algo falhar, o log conta a história

## Claude Code Operating Rules

Claude must respect the project's layered data architecture.

### Layer Access Rules
Data flows strictly downward: L1 → L2 → L3 → L4 → Modeling.
Forbidden: L3 reading L1, L4 reading L2 directly, Modeling reading L2 or L1.

### Code Generation Rules
- Never hardcode dates or asset lists
- Always use parameters.yml for configuration
- Always use DatetimeIndex UTC
- Prefer pure functions in nodes
- Avoid side effects in Kedro nodes
- Never introduce lookahead bias

### Workflow
1. Prototipar em scripts/ ou notebooks/
2. Validar resultado
3. Migrar para node Kedro quando estável
4. Scripts mantidos como referência, Kedro é a fonte oficial
