# TASK: Paper Trading — Specialist 4h (Modelo AB)

## Contexto
Projeto: crypto-market-state (Kedro 1.2.0, conda env: crypto_market_state)
Modelo AB (técnico 4h + R5C regime) aprovado: Sharpe s4=+3.872

Paper trading R11 já existe e serve como referência de padrão:
- Script: `scripts/paper_trading/r11_paper_trader.py`
- State: `scripts/paper_trading/state/`
- Modelo R11: `scripts/paper_trading/state/r11_hmm_model.pkl`
- Dashboard: `apps/r11_dashboard.py` (streamlit)
- Cron: 08:00 UTC diário

## Objetivo
Criar paper trading paralelo para o specialist AB 4h, rodando a cada 4h, sem alterar o R11 existente.

## Passo 1 — Inspecionar antes de codar
1. Ler `scripts/paper_trading/r11_paper_trader.py` inteiro — entender padrão de: leitura de dados, inferência, state management, logging
2. Verificar onde está o modelo AB treinado — procurar em `data/05_models/` ou `data/06_models/` ou `data/06_reports/`
3. Verificar como R11 lê dados recentes (data lake? API? catalog?)
4. Ler `apps/r11_dashboard.py` — entender padrão do dashboard

## Passo 2 — Script de inferência
Criar: `scripts/paper_trading/specialist_4h_paper_trader.py`

Seguir EXATAMENTE o padrão do r11_paper_trader.py mas adaptado para 4h.

Fluxo de cada execução:
```
1. Ler btc_spot_4h mais recente (mesmo método que R11 usa para daily)
2. Calcular features grupo A (técnico 4h):
   - returns_4h, returns_12h, volatility_24h, volume_zscore, buy_pressure, price_range_4h
   IMPORTANTE: usar mesmas janelas/fórmulas de L3 (primary.spot_4h)
3. Ler R5C regime mais recente (grupo B):
   - allocation_r5c, prob_bear_r5c, days_in_state_r5c
   CONTRATO L5: regime do dia D aplica-se a candles de D+1 em diante
4. Carregar modelo AB treinado
5. predict_proba → allocation_raw (valor entre 0 e 1)
6. Aplicar gates (pass-through por enquanto — ver Passo 5)
7. Converter allocation_final em posição
8. Logar sinal + atualizar state
```

Features do grupo A — referência de cálculo (L3 primary.spot_4h):
```python
returns_4h     = close.pct_change(1)
returns_12h    = close.pct_change(3)   # 3 candles de 4h
volatility_24h = close.pct_change(1).rolling(6).std()  # 6 candles = 24h
volume_zscore  = (volume - volume.rolling(50).mean()) / volume.rolling(50).std()
buy_pressure   = taker_buy_volume / volume  # ou taker_buy_base_vol
price_range_4h = (high - low) / close
```
NOTA: confirmar nomes e janelas exatos inspecionando o código L3 real em
`src/crypto_mkt_state/pipelines/primary/spot_4h/nodes.py`

Features do grupo B (R5C regime — vem do L4 daily):
```python
allocation_r5c    = max(0, prob_bull - prob_bear)  # contínuo 0-1
prob_bear_r5c     = prob_bear do HMM R5C
days_in_state_r5c = dias consecutivos no estado atual
```
Ler de `data/04_model_input/regime_context/daily/BTCUSDT.parquet` ou de onde o R11 paper trader lê.

## Passo 3 — State management
Criar em: `scripts/paper_trading/state/specialist_4h/`

Arquivos:
- `config.json` — capital inicial, fees, thresholds
- `portfolio.json` — estado atual (posição, cash, valor total)
- `signals.csv` — log incremental (append a cada candle)

Config padrão:
```json
{
  "initial_capital": 100000,
  "fee_rate": 0.0004,
  "min_delta_allocation": 0.05,
  "model_path": "<path do modelo AB detectado no Passo 1>"
}
```

Colunas do signals.csv:
```
timestamp, candle_close, price_close,
allocation_raw, allocation_final,
regime_r5c, prob_bear_r5c, days_in_state,
returns_4h, returns_12h, volatility_24h, volume_zscore, buy_pressure, price_range_4h,
position_btc, position_usdt, portfolio_value,
action, delta_allocation, fee_paid
```

## Passo 4 — Regra de posicionamento
```python
if allocation_final > 0.5:
    target_exposure = (allocation_final - 0.5) * 2  # 0 a 1
else:
    target_exposure = 0  # cash, sem short

# Só executa se mudança > threshold
if abs(target_exposure - current_exposure) > min_delta_allocation:
    execute_trade(target_exposure)
    fee = abs(delta_notional) * fee_rate
```

## Passo 5 — Control layer (preparar, não implementar gates)
Criar função que será estendida depois:
```python
def apply_control_layer(allocation_raw: float, context: dict) -> float:
    """
    Control layer: intercepta allocation_raw e aplica gates.
    
    Arquitetura futura:
      specialist_4h (AB) → allocation_raw → [gates] → allocation_final
      Gates planejados:
        - macro_gate (MOVE-based, já existe no daily)
        - derivatives_gate (funding/OI-based, a implementar)
    
    Por enquanto: pass-through.
    """
    allocation_final = allocation_raw
    # TODO: macro_gate
    # TODO: derivatives_gate
    return allocation_final
```

## Passo 6 — Cron
Adicionar ao crontab (mesmo padrão do R11):
```bash
0 0,4,8,12,16,20 * * * cd /path/to/project && conda run -n crypto_market_state python scripts/paper_trading/specialist_4h_paper_trader.py >> logs/specialist_4h_paper.log 2>&1
```
NOTA: ajustar o path conforme o que o R11 usa.

## Passo 7 — Dashboard
Criar: `apps/specialist_4h_dashboard.py` (streamlit, arquivo separado do R11)

Mínimo viável:
- Equity curve do paper trading (portfolio_value ao longo do tempo)
- Sinais recentes (últimas 24h = 6 candles)
- Allocation atual + regime R5C atual
- Comparativo simples: specialist AB vs buy&hold BTC no mesmo período

Seguir padrão visual do r11_dashboard.py.

## Restrições
- NUNCA trades reais — paper trading only
- NÃO modificar R11 (scripts, state, dashboard)
- NÃO hardcodar paths de modelos — ler de config.json
- Index sempre DatetimeIndex UTC
- Regime R5C com shift de 1 dia (contrato L5)
- Features calculadas com mesmas janelas do treinamento (confirmar no código L3)
- Sem prints — usar logging

## Validação
1. Rodar manualmente 1x: `python scripts/paper_trading/specialist_4h_paper_trader.py`
2. Verificar que signals.csv tem 1 linha nova com valores coerentes
3. Verificar allocation_raw entre 0 e 1
4. Verificar que regime R5C usa shift correto (dia anterior)
5. Rodar dashboard: `streamlit run apps/specialist_4h_dashboard.py`
