# TASK: Control Layer — Fase 1 (Stop Loss) + Estrutura para Fases 2-3

## Contexto
O paper trading do specialist 4h AB está ativo desde 2026-03-24.
A função `apply_control_layer()` já existe como stub (pass-through) em:
`scripts/paper_trading/specialist_4h_paper_trader.py`

State em: `scripts/paper_trading/state/specialist_4h/`
Config em: `scripts/paper_trading/state/specialist_4h/config.json`

## Objetivo
1. Implementar stop loss fixo (Fase 1) — funcional agora
2. Estruturar a control layer para receber macro_gate (Fase 2) e derivatives_gate (Fase 3) — stubs prontos
3. Cada gate on/off via config.json
4. Log de quando um gate dispara no signals.csv

## Passo 1 — Inspecionar antes de codar
1. Ler `scripts/paper_trading/specialist_4h_paper_trader.py` — encontrar:
   - A função `apply_control_layer()` atual (stub)
   - Como `portfolio.json` armazena estado (btc_held, usdt_free, btc_price)
   - Como `signals.csv` é escrito (colunas, formato)
   - Onde o `context` é montado antes de chamar apply_control_layer
2. Ler `scripts/paper_trading/state/specialist_4h/config.json` — entender estrutura
3. Ler `scripts/paper_trading/state/specialist_4h/portfolio.json` — verificar campos disponíveis

## Passo 2 — Atualizar config.json
Adicionar seção `control_layer`:
```json
{
  "...campos existentes...",
  "control_layer": {
    "stop_loss": {
      "enabled": true,
      "pct": 0.03,
      "cooldown_candles": 3
    },
    "macro_gate": {
      "enabled": false,
      "move_stress_min": 1.0,
      "move_stress_max": 3.0,
      "max_cut": 0.70
    },
    "derivatives_gate": {
      "enabled": false,
      "funding_z_threshold": 2.0,
      "oi_div_threshold": 0.02,
      "max_cut": 0.50
    }
  }
}
```

## Passo 3 — Atualizar portfolio.json
Adicionar campos para tracking de trades:
```json
{
  "...campos existentes...",
  "entry_price": null,
  "entry_time": null,
  "max_price_since_entry": null,
  "stop_loss_cooldown_until": null
}
```
- `entry_price`: preço do close no candle em que entrou (position > 0)
- `entry_time`: timestamp da entrada
- `max_price_since_entry`: track do máximo para trailing stop futuro
- `stop_loss_cooldown_until`: timestamp até o qual não pode reentrar após stop

## Passo 4 — Implementar apply_control_layer()
Substituir o stub por implementação real:

```python
def apply_control_layer(allocation_raw: float, context: dict, config: dict) -> tuple[float, dict]:
    """
    Control layer: intercepta allocation_raw e aplica gates sequenciais.
    Cada gate pode reduzir ou zerar alocação, nunca aumentar.
    
    Returns:
        allocation_final: float (0 a 1)
        gate_log: dict com info de quais gates dispararam
    """
    allocation = allocation_raw
    gate_log = {
        "stop_loss_triggered": False,
        "macro_gate_multiplier": 1.0,
        "derivatives_gate_multiplier": 1.0,
    }
    cl_config = config.get("control_layer", {})
    
    # ── FASE 1: Stop Loss ──
    sl = cl_config.get("stop_loss", {})
    if sl.get("enabled", False):
        allocation, triggered = _apply_stop_loss(allocation, context, sl)
        gate_log["stop_loss_triggered"] = triggered
    
    # ── FASE 2: Macro Gate (stub) ──
    mg = cl_config.get("macro_gate", {})
    if mg.get("enabled", False):
        allocation, multiplier = _apply_macro_gate(allocation, context, mg)
        gate_log["macro_gate_multiplier"] = multiplier
    
    # ── FASE 3: Derivatives Gate (stub) ──
    dg = cl_config.get("derivatives_gate", {})
    if dg.get("enabled", False):
        allocation, multiplier = _apply_derivatives_gate(allocation, context, dg)
        gate_log["derivatives_gate_multiplier"] = multiplier
    
    return allocation, gate_log
```

## Passo 5 — Implementar _apply_stop_loss()
```python
def _apply_stop_loss(allocation: float, context: dict, config: dict) -> tuple[float, bool]:
    """
    Stop loss fixo: se posição aberta e drawdown > pct, força saída.
    Após stop, aplica cooldown (não reentra por N candles).
    """
    entry_price = context.get("entry_price")
    current_price = context.get("current_price")
    cooldown_until = context.get("stop_loss_cooldown_until")
    current_time = context.get("current_time")
    stop_pct = config.get("pct", 0.03)
    cooldown_candles = config.get("cooldown_candles", 3)
    
    # Check cooldown — se em cooldown, força cash
    if cooldown_until and current_time < cooldown_until:
        return 0.0, False  # ainda em cooldown, não reentra
    
    # Check stop loss — só se tem posição aberta
    if entry_price and current_price and entry_price > 0:
        drawdown = (current_price - entry_price) / entry_price
        if drawdown < -stop_pct:
            # STOP LOSS TRIGGERED
            # Setar cooldown = current_time + cooldown_candles * 4h
            return 0.0, True
    
    return allocation, False
```

## Passo 6 — Stubs para Fases 2 e 3
```python
def _apply_macro_gate(allocation: float, context: dict, config: dict) -> tuple[float, float]:
    """
    Macro gate: MOVE-based continuous gate.
    Reduz exposição quando MOVE alto + HMM bearish.
    
    Formula (do daily specialist, validada):
      stress_factor = clip((MOVE - min) / (max - min), 0, 1)
      bearish_strength = clip(1 - allocation_r5c, 0, 1)
      multiplier = (1 - stress_factor * bearish_strength * max_cut).clip(0, 1)
    
    TODO: implementar leitura do MOVE mais recente
    """
    multiplier = 1.0  # pass-through
    return allocation * multiplier, multiplier


def _apply_derivatives_gate(allocation: float, context: dict, config: dict) -> tuple[float, float]:
    """
    Derivatives gate: funding/OI-based.
    Reduz exposição quando funding extremo + OI divergindo do preço.
    
    Logic:
      funding_z e oi_price_div têm sinal forte (#2 e #6 importance na ablation)
      mas degradam Sharpe como features do modelo.
      Como gate pós-modelo: regra fixa baseada em conhecimento de mercado.
    
    TODO: implementar leitura de funding_z e oi_price_div
    """
    multiplier = 1.0  # pass-through
    return allocation * multiplier, multiplier
```

## Passo 7 — Atualizar signals.csv
Adicionar colunas ao log:
```
...colunas existentes...,
stop_loss_triggered, macro_gate_mult, deriv_gate_mult, entry_price, drawdown_pct
```
IMPORTANTE: manter backward compatibility — se signals.csv já tem linhas, as novas colunas ficam vazias nas linhas antigas.

## Passo 8 — Atualizar fluxo principal
No loop principal do paper trader, onde hoje chama apply_control_layer():
1. Montar context com entry_price, current_price, current_time, cooldown_until (do portfolio.json)
2. Chamar apply_control_layer(allocation_raw, context, config)
3. Se stop_loss_triggered: atualizar portfolio.json com cooldown_until = current_time + timedelta(hours=4*cooldown_candles)
4. Se nova entrada (era cash, agora position > 0): setar entry_price e entry_time no portfolio.json
5. Se saiu (era position > 0, agora cash): limpar entry_price, entry_time, max_price_since_entry
6. Atualizar max_price_since_entry a cada candle se position > 0
7. Logar gate_log no signals.csv

## Passo 9 — Atualizar dashboard
Em `apps/specialist_4h_dashboard.py`, adicionar:
- Indicador visual de stop loss (se triggered recentemente)
- Mostrar entry_price e drawdown atual quando em posição
- Mostrar status dos gates (enabled/disabled, último trigger)

## Restrições
- NÃO modificar a lógica do modelo AB (predict_proba não muda)
- NÃO modificar r11_paper_trader.py
- Gates NUNCA aumentam alocação, só reduzem
- Manter signals.csv backward compatible
- Cooldown é em candles (não tempo fixo) — 3 candles = 12h
- entry_price = close do candle onde a posição foi aberta (não o preço de execução simulado)
- Sem prints — usar logging

## Validação
1. Simular cenário de stop loss: editar portfolio.json com entry_price alto e btc_price baixo (drawdown > 3%), rodar o script, verificar que força saída e seta cooldown
2. Verificar que signals.csv tem as novas colunas
3. Verificar que config.json tem control_layer com stop_loss enabled e os outros disabled
4. Verificar que macro_gate e derivatives_gate são ignorados quando disabled
5. Rodar dashboard e verificar que mostra info de stop loss
