# TASK: Implementar Nova Lógica de Entrada (baseada na análise cruzada)

## Projeto: crypto-market-state
Arquivo: `scripts/paper_trading/specialist_4h_paper_trader.py`

## Contexto
A análise cruzada de 82 sinais (13 dias) revelou:
1. alloc_raw NÃO tem poder discriminatório (correlação ~0 com retorno)
2. alloc_raw extremo alto (>0.54) performa PIOR que o meio
3. O edge está em: alloc mid (Q40-Q70) + RSI 30-50 + sem queda forte
4. Win rate 24h = 80% com esses filtros

## Mudanças

### Passo 1 — Remover threshold fixo de entrada

O threshold fixo de 0.38 não funciona (99% dos sinais passam). Substituir por lógica percentil dinâmica.

**ANTES:**
```python
if allocation_final > entry_threshold:  # 0.38
    generate_pending_signal()
```

**DEPOIS:**
```python
# A decisão de entrada usa filtros combinados, não threshold fixo
entry_decision = _evaluate_entry(allocation_final, gate_log, config, portfolio)
if entry_decision["enter"]:
    generate_pending_signal()
```

### Passo 2 — Criar função _evaluate_entry()

```python
def _evaluate_entry(allocation: float, gate_log: dict, cfg: dict, portfolio: dict) -> dict:
    """
    Avalia se deve gerar intenção de entrada.
    Baseado na análise cruzada: alloc zona intermediária + RSI não esticado + BB não topo.
    SIMPLES: 3 filtros. Overfiltrar mata o edge.
    
    Retorna: {"enter": bool, "reason": str, "confidence": float}
    """
    entry_cfg = cfg.get("control_layer", {}).get("entry_filter", {})
    if not entry_cfg.get("enabled", True):
        # Fallback: threshold fixo
        threshold = float(cfg.get("entry_threshold", 0.38))
        return {"enter": allocation > threshold, "reason": "threshold_fixed", "confidence": allocation}
    
    reasons = []
    
    # 1. Regime check (já passou pelo gate, mas double-check)
    regime = gate_log.get("regime_gate_regime", "")
    if regime.lower() == "bear":
        return {"enter": False, "reason": "regime_bear", "confidence": 0}
    
    # 2. Alloc na zona intermediária (edge está no meio, não nos extremos)
    alloc_low = float(entry_cfg.get("alloc_quantile_low", 0.50))
    alloc_high = float(entry_cfg.get("alloc_quantile_high", 0.54))
    
    if not (alloc_low <= allocation <= alloc_high):
        reasons.append(f"alloc_outside({allocation:.3f} not in [{alloc_low:.3f}, {alloc_high:.3f}])")
    
    # 3. RSI não esticado (evitar topo sem ser restritivo)
    rsi = gate_log.get("tech_gate_rsi", 50.0)
    rsi_max = float(entry_cfg.get("rsi_max", 55))
    
    if rsi >= rsi_max:
        reasons.append(f"rsi_high({rsi:.1f} >= {rsi_max})")
    
    # 4. BB não no topo extremo
    bb = gate_log.get("tech_gate_bb", 0.50)
    bb_max = float(entry_cfg.get("bb_max", 0.80))
    
    if bb >= bb_max:
        reasons.append(f"bb_high({bb:.2f} >= {bb_max})")
    
    # Decisão
    if len(reasons) == 0:
        return {
            "enter": True,
            "reason": f"all_pass(alloc={allocation:.3f} rsi={rsi:.1f} bb={bb:.2f})",
            "confidence": allocation,
        }
    else:
        return {
            "enter": False,
            "reason": " | ".join(reasons),
            "confidence": allocation,
        }
```

### Passo 3 — Integrar no fluxo de decisão

No bloco onde compara allocation_final com threshold e decide gerar pending signal:

```python
# ANTES:
# if target_exposure > current_exposure + min_delta:
#     generate_pending_signal(...)

# DEPOIS:
entry = _evaluate_entry(allocation_final, gate_log, cfg, portfolio)
logger.info(
    "[entry_filter] enter=%s reason=%s confidence=%.3f",
    entry["enter"], entry["reason"], entry["confidence"],
)

if entry["enter"]:
    # Gerar intenção pendente (como antes)
    ...
else:
    logger.info("[SPECIALIST] Entry filtered: %s", entry["reason"])
```

### Passo 4 — Atualizar config.json

IMPORTANTE: Manter filtros SIMPLES. A análise mostrou que overfiltrar mata o edge.
O edge original (C7) = alloc 0.50-0.54 + RSI < 50 → 20 trades, ret_12h=+0.635%, win=65%.
Adicionar prev_ret ou RSI > 30 reduziu trades para 10 e piorou retorno. Menos é mais.

```json
{
  "control_layer": {
    "entry_filter": {
      "enabled": true,
      "alloc_quantile_low": 0.50,
      "alloc_quantile_high": 0.54,
      "rsi_max": 55,
      "bb_max": 0.80
    }
  }
}
```

Apenas 3 filtros:
1. Alloc raw na zona intermediária (0.50-0.54)
2. RSI < 55 (evita topo sem ser restritivo)
3. BB < 0.80 (evita topo extremo)

SEM filtro de prev_ret (mata trades de recuperação).
SEM filtro de RSI mínimo (mata trades de oversold).

Todos os parâmetros editáveis sem mexer no código.

### Passo 5 — Logging no signals.csv

Adicionar colunas ao log:
```
entry_filter_result, entry_filter_reason, entry_rsi, entry_prev_ret
```

### Passo 6 — Manter compatibilidade

Se `entry_filter.enabled = false`, usa o threshold fixo antigo (fallback). Permite reverter rapidamente se a nova lógica não funcionar.

## Fluxo completo após mudança

```
R5C regime → regime gate (Bear bloqueia, Sideways/Bull passa)
    ↓
Specialist → alloc_raw
    ↓
News gate (Bull/Sideways/Bear)
    ↓
Technical gate (RSI + BB → bloqueia topo)
    ↓
Entry filter (NOVO):
    alloc Q40-Q70? ✓
    RSI 30-50? ✓
    prev_ret > -1%? ✓
    → GERA INTENÇÃO
    ↓
Timing Layer (monitor 1h) → EXECUTA
    ↓
Position sizing (Sideways × 0.5)
```

## O que NÃO mudar
- shared/execution.py → NÃO MUDAR
- Monitor 1h → NÃO MUDAR
- Timing layer → NÃO MUDAR
- Regime gate (R5C) → NÃO MUDAR
- News gate → NÃO MUDAR
- Technical gate (RSI + BB) → NÃO MUDAR
- Stop loss/gain/trailing → NÃO MUDAR
- Sideways position sizing → NÃO MUDAR

## Validação

```bash
python scripts/paper_trading/specialist_4h_paper_trader.py
```

Com RSI=59, alloc=0.47:
- RSI 59 > rsi_max 50 → FILTERED (rsi_outside) ✅
- Não gera intenção ✅ (correto, preço esticado)

Com RSI=35, alloc=0.52, prev_ret=+0.3%:
- alloc 0.52 in [0.50, 0.54] ✅
- RSI 35 in (30, 50) ✅
- prev_ret +0.3% > -1% ✅
- → GERA INTENÇÃO ✅ (correto, boa entrada)
