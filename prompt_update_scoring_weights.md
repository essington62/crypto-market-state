# TASK: Atualizar Pesos do Scoring System (validado com walk-forward 2025+)

## Projeto: crypto-market-state
Arquivo: `scripts/paper_trading/specialist_4h_paper_trader.py`

## Contexto
Walk-forward em 208 dias de Sideways (2025+) validou:
- BB < 0.30: 51 sinais, win_3d=75%, ret_3d=+1.37%
- Score >= 2.5: 32 sinais, win_3d=72%, ret_3d=+1.18%
- BB > 0.30: retorno NEGATIVO (win 41-42%)
- BB < 0.20: win 88%, ret_3d=+1.75% (sinal mais forte)

## Mudança (cirúrgica, só os pesos e threshold)

Localizar a função `_compute_entry_score()` e atualizar APENAS os pesos:

### BB score (dominante — calibrado pelos dados):
```python
# ANTES:
if bb < 0.20:
    bb_score = 2.0
elif bb < 0.30:
    bb_score = 1.5
elif bb < 0.40:
    bb_score = 0.5
else:
    bb_score = 0.0

# DEPOIS:
if bb > 0.80:
    bb_score = -2.0   # kill switch no topo (win 43%, ret negativo)
elif bb < 0.20:
    bb_score = 3.0    # sinal forte (win 88%, ret_3d +1.75%)
elif bb < 0.30:
    bb_score = 2.0    # sinal bom (win 77%, ret_3d +1.65%)
elif bb < 0.40:
    bb_score = 0.5    # zona favorável mas fraca
else:
    bb_score = 0.0    # neutro
```

### RSI score (sem mudança, já está correto):
```python
if rsi < 35:
    rsi_score = 1.0
elif rsi < 45:
    rsi_score = 0.5
elif rsi > 60:
    rsi_score = -1.0
else:
    rsi_score = 0.0
```

### Alloc score (reduzir para max 0.5):
```python
# ANTES:
if alloc_low <= allocation <= alloc_high:
    alloc_score = 1.0
elif alloc_marginal <= allocation < alloc_low:
    alloc_score = 0.5
else:
    alloc_score = 0.0

# DEPOIS:
if alloc_low <= allocation <= alloc_high:
    alloc_score = 0.5   # fraco — alloc não tem edge forte
elif alloc_marginal <= allocation < alloc_low:
    alloc_score = 0.25
else:
    alloc_score = 0.0
```

### News score (sem mudança, já está correto)

### Threshold (atualizar):
```python
# ANTES:
min_score = float(score_cfg.get("min_score", 2.0))

# DEPOIS:
min_score = float(score_cfg.get("min_score", 2.5))
```

### Atualizar config.json:
```json
{
  "control_layer": {
    "entry_scoring": {
      "enabled": true,
      "min_score": 2.5
    }
  }
}
```

## O que NÃO mudar
- Estrutura da função _compute_entry_score() → NÃO MUDAR
- News score → NÃO MUDAR
- Position sizing baseado em score → NÃO MUDAR (mas ajustar thresholds se necessário)
- Tudo fora de _compute_entry_score() → NÃO MUDAR

## Validação
```bash
python scripts/paper_trading/specialist_4h_paper_trader.py
```

Cenários esperados:
- BB=0.15 RSI=30: score=3.0+1.0=4.0 → ENTER ✅
- BB=0.25 RSI=40: score=2.0+0.5=2.5 → ENTER (limite) ✅
- BB=0.25 RSI=55: score=2.0+0.0=2.0 → HOLD ✅
- BB=0.35 RSI=33: score=0.5+1.0=1.5 → HOLD ✅
- BB=0.85 RSI=any: score=-2.0+X → HOLD ✅
- BB=0.25 RSI=40 news=BEAR forte: score=2.0+0.5-1.5=1.0 → HOLD ✅
