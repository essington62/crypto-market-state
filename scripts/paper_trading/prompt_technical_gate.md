# TASK: Adicionar Gate Técnico (RSI + BB) no Specialist

## Projeto: crypto-market-state
Arquivo: `scripts/paper_trading/specialist_4h_paper_trader.py`

## Objetivo
Evitar entradas quando o preço está esticado (RSI alto, BB alto). Favorecer entradas em oversold (RSI baixo, BB baixo). É um gate de "posição no range" — grafismo básico.

## Lógica

O gate técnico avalia RSI e BB%B combinados para decidir:

```
BLOQUEIA entrada (preço esticado):
  RSI > 65 OU BB%B > 0.75
  → preço no topo do range, risco de rejeição

REDUZ allocation (zona neutra/alta):
  RSI 50-65 E BB%B 0.50-0.75
  → allocation × 0.7 (reduz 30%)

PASSA normal (zona neutra):
  RSI 40-50 E BB%B 0.30-0.50
  → allocation intacta

BOOST allocation (oversold, boa entrada):
  RSI < 35 E BB%B < 0.25
  → allocation × 1.2 (boost 20%)

FORTE BOOST (fortemente oversold):
  RSI < 30 E BB%B < 0.15
  → allocation × 1.3 (boost 30%)
```

## Implementação

### Passo 1 — Buscar RSI e BB do candle 1h mais recente

No specialist, após buscar candles 4h, também buscar dados técnicos 1h:

```python
def _compute_technical_gate_features() -> dict:
    """Calcular RSI e BB %B a partir dos candles 1h mais recentes."""
    try:
        # Ler parquet 1h
        import pandas as pd
        ohlcv_1h_path = PROJECT / "data" / "01_raw" / "spot" / "crypto" / "1h" / "BTCUSDT_1h.parquet"
        df = pd.read_parquet(ohlcv_1h_path)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        close = df["close"].iloc[-50:]  # últimas 50 horas

        # RSI 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_14 = float(rsi.iloc[-1])

        # Bollinger Bands %B (20 períodos)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        bb_pct_b = float((close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]))

        return {"rsi_14": rsi_14, "bb_pct_b": bb_pct_b, "valid": True}
    except Exception as e:
        logger.warning("[tech_gate] Failed to compute features: %s — fail open", e)
        return {"rsi_14": 50.0, "bb_pct_b": 0.50, "valid": False}
```

### Passo 2 — Criar função do gate técnico

```python
def _apply_technical_gate(allocation: float, cfg: dict) -> tuple:
    """
    Gate técnico: RSI + BB %B combinados.
    Bloqueia entradas no topo do range, favorece entradas em oversold.
    Retorna (allocation_ajustada, triggered: bool, reason: str)
    """
    tg_cfg = cfg.get("control_layer", {}).get("technical_gate", {})
    if not tg_cfg.get("enabled", True):
        return allocation, False, "disabled"

    features = _compute_technical_gate_features()
    rsi = features["rsi_14"]
    bb = features["bb_pct_b"]

    # Thresholds (configuráveis)
    rsi_block     = float(tg_cfg.get("rsi_block", 65))
    rsi_reduce    = float(tg_cfg.get("rsi_reduce", 50))
    rsi_boost     = float(tg_cfg.get("rsi_boost", 35))
    rsi_strong    = float(tg_cfg.get("rsi_strong_boost", 30))
    bb_block      = float(tg_cfg.get("bb_block", 0.75))
    bb_reduce     = float(tg_cfg.get("bb_reduce", 0.50))
    bb_boost      = float(tg_cfg.get("bb_boost", 0.25))
    bb_strong     = float(tg_cfg.get("bb_strong_boost", 0.15))
    reduce_factor = float(tg_cfg.get("reduce_factor", 0.7))
    boost_factor  = float(tg_cfg.get("boost_factor", 1.2))
    strong_factor = float(tg_cfg.get("strong_boost_factor", 1.3))

    logger.info(
        "[tech_gate] RSI=%.1f BB=%.2f (block: RSI>%.0f or BB>%.2f)",
        rsi, bb, rsi_block, bb_block,
    )

    # BLOQUEIA: preço esticado
    if rsi > rsi_block or bb > bb_block:
        logger.info(
            "[tech_gate] BLOCKED — preço esticado (RSI=%.1f>%.0f or BB=%.2f>%.2f)",
            rsi, rsi_block, bb, bb_block,
        )
        return 0.0, True, f"overbought_rsi{rsi:.0f}_bb{bb:.2f}"

    # FORTE BOOST: fortemente oversold
    if rsi < rsi_strong and bb < bb_strong:
        boosted = min(allocation * strong_factor, 1.0)
        logger.info(
            "[tech_gate] STRONG BOOST — oversold (RSI=%.1f BB=%.2f) alloc %.3f → %.3f",
            rsi, bb, allocation, boosted,
        )
        return boosted, False, f"strong_oversold_rsi{rsi:.0f}_bb{bb:.2f}"

    # BOOST: oversold
    if rsi < rsi_boost and bb < bb_boost:
        boosted = min(allocation * boost_factor, 1.0)
        logger.info(
            "[tech_gate] BOOST — oversold (RSI=%.1f BB=%.2f) alloc %.3f → %.3f",
            rsi, bb, allocation, boosted,
        )
        return boosted, False, f"oversold_rsi{rsi:.0f}_bb{bb:.2f}"

    # REDUZ: zona neutra/alta
    if rsi > rsi_reduce and bb > bb_reduce:
        reduced = allocation * reduce_factor
        logger.info(
            "[tech_gate] REDUCED — zona alta (RSI=%.1f BB=%.2f) alloc %.3f → %.3f",
            rsi, bb, allocation, reduced,
        )
        return reduced, False, f"high_zone_rsi{rsi:.0f}_bb{bb:.2f}"

    # NEUTRO: passa
    logger.info("[tech_gate] NEUTRAL — RSI=%.1f BB=%.2f — pass through", rsi, bb)
    return allocation, False, f"neutral_rsi{rsi:.0f}_bb{bb:.2f}"
```

### Passo 3 — Integrar na control layer

Em `apply_control_layer()`, adicionar o gate técnico DEPOIS do news gate e ANTES do macro/deriv gates:

```python
# Ordem dos gates:
# 1. Stop gain
# 2. Stop loss
# 3. Regime gate (R5C)
# 4. News gate
# 5. Technical gate (RSI + BB) ← NOVO
# 6. Macro gate (stub)
# 7. Derivatives gate (stub)

# Adicionar:
allocation, tg_triggered, tg_reason = _apply_technical_gate(allocation, cfg)
gate_log["technical_gate_triggered"] = tg_triggered
gate_log["technical_gate_reason"] = tg_reason
if tg_triggered:
    gate_log["technical_gate_blocked"] = True
```

### Passo 4 — Atualizar config.json

Adicionar seção configurável:

```json
{
  "control_layer": {
    "technical_gate": {
      "enabled": true,
      "rsi_block": 65,
      "rsi_reduce": 50,
      "rsi_boost": 35,
      "rsi_strong_boost": 30,
      "bb_block": 0.75,
      "bb_reduce": 0.50,
      "bb_boost": 0.25,
      "bb_strong_boost": 0.15,
      "reduce_factor": 0.7,
      "boost_factor": 1.2,
      "strong_boost_factor": 1.3
    }
  }
}
```

Todos os thresholds editáveis sem mexer no código.

### Passo 5 — Atualizar log report

No `_print_report()`, adicionar linha:
```python
logger.info("    technical_gate : RSI=%.1f BB=%.2f → %s", rsi, bb, tg_reason)
```

### Passo 6 — Atualizar signals.csv

Adicionar colunas ao signals log:
```
tech_gate_rsi, tech_gate_bb, tech_gate_reason
```

## Exemplos de comportamento

| Cenário | RSI | BB %B | Ação | Resultado |
|---------|-----|-------|------|-----------|
| BTC $65k (base range) | 27 | 0.11 | STRONG BOOST +30% | Entrada favorecida |
| BTC $66k (oversold) | 33 | 0.22 | BOOST +20% | Entrada favorecida |
| BTC $68k (meio) | 45 | 0.45 | NEUTRO | Passa normal |
| BTC $69k (zona alta) | 55 | 0.60 | REDUCE -30% | Allocation reduzida |
| BTC $70k (topo) | 59 | 0.67 | REDUCE -30% | Allocation reduzida |
| BTC $71k (esticado) | 68 | 0.80 | BLOQUEIA | Não entra |

Com os dados de hoje (RSI=59, BB=0.67):
- Gate teria classificado como REDUCE (RSI 50-65 e BB 0.50-0.75)
- Allocation reduzida 30% → 0.467 × 0.7 = 0.327
- 0.327 < threshold 0.38 → NÃO ENTRA ✅ (correto, preço esticado)

Com dados de sexta (RSI=27, BB=0.11):
- Gate teria classificado como STRONG BOOST
- Allocation aumentada 30% → 0.54 × 1.3 = 0.70
- Position sizing generoso → ENTRA com exposição maior ✅

## O que NÃO mudar
- shared/execution.py → NÃO MUDAR
- Monitor 1h → NÃO MUDAR
- Timing layer → NÃO MUDAR (timing avalia no 1h, este gate avalia no specialist 4h)
- Regime gate → NÃO MUDAR
- News gate → NÃO MUDAR
- Stop loss/gain/trailing → NÃO MUDAR

## Validação

```bash
python scripts/paper_trading/specialist_4h_paper_trader.py
```

Deve mostrar no log:
```
[tech_gate] RSI=59.0 BB=0.67 (block: RSI>65 or BB>0.75)
[tech_gate] REDUCED — zona alta (RSI=59.0 BB=0.67) alloc 0.467 → 0.327
```

E NÃO gerar intenção pendente (0.327 < threshold 0.38).
