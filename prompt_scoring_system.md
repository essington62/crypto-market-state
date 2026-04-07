# TASK: Implementar Scoring System de Entrada (substitui entry filter)

## Projeto: crypto-market-state
Arquivo: `scripts/paper_trading/specialist_4h_paper_trader.py`

## Contexto
A análise cruzada e testes de cenários mostraram:
- ML supervisionado (XGBoost, LR) NÃO funciona em Sideways (AUC < 0.50)
- alloc_raw NÃO tem poder discriminatório (correlação ~0 com retorno)
- O edge está em regras estatísticas simples: BB < 0.30 (100% win rate, 6 trades)
- Scoring system ponderado > filtros AND rígidos (menos overfitting)

## Nova lógica: Scoring System

Cada condição contribui com um peso. A soma decide se entra.
Threshold para entrar: score >= 2.0

### Componentes do score:

```
SCORE = BB_score + RSI_score + ALLOC_score + NEWS_score

BB (sinal dominante — Bollinger Band %B):
  BB < 0.20  → +2.0   (preço muito perto do suporte, sinal forte)
  BB < 0.30  → +1.5   (perto do suporte)
  BB < 0.40  → +0.5   (zona favorável)
  BB >= 0.40 → +0.0   (neutro)

RSI (complementar):
  RSI < 35   → +1.0   (oversold)
  RSI < 45   → +0.5   (abaixo da média)
  RSI 45-60  → +0.0   (neutro)
  RSI > 60   → -1.0   (penaliza topo)

ALLOC (zona intermediária do specialist):
  0.50 <= alloc <= 0.54 → +1.0   (edge do C7)
  0.48 <= alloc < 0.50  → +0.5   (marginal)
  else                   → +0.0   (fora da zona)

NEWS (regime das notícias):
  news_regime == "BULL" and news_score > 3   → +1.0   (notícias fortemente positivas)
  news_regime == "BULL"                       → +0.5   (notícias positivas)
  news_regime == "SIDEWAYS"                   → +0.0   (neutro)
  news_regime == "BEAR"                       → -0.5   (notícias negativas)
  news_regime == "BEAR" and news_score < -3   → -1.5   (notícias fortemente negativas)

THRESHOLD: score >= 2.0 → ENTER
```

### Exemplos de decisão:

```
BB=0.15 RSI=33 alloc=0.52 news=SIDEWAYS
  score = 2.0 + 1.0 + 1.0 + 0.0 = 4.0 → ENTRA (forte)

BB=0.25 RSI=38 alloc=0.51 news=BULL
  score = 1.5 + 0.5 + 1.0 + 0.5 = 3.5 → ENTRA (bom)

BB=0.28 RSI=42 alloc=0.48 news=SIDEWAYS
  score = 1.5 + 0.5 + 0.5 + 0.0 = 2.5 → ENTRA (marginal)

BB=0.50 RSI=55 alloc=0.47 news=SIDEWAYS
  score = 0.0 + 0.0 + 0.0 + 0.0 = 0.0 → NÃO ENTRA

BB=0.60 RSI=62 alloc=0.52 news=BEAR
  score = 0.0 + (-1.0) + 1.0 + (-0.5) = -0.5 → NÃO ENTRA

BB=0.20 RSI=30 alloc=0.45 news=BEAR(forte)
  score = 2.0 + 1.0 + 0.0 + (-1.5) = 1.5 → NÃO ENTRA (BB forte mas news muito ruim)

Entrada que fizemos a $70k (RSI=59, BB=0.67, alloc=0.47):
  score = 0.0 + 0.0 + 0.0 + 0.0 = 0.0 → NÃO ENTRA (correto)

Entrada ideal a $65k (RSI=27, BB=0.11):
  score = 2.0 + 1.0 + X + Y = 3.0+ → ENTRA (correto)
```

## Implementação

### Passo 1 — Substituir _evaluate_entry() por _compute_entry_score()

```python
def _compute_entry_score(allocation: float, gate_log: dict, cfg: dict) -> dict:
    """
    Scoring system para decisão de entrada.
    Cada condição contribui com um peso. Score >= threshold → enter.
    Mais robusto que filtros AND (não precisa que todas as condições sejam perfeitas).
    
    Retorna: {"enter": bool, "score": float, "components": dict, "reason": str}
    """
    score_cfg = cfg.get("control_layer", {}).get("entry_scoring", {})
    if not score_cfg.get("enabled", True):
        threshold = float(cfg.get("entry_threshold", 0.38))
        return {"enter": allocation > threshold, "score": allocation, "components": {}, "reason": "threshold_fixed"}
    
    min_score = float(score_cfg.get("min_score", 2.0))
    components = {}
    total = 0.0
    
    # ── 1. BB score (sinal dominante) ──
    bb = gate_log.get("tech_gate_bb", 0.50)
    if bb < 0.20:
        bb_score = 2.0
    elif bb < 0.30:
        bb_score = 1.5
    elif bb < 0.40:
        bb_score = 0.5
    else:
        bb_score = 0.0
    components["bb"] = bb_score
    total += bb_score
    
    # ── 2. RSI score (complementar) ──
    rsi = gate_log.get("tech_gate_rsi", 50.0)
    if rsi < 35:
        rsi_score = 1.0
    elif rsi < 45:
        rsi_score = 0.5
    elif rsi > 60:
        rsi_score = -1.0
    else:
        rsi_score = 0.0
    components["rsi"] = rsi_score
    total += rsi_score
    
    # ── 3. Alloc score (zona intermediária) ──
    alloc_low = float(score_cfg.get("alloc_low", 0.50))
    alloc_high = float(score_cfg.get("alloc_high", 0.54))
    alloc_marginal = float(score_cfg.get("alloc_marginal", 0.48))
    
    if alloc_low <= allocation <= alloc_high:
        alloc_score = 1.0
    elif alloc_marginal <= allocation < alloc_low:
        alloc_score = 0.5
    else:
        alloc_score = 0.0
    components["alloc"] = alloc_score
    total += alloc_score
    
    # ── 4. News score ──
    try:
        import json
        metrics_path = PROJECT / "data" / "01_raw" / "news" / "cryptocompare" / "sentiment_metrics.json"
        with open(metrics_path) as f:
            metrics = json.load(f)
        combined = metrics.get("combined", {}).get("4h", {})
        news_regime = combined.get("regime", "SIDEWAYS")
        news_score_val = float(combined.get("combined_score", 0))
    except Exception:
        news_regime = "SIDEWAYS"
        news_score_val = 0.0
    
    if news_regime == "BULL" and news_score_val > 3:
        news_score = 1.0
    elif news_regime == "BULL":
        news_score = 0.5
    elif news_regime == "BEAR" and news_score_val < -3:
        news_score = -1.5
    elif news_regime == "BEAR":
        news_score = -0.5
    else:
        news_score = 0.0
    components["news"] = news_score
    total += news_score
    
    # ── Decisão ──
    enter = total >= min_score
    
    # Regime check (hard block)
    regime = gate_log.get("regime_gate_regime", "")
    if regime.lower() == "bear":
        enter = False
        total = -99
        components["regime_block"] = True
    
    reason_parts = [f"{k}={v:+.1f}" for k, v in components.items() if isinstance(v, (int, float))]
    reason = f"score={total:.1f} ({' '.join(reason_parts)}) {'ENTER' if enter else 'HOLD'}"
    
    logger.info(
        "[entry_score] total=%.1f (bb=%.1f rsi=%.1f alloc=%.1f news=%.1f) min=%.1f → %s",
        total, bb_score, rsi_score, alloc_score, news_score, min_score,
        "ENTER" if enter else "HOLD",
    )
    
    return {
        "enter": enter,
        "score": total,
        "components": components,
        "reason": reason,
    }
```

### Passo 2 — Integrar no fluxo de decisão

Substituir a chamada de `_evaluate_entry()` por `_compute_entry_score()`:

```python
# ANTES (entry filter):
# entry = _evaluate_entry(allocation_final, gate_log, cfg, portfolio)

# DEPOIS (scoring):
entry = _compute_entry_score(allocation_final, gate_log, cfg)
logger.info("[entry] %s", entry["reason"])

if entry["enter"]:
    # Gerar intenção pendente
    ...
else:
    logger.info("[SPECIALIST] Entry scored below threshold: %s", entry["reason"])
```

### Passo 3 — Atualizar config.json

Substituir seção `entry_filter` por `entry_scoring`:

```json
{
  "control_layer": {
    "entry_scoring": {
      "enabled": true,
      "min_score": 2.0,
      "alloc_low": 0.50,
      "alloc_high": 0.54,
      "alloc_marginal": 0.48
    }
  }
}
```

Pesos do BB, RSI e news estão no código (não no config) porque são baseados na análise estatística. Se precisar ajustar, muda no código ou adiciona ao config depois.

### Passo 4 — Atualizar signals.csv

Substituir colunas entry_filter por entry_score:
```
entry_score_total, entry_score_bb, entry_score_rsi, entry_score_alloc, entry_score_news, entry_score_reason
```

### Passo 5 — Atualizar _print_report()

```python
logger.info("    entry_score   : total=%.1f (bb=%.1f rsi=%.1f alloc=%.1f news=%.1f) → %s",
    entry["score"],
    entry["components"].get("bb", 0),
    entry["components"].get("rsi", 0),
    entry["components"].get("alloc", 0),
    entry["components"].get("news", 0),
    "ENTER" if entry["enter"] else "HOLD",
)
```

### Passo 6 — Position sizing baseado no score

O score também pode modular o tamanho da posição:

```python
if entry["enter"]:
    # Score mais alto = posição maior (dentro do Sideways sizing)
    base_exposure = target_exposure  # já reduzido pelo sideways factor
    if entry["score"] >= 3.5:
        # Sinal muito forte (BB+RSI+alloc+news todos alinhados)
        size_mult = 1.2
    elif entry["score"] >= 2.5:
        size_mult = 1.0
    else:  # score 2.0-2.5
        size_mult = 0.8
    
    final_exposure = base_exposure * size_mult
```

## O que REMOVER
- Remover `_evaluate_entry()` (substituída por `_compute_entry_score()`)
- Remover seção `entry_filter` do config.json (substituída por `entry_scoring`)

## O que NÃO mudar
- shared/execution.py → NÃO MUDAR
- Monitor 1h → NÃO MUDAR
- Timing layer → NÃO MUDAR
- Regime gate (R5C) → NÃO MUDAR
- News gate → NÃO MUDAR (continua como gate independente, scoring usa o resultado)
- Technical gate → NÃO MUDAR (continua computando RSI/BB, scoring lê do gate_log)
- Stop loss/gain/trailing → NÃO MUDAR
- Sideways position sizing → NÃO MUDAR

## Validação

```bash
python scripts/paper_trading/specialist_4h_paper_trader.py
```

Cenário atual (RSI~59, BB~0.67, alloc~0.45):
```
[entry_score] total=0.0 (bb=0.0 rsi=0.0 alloc=0.0 news=0.0) min=2.0 → HOLD
```
Correto — nenhuma condição favorável.

Cenário ideal (RSI=27, BB=0.11, alloc=0.52, news=SIDEWAYS):
```
[entry_score] total=4.0 (bb=2.0 rsi=1.0 alloc=1.0 news=0.0) min=2.0 → ENTER
```
Correto — sinal forte, entra.

Cenário intermediário (RSI=42, BB=0.28, alloc=0.51, news=BULL):
```
[entry_score] total=3.5 (bb=1.5 rsi=0.5 alloc=1.0 news=0.5) min=2.0 → ENTER
```
Correto — combinação boa, entra.

Cenário com news negativa forte (RSI=30, BB=0.18, alloc=0.52, news=BEAR forte):
```
[entry_score] total=1.5 (bb=2.0 rsi=1.0 alloc=1.0 news=-1.5) min=2.0 → HOLD
```
Correto — BB forte mas news muito negativas cancelam. Proteção contra entrar em crash com notícia de escalada.
