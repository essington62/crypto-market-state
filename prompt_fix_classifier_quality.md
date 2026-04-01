# TASK: Fix Classificador DeepSeek — 5 Ajustes de Qualidade

## Projeto: crypto-market-state
Path: `/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/`
Arquivo principal: `scripts/cron/classify_news.py`

## 5 Problemas e Fixes

### 1. Remover viés de input — NÃO enviar [SENTIMENT] do CryptoCompare
O CryptoCompare classifica como NEUTRAL coisas que são claramente BULLISH/BEARISH. Se enviamos esse label, o DeepSeek é contaminado.

**Fix:** Remover o sentiment do CryptoCompare do input. Enviar APENAS o título:

```python
# ANTES (contaminado):
news_list = "\n".join([
    f"[{i}] [{n['sentiment']}] {n['title']}"
    ...
])

# DEPOIS (limpo):
news_list = "\n".join([
    f"[{i}] {n['title']}"
    ...
])
```

### 2. Remover `direction` — derivar do score
Direction é redundante com score e cria inconsistência (score=-3 com direction=NEUTRAL).

**Fix:** Remover `direction` do output. Derivar automaticamente:

```python
# Após receber classificação:
for item in classifications:
    score = item["score"]
    if score >= 2:
        item["direction"] = "BULLISH"
    elif score <= -2:
        item["direction"] = "BEARISH"
    else:
        item["direction"] = "NEUTRAL"
```

Remover `direction` do prompt do DeepSeek — só pedir score.

### 3. Adicionar `event_type` — classificar o tipo de evento
Sem event_type, o modelo não entende contexto. "War" pode ser escalada (bearish) ou desescalada (bullish).

**Fix:** Adicionar event_type ao prompt:

```
"event_type": "ESCALATION|DEESCALATION|POLICY_SHIFT|CAPITAL_FLOW|MARKET_SHOCK|DATA_RELEASE|INSTITUTIONAL_MOVE|NOISE"
```

Definições:
- ESCALATION: conflito/tensão aumenta (bearish)
- DEESCALATION: conflito/tensão diminui (bullish)
- POLICY_SHIFT: mudança de política (direção depende do conteúdo)
- CAPITAL_FLOW: dinheiro entrando/saindo (ETF inflow/outflow)
- MARKET_SHOCK: evento súbito (hack, crash, liquidação)
- DATA_RELEASE: dados econômicos (CPI, jobs, rate decision)
- INSTITUTIONAL_MOVE: grande player comprando/vendendo
- NOISE: informativo, sem impacto de preço

### 4. Forçar classificação pelo RESULTADO FINAL do evento
O prompt deve instruir o DeepSeek a classificar pelo impacto no BTC, não pelas palavras.

### 5. Explicitar que eventos macro = HIGH impact
Qualquer evento com potencial de alterar comportamento global de risco deve ser HIGH.

## Novo Prompt DeepSeek

Substituir o CLASSIFY_PROMPT em classify_news.py por:

```python
CLASSIFY_PROMPT = """Você é um analista sênior de risco para trading de Bitcoin.

REGRA PRINCIPAL: Classifique pelo RESULTADO FINAL do evento no preço do Bitcoin, NÃO pelas palavras individuais.
Exemplos:
- "Trump signals end of Iran war" → DEESCALATION → score +7 (fim de guerra = risk-on = BTC sobe)
- "Trump threatens ground invasion of Iran" → ESCALATION → score -8 (guerra escala = risk-off = BTC cai)
- "Fed signals rate cut" → POLICY_SHIFT → score +6 (juros caem = liquidez = BTC sobe)
- "Investors pull $414M from crypto funds" → CAPITAL_FLOW → score -5 (outflow = pressão vendedora)
- "Google warns quantum threatens crypto" → NOISE → score 0 (sem impacto imediato no preço)

REGRA DE IMPACTO:
- HIGH: qualquer evento que mude narrativa macro (guerra, paz, decisão do Fed, tarifa, ban nacional). SEMPRE HIGH se envolve Trump+guerra, Fed+rate, ou regulação nacional.
- MEDIUM: evento significativo mas não muda narrativa (ETF flow, miner sell-off, whale move)
- LOW: informativo, análise, previsão, opinião

Classifique cada notícia. Responda APENAS em JSON válido, sem markdown, sem explicação:
[
  {{
    "index": 0,
    "topic": "geopolitical_war|trump_policy|fed_monetary|institutional_btc|oil_energy|regulatory|market_stress|mining|noise",
    "event_type": "ESCALATION|DEESCALATION|POLICY_SHIFT|CAPITAL_FLOW|MARKET_SHOCK|DATA_RELEASE|INSTITUTIONAL_MOVE|NOISE",
    "impact": "HIGH|MEDIUM|LOW",
    "score": -10 a +10,
    "reason": "máximo 8 palavras"
  }}
]

Score:
+7 a +10: muda narrativa inteira para bullish
+3 a +6: pressão de alta significativa
+1 a +2: leve positivo
0: neutro/irrelevante
-1 a -2: leve negativo
-3 a -6: pressão de baixa significativa
-7 a -10: muda narrativa inteira para bearish

NOTÍCIAS (classifique TODAS):
{news_list}
"""
```

## Atualizar colunas do parquet

Adicionar/renomear:
```python
# Novas colunas
"ds_event_type"   # ESCALATION, DEESCALATION, etc
# Remover ds_direction — derivar do score
# ds_direction calculado automaticamente: score >= 2 = BULLISH, <= -2 = BEARISH, else NEUTRAL
```

## Atualizar sentiment_metrics.json

Seção impact agora inclui:
```json
{
  "impact": {
    "4h": {
      "impact_score": -3.5,
      "high_impact_count": 4,
      "escalation_count": 2,
      "deescalation_count": 1,
      "dominant_topic": "geopolitical_war",
      "dominant_event_type": "ESCALATION",
      "top_stories": [
        {"title": "...", "score": +7, "event_type": "DEESCALATION", "topic": "geopolitical_war"},
        {"title": "...", "score": -6, "event_type": "CAPITAL_FLOW", "topic": "institutional_btc"}
      ]
    }
  }
}
```

## Atualizar news gate

```python
# Escalation cluster = bloqueia
escalations = [n for n in high_impact if n["event_type"] == "ESCALATION"]
if len(escalations) >= 2:
    return 0.0, True, "escalation_cluster"

# Deescalation = boost
deescalations = [n for n in high_impact if n["event_type"] == "DEESCALATION"]
if len(deescalations) >= 1 and impact_score > 3:
    allocation = min(allocation * 1.15, 1.0)  # +15% boost
    return allocation, False, "deescalation_boost"
```

## Validação

Testar com notícias reais de hoje:

```python
test_news = [
    "Trump signals he may end Iran war with Hormuz still shut",
    "Investors Pull $414M From Crypto Funds As Inflation Jitters Mount",
    "Bitcoin falls below $70k as risk assets take hit on Iran peace talks uncertainty",
    "Google Warns Quantum Computers Threaten Crypto",
    "Senators Reveal Mined in America Bill to Boost Bitcoin Mining",
    "Bitcoin Price Flashes Warning as Nearly Half of Supply Sits at Loss",
    "Fed Nominee Kevin Warsh Confirmation Hearing Expected Week of April 13",
    "Strategy Pauses Bitcoin Buying as Trump Renews Iran Attack Threats",
]
```

Resultados esperados:
- "Trump signals end war" → DEESCALATION, HIGH, score +7
- "Investors Pull $414M" → CAPITAL_FLOW, MEDIUM, score -5
- "Google quantum" → NOISE, LOW, score 0
- "Fed Nominee hearing" → DATA_RELEASE, MEDIUM, score +2
- "Strategy Pauses Buying" → INSTITUTIONAL_MOVE, MEDIUM, score -4
