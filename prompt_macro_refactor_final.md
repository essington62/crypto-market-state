# TASK: Refatorar Macro News Pipeline (Google News RSS) com foco em qualidade de sinal e consistência L3/L4

## Projeto: crypto-market-state
Path: `/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/`

## CONTEXTO
O pipeline atual de macro news ingest via Google News RSS já está funcional, mas precisa de correções estruturais para evitar ruído, duplicação e inconsistência com o classificador DeepSeek.

## OBJETIVO
Transformar o pipeline em um gerador de sinal de regime macro robusto, alinhado com arquitetura modular (L1→L4), garantindo:
- ingest limpo (L1)
- classificação consistente (L3)
- agregação baseada em score real (L4)

## Passo 0 — Inspecionar antes de mexer
```bash
cat scripts/cron/macro_news_ingest.py
cat scripts/cron/classify_news.py
cat conf/macro_keywords.json
ls data/01_raw/news/macro/
```

---

## CORREÇÃO 1 — REMOVER MÉTRICAS BASEADAS EM CONTAGEM (CRÍTICO)

Problema: compute_macro_metrics usa apenas contagem de notícias → não gera sinal.

Correção:
- REMOVER qualquer uso de count, negative_pct ou métricas baseadas em volume de `macro_news_ingest.py`
- As únicas métricas válidas vêm PÓS-CLASSIFICAÇÃO e usam exclusivamente:
  - `ds_score`
  - `ds_impact`
  - `event_type`

Nova lógica de agregação (implementar em classify_news.py, NÃO no ingest):
```python
def aggregate_macro_metrics(df: pd.DataFrame, hours: int, source_weights: dict) -> dict:
    """Agregar métricas macro a partir de notícias JÁ CLASSIFICADAS pelo DeepSeek."""
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    recent = df[(df["published_at"] >= cutoff) & (df["ds_classified"] == True)]
    
    if len(recent) == 0:
        return {"macro_score": 0.0, "escalation_count": 0, "deescalation_count": 0,
                "dominant_group": None, "classified_count": 0}
    
    # Score ponderado por fonte
    recent["source_weight"] = recent["source"].map(
        lambda s: max([w for k, w in source_weights.items() if k.lower() in s.lower()], default=0.5)
    )
    recent["weighted_score"] = recent["ds_score"] * recent["source_weight"]
    
    macro_score = recent["weighted_score"].mean()
    escalation_count = (recent["ds_event_type"] == "ESCALATION").sum()
    deescalation_count = (recent["ds_event_type"] == "DEESCALATION").sum()
    
    # Grupo dominante por peso
    by_group = recent.groupby("search_group")["weighted_score"].mean()
    dominant_group = by_group.idxmin() if macro_score < 0 else by_group.idxmax()
    
    return {
        "macro_score": float(macro_score),
        "escalation_count": int(escalation_count),
        "deescalation_count": int(deescalation_count),
        "dominant_group": dominant_group,
        "classified_count": int(len(recent)),
        "top_stories": _get_top_stories(recent, n=5),
    }

def _get_top_stories(df, n=5):
    """Top N notícias por abs(score) — as mais impactantes."""
    top = df.nlargest(n, "ds_score", keep="first") if df["ds_score"].mean() > 0 else df.nsmallest(n, "ds_score", keep="first")
    return [
        {"title": r["title"][:100], "score": float(r["ds_score"]), 
         "event_type": r["ds_event_type"], "group": r["search_group"], "source": r["source"]}
        for _, r in top.iterrows()
    ]
```

IMPORTANTE: `aggregate_macro_metrics` roda APENAS após classificação DeepSeek, DENTRO do classify_news.py.

---

## CORREÇÃO 2 — MOVER AGREGAÇÃO PARA PÓS-CLASSIFICAÇÃO

Problema: macro_metrics está sendo calculado no ingest (L1).

Correção:
- REMOVER `compute_macro_metrics()` de `macro_news_ingest.py`
- O ingest faz APENAS: fetch RSS → parse → dedup → salvar parquet (L1 puro)
- A agregação roda em `classify_news.py` após classificar crypto + macro

Fluxo correto:
```
:55  crypto ingest (L1 puro — salva btc_news.parquet)
:56  macro ingest (L1 puro — salva google_news.parquet)
:57  classify + aggregate (L3 + L4):
     1. Classifica crypto não-classificadas (DeepSeek)
     2. Classifica macro não-classificadas (DeepSeek)
     3. Agrega métricas crypto (sentiment_score)
     4. Agrega métricas macro (macro_score)
     5. Calcula score combinado
     6. Salva sentiment_metrics.json com TUDO
```

---

## CORREÇÃO 3 — IMPLEMENTAR FILTRO DE RELEVÂNCIA (ANTES DO DEEPSEEK)

Problema: Google News RSS traz ruído (opinião, política irrelevante).

Implementar em classify_news.py:
```python
MACRO_RELEVANCE_KEYWORDS = [
    "oil", "crude", "petroleum", "fuel", "energy",
    "inflation", "cpi", "prices",
    "fed", "federal reserve", "rate", "fomc", "powell", "monetary",
    "war", "conflict", "iran", "military", "sanctions", "hormuz",
    "china", "tariff", "trade war",
    "economy", "recession", "gdp", "unemployment", "jobs",
    "yield", "bond", "treasury", "debt",
]

def macro_relevance_filter(title: str) -> bool:
    """Filtrar notícias macro irrelevantes ANTES de enviar ao DeepSeek."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in MACRO_RELEVANCE_KEYWORDS)
```

Aplicar:
- Filtrar notícias macro antes de enviar ao DeepSeek → reduz custo
- Notícias que não passam o filtro → marcar como `ds_classified=True, ds_score=0, ds_event_type="NOISE"` sem custo de API

---

## CORREÇÃO 4 — DEDUP ROBUSTO (CRÍTICO)

Problema: Mesma notícia aparece com títulos levemente diferentes (Reuters vs Yahoo).

Substituir hash atual em `macro_news_ingest.py`:
```python
import re
import hashlib

def normalize_title(title: str) -> str:
    """Normalizar título para dedup: remove pontuação, lowercase, strip."""
    return re.sub(r'[^a-z0-9 ]', '', title.lower()).strip()

def generate_title_hash(title: str) -> str:
    """Hash robusto do título normalizado."""
    normalized = normalize_title(title)
    return hashlib.sha1(normalized.encode()).hexdigest()
```

Aplicar:
- No ingest ao criar `title_hash`
- No merge parquet: `drop_duplicates(subset=["title_hash"])`
- Antes da classificação: evitar classificar duplicatas

---

## CORREÇÃO 5 — SOURCE WEIGHTING (QUALIDADE DE FONTE)

Problema: Todas as fontes têm o mesmo peso na agregação.

Adicionar config (em `conf/macro_keywords.json` ou hardcoded):
```python
SOURCE_WEIGHTS = {
    "Reuters": 1.0,
    "Bloomberg": 1.0,
    "Wall Street Journal": 0.9,
    "WSJ": 0.9,
    "Financial Times": 0.9,
    "New York Times": 0.9,
    "CNBC": 0.8,
    "CNN": 0.7,
    "Al Jazeera": 0.8,
    "BBC": 0.8,
    "Associated Press": 0.9,
    "Yahoo Finance": 0.6,
    "Yahoo": 0.5,
    "The Motley Fool": 0.4,
    "Substack": 0.5,
}
DEFAULT_WEIGHT = 0.5
```

Na agregação: `weighted_score = ds_score × source_weight`

---

## CORREÇÃO 6 — REDUZIR VOLUME DE INPUT

Problema: 5 grupos × 20 notícias = 100/h → excesso de ruído + custo DeepSeek.

Correção em `conf/macro_keywords.json`:
```json
{
  "max_results_per_query": 10
}
```

Pipeline de redução:
```
RSS retorna: ~50 notícias (5 grupos × 10)
Dedup: ~35-40 únicas
Relevance filter: ~20-25 relevantes
→ DeepSeek classifica: ~20-25 (custo ~$0.003/hora)
```

---

## CORREÇÃO 7 — GARANTIR CONSISTÊNCIA COM CLASSIFICADOR

Regras rígidas:
- NÃO usar sentiment do Google/CryptoCompare externo na agregação
- NÃO criar métricas paralelas (tudo vem do DeepSeek pós-classificação)
- DeepSeek é a ÚNICA fonte de verdade para:
  - direção (derivada do score: ≥+2 BULLISH, ≤-2 BEARISH)
  - impacto (HIGH/MEDIUM/LOW)
  - event_type (ESCALATION/DEESCALATION/etc)
- O sentiment nativo do CryptoCompare pode ser mantido no parquet como campo informativo, mas NUNCA usado para decisão

---

## AJUSTE ADICIONAL — AGREGAÇÃO DENTRO DO CLASSIFY_NEWS.PY

O `classify_news.py` (:57) faz TUDO após os ingests:

```python
def run():
    # ── L3: Classificar ──
    # Crypto
    crypto_df = load_and_classify_crypto()
    
    # Macro
    macro_df = load_and_classify_macro()
    
    # ── L4: Agregar ──
    # Métricas crypto (já existe)
    crypto_metrics = compute_crypto_impact(crypto_df)
    
    # Métricas macro (NOVO)
    macro_metrics = {}
    for label, hours in [("1h", 1), ("4h", 4), ("24h", 24)]:
        macro_metrics[label] = aggregate_macro_metrics(macro_df, hours, SOURCE_WEIGHTS)
    
    # Score combinado
    combined = {}
    for label in ["1h", "4h", "24h"]:
        crypto_score = crypto_metrics.get(label, {}).get("impact_score", 0)
        macro_score = macro_metrics.get(label, {}).get("macro_score", 0)
        # Peso: macro 60%, crypto 40% (macro tem mais impacto em regime)
        combined[label] = {
            "crypto_score": crypto_score,
            "macro_score": macro_score,
            "combined_score": 0.4 * crypto_score + 0.6 * macro_score,
            "dominant_driver": _get_dominant_driver(crypto_score, macro_score, macro_metrics[label]),
        }
    
    # ── Salvar TUDO em sentiment_metrics.json ──
    metrics = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        # Crypto (existente)
        "1h": crypto_metrics.get("1h", {}),
        "4h": crypto_metrics.get("4h", {}),
        "24h": crypto_metrics.get("24h", {}),
        # Macro (NOVO)
        "macro": macro_metrics,
        # Combinado (NOVO)
        "combined": combined,
        # Flags existentes
        "news_zscore_24h": compute_news_zscore(crypto_df),
        "high_stress": _compute_high_stress(combined),
    }
    
    save_metrics(metrics)
```

high_stress refinado:
```python
def _compute_high_stress(combined: dict) -> bool:
    """High stress = escalation cluster OU macro+crypto ambos muito negativos."""
    c4h = combined.get("4h", {})
    return (
        c4h.get("combined_score", 0) < -3.0
        or (c4h.get("macro_score", 0) < -4.0 and c4h.get("crypto_score", 0) < -0.3)
    )
```

---

## OUTPUT ESPERADO

1. `macro_news_ingest.py` refatorado:
   - APENAS ingestão + dedup robusto
   - SEM métricas, SEM compute_macro_metrics
   - Dedup com normalize_title + sha1
   - Salva em `data/01_raw/news/macro/google_news.parquet`

2. `classify_news.py` atualizado:
   - Classifica crypto + macro
   - Filtro de relevância macro antes do DeepSeek
   - Agregação L4 pós-classificação (macro_score, escalation_count, top_stories)
   - Source weighting na agregação
   - Score combinado (crypto 40% + macro 60%)
   - Salva TUDO em sentiment_metrics.json

3. `sentiment_metrics.json` atualizado:
   ```json
   {
     "1h": { "...crypto..." },
     "4h": { "...crypto..." },
     "24h": { "...crypto..." },
     "macro": {
       "1h": { "macro_score": -2.5, "escalation_count": 1, "...": "..." },
       "4h": { "macro_score": -4.1, "escalation_count": 3, "dominant_group": "geopolitical", "top_stories": [...] },
       "24h": { "..." }
     },
     "combined": {
       "4h": { "crypto_score": -0.23, "macro_score": -4.1, "combined_score": -2.55, "dominant_driver": "geopolitical" }
     },
     "high_stress": true
   }
   ```

---

## CONSTRAINTS (OBRIGATÓRIO)
- Sem hardcoded paths — tudo derivado de PROJECT root ou config
- Tudo em UTC (usar parse_utc do shared/execution.py se necessário)
- Funções puras (sem side effects ocultos)
- Output em Parquet
- Separação clara L1 (raw) vs L3 (classificado) vs L4 (agregado)
- DeepSeek é única fonte de verdade para score/impact/event_type
- Source weighting na agregação (Reuters > Yahoo)
- Filtro de relevância antes do DeepSeek (economizar tokens)

---

## CRITÉRIO DE SUCESSO
- Macro score reflete eventos reais (ex: guerra → score negativo, paz → positivo)
- Redução de ruído irrelevante (notícias filtradas antes do DeepSeek)
- Sem duplicação entre fontes (normalize_title + sha1)
- Consistência entre crypto e macro pipelines (ambos usam DeepSeek como verdade)
- News gate responde corretamente a macro_stress
- combined_score influencia decisão do pipeline integrado
