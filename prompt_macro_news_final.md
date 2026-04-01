# TASK: Macro News Ingest — Com 5 Fixes de Qualidade

## Projeto: crypto-market-state
Path: `/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/`

## Contexto
Implementar ingestão de notícias macro exógenas (oil, fed, war, inflation) que impactam BTC indiretamente. O classificador DeepSeek já existe em classify_news.py.

## 5 Fixes obrigatórios (aplicar durante implementação)

### Fix 1 — Macro metrics usa APENAS ds_score, NUNCA sentiment CryptoCompare
```python
# ERRADO: usar sentiment do CryptoCompare
sentiment_score = (pos - neg) / n  # contaminado

# CORRETO: usar APENAS ds_score classificado pelo DeepSeek
# Se artigo não foi classificado → não entra no score
classified = df[df["ds_classified"] == True]
impact_score = classified["ds_score"].mean()
```
Toda métrica agregada (macro_stress_score, combined_score) deve derivar exclusivamente de ds_score.

### Fix 2 — Deduplicação global entre crypto e macro
A mesma notícia pode aparecer no feed BTC e no search macro.
```python
# Antes de classificar, dedup global por title_hash
import hashlib

def title_hash(title: str) -> str:
    return hashlib.md5(title.lower().strip().encode()).hexdigest()

# No classify_news.py:
crypto_df["title_hash"] = crypto_df["title"].apply(title_hash)
macro_df["title_hash"] = macro_df["title"].apply(title_hash)

# Remover do macro o que já existe no crypto
existing_hashes = set(crypto_df["title_hash"])
macro_df = macro_df[~macro_df["title_hash"].isin(existing_hashes)]
```

### Fix 3 — Combined score com pesos explícitos e auditáveis
```python
COMBINE_WEIGHTS = {
    "macro_weight": 0.60,  # macro pesa mais (driver exógeno)
    "crypto_weight": 0.40,  # crypto direto
}

combined_score = (
    macro_score * COMBINE_WEIGHTS["macro_weight"] +
    crypto_score * COMBINE_WEIGHTS["crypto_weight"]
)

# Persistir no output para auditoria
metrics["combined"]["4h"] = {
    "crypto_score": crypto_score,
    "macro_score": macro_score,
    "combined_score": combined_score,
    "weights": COMBINE_WEIGHTS,  # sempre visível
    "dominant_driver": "macro" if abs(macro_score) > abs(crypto_score) else "crypto",
}
```

### Fix 4 — Prompt unificado, contexto parametrizado
NÃO criar prompt separado para macro. Usar o MESMO prompt do classify_news.py, apenas indicar o tipo:

```python
# No news_list, marcar tipo mas usar mesma escala:
news_list = "\n".join([
    f"[{i}] [MACRO] {n['title']}" if n["news_type"] == "macro"
    else f"[{i}] [CRYPTO] {n['title']}"
    for i, n in enumerate(news_items)
])

# No prompt, adicionar contexto para MACRO:
CLASSIFY_PROMPT = """...prompt existente...

ATENÇÃO para notícias marcadas [MACRO]:
Estas não mencionam BTC diretamente. Classifique pelo impacto INDIRETO:
- Petróleo sobe → inflação → juros altos → risk-off → BTC cai
- Guerra escala → risk-off → flight to safety → BTC cai (curto)
- Fed dovish → liquidez → risk-on → BTC sobe
- Recessão → pode ser bearish (risk-off) ou bullish (digital gold)

Mesma escala de score (-10 a +10) para [CRYPTO] e [MACRO].

NOTÍCIAS:
{news_list}
"""
```

### Fix 5 — Pipeline order: ingest → classify → aggregate
```
:55  Crypto news ingest    → btc_news.parquet (raw, sem ds_*)
:56  Macro news ingest     → macro_news.parquet (raw, sem ds_*)
:57  Classify ALL news     → atualiza ds_* em AMBOS parquets
                           → calcula macro_metrics.json (só com ds_score)
                           → calcula sentiment_metrics.json (atualiza com combined)
                           → NUNCA calcular metrics antes da classificação
```

O macro_news_ingest.py NÃO calcula métricas. Só faz ingestão. O classify_news.py faz classificação E agregação.

---

## Implementação

### Passo 1 — Criar conf/macro_keywords.json
```json
{
  "version": "1.0",
  "updated_at": "2026-03-31",
  "search_groups": {
    "energy_oil": {
      "queries": ["oil crisis", "oil price surge", "hormuz strait", "opec cut", "fuel shortage", "energy crisis"],
      "enabled": true
    },
    "fed_monetary": {
      "queries": ["federal reserve rate", "fed inflation", "fomc decision", "powell speech", "rate hike", "rate cut"],
      "enabled": true
    },
    "geopolitical": {
      "queries": ["iran war", "trump iran", "middle east conflict", "sanctions iran"],
      "enabled": true
    },
    "inflation": {
      "queries": ["inflation data", "cpi report", "consumer prices"],
      "enabled": true
    },
    "global_risk": {
      "queries": ["recession risk", "market crash", "bank failure", "debt crisis"],
      "enabled": true
    }
  },
  "max_queries_per_hour": 5,
  "max_results_per_query": 10,
  "combine_weights": {
    "macro_weight": 0.60,
    "crypto_weight": 0.40
  }
}
```

### Passo 2 — Criar scripts/cron/macro_news_ingest.py
Responsabilidade: APENAS ingestão. Sem métricas, sem classificação.

```python
"""
Macro News Ingest — busca notícias macro exógenas via CryptoCompare search.
APENAS ingestão — classificação e métricas em classify_news.py.

Output: data/01_raw/news/cryptocompare/macro_news.parquet
"""

def run_ingest():
    config = load_macro_keywords()
    
    all_posts = []
    for group_id, group in config["search_groups"].items():
        if not group.get("enabled", True):
            continue
        
        # Rotacionar query baseado na hora atual
        queries = group["queries"]
        hour = datetime.now(timezone.utc).hour
        query = queries[hour % len(queries)]
        
        articles = search_news(query, limit=config["max_results_per_query"])
        
        for a in articles:
            all_posts.append({
                "id": a["ID"],
                "title": a["TITLE"],
                "title_hash": hashlib.md5(a["TITLE"].lower().strip().encode()).hexdigest(),
                "body": a.get("BODY", "")[:500],
                "url": a.get("URL", ""),
                "published_at": pd.Timestamp(a["PUBLISHED_ON"], unit="s", tz="UTC"),
                "sentiment_raw": a.get("SENTIMENT", "NEUTRAL"),  # renomear para _raw
                "source": a.get("SOURCE_DATA", {}).get("NAME", ""),
                "search_group": group_id,
                "search_query": query,
                "news_type": "macro",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                # ds_* fields iniciam vazios — preenchidos pelo classify_news.py
                "ds_classified": False,
                "ds_topic": None,
                "ds_event_type": None,
                "ds_impact": None,
                "ds_score": None,
                "ds_reason": None,
            })
        
        logger.info("Group %s query '%s': %d articles", group_id, query, len(articles))
    
    if not all_posts:
        logger.info("No macro articles found")
        return
    
    new_df = pd.DataFrame(all_posts)
    
    # Merge com existente
    if MACRO_PARQUET.exists():
        existing = pd.read_parquet(MACRO_PARQUET)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    
    # Dedup por ID e title_hash
    combined = combined.drop_duplicates(subset=["id"], keep="last")
    combined = combined.drop_duplicates(subset=["title_hash"], keep="last")
    
    # Janela 90 dias
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)
    combined["published_at"] = pd.to_datetime(combined["published_at"], utc=True)
    combined = combined[combined["published_at"] >= cutoff]
    combined = combined.sort_values("published_at").reset_index(drop=True)
    
    combined.to_parquet(MACRO_PARQUET, index=False)
    logger.info("Macro parquet: %d rows (%d new)", len(combined), len(all_posts))
```

### Passo 3 — Atualizar classify_news.py

Mudanças:
1. Ler AMBOS parquets (crypto + macro)
2. Dedup global por title_hash antes de classificar
3. Batch unificado com tag [CRYPTO]/[MACRO]
4. Após classificar, salvar ds_* em AMBOS parquets
5. Calcular metrics usando APENAS ds_score
6. Combined score com pesos do config

```python
def run_classify():
    # 1. Ler ambos
    crypto_df = read_parquet(CRYPTO_PARQUET)
    macro_df = read_parquet(MACRO_PARQUET)
    
    # 2. Dedup global
    if crypto_df is not None and macro_df is not None:
        crypto_hashes = set(crypto_df["title_hash"])
        macro_df = macro_df[~macro_df["title_hash"].isin(crypto_hashes)]
    
    # 3. Filtrar não classificados
    to_classify = []
    if crypto_df is not None:
        crypto_new = crypto_df[crypto_df["ds_classified"] != True]
        for _, r in crypto_new.iterrows():
            to_classify.append({"title": r["title"], "news_type": "crypto", "id": r["id"]})
    
    if macro_df is not None:
        macro_new = macro_df[macro_df["ds_classified"] != True]
        for _, r in macro_new.iterrows():
            to_classify.append({"title": r["title"], "news_type": "macro", "id": r["id"]})
    
    if not to_classify:
        logger.info("Nothing to classify")
        return
    
    # 4. Classificar em batch (prompt unificado)
    results = classify_batch_deepseek(to_classify)
    
    # 5. Atualizar parquets com ds_*
    update_parquet(crypto_df, results, CRYPTO_PARQUET)
    update_parquet(macro_df, results, MACRO_PARQUET)
    
    # 6. Calcular metrics (APENAS ds_score)
    compute_all_metrics(crypto_df, macro_df, config)
```

### Passo 4 — compute_all_metrics usa APENAS ds_score

```python
def compute_all_metrics(crypto_df, macro_df, config):
    weights = config.get("combine_weights", {"macro_weight": 0.6, "crypto_weight": 0.4})
    
    for hours in [1, 4, 24]:
        # Crypto score (apenas classificados)
        crypto_classified = filter_classified(crypto_df, hours)
        crypto_score = crypto_classified["ds_score"].mean() if len(crypto_classified) > 0 else 0
        
        # Macro score (apenas classificados)
        macro_classified = filter_classified(macro_df, hours)
        macro_score = macro_classified["ds_score"].mean() if len(macro_classified) > 0 else 0
        
        # Combined
        combined = (
            macro_score * weights["macro_weight"] +
            crypto_score * weights["crypto_weight"]
        )
        
        metrics[f"{hours}h"] = {
            "crypto_score": crypto_score,
            "macro_score": macro_score,
            "combined_score": combined,
            "weights": weights,
            "dominant_driver": "macro" if abs(macro_score) > abs(crypto_score) else "crypto",
            # ... contagens, top stories etc
        }
```

### Passo 5 — Crontab
```
:50  Update 1h candles
:55  CryptoCompare crypto news
:56  Macro news search
:57  Classify ALL news (DeepSeek) + aggregate metrics
:00  Daily pipeline (07:00)
:02  Monitor 1h (stops + timing)
:05  Specialist 4h (decisão + intenção)
```

### Passo 6 — Dashboard (crypto-trading-dashboard)

Atualizar seção sentimento para mostrar:
- Crypto score e Macro score separados
- Combined score com pesos visíveis
- Feed com notícias de ambos os tipos (tag [CRYPTO]/[MACRO])
- Macro stress por grupo (energy, geopolitical, fed)

### Passo 7 — AI Analysis

Atualizar _gather_market_context() para incluir macro news:
```
MACRO NEWS (drivers exógenos):
  Macro stress 4h: -0.40
  Energy: score=-5 (3 articles)
  Geopolitical: score=-6 (4 articles)
  Fed: score=0 (2 articles)
  
  Top macro stories:
    🔴 -7 [MACRO] Asia barters scarce energy as Iran crisis throttles supplies
    🟢 +5 [MACRO] Trump signals willingness to end Iran conflict
    
COMBINED: crypto=-0.23 × 0.4 + macro=-0.40 × 0.6 = -0.33
```

## Validação
1. `python scripts/cron/macro_news_ingest.py` → macro_news.parquet criado
2. `python scripts/cron/classify_news.py` → classifica crypto + macro
3. sentiment_metrics.json tem crypto_score, macro_score, combined_score com pesos
4. Dedup: mesma notícia não aparece em ambos parquets
5. Metrics usa APENAS ds_score (grep -n "sentiment_raw" classify_news.py → zero uso em cálculos)
6. Dashboard mostra crypto + macro separados + combined
7. Notícias sobre oil/iran/fed aparecem no macro feed
