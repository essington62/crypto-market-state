# TASK: CryptoCompare News — Ingestão + Dashboard

## Contexto
CryptoCompare API gratuita retorna notícias BTC com sentimento nativo (POSITIVE/NEGATIVE/NEUTRAL via GPT-3.5), votos, keywords, texto completo. Substitui CryptoPanic (removido).

Campos disponíveis no free tier:
```
ID, GUID, TITLE, SUBTITLE, BODY, URL, IMAGE_URL,
PUBLISHED_ON (unix timestamp), SENTIMENT (POSITIVE/NEGATIVE/NEUTRAL),
UPVOTES, DOWNVOTES, SCORE, KEYWORDS, LANG,
SOURCE_DATA, CATEGORY_DATA, AUTHORS
```

## Arquitetura
```
CryptoCompare API (1x/hora, gratuito, 50 calls/hora)
       ↓
Parquet L1 (event-level, acumula histórico 90d)
       ↓
Feature aggregation (sentiment_ratio, news_volume, stress_flags)
       ↓
Dashboard (feed + indicadores + timeline)
       ↓ futuro
News gate (control layer) + modelo de sentimento
```

## PARTE 1 — Ingestão (crypto-market-state)

Projeto: `/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/`

### Passo 1 — Verificar que CryptoPanic foi removido
```bash
grep -rn "cryptopanic\|CryptoPanic\|CRYPTOPANIC" scripts/ conf/ data/ 2>/dev/null
```
Deve retornar zero resultados. Se encontrar algo, remover.

### Passo 2 — Criar script de ingestão
Arquivo: `scripts/cron/cryptocompare_news_ingest.py`

```python
"""
CryptoCompare News Ingest — roda a cada hora via cron.
Busca notícias BTC com sentimento nativo, salva em parquet incremental.
1 request por execução (respeita 50 calls/hora do free tier).

Outputs:
    data/01_raw/news/cryptocompare/btc_news.parquet       — event-level history
    data/01_raw/news/cryptocompare/sentiment_metrics.json  — aggregated metrics
"""
```

API endpoint:
```
GET https://data-api.cryptocompare.com/news/v1/article/list
    ?lang=EN
    &categories=BTC
    &limit=50
```

NOTA: NÃO precisa de API key para o free tier. Se quiser mais requests, registrar em https://www.cryptocompare.com/cryptopian/api-keys e adicionar `&api_key=KEY`.

Fluxo de cada execução:
```
1. Fetch até 50 artigos mais recentes sobre BTC
2. Parse: extrair campos relevantes, converter timestamps
3. Carregar parquet existente (se houver)
4. Dedup por ID
5. Manter janela de 90 dias
6. Salvar parquet atualizado
7. Calcular métricas de sentimento (1h, 4h, 24h)
8. Salvar metrics JSON
9. Log resultado
```

Campos a salvar no parquet:
```python
{
    "id": r["ID"],
    "guid": r["GUID"],
    "title": r["TITLE"],
    "body": r.get("BODY", "")[:500],  # truncar body para economizar espaço
    "url": r["URL"],
    "published_at": pd.Timestamp(r["PUBLISHED_ON"], unit="s", tz="UTC"),
    "sentiment": r["SENTIMENT"],  # POSITIVE / NEGATIVE / NEUTRAL
    "upvotes": r.get("UPVOTES", 0),
    "downvotes": r.get("DOWNVOTES", 0),
    "score": r.get("SCORE", 0),
    "keywords": r.get("KEYWORDS", ""),
    "source": r.get("SOURCE_DATA", {}).get("NAME", ""),
    "categories": str(r.get("CATEGORY_DATA", [])),
    "ingested_at": datetime.now(timezone.utc).isoformat(),
}
```

Cálculo de métricas:
```python
def compute_metrics(df, hours):
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    recent = df[df["published_at"] >= cutoff]
    
    if len(recent) == 0:
        return {"news_count": 0, "positive_pct": 0, "negative_pct": 0, 
                "sentiment_score": 0, "importance": 0}
    
    n = len(recent)
    pos = (recent["sentiment"] == "POSITIVE").sum()
    neg = (recent["sentiment"] == "NEGATIVE").sum()
    
    # Sentiment score: -1 (all negative) to +1 (all positive)
    sentiment_score = (pos - neg) / n
    
    # Volume-weighted: score × (upvotes + 1)
    weighted = recent.apply(
        lambda r: (1 if r["sentiment"] == "POSITIVE" else 
                   -1 if r["sentiment"] == "NEGATIVE" else 0)
        * (1 + r.get("upvotes", 0)), axis=1
    )
    
    return {
        "news_count": int(n),
        "positive_pct": float(pos / n * 100),
        "negative_pct": float(neg / n * 100),
        "neutral_pct": float((n - pos - neg) / n * 100),
        "sentiment_score": float(sentiment_score),
        "weighted_sentiment": float(weighted.sum() / n),
        "total_score": float(recent["score"].sum()),
    }
```

News volume z-score (mesmo conceito do script anterior):
```python
def compute_news_zscore(df):
    series = df.set_index("published_at").resample("1h").size()
    if len(series) < 2:
        return 0.0
    mean = series.rolling(24, min_periods=2).mean()
    std = series.rolling(24, min_periods=2).std().replace(0, np.nan)
    z = (series - mean) / std
    return float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else 0.0
```

High stress flag:
```python
"high_stress": bool(zscore > 2.0 and m1h["sentiment_score"] < -0.3)
```

### Passo 3 — Paths
```python
PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "data" / "01_raw" / "news" / "cryptocompare"
PARQUET_PATH = DATA_DIR / "btc_news.parquet"
METRICS_PATH = DATA_DIR / "sentiment_metrics.json"
BASE_URL = "https://data-api.cryptocompare.com/news/v1/article/list"
```

### Passo 4 — Crontab
Adicionar ao crontab (substituindo o antigo CryptoPanic):
```bash
# ─────────────────────────────────────────────
# CryptoCompare News Ingest — BTC sentiment
# Roda a cada hora no minuto :55
# Antes do monitor (:02) para ter métricas frescas
# ─────────────────────────────────────────────
55 * * * * cd /Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state && /opt/homebrew/Caskroom/miniforge/base/envs/crypto_market_state/bin/python scripts/cron/cryptocompare_news_ingest.py >> /Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/logs/cryptocompare_news.log 2>&1
```

Crontab final deve ter 6 entries:
```
:55  CryptoCompare news
:00  Daily pipeline (07:00 only)
:02  Monitor 1h
:05  Specialist 4h (candle hours only)
:08  XGBoost 1h
:15  R11 daily (07:15 only)
```

---

## PARTE 2 — Dashboard (crypto-trading-dashboard)

Projeto: `/Users/brown/Documents/MLGeral/crypto_v2/crypto-trading-dashboard/`

### Passo 5 — Atualizar config/settings.py
```python
# CryptoCompare News
CRYPTOCOMPARE_DATA = DATA_LAKE / "data" / "01_raw" / "news" / "cryptocompare"
CRYPTOCOMPARE_NEWS = CRYPTOCOMPARE_DATA / "btc_news.parquet"
CRYPTOCOMPARE_METRICS = CRYPTOCOMPARE_DATA / "sentiment_metrics.json"
```

### Passo 6 — Atualizar data/readers.py
```python
def read_news_feed(hours: int = 24) -> pd.DataFrame | None:
    """Ler notícias BTC das últimas N horas do CryptoCompare."""

def read_news_metrics() -> dict | None:
    """Ler métricas de sentimento pré-calculadas."""
```

### Passo 7 — Criar components/sentiment.py
Nova seção do dashboard entre Market e Performance.

**7a. Sentiment Banner (topo da seção):**
Se high_stress = True:
```
🚨 HIGH MARKET STRESS — sentiment negativo + volume de notícias anormal
```
Se sentiment_score < -0.3:
```
⚠️ Sentimento bearish predominante (XX% negativo nas últimas 4h)
```
Se sentiment_score > 0.3:
```
🟢 Sentimento bullish predominante (XX% positivo nas últimas 4h)
```

**7b. Sentiment Overview (3 cards):**

| 1h | 4h | 24h |
|----|-----|------|
| 📰 X notícias | 📰 X notícias | 📰 X notícias |
| 🟢 X% positivo | 🟢 X% positivo | 🟢 X% positivo |
| 🔴 X% negativo | 🔴 X% negativo | 🔴 X% negativo |
| Score: +0.XX | Score: +0.XX | Score: +0.XX |

Cor do card: verde se score > 0.2, vermelho se score < -0.2, cinza se neutro.

**7c. News Feed (últimas 15 notícias):**
Tabela:
```
Timestamp (BRT) | Sentiment | Score | Título | Source
17:30           | 🔴 NEG   | 42    | Bitcoin Under Pressure After... | CoinDesk
17:25           | 🟢 POS   | 28    | BTC DeFi Milestone...          | Decrypt
17:10           | 🔴 NEG   | 67    | ETF Flows Plunge...            | Bloomberg
```

Colorir linha inteira: verde para POSITIVE, vermelho para NEGATIVE, cinza para NEUTRAL.
Score alto = mais relevante.

**7d. Sentiment Timeline (plotly):**
Gráfico de linha das últimas 48h:
- Eixo X: tempo (hora)
- Eixo Y: sentiment_score por hora (-1 a +1)
- Agregar: média de sentiment por hora
- Linha vermelha em -0.3 (bearish threshold)
- Linha verde em +0.3 (bullish threshold)
- Área sombreada: verde acima de 0, vermelha abaixo

**7e. News Volume (plotly bar chart):**
Barras horárias (últimas 48h):
- Altura = número de notícias
- Cor = sentiment médio da hora (verde/vermelho)
- Linha de referência: média de volume
- Spike visível quando muitas notícias aparecem de repente

**7f. Keywords Cloud (opcional, se simples):**
Top 10 keywords das últimas 24h com contagem:
```
ETF (15) | sell-off (12) | regulation (8) | mining (7) | Trump (6)
```

### Passo 8 — Atualizar app.py
```python
from components.sentiment import render_sentiment_section

# Layout:
render_health_section()
st.divider()
render_signals_section()
st.divider()
render_technical_section()
st.divider()
render_sentiment_section()    # ← NEWS + SENTIMENT
st.divider()
render_market_section()
st.divider()
render_performance_section()
```

---

## PARTE 3 — Keyword search futuro (preparar, não implementar)

O endpoint `/news/v1/search` permite buscar por keywords:
```
GET /news/v1/search?query=trump+iran&lang=EN&limit=10
```

Preparar stub em readers.py:
```python
def search_news(query: str, limit: int = 10) -> pd.DataFrame | None:
    """
    Busca notícias por keyword via CryptoCompare search.
    Útil para: FED, oil, tariff, war, SEC, hack, ETF.
    TODO: implementar quando news gate for ativado.
    """
    pass
```

## Restrições
- Free tier: 50 calls/hora, sem API key obrigatória
- 1 request por execução (1x/hora = 24 calls/dia, bem dentro do limite)
- Dedup por ID (campo ID do CryptoCompare é único)
- Body truncado a 500 chars no parquet (economizar espaço, texto completo no URL)
- Manter janela de 90 dias no parquet
- Se API falha → log warning, skip, tentar próxima hora
- Dashboard degrada gracefully se parquet não existe
- ZERO referência ao CryptoPanic em qualquer arquivo

## Validação
1. Rodar ingestão:
   ```bash
   cd /Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state
   python scripts/cron/cryptocompare_news_ingest.py
   ```
   → Parquet criado com ~50 rows

2. Verificar dados:
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_parquet('data/01_raw/news/cryptocompare/btc_news.parquet')
   print(f'Rows: {len(df)}')
   print(f'Sentiment: {df[\"sentiment\"].value_counts().to_dict()}')
   print(df[['title','sentiment','score']].head(5))
   "
   ```

3. Verificar métricas:
   ```bash
   cat data/01_raw/news/cryptocompare/sentiment_metrics.json
   ```

4. Dashboard com seção Sentiment:
   ```bash
   cd /Users/brown/Documents/MLGeral/crypto_v2/crypto-trading-dashboard
   streamlit run app.py
   ```
   → Banner de stress, cards, feed, timeline, volume

5. Crontab: 6 entries (news, daily, monitor, specialist, xgb, r11)

6. Zero referência a CryptoPanic:
   ```bash
   grep -rn "cryptopanic\|CryptoPanic" /Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/scripts/
   grep -rn "cryptopanic\|CryptoPanic" /Users/brown/Documents/MLGeral/crypto_v2/crypto-trading-dashboard/
   ```
   → Zero resultados
