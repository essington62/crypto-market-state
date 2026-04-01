# TASK: Fix macro_news_ingest.py — Trocar CryptoCompare Search por Google News RSS

## Projeto: crypto-market-state
Arquivo: `scripts/cron/macro_news_ingest.py`

## Problema
O script usa `data-api.cryptocompare.com/news/v1/search` que retorna 400 Bad Request (feature paga). Google News RSS funciona perfeitamente e é gratuito (testamos e retorna 100 resultados por query com Reuters, Bloomberg, WSJ, CNBC, NYT).

## Mudança (cirúrgica)
Substituir APENAS a função de fetch. O resto (dedup, rotação, salvamento, config) NÃO MUDA.

### Substituir a função de fetch existente por:

```python
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

def _fetch_rss(query: str, max_results: int = 10) -> list[dict]:
    """Buscar notícias via Google News RSS."""
    url = f"{GOOGLE_NEWS_RSS}?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        root = ET.fromstring(resp.read())
    except Exception as e:
        logger.warning("RSS failed for '%s': %s", query, e)
        return []
    
    items = []
    for item in root.findall(".//item")[:max_results]:
        title_el = item.find("title")
        link_el = item.find("link")
        pub_el = item.find("pubDate")
        source_el = item.find("source")
        
        title = title_el.text if title_el is not None else ""
        link = link_el.text if link_el is not None else ""
        pub_date = pub_el.text if pub_el is not None else ""
        source = source_el.text if source_el is not None else ""
        
        # Parse date
        try:
            published_at = parsedate_to_datetime(pub_date)
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
        except Exception:
            published_at = datetime.now(timezone.utc)
        
        items.append({
            "title": title,
            "url": link,
            "source": source,
            "published_at": published_at,
            "title_hash": generate_title_hash(title),
        })
    
    return items
```

### Atualizar a chamada no loop principal
Onde o script chama a função de search antiga, trocar por `_fetch_rss(query, max_results)`.

Garantir que `import urllib.parse` está presente no topo do arquivo.

### NÃO MUDAR:
- Rotação de queries ✓
- Dedup (normalize_title + sha1) ✓
- Config (macro_keywords.json) ✓
- Salvamento em parquet ✓
- Logging ✓
- Tudo que não é a função de fetch ✓

## Validação
```bash
python scripts/cron/macro_news_ingest.py
```
Deve retornar artigos de Reuters, Bloomberg, WSJ sobre oil, iran, fed, inflation.
