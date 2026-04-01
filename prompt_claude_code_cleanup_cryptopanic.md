# TASK: Remover CryptoPanic + Preparar CryptoCompare

## Objetivo
Remover TODA referência ao CryptoPanic dos projetos. Preparar estrutura para CryptoCompare como fonte de notícias.

## PARTE 1 — Remover CryptoPanic

### Projeto crypto-market-state
```
/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/
```

1. Deletar script de ingestão:
   ```
   rm scripts/cron/cryptopanic_ingest.py
   ```

2. Deletar dados gerados (se existirem):
   ```
   rm -rf data/01_raw/news/cryptopanic/
   ```

3. Remover do secrets.yml — abrir `conf/local/secrets.yml` e remover a linha:
   ```
   cryptopanic: "e464878e40aa1d153c0b5cbbdae4fb35a792ff88"
   ```

4. Remover do crontab — editar `crontab -e` e remover a entrada do CryptoPanic:
   ```
   # Remover esta linha:
   55 * * * * cd ... cryptopanic_ingest.py ...
   ```

5. Remover do daily_update.sh — se foi adicionado, remover o bloco CryptoPanic.
   ```
   grep -n "cryptopanic\|CryptoPanic" scripts/cron/daily_update.sh
   ```
   Se encontrar, remover.

### Projeto crypto-trading-dashboard
```
/Users/brown/Documents/MLGeral/crypto_v2/crypto-trading-dashboard/
```

1. Remover referências em config/settings.py:
   ```
   grep -n "cryptopanic\|CRYPTOPANIC" config/settings.py
   ```
   Remover todas as linhas.

2. Remover readers em data/readers.py:
   ```
   grep -n "cryptopanic\|CRYPTOPANIC" data/readers.py
   ```
   Remover funções read_cryptopanic_news() e read_cryptopanic_metrics().

3. Remover componente do dashboard:
   ```
   grep -rn "cryptopanic\|CRYPTOPANIC\|CryptoPanic" components/
   ```
   Remover _render_cryptopanic_banner() e qualquer referência em market.py ou sentiment.py.

4. Remover import/chamada em app.py:
   ```
   grep -n "cryptopanic\|sentiment" app.py
   ```
   Remover se houver referência.

## PARTE 2 — Preparar estrutura para CryptoCompare

### Testar API gratuita
Primeiro, investigar a API do CryptoCompare:
```bash
python -c "
import urllib.request, json

# Testar endpoint de notícias
url = 'https://min-api.cryptocompare.com/data/v2/news/?lang=EN'
resp = urllib.request.urlopen(url, timeout=15)
data = json.loads(resp.read())
print('Top keys:', list(data.keys()))
print(json.dumps(data, indent=2, default=str)[:1500])
"
```

Depois testar com filtro BTC:
```bash
python -c "
import urllib.request, json

# Com categoria BTC
url = 'https://min-api.cryptocompare.com/data/v2/news/?categories=BTC&lang=EN'
resp = urllib.request.urlopen(url, timeout=15)
data = json.loads(resp.read())
print('Response:')
print(json.dumps(data, indent=2, default=str)[:1500])
"
```

### Verificar campos disponíveis
Precisamos saber se o free tier retorna:
- title ✓ (provável)
- source ✓ (provável)
- published_at ✓ (provável)
- sentiment / score (verificar)
- categories (verificar)
- url (verificar)

### Preparar paths (não criar scripts ainda)
Em config/settings.py do dashboard, preparar:
```python
# CryptoCompare (substituiu CryptoPanic)
CRYPTOCOMPARE_DATA = DATA_LAKE / "data" / "01_raw" / "news" / "cryptocompare"
CRYPTOCOMPARE_NEWS = CRYPTOCOMPARE_DATA / "btc_news.parquet"
CRYPTOCOMPARE_METRICS = CRYPTOCOMPARE_DATA / "sentiment_metrics.json"
```

## Validação
1. `grep -rn "cryptopanic\|CryptoPanic\|CRYPTOPANIC" .` em ambos os projetos → zero resultados
2. `crontab -l` → sem entrada do CryptoPanic (deve ter 5 entries, não 6)
3. `cat conf/local/secrets.yml` → sem key do CryptoPanic
4. Dashboard abre sem erros (seções que dependiam do CryptoPanic omitidas gracefully)
5. API CryptoCompare testada com output documentado
