# TASK: Evolução do Pipeline — R5C + Classificador Unificado Bull/Sideways/Bear

## Projetos
- crypto-market-state: `/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/`
- Dashboard: `/Users/brown/Documents/MLGeral/crypto-trading-dashboard/`

## Objetivo
3 mudanças integradas:
1. Classificador de notícias simplificado: Bull / Sideways / Bear
2. R5C (3 estados) substitui R11 (2 estados) como Layer 1
3. Retreinar specialist com features do R5C

---

## PARTE 1 — Classificador de Notícias: Bull / Sideways / Bear

### Passo 1 — Simplificar classify_news.py

Remover os 9 event_types complexos. Substituir por 3 regimes alinhados com o R5C:

**Novo prompt DeepSeek:**

```python
CLASSIFY_PROMPT = """Você é um analista sênior de trading de Bitcoin.

Classifique cada notícia pelo IMPACTO REAL no preço do Bitcoin usando 3 categorias:

BULL (score +3 a +10): Notícia claramente positiva para o preço do BTC.
  Exemplos: ceasefire/paz, rate cut, ETF aprovado, compras institucionais, adoção.

SIDEWAYS (score -2 a +2): Notícia mista, incerta, contraditória ou já precificada.
  Exemplos: "Trump mixes threats and talks", profit taking após rally,
  incerteza sem direção clara, análise/opinião sem novo fato.
  REGRA: Se a notícia contém TANTO sinais positivos QUANTO negativos → SIDEWAYS.
  REGRA: Se o BTC está SUBINDO apesar de notícia negativa → mercado precificou → SIDEWAYS.
  REGRA: Profit taking após rally → SIDEWAYS +1 (saudável, não bearish).

BEAR (score -3 a -10): Notícia claramente negativa para o preço do BTC.
  Exemplos: guerra escala sem negociação, rate hike, hack, ban, sell-off forçado.

IMPORTANTE:
- Classifique pelo RESULTADO FINAL no preço, não pelas palavras isoladas.
- "War" + "ceasefire" na mesma notícia = SIDEWAYS (contraditório).
- Preço subindo + notícia "uncertain" = SIDEWAYS (mercado resiliente).
- Impacto HIGH se muda narrativa macro (guerra/paz, Fed, regulação nacional).
- Impacto MEDIUM se evento significativo mas não muda narrativa.
- Impacto LOW se opinião, análise, ou genérico.

Responda APENAS em JSON válido, sem markdown:
[
  {{
    "index": 0,
    "regime": "BULL|SIDEWAYS|BEAR",
    "impact": "HIGH|MEDIUM|LOW",
    "score": -10 a +10,
    "reason": "máximo 8 palavras"
  }}
]

NOTÍCIAS:
{news_list}
"""
```

### Passo 2 — Atualizar colunas do parquet

Renomear campos:
```
ds_event_type → ds_regime (BULL / SIDEWAYS / BEAR)
ds_direction  → REMOVER (redundante, derivado do score)
ds_topic      → manter (informativo)
```

Derivação automática:
```python
if score >= 3:
    ds_regime = "BULL"
elif score <= -3:
    ds_regime = "BEAR"
else:
    ds_regime = "SIDEWAYS"
```

### Passo 3 — Atualizar agregação de métricas

```python
def compute_news_regime(df, hours):
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    recent = df[(df["published_at"] >= cutoff) & (df["ds_classified"] == True)]
    
    if len(recent) == 0:
        return {"regime": "SIDEWAYS", "score": 0, "confidence": 0}
    
    # Source-weighted score
    recent["weighted"] = recent["ds_score"] * recent["source_weight"]
    avg_score = recent["weighted"].mean()
    
    bull_count = (recent["ds_regime"] == "BULL").sum()
    bear_count = (recent["ds_regime"] == "BEAR").sum()
    sideways_count = (recent["ds_regime"] == "SIDEWAYS").sum()
    total = len(recent)
    
    # Regime por maioria ponderada
    if avg_score >= 2:
        regime = "BULL"
    elif avg_score <= -2:
        regime = "BEAR"
    else:
        regime = "SIDEWAYS"
    
    return {
        "regime": regime,
        "score": float(avg_score),
        "bull_pct": float(bull_count / total * 100),
        "sideways_pct": float(sideways_count / total * 100),
        "bear_pct": float(bear_count / total * 100),
        "news_count": int(total),
        "confidence": float(abs(avg_score) / 10),  # 0 a 1
    }
```

### Passo 4 — Atualizar sentiment_metrics.json

```json
{
  "crypto_news": {
    "4h": { "regime": "BULL", "score": 3.2, "bull_pct": 55, "sideways_pct": 30, "bear_pct": 15, "confidence": 0.32 }
  },
  "macro_news": {
    "4h": { "regime": "BEAR", "score": -3.9, "bull_pct": 10, "sideways_pct": 20, "bear_pct": 70, "confidence": 0.39 }
  },
  "combined_news": {
    "4h": { "regime": "SIDEWAYS", "score": -0.8, "confidence": 0.08 }
  }
}
```

---

## PARTE 2 — R5C como Layer 1

### Passo 5 — Copiar modelo R5C para crypto-market-state

O R5C está em: `/Users/brown/Documents/MLGeral/xgboost_R5c_1h/data/03_models/r5c_hmm.pkl`

Copiar para o data lake:
```bash
cp /Users/brown/Documents/MLGeral/xgboost_R5c_1h/data/03_models/r5c_hmm.pkl \
   /Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/05_models/r5c_hmm.pkl
```

### Passo 6 — Criar R5C signal generator

Arquivo: `scripts/paper_trading/r5c_signal_generator.py`

Similar ao `r11_signal_generator.py` existente mas com 3 estados:

```python
class R5CSignalGenerator:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.hmm = pickle.load(f)
        
        # Identificar estados pelos means
        means = self.hmm.means_[:, 0]  # log_return médio
        self.bull_state = int(means.argmax())
        self.bear_state = int(means.argmin())
        self.sideways_state = [i for i in range(3) if i != self.bull_state and i != self.bear_state][0]
        
        self.state_names = {
            self.bull_state: "Bull",
            self.bear_state: "Bear",
            self.sideways_state: "Sideways"
        }
    
    def predict(self, daily_df, proba_window=10):
        """Retorna regime e probabilidades dos últimos proba_window dias."""
        features = self._compute_features(daily_df)
        recent = features.iloc[-proba_window:]
        X = recent[['log_return', 'vol_short', 'vol_ratio', 'drawdown', 'volume_z', 'slope_21d']].values
        
        probs = self.hmm.predict_proba(X)
        states = self.hmm.predict(X)
        
        last_state = states[-1]
        last_probs = probs[-1]
        
        return {
            "regime": self.state_names[last_state],
            "prob_bull": float(last_probs[self.bull_state]),
            "prob_bear": float(last_probs[self.bear_state]),
            "prob_sideways": float(last_probs[self.sideways_state]),
            "entropy": float(-sum(p * np.log(p + 1e-10) for p in last_probs)),
            "regime_age": self._compute_regime_age(states),
        }
    
    def _compute_features(self, df):
        close = df['close']
        features = pd.DataFrame(index=df.index)
        features['log_return'] = np.log(close / close.shift(1))
        features['vol_short'] = features['log_return'].rolling(7).std()
        features['vol_ratio'] = features['vol_short'] / features['log_return'].rolling(30).std()
        features['drawdown'] = close / close.rolling(63).max() - 1
        features['volume_z'] = (df['volume'] - df['volume'].rolling(30).mean()) / df['volume'].rolling(30).std()
        features['slope_21d'] = close.pct_change(21)
        return features.dropna()
    
    def _compute_regime_age(self, states):
        """Dias consecutivos no estado atual."""
        current = states[-1]
        age = 1
        for s in reversed(states[:-1]):
            if s == current:
                age += 1
            else:
                break
        return age
```

### Passo 7 — Atualizar specialist_4h_paper_trader.py

Substituir R11 por R5C:

**7a. Trocar signal generator:**
```python
# ANTES:
from r11_signal_generator import SignalGenerator
gen = SignalGenerator(model_path=r11_model_path)

# DEPOIS:
from r5c_signal_generator import R5CSignalGenerator
gen = R5CSignalGenerator(model_path=r5c_model_path)
```

**7b. Atualizar features do Group B:**
```python
# ANTES (R11 — 3 features):
group_b = {
    "r11_prob_bull": signal["prob_bull"],
    "r11_entropy": signal["entropy"],
    "regime_age_log": np.log1p(signal["regime_age"]),
}

# DEPOIS (R5C — 5 features):
group_b = {
    "r5c_prob_bull": signal["prob_bull"],
    "r5c_prob_bear": signal["prob_bear"],
    "r5c_prob_sideways": signal["prob_sideways"],
    "r5c_entropy": signal["entropy"],
    "regime_age_log": np.log1p(signal["regime_age"]),
}
```

**7c. Atualizar regime gate (3 estados):**
```python
def _apply_regime_gate(allocation, context, config):
    regime = context["r5c_regime"]
    prob_bull = context["r5c_prob_bull"]
    prob_bear = context["r5c_prob_bear"]
    
    if regime == "Bear" or prob_bear > 0.60:
        # Bear: bloqueia 100%
        return 0.0, True, "bear"
    
    elif regime == "Sideways":
        # Sideways: permite com exposição reduzida (50%)
        sideways_factor = config.get("sideways_allocation_factor", 0.5)
        return allocation * sideways_factor, False, "sideways_reduced"
    
    elif regime == "Bull":
        # Bull: exposição total
        return allocation, False, "bull"
    
    return allocation, False, "unknown"
```

**7d. Atualizar config.json:**
```json
{
  "r5c_model_path": "data/05_models/r5c_hmm.pkl",
  "control_layer": {
    "regime_gate": {
      "enabled": true,
      "min_prob_bull": 0.30,
      "max_prob_bear": 0.60,
      "sideways_allocation_factor": 0.5,
      "max_entropy": 0.85
    }
  }
}
```

**7e. Atualizar portfolio.json campos:**
```json
{
  "r5c_regime": "Sideways",
  "r5c_prob_bull": 0.000,
  "r5c_prob_bear": 0.000,
  "r5c_prob_sideways": 1.000
}
```

---

## PARTE 3 — Retreinar Specialist com R5C Features

### Passo 8 — IMPORTANTE: O modelo precisa ser retreinado

O specialist atual (SET_B_split_4_recent.pkl) foi treinado com 3 features R11:
```
r11_prob_bull, r11_entropy, regime_age_log
```

Agora queremos 5 features R5C:
```
r5c_prob_bull, r5c_prob_bear, r5c_prob_sideways, r5c_entropy, regime_age_log
```

Opções:
- **Opção A (rápida, sem retreino):** Usar as 3 features que o modelo já conhece mas alimentar do R5C:
  ```
  r11_prob_bull → r5c_prob_bull (similar, ambos medem "bullishness")
  r11_entropy → r5c_entropy (fórmula diferente com 3 estados mas conceito similar)
  regime_age_log → regime_age_log (mesmo)
  ```
  Risco: features não são idênticas, modelo pode se comportar diferente.

- **Opção B (correta, com retreino):** Retreinar specialist com as 5 features do R5C.
  Precisa: gerar R5C features para todo o período de treino, refazer o walk-forward.

**RECOMENDAÇÃO: Opção A primeiro (testar), depois Opção B (retreinar).**

Para Opção A, mapear:
```python
# No specialist, ao computar Group B:
group_b = {
    "r11_prob_bull": signal["prob_bull"],     # r5c_prob_bull mapeado para r11_prob_bull
    "r11_entropy": signal["entropy"],          # r5c_entropy mapeado para r11_entropy
    "regime_age_log": np.log1p(signal["regime_age"]),
}
# O modelo recebe os mesmos nomes de features que foi treinado
# Mas os valores vêm do R5C em vez do R11
```

Isso permite testar R5C sem retreinar. Se funcionar, fazemos o retreino completo depois.

---

## PARTE 4 — News Gate Unificado

### Passo 9 — Atualizar news gate

```python
def _apply_news_gate(allocation, context, config):
    metrics = read_sentiment_metrics()
    
    combined = metrics.get("combined_news", {}).get("4h", {})
    news_regime = combined.get("regime", "SIDEWAYS")
    news_score = combined.get("score", 0)
    
    if news_regime == "BEAR" and news_score < -3:
        # News Bear forte: bloqueia
        return 0.0, True, "news_bear"
    
    elif news_regime == "BEAR":
        # News Bear moderado: reduz
        return allocation * 0.5, False, "news_bear_reduced"
    
    elif news_regime == "SIDEWAYS":
        # News misto: sem mudança
        return allocation, False, "news_sideways"
    
    elif news_regime == "BULL" and news_score > 3:
        # News Bull forte: boost
        return min(allocation * 1.15, 1.0), False, "news_bull_boost"
    
    return allocation, False, "news_neutral"
```

---

## PARTE 5 — Dashboard

### Passo 10 — Atualizar card Pipeline Integrado

```
PIPELINE INTEGRADO — HMM → SPECIALIST → TIMING
Layer 1 (R5C): 🟡 Sideways  bull=0.00 bear=0.00 side=1.00
Layer 2 (Specialist): Alloc raw: 0.54 — sideways gate reduzido 50%
Layer 3 (Timing): aguardando intenção
News: 🟢 BULL score=+3.2 (4h)
```

### Passo 11 — Atualizar sentimento no dashboard

Mostrar regime das notícias em vez de score numérico:
```
Crypto: 🟢 BULL (55% bull, 30% side, 15% bear)
Macro:  🔴 BEAR (10% bull, 20% side, 70% bear)
Combined: 🟡 SIDEWAYS (score -0.8)
```

### Passo 12 — Atualizar AI Analysis
Contexto para DeepSeek agora usa linguagem unificada:
```
REGIME R5C: Sideways (prob: bull=0.00, bear=0.00, sideways=1.00)
NEWS REGIME: Crypto=BULL, Macro=BEAR, Combined=SIDEWAYS
DECISÃO: Sideways regime → allocation reduzida 50%
```

---

## Validação

1. Classificador: "Trump mixes threats and talks" → SIDEWAYS ✅
2. Classificador: "Ceasefire hopes" → BULL +7 ✅
3. Classificador: "Profit taking below $70K" → SIDEWAYS +1 ✅
4. R5C: retorna Sideways para os últimos 10 dias ✅
5. Regime gate: Sideways → allocation × 0.5 (não bloqueia 100%) ✅
6. Specialist roda sem erro com features R5C mapeadas ✅
7. News gate: BULL → boost, SIDEWAYS → neutro, BEAR → bloqueia ✅
8. Dashboard mostra 3 regimes (Bull/Sideways/Bear) em todas as seções ✅
9. Pipeline gera intenção em regime Sideways (antes bloqueava) ✅

## ATENÇÃO
- Manter shared/execution.py intacto
- Manter timing layer intacto
- Manter stop loss/gain/trailing intactos
- Manter macro_news_ingest.py intacto
- Manter crontab intacto
- Salvar modelo R11 como backup (não deletar)
- Se algo falhar com R5C, poder reverter para R11 rapidamente
