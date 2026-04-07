# TASK: Treinar XGBoost Regime-Aware (Sideways vs Trend)

## Projeto: xgboost_R5c_1h
Path: `/Users/brown/Documents/MLGeral/xgboost_R5c_1h/`
Data lake: `/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/`

## Contexto
O XGBoost atual tem target muito restritivo (ret_4h >= 1% AND max_dd >= -2%), treinado sem distinção de regime. Em Sideways (regime atual), o modelo não ativa (alloc_raw ~0.2, zero trades). A solução: dois modelos especializados.

## Passo 0 — Inspecionar dados disponíveis
```bash
ls data/03_models/
ls data/02_features/
```
Verificar features existentes e estrutura.

---

## PARTE 1 — Preparar Datasets por Regime

### Passo 1 — Classificar candles 1h com R5C regime diário

O R5C classifica em daily. Cada candle 1h herda o regime do dia anterior (day-shift para evitar lookahead):

```python
import pickle, pandas as pd, numpy as np

# Carregar R5C
with open('data/03_models/r5c_hmm.pkl', 'rb') as f:
    hmm = pickle.load(f)

means = hmm.means_[:, 0]
bull_st, bear_st = int(means.argmax()), int(means.argmin())
side_st = [i for i in range(3) if i != bull_st and i != bear_st][0]
state_names = {bull_st: "Bull", bear_st: "Bear", side_st: "Sideways"}

# Carregar daily e computar features R5C
daily = pd.read_parquet(DATA_LAKE / "data/02_intermediate/spot/daily/BTCUSDT.parquet")
daily.index = pd.to_datetime(daily.index, utc=True)
close_d = daily["close"]

feat = pd.DataFrame(index=daily.index)
feat["log_return"] = np.log(close_d / close_d.shift(1))
feat["vol_short"] = feat["log_return"].rolling(7).std()
feat["vol_ratio"] = feat["vol_short"] / feat["log_return"].rolling(30).std()
feat["drawdown"] = close_d / close_d.rolling(63).max() - 1
feat["volume_z"] = (daily["volume"] - daily["volume"].rolling(30).mean()) / daily["volume"].rolling(30).std()
feat["slope_21d"] = close_d.pct_change(21)
feat = feat.dropna()

X_daily = feat[["log_return", "vol_short", "vol_ratio", "drawdown", "volume_z", "slope_21d"]].values
states = hmm.predict(X_daily)
feat["r5c_regime"] = [state_names[s] for s in states]

# Day-shift: regime do dia D aplica-se aos candles do dia D+1
regime_daily = feat[["r5c_regime"]].shift(1).dropna()
regime_daily.index = regime_daily.index.normalize()
```

### Passo 2 — Merge regime com candles 1h

```python
# Carregar candles 1h
candles_1h = pd.read_parquet(DATA_LAKE / "data/01_raw/spot/crypto/1h/BTCUSDT_1h.parquet")
candles_1h.index = pd.to_datetime(candles_1h.index, utc=True)
candles_1h = candles_1h.sort_index()

# Adicionar coluna de data (para join)
candles_1h["date"] = candles_1h.index.normalize()

# Merge com regime (day-shifted)
candles_1h = candles_1h.join(regime_daily, on="date", how="left")
candles_1h = candles_1h.dropna(subset=["r5c_regime"])
```

### Passo 3 — Computar features 1h (Group A)

```python
close = candles_1h["close"]
high = candles_1h["high"]
low = candles_1h["low"]
volume = candles_1h["volume"]

features = pd.DataFrame(index=candles_1h.index)
features["returns_1h"] = close.pct_change(1)
features["returns_4h"] = close.pct_change(4)
features["returns_12h"] = close.pct_change(12)
features["volatility_6h"] = features["returns_1h"].rolling(6).std()
features["volatility_24h"] = features["returns_1h"].rolling(24).std()
features["volume_zscore"] = (volume - volume.rolling(24).mean()) / volume.rolling(24).std()
features["buy_pressure"] = (close - low) / (high - low + 1e-10)
features["price_range_1h"] = (high - low) / close
features["rsi_14"] = _compute_rsi(close, 14)
features["bb_pct_b"] = _compute_bb_pctb(close, 20)

# Feature NOVA para Sideways: posição no range
features["range_position"] = (close - close.rolling(720).min()) / (close.rolling(720).max() - close.rolling(720).min() + 1e-10)
# 720h = 30 dias → onde está no range dos últimos 30 dias

# Merge com regime
features["r5c_regime"] = candles_1h["r5c_regime"]
features = features.dropna()
```

### Passo 4 — Computar targets

```python
# Target TREND (original): +1% em 4h sem cair 2%
features["ret_fwd_4h"] = close.shift(-4) / close - 1
features["max_dd_fwd_4h"] = close.shift(-4).rolling(4).min() / close - 1  # simplificado
# Calcular max_dd corretamente:
for i in range(len(features)):
    if i + 4 < len(close):
        window = close.iloc[i+1:i+5]
        features.iloc[i, features.columns.get_loc("max_dd_fwd_4h")] = (window.min() / close.iloc[i]) - 1

features["target_trend"] = (
    (features["ret_fwd_4h"] >= 0.01) & 
    (features["max_dd_fwd_4h"] >= -0.02)
).astype(int)

# Target SIDEWAYS (novo): +0.5% em 12h sem cair 1%
features["ret_fwd_12h"] = close.shift(-12) / close - 1
# Max DD em 12h
for i in range(len(features)):
    if i + 12 < len(close):
        window = close.iloc[i+1:i+13]
        features.iloc[i, features.columns.get_loc("max_dd_fwd_12h")] = (window.min() / close.iloc[i]) - 1
    else:
        features.iloc[i, features.columns.get_loc("max_dd_fwd_12h")] = np.nan

features["target_sideways"] = (
    (features["ret_fwd_12h"] >= 0.005) & 
    (features["max_dd_fwd_12h"] >= -0.01)
).astype(int)

# Alternativa: regressão (ret_12h direto)
features["target_sideways_reg"] = features["ret_fwd_12h"]
```

### Passo 5 — Verificar pos_rate

```python
df_side = features[features["r5c_regime"] == "Sideways"]
df_trend = features[features["r5c_regime"].isin(["Bull", "Bear"])]

print(f"Sideways: {len(df_side)} candles, pos_rate target_sideways: {df_side['target_sideways'].mean()*100:.1f}%")
print(f"Trend:    {len(df_trend)} candles, pos_rate target_trend:   {df_trend['target_trend'].mean()*100:.1f}%")
```

**CRITÉRIO:** pos_rate sideways deve estar entre 25-40%. Se < 20%, afrouxar target. Se > 50%, apertar.

---

## PARTE 2 — Treinar Modelos

### Passo 6 — Split temporal (walk-forward)

```python
# Split: treino até 2025-12-31, teste 2026-01-01+
train_cutoff = pd.Timestamp("2025-12-31", tz="UTC")

# SIDEWAYS
side_train = df_side[df_side.index <= train_cutoff]
side_test = df_side[df_side.index > train_cutoff]

# TREND
trend_train = df_trend[df_trend.index <= train_cutoff]
trend_test = df_trend[df_trend.index > train_cutoff]

print(f"Sideways train: {len(side_train)}, test: {len(side_test)}")
print(f"Trend train:    {len(trend_train)}, test: {len(trend_test)}")
```

### Passo 7 — Treinar XGBoost Sideways

```python
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

FEATURE_COLS = [
    "returns_1h", "returns_4h", "returns_12h",
    "volatility_6h", "volatility_24h",
    "volume_zscore", "buy_pressure", "price_range_1h",
    "rsi_14", "bb_pct_b",
    "range_position",  # NOVA feature para Sideways
]

# Sideways model
X_train_s = side_train[FEATURE_COLS]
y_train_s = side_train["target_sideways"]
X_test_s = side_test[FEATURE_COLS]
y_test_s = side_test["target_sideways"]

model_sideways = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric="auc",
)

model_sideways.fit(
    X_train_s, y_train_s,
    eval_set=[(X_test_s, y_test_s)],
    verbose=False,
)

# Avaliar
y_pred_s = model_sideways.predict_proba(X_test_s)[:, 1]
auc_s = roc_auc_score(y_test_s, y_pred_s)
print(f"\nSideways model — AUC: {auc_s:.3f}")
print(f"Score distribution: mean={y_pred_s.mean():.3f} std={y_pred_s.std():.3f} min={y_pred_s.min():.3f} max={y_pred_s.max():.3f}")

# Threshold dinâmico
threshold_s = np.quantile(y_pred_s, 0.7)
print(f"Threshold Q70: {threshold_s:.3f}")

# Backtest simples
entries_s = y_pred_s >= threshold_s
if entries_s.sum() > 0:
    ret_entries = side_test.loc[entries_s, "ret_fwd_12h"]
    print(f"Trades: {entries_s.sum()}")
    print(f"Ret 12h médio: {ret_entries.mean()*100:+.3f}%")
    print(f"Win rate: {(ret_entries > 0).mean()*100:.0f}%")
```

### Passo 8 — Treinar XGBoost Trend (manter configuração original)

```python
X_train_t = trend_train[FEATURE_COLS[:10]]  # sem range_position
y_train_t = trend_train["target_trend"]
X_test_t = trend_test[FEATURE_COLS[:10]]
y_test_t = trend_test["target_trend"]

model_trend = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric="auc",
)

model_trend.fit(
    X_train_t, y_train_t,
    eval_set=[(X_test_t, y_test_t)],
    verbose=False,
)

y_pred_t = model_trend.predict_proba(X_test_t)[:, 1]
auc_t = roc_auc_score(y_test_t, y_pred_t) if y_test_t.sum() > 0 else 0
print(f"\nTrend model — AUC: {auc_t:.3f}")
print(f"Score distribution: mean={y_pred_t.mean():.3f} std={y_pred_t.std():.3f}")
```

### Passo 9 — Feature importance

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for name, model in [("Sideways", model_sideways), ("Trend", model_trend)]:
    imp = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    imp = imp.sort_values(ascending=True)
    print(f"\n{name} — Feature Importance:")
    for feat, val in imp.items():
        print(f"  {feat:20s}: {val:.3f}")
```

### Passo 10 — Salvar modelos

```python
import pickle

with open("data/03_models/xgb_sideways.pkl", "wb") as f:
    pickle.dump({
        "model": model_sideways,
        "features": FEATURE_COLS,
        "target": "target_sideways",
        "threshold_q70": float(threshold_s),
        "auc": float(auc_s),
        "train_end": str(train_cutoff),
        "regime": "Sideways",
    }, f)

with open("data/03_models/xgb_trend.pkl", "wb") as f:
    pickle.dump({
        "model": model_trend,
        "features": FEATURE_COLS[:10],
        "target": "target_trend",
        "threshold_q70": float(np.quantile(y_pred_t, 0.7)),
        "auc": float(auc_t),
        "train_end": str(train_cutoff),
        "regime": "Trend",
    }, f)

print("\nModelos salvos em data/03_models/")
```

---

## PARTE 3 — Atualizar Paper Trader

### Passo 11 — Atualizar xgb_1h_trader.py

```python
# No início do script:
def _load_model(regime: str):
    """Carregar modelo baseado no regime."""
    if regime == "Sideways":
        path = MODEL_DIR / "xgb_sideways.pkl"
    else:
        path = MODEL_DIR / "xgb_trend.pkl"
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["features"], data["threshold_q70"]

# Na inferência:
regime = get_r5c_regime()  # ler do data lake
model, feature_cols, threshold = _load_model(regime)

X = compute_features(candles_1h)[feature_cols]
score = model.predict_proba(X)[:, 1][-1]

# Threshold dinâmico
if score >= threshold:
    enter()
```

### Passo 12 — Config

```json
{
  "regime_model_selection": true,
  "r5c_model_path": "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/05_models/r5c_hmm.pkl",
  "models": {
    "sideways": "data/03_models/xgb_sideways.pkl",
    "trend": "data/03_models/xgb_trend.pkl"
  },
  "control_layer": {
    "stop_loss": { "enabled": true, "pct": 0.02 },
    "stop_gain": { "enabled": true, "pct": 0.015 },
    "trailing_stop": { "enabled": false },
    "timing_gate": { "enabled": false },
    "news_gate": { "enabled": false }
  }
}
```

---

## PARTE 4 — Comparação

### Passo 13 — Backtest comparativo

Rodar backtest no período de teste (2026-01-01+) comparando:

| Modelo | Target | Regime | Resultado |
|--------|--------|--------|-----------|
| XGB original | ret_4h>=1% | Todos | Baseline |
| XGB Sideways | ret_12h>=0.5% | Sideways | NOVO |
| XGB Trend | ret_4h>=1% | Bull+Bear | NOVO |
| Regra C7 | alloc 0.50-0.54 + RSI<50 | Sideways | Benchmark análise cruzada |

Métricas: Sharpe, ret médio (4h/12h), win rate, nº trades, max DD.

---

## Validação

1. Sideways pos_rate entre 25-40%
2. Score distribution mais ampla (não travada em <0.3)
3. AUC > 0.55 em ambos os modelos
4. Sideways gera trades em regime lateral (não zero trades)
5. Trend mantém performance similar ao original
6. Feature importance: range_position relevante no Sideways
7. Backtest Sideways supera buy & hold no período lateral

## CONSTRAINTS
- Sem paths hardcoded (usar variáveis PROJECT, DATA_LAKE)
- UTC only
- Walk-forward split (treino antes de 2026, teste 2026+)
- Day-shift no regime (evitar lookahead)
- Modelos salvos com metadata (features, threshold, AUC, train_end)
