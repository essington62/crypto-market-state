# TASK: Reativar XGBoost 1h Paper Trading (baseline sem timing/news gates)

## Projetos
- XGBoost: `/Users/brown/Documents/MLGeral/xgboost_R5c_1h/`
- Dashboard: `/Users/brown/Documents/MLGeral/crypto-trading-dashboard/`
- Crontab: sistema

## Objetivo
Reativar o XGBoost 1h como paper trader independente para comparação com o pipeline integrado. Sem timing gate e news gate — modelo puro + stops básicos.

---

## PARTE 1 — Reativar XGBoost 1h

### Passo 1 — Atualizar config.json
Arquivo: `/Users/brown/Documents/MLGeral/xgboost_R5c_1h/scripts/paper_trading/state/config.json`

Desativar trailing, timing gate e news gate. Manter apenas stop loss e stop gain:

```json
{
  "control_layer": {
    "entry_threshold": 0.50,
    "stop_loss": { "enabled": true, "pct": 0.02, "cooldown_candles": 1 },
    "stop_gain": { "enabled": true, "pct": 0.015 },
    "trailing_stop": { "enabled": false },
    "timing_gate": { "enabled": false },
    "news_gate": { "enabled": false }
  }
}
```

IMPORTANTE: Só mudar os `enabled` dos gates. Não mexer em nenhum outro campo do config.

### Passo 2 — Resetar portfolio para fresh start
Arquivo: `/Users/brown/Documents/MLGeral/xgboost_R5c_1h/scripts/paper_trading/state/portfolio.json`

```json
{
  "usdt_free": 100000.0,
  "btc_held": 0.0,
  "total_usdt": 100000.0,
  "btc_price": null,
  "position_pct": 0.0,
  "last_update": null,
  "entry_price": null,
  "entry_time": null,
  "max_price_since_entry": null,
  "stop_loss_cooldown_until": null,
  "last_monitor_price": null,
  "initial_capital": 100000.0
}
```

### Passo 3 — Limpar signals.csv (manter header, apagar dados antigos)
Ler o header atual do signals.csv e recriar com header only (backup do antigo):

```bash
cd /Users/brown/Documents/MLGeral/xgboost_R5c_1h/scripts/paper_trading/state
cp signals.csv signals_backup_20260401.csv
head -1 signals.csv > signals_new.csv
mv signals_new.csv signals.csv
```

### Passo 4 — Remover comentário de desativação
No arquivo `scripts/paper_trading/xgb_1h_trader.py`, remover ou comentar o bloco de desativação que foi adicionado em 2026-03-31. O script deve voltar a funcionar normalmente.

### Passo 5 — Testar manualmente
```bash
cd /Users/brown/Documents/MLGeral/xgboost_R5c_1h
conda activate crypto_xgb_1h
python scripts/paper_trading/xgb_1h_trader.py
cat scripts/paper_trading/state/portfolio.json
tail -1 scripts/paper_trading/state/signals.csv
```

### Passo 6 — Adicionar ao crontab
```bash
crontab -e
```
Adicionar:
```
8 * * * * cd /Users/brown/Documents/MLGeral/xgboost_R5c_1h && /opt/homebrew/Caskroom/miniforge/base/envs/crypto_xgb_1h/bin/python scripts/paper_trading/xgb_1h_trader.py >> /Users/brown/Documents/MLGeral/xgboost_R5c_1h/logs/xgb_1h_paper.log 2>&1
```

Crontab final (7 entries):
```
:50  Update 1h candles
:55  CryptoCompare crypto news
:56  Macro news (Google RSS)
:57  Classify news (DeepSeek)
:00  Daily pipeline (07:00 only)
:02  Monitor 1h (stops + timing)
:05  Specialist 4h (decisão + intenção)
:08  XGBoost 1h (reativado)  ← NOVO
```

---

## PARTE 2 — Atualizar Dashboard

### Passo 7 — Card XGBoost: de "CONGELADO" para ativo
Em `components/signals.py`, o card do XGBoost deve mudar:

**De:**
```
XGBOOST 1H — BENCHMARK
⏸ CONGELADO
Desativado em 2026-03-31 (congelado)
Portfolio: $99,995.27
```

**Para:**
```
XGBOOST 1H — BASELINE (sem gates)
Alloc: X.X% CASH/IN POS
Portfolio: $100,000.00
SL:2% ✅  SG:1.5% ✅  Trail:OFF  TG:OFF  NG:OFF
Última execução: DD/MM HH:MM BRT
```

Mostrar quais gates estão OFF (trailing, timing, news) para deixar claro que é modelo puro.

### Passo 8 — System status
De:
```
⏸ XGB Benchmark
```
Para:
```
🟢 XGB 1h Baseline (Xmin atrás)
```

### Passo 9 — AI Analysis
Em `_gather_market_context()`, atualizar seção XGBoost:
- De "BENCHMARK (congelado)" para "BASELINE (ativo, sem gates)"
- Incluir allocation_raw e ação do último sinal

### Passo 10 — Data Quality
XGB 1h Signals e XGB 1h Portfolio devem voltar a ficar 🟢 quando o cron começar a rodar.

---

## PARTE 3 — Validação

1. XGBoost roda manualmente sem erro
2. Config tem trailing/timing/news desativados
3. Portfolio resetado para $100,000
4. signals.csv limpo (só header)
5. Crontab tem 7 entries (XGB de volta)
6. Dashboard mostra "BASELINE (sem gates)" em vez de "CONGELADO"
7. System status mostra XGB ativo
8. Data Quality: XGB signals/portfolio 🟢
