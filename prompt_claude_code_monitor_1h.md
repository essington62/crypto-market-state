# TASK: Monitor 1h — Stop Loss/Gain com frequência horária

## Contexto
Análise estatística mostrou que:
- Alvo de 1.5% demora ~510min (mediana) para ser atingido
- Stop loss 2% é atingido em 150min nos 10% mais rápidos (P10)
- RSI/BB em 4h são 14pp melhores que em timeframes menores
- Decisão de entrada deve ficar em 4h, monitoramento em 1h

## Arquitetura final (data-driven)
```
R11 HMM (daily) → regime
AB + Timing Gate (4h) → decisão de entrada (a cada 4h)
Monitor (1h) → checa stop loss/gain (a cada 1h) ← ESTE TASK
```

## Objetivo
Criar script de monitoramento que roda a cada 1h e:
1. Se NÃO tem posição aberta → não faz nada (log e sai)
2. Se TEM posição aberta → busca preço atual via Binance API, checa stop loss e stop gain
3. Se stop atingido → executa saída, atualiza portfolio.json e signals.csv
4. Se stop NÃO atingido → loga o check e sai

O monitor NÃO toma decisão de entrada. Entrada é responsabilidade do specialist_4h (4h).

## Passo 1 — Inspecionar antes de codar
1. Ler `scripts/paper_trading/specialist_4h_paper_trader.py`:
   - Como lê preço atual (Binance API)
   - Como executa trade (atualiza portfolio.json)
   - Como loga em signals.csv
   - Funções _apply_stop_loss() e _apply_stop_gain()
2. Ler `scripts/paper_trading/state/specialist_4h/portfolio.json` — campos disponíveis
3. Ler `scripts/paper_trading/state/specialist_4h/config.json` — stop loss/gain config

## Passo 2 — Criar script monitor
Arquivo: `scripts/paper_trading/specialist_4h_monitor_1h.py`

Fluxo:
```python
def run_monitor():
    """
    Monitor de 1h — checa stop loss/gain em posições abertas.
    Roda independente do specialist_4h_paper_trader.py.
    Compartilha os mesmos state files (portfolio.json, signals.csv, config.json).
    """
    # 1. Ler config e portfolio
    config = load_config()
    portfolio = load_portfolio()
    
    # 2. Se não tem posição, sair
    if portfolio["btc_held"] <= 0:
        logger.info("No position open — monitor skip")
        return
    
    # 3. Buscar preço atual (Binance API — candle 1h mais recente ou ticker)
    current_price = fetch_current_price()
    
    # 4. Montar context
    context = {
        "entry_price": portfolio["entry_price"],
        "current_price": current_price,
        "position_btc": portfolio["btc_held"],
        "current_time": datetime.now(timezone.utc),
        "stop_loss_cooldown_until": portfolio.get("stop_loss_cooldown_until"),
    }
    
    # 5. Checar stop gain
    cl = config.get("control_layer", {})
    sg = cl.get("stop_gain", {})
    if sg.get("enabled", False):
        entry_price = context["entry_price"]
        profit_pct = (current_price - entry_price) / entry_price
        if profit_pct >= sg["pct"]:
            execute_exit(portfolio, config, current_price, "TAKE_PROFIT")
            return
    
    # 6. Checar stop loss
    sl = cl.get("stop_loss", {})
    if sl.get("enabled", False):
        drawdown_pct = (current_price - entry_price) / entry_price
        if drawdown_pct <= -sl["pct"]:
            execute_exit(portfolio, config, current_price, "STOP_LOSS")
            return
    
    # 7. Nenhum stop atingido — logar check
    drawdown = (current_price - portfolio["entry_price"]) / portfolio["entry_price"]
    logger.info(
        f"Monitor check: price=${current_price:.2f}, "
        f"entry=${portfolio['entry_price']:.2f}, "
        f"drawdown={drawdown*100:.2f}%, "
        f"stop_loss={-sl.get('pct', 0.02)*100:.1f}%, "
        f"stop_gain={+sg.get('pct', 0.015)*100:.1f}%"
    )
    
    # 8. Atualizar max_price_since_entry (para trailing stop futuro)
    if current_price > (portfolio.get("max_price_since_entry") or 0):
        portfolio["max_price_since_entry"] = current_price
        save_portfolio(portfolio)
```

## Passo 3 — Função fetch_current_price()
Reutilizar o mesmo método que o specialist_4h_paper_trader usa para buscar preço.
Se usa Binance API REST:
```python
def fetch_current_price() -> float:
    """Busca preço atual do BTCUSDT via Binance API (ticker)."""
    import urllib.request
    import json
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return float(data["price"])
```

Nota: usar ticker price (instantâneo) em vez de candle — mais rápido e preciso para check de stop.

## Passo 4 — Função execute_exit()
Reutilizar/adaptar a lógica de saída do specialist_4h_paper_trader:
```python
def execute_exit(portfolio, config, exit_price, reason):
    """
    Executa saída (TAKE_PROFIT ou STOP_LOSS).
    Atualiza portfolio.json e signals.csv.
    """
    btc_held = portfolio["btc_held"]
    fee_rate = config.get("fee_rate", 0.0004)
    
    # Calcular valor da venda
    gross_value = btc_held * exit_price
    fee = gross_value * fee_rate
    net_value = gross_value - fee
    
    # P&L
    entry_price = portfolio["entry_price"]
    pnl_pct = (exit_price - entry_price) / entry_price
    
    # Atualizar portfolio
    portfolio["usdt_free"] += net_value
    portfolio["btc_held"] = 0.0
    portfolio["total_usdt"] = portfolio["usdt_free"]
    portfolio["btc_price"] = exit_price
    portfolio["position_pct"] = 0.0
    portfolio["last_update"] = datetime.now(timezone.utc).isoformat()
    
    # Limpar tracking de entrada
    portfolio["entry_price"] = None
    portfolio["entry_time"] = None
    portfolio["max_price_since_entry"] = None
    
    # Cooldown (só para stop loss)
    if reason == "STOP_LOSS":
        cooldown_candles = config.get("control_layer", {}).get("stop_loss", {}).get("cooldown_candles", 1)
        cooldown_until = datetime.now(timezone.utc) + timedelta(hours=4 * cooldown_candles)
        portfolio["stop_loss_cooldown_until"] = cooldown_until.isoformat()
    
    save_portfolio(portfolio)
    
    # Logar em signals.csv
    log_signal(
        candle_close=datetime.now(timezone.utc).isoformat(),
        price_close=exit_price,
        action=reason,
        portfolio_value=portfolio["total_usdt"],
        entry_price=entry_price,
        drawdown_pct=pnl_pct,
        fee_paid=fee,
        # Marcar como monitor exit
        source="monitor_1h",
    )
    
    logger.info(
        f"{'🎯' if reason == 'TAKE_PROFIT' else '🛑'} {reason}: "
        f"exit=${exit_price:.2f}, entry=${entry_price:.2f}, "
        f"P&L={pnl_pct*100:.2f}%, fee=${fee:.2f}"
    )
```

## Passo 5 — Compartilhamento de state
CRÍTICO: O monitor e o specialist_4h compartilham os mesmos arquivos de state.
Precisa de cuidado para evitar race conditions:

- portfolio.json: ambos leem e escrevem
- signals.csv: ambos fazem append
- config.json: ambos só leem

Risco: se o cron de 4h e o de 1h rodam no mesmo minuto (ex: 00:05 e 00:05), podem conflitar.

Solução simples: usar file lock
```python
import fcntl

def load_portfolio_locked():
    with open(PORTFOLIO_PATH, 'r') as f:
        fcntl.flock(f, fcntl.LOCK_SH)  # shared lock para leitura
        data = json.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return data

def save_portfolio_locked(data):
    with open(PORTFOLIO_PATH, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # exclusive lock para escrita
        json.dump(data, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)
```

Aplicar file lock TANTO no monitor quanto no specialist_4h_paper_trader.py.

## Passo 6 — Adicionar coluna source ao signals.csv
Para distinguir sinais do specialist (4h) vs monitor (1h):
- specialist_4h gera: BUY, HOLD, SELL
- monitor_1h gera: TAKE_PROFIT, STOP_LOSS (checado a cada hora)

Adicionar coluna `source` ao signals.csv: "specialist_4h" ou "monitor_1h".
Backward compatible: linhas existentes ficam com source vazio.

## Passo 7 — Crontab
Adicionar ao crontab:
```bash
# ─────────────────────────────────────────────
# Monitor 1h — Stop Loss/Gain check
# Roda a cada hora, 2min após o candle fechar
# Nos horários do specialist (21,01,05,09,13,17 BRT), o specialist roda em :05
# Monitor roda em :02, antes do specialist, para pegar stops antes de nova decisão
# ─────────────────────────────────────────────
2 * * * * cd /Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state && /opt/homebrew/Caskroom/miniforge/base/envs/crypto_market_state/bin/python scripts/paper_trading/specialist_4h_monitor_1h.py >> /Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/logs/specialist_4h_monitor.log 2>&1
```

Nota: roda no minuto :02, ANTES do specialist que roda no :05. Assim:
- :02 monitor checa stop → se atingido, vende e limpa posição
- :05 specialist roda → vê que não tem posição, decide se reentra

## Passo 8 — Atualizar dashboard
No trading dashboard (`crypto-trading-dashboard`):
- Mostrar "Monitor 1h: ativo" no health check
- Mostrar última checagem do monitor (timestamp do log)
- Distinguir exits do specialist vs exits do monitor no histórico

No specialist dashboard (`apps/specialist_4h_dashboard.py`):
- Adicionar status do monitor

## Restrições
- Monitor NUNCA toma decisão de entrada — só checa saída
- Monitor compartilha state com specialist (file lock obrigatório)
- Se não tem posição → log e sai (custo zero)
- Se Binance API falha → log warning e sai (não crashar, tentar no próximo ciclo)
- Preço via ticker (não candle) — mais rápido e preciso
- Cooldown de stop loss respeita a config (4h * cooldown_candles)
- Sem hardcode de paths — usar mesma estrutura de config que specialist

## Validação
1. Rodar monitor manualmente sem posição aberta:
   ```bash
   python scripts/paper_trading/specialist_4h_monitor_1h.py
   ```
   Deve logar "No position open — monitor skip"

2. Editar portfolio.json com posição fictícia (entry_price alto, btc_held > 0).
   Rodar monitor. Se preço atual < entry - 2%, deve executar STOP_LOSS.

3. Editar portfolio.json com entry_price baixo.
   Rodar monitor. Se preço atual > entry + 1.5%, deve executar TAKE_PROFIT.

4. Verificar que signals.csv tem nova linha com source="monitor_1h"

5. Verificar file lock: rodar specialist e monitor simultaneamente, confirmar que não corrompe portfolio.json

6. Verificar crontab: `crontab -l` mostra 4 entries (daily_update, r11, specialist_4h, monitor_1h)
