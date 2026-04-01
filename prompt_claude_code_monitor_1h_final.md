# TASK: Monitor 1h — Stop Loss/Gain/Trailing (VERSÃO FINAL CORRIGIDA)

## Contexto
Sistema em produção com:
- Entrada decidida no specialist_4h (4h) com regime gate + timing gate
- Monitoramento a cada 1h (este script)
- State compartilhado via portfolio.json e signals.csv

Correções aplicadas nesta versão:
1. Smoothed price: raw para stop loss (segurança), smoothed para stop gain/trailing (anti-ruído)
2. Trailing stop: só ativa após lucro mínimo, threshold configurável
3. Log em signals.csv com coluna source para distinguir specialist vs monitor
4. Paths absolutos via Path(__file__) para funcionar no cron
5. Race condition check via double-read de portfolio
6. File lock em todas as operações de I/O compartilhadas

## Passo 1 — Inspecionar implementação atual
1. Ler `scripts/paper_trading/specialist_4h_paper_trader.py`:
   - Como loga em signals.csv (colunas, formato, função de append)
   - Como atualiza portfolio.json
   - Verificar se já tem file lock implementado (se não, adicionar também)
2. Ler signals.csv — verificar colunas atuais
3. Ler portfolio.json — campos disponíveis
4. Ler config.json — estrutura control_layer

## Passo 2 — Atualizar config.json
Adicionar seção trailing_stop:
```json
{
  "control_layer": {
    "...existente...",
    "stop_loss": { "enabled": true, "pct": 0.02, "cooldown_candles": 1 },
    "stop_gain": { "enabled": true, "pct": 0.015 },
    "trailing_stop": {
      "enabled": true,
      "pct": 0.01,
      "min_profit_to_activate": 0.005
    },
    "regime_gate": { "...existente..." },
    "timing_gate": { "...existente..." }
  }
}
```

## Passo 3 — Criar script monitor
Arquivo: `scripts/paper_trading/specialist_4h_monitor_1h.py`

```python
#!/usr/bin/env python3
"""
Monitor 1h — checa stop loss/gain/trailing em posições abertas.
Roda a cada hora via cron (:02 de cada hora).
NÃO toma decisão de entrada — só monitora saídas.
Compartilha state com specialist_4h_paper_trader.py via file lock.
"""

import json
import csv
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
import fcntl
import os

# ── Paths absolutos (funciona no cron) ──
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATE_DIR = BASE_DIR / "scripts" / "paper_trading" / "state" / "specialist_4h"
PORTFOLIO_PATH = STATE_DIR / "portfolio.json"
CONFIG_PATH = STATE_DIR / "config.json"
SIGNALS_PATH = STATE_DIR / "signals.csv"
LOG_DIR = BASE_DIR / "logs"

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MONITOR-1H] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════
# FILE LOCK HELPERS
# ═══════════════════════════════════════

def load_json_locked(path):
    """Lê JSON com shared lock."""
    with open(path, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        data = json.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return data


def save_json_locked(path, data):
    """Escreve JSON com exclusive lock."""
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(data, f, indent=2)
        f.write("\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def append_signal_locked(row: dict):
    """Append uma linha ao signals.csv com exclusive lock."""
    file_exists = SIGNALS_PATH.exists()
    with open(SIGNALS_PATH, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists or os.path.getsize(SIGNALS_PATH) == 0:
            writer.writeheader()
        writer.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


# ═══════════════════════════════════════
# PRICE FETCH
# ═══════════════════════════════════════

def fetch_current_price() -> float:
    """Busca preço atual BTCUSDT via Binance ticker."""
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    response = urllib.request.urlopen(url, timeout=10)
    data = json.loads(response.read())
    return float(data["price"])


def get_prices(portfolio: dict) -> tuple:
    """
    Retorna (raw_price, smoothed_price).
    Raw: para stop loss (mais sensível, protege capital).
    Smoothed: para stop gain e trailing (menos sensível a spikes).
    """
    raw = fetch_current_price()
    last = portfolio.get("last_monitor_price", raw)
    smoothed = 0.7 * raw + 0.3 * last
    return raw, smoothed


# ═══════════════════════════════════════
# EXIT EXECUTION
# ═══════════════════════════════════════

def execute_exit(portfolio: dict, config: dict, exit_price: float, reason: str):
    """
    Executa saída e atualiza portfolio + signals.
    Reasons: STOP_LOSS, TAKE_PROFIT, TRAILING_STOP
    """
    btc_held = portfolio["btc_held"]
    entry_price = portfolio["entry_price"]
    fee_rate = config.get("fee_rate", 0.0004)

    # Calcular venda
    gross_value = btc_held * exit_price
    fee = gross_value * fee_rate
    net_value = gross_value - fee
    pnl_pct = (exit_price - entry_price) / entry_price

    # Atualizar portfolio
    portfolio["usdt_free"] = portfolio.get("usdt_free", 0) + net_value
    portfolio["btc_held"] = 0.0
    portfolio["total_usdt"] = portfolio["usdt_free"]
    portfolio["btc_price"] = exit_price
    portfolio["position_pct"] = 0.0
    portfolio["last_update"] = datetime.now(timezone.utc).isoformat()

    # Limpar tracking de entrada
    portfolio["entry_price"] = None
    portfolio["entry_time"] = None
    portfolio["max_price_since_entry"] = None
    portfolio["last_monitor_price"] = None

    # Cooldown (só para stop loss)
    if reason == "STOP_LOSS":
        cl = config.get("control_layer", {})
        cooldown_candles = cl.get("stop_loss", {}).get("cooldown_candles", 1)
        cooldown_until = datetime.now(timezone.utc) + timedelta(hours=4 * cooldown_candles)
        portfolio["stop_loss_cooldown_until"] = cooldown_until.isoformat()

    # Salvar portfolio
    save_json_locked(PORTFOLIO_PATH, portfolio)

    # Logar em signals.csv
    now = datetime.now(timezone.utc)
    signal_row = {
        "timestamp": now.isoformat(),
        "candle_close": now.isoformat(),
        "price_close": exit_price,
        "allocation_raw": 0.0,
        "allocation_final": 0.0,
        "r11_prob_bull": portfolio.get("r11_prob_bull", ""),
        "r11_entropy": portfolio.get("r11_entropy", ""),
        "regime_age_log": "",
        "stress_score": "",
        "returns_4h": "",
        "volatility_24h": "",
        "volume_zscore": "",
        "buy_pressure": "",
        "price_range_4h": "",
        "position_btc": 0.0,
        "position_usdt": portfolio["usdt_free"],
        "portfolio_value": portfolio["total_usdt"],
        "action": reason,
        "delta_allocation": -portfolio.get("position_pct", 0),
        "fee_paid": fee,
        "stop_loss_triggered": reason == "STOP_LOSS",
        "stop_gain_triggered": reason == "TAKE_PROFIT",
        "macro_gate_mult": 1.0,
        "deriv_gate_mult": 1.0,
        "entry_price": entry_price,
        "drawdown_pct": pnl_pct,
        "regime_gate_triggered": "",
        "timing_gate_score": "",
        "rsi_4h": "",
        "bb_pct_b": "",
        "source": "monitor_1h",
    }

    # Ler header atual para garantir compatibilidade
    try:
        if SIGNALS_PATH.exists():
            import pandas as pd
            existing = pd.read_csv(SIGNALS_PATH, nrows=0)
            existing_cols = list(existing.columns)
            # Adicionar colunas novas que não existem
            for col in ["source"]:
                if col not in existing_cols:
                    existing_cols.append(col)
            # Filtrar row para só ter colunas do CSV
            filtered_row = {k: signal_row.get(k, "") for k in existing_cols}
            append_signal_locked(filtered_row)
        else:
            append_signal_locked(signal_row)
    except Exception as e:
        logger.warning(f"Failed to log signal: {e}")

    emoji = {"TAKE_PROFIT": "🎯", "STOP_LOSS": "🛑", "TRAILING_STOP": "📉"}.get(reason, "⚠️")
    logger.info(
        f"{emoji} {reason}: exit=${exit_price:.2f}, "
        f"entry=${entry_price:.2f}, P&L={pnl_pct*100:+.2f}%, "
        f"fee=${fee:.2f}, portfolio=${portfolio['total_usdt']:.2f}"
    )


# ═══════════════════════════════════════
# MAIN MONITOR
# ═══════════════════════════════════════

def run_monitor():
    """
    Monitor principal. Roda a cada hora.
    Só atua em posições abertas — nunca decide entrada.
    """
    config = load_json_locked(CONFIG_PATH)
    portfolio = load_json_locked(PORTFOLIO_PATH)
    initial_update = portfolio.get("last_update")

    # ── 1. Sem posição → skip ──
    if portfolio.get("btc_held", 0) <= 0:
        logger.info("No position — skip")
        return

    # ── 2. Cooldown check ──
    cooldown_until = portfolio.get("stop_loss_cooldown_until")
    if cooldown_until:
        try:
            if datetime.now(timezone.utc) < datetime.fromisoformat(cooldown_until):
                logger.info(f"In cooldown until {cooldown_until} — skip")
                return
        except (ValueError, TypeError):
            pass  # cooldown inválido, ignorar

    # ── 3. Buscar preço ──
    try:
        raw_price, smoothed_price = get_prices(portfolio)
    except Exception as e:
        logger.warning(f"Price fetch failed: {e} — skip")
        return

    entry_price = portfolio.get("entry_price")
    if not entry_price or entry_price <= 0:
        logger.warning("entry_price missing/invalid — skip")
        return

    # ── 4. Race condition check ──
    portfolio_check = load_json_locked(PORTFOLIO_PATH)
    if portfolio_check.get("last_update") != initial_update:
        logger.warning("Portfolio changed during execution — abort")
        return

    # ── 5. Atualizar tracking ──
    max_price = portfolio.get("max_price_since_entry") or entry_price
    if raw_price > max_price:
        max_price = raw_price
        portfolio["max_price_since_entry"] = max_price
    portfolio["last_monitor_price"] = raw_price

    cl = config.get("control_layer", {})

    # ── 6. STOP GAIN (usa smoothed — anti-spike) ──
    sg = cl.get("stop_gain", {})
    if sg.get("enabled", False):
        profit_pct = (smoothed_price - entry_price) / entry_price
        if profit_pct >= sg["pct"]:
            # Executar com raw_price (preço real, não smoothed)
            execute_exit(portfolio, config, raw_price, "TAKE_PROFIT")
            return

    # ── 7. STOP LOSS (usa raw — mais sensível, protege capital) ──
    sl = cl.get("stop_loss", {})
    if sl.get("enabled", False):
        drawdown_pct = (raw_price - entry_price) / entry_price
        if drawdown_pct <= -sl["pct"]:
            execute_exit(portfolio, config, raw_price, "STOP_LOSS")
            return

    # ── 8. TRAILING STOP (usa smoothed, só se já em lucro) ──
    ts = cl.get("trailing_stop", {})
    if ts.get("enabled", False):
        trailing_pct = ts.get("pct", 0.01)
        min_profit = ts.get("min_profit_to_activate", 0.005)

        # Só ativa se já subiu pelo menos min_profit desde a entrada
        profit_from_entry = (max_price - entry_price) / entry_price
        if profit_from_entry >= min_profit:
            trailing_dd = (smoothed_price - max_price) / max_price
            if trailing_dd <= -trailing_pct:
                execute_exit(portfolio, config, raw_price, "TRAILING_STOP")
                return

    # ── 9. Log check (sem ação) ──
    drawdown = (raw_price - entry_price) / entry_price
    profit_from_max = (raw_price - max_price) / max_price if max_price else 0

    logger.info(
        f"Check: raw=${raw_price:.2f}, smooth=${smoothed_price:.2f}, "
        f"entry=${entry_price:.2f}, max=${max_price:.2f}, "
        f"dd={drawdown*100:+.2f}%, from_max={profit_from_max*100:+.2f}%"
    )

    # ── 10. Salvar state atualizado ──
    save_json_locked(PORTFOLIO_PATH, portfolio)


# ═══════════════════════════════════════

if __name__ == "__main__":
    run_monitor()
```

## Passo 4 — Adicionar file lock ao specialist_4h_paper_trader.py
IMPORTANTE: o specialist também precisa usar file lock nas mesmas operações.
Adicionar as mesmas funções `load_json_locked` e `save_json_locked` ao specialist.
Substituir todas as chamadas diretas de json.load/json.dump por versões locked.

## Passo 5 — Migrar signals.csv (coluna source)
Adicionar coluna `source` ao signals.csv.
- Linhas existentes: source = "" (vazio, backward compatible)
- Novas linhas do specialist: source = "specialist_4h"
- Novas linhas do monitor: source = "monitor_1h"

Atualizar `_migrate_signals_csv()` no specialist para adicionar coluna source se não existir.
Atualizar o specialist para incluir `"source": "specialist_4h"` em cada linha logada.

## Passo 6 — Crontab
Adicionar ao crontab:
```bash
# ─────────────────────────────────────────────
# Monitor 1h — Stop Loss/Gain/Trailing check
# Roda no minuto :02 de cada hora
# ANTES do specialist (:05) para pegar stops primeiro
# ─────────────────────────────────────────────
2 * * * * cd /Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state && /opt/homebrew/Caskroom/miniforge/base/envs/crypto_market_state/bin/python scripts/paper_trading/specialist_4h_monitor_1h.py >> /Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/logs/specialist_4h_monitor.log 2>&1
```

Verificar que crontab final tem 4 entries:
1. daily_update.sh (07:00)
2. r11_paper_trader.py (07:15)
3. specialist_4h_monitor_1h.py (:02 de cada hora) ← NOVO
4. specialist_4h_paper_trader.py (:05 nos 21,01,05,09,13,17)

## Passo 7 — Atualizar dashboards
Em `apps/specialist_4h_dashboard.py`:
- Mostrar trailing_stop na seção Control Layer (enabled, pct, min_profit)
- Mostrar "Monitor 1h: ativo" com último check timestamp

Em `/Users/brown/Documents/MLGeral/crypto_v2/crypto-trading-dashboard/`:
- health.py: adicionar card do monitor 1h (último run, status)
- signals.py: mostrar coluna source no histórico (specialist_4h vs monitor_1h)
- signals.py: ícones distintos para TAKE_PROFIT (🎯), STOP_LOSS (🛑), TRAILING_STOP (📉)

## Restrições
- Monitor NUNCA decide entrada — só monitora saídas
- Stop loss usa raw_price (sensível, protege capital)
- Stop gain e trailing usam smoothed_price (anti-spike)
- Execução usa raw_price (preço real do mercado)
- Trailing só ativa após lucro mínimo configurável
- File lock obrigatório em AMBOS os scripts (monitor + specialist)
- Paths absolutos via Path(__file__) — funciona no cron
- Todos os thresholds configuráveis no config.json
- Se Binance API falha → log warning e skip (não crashar)
- Backward compatible com signals.csv existente

## Validação
1. Sem posição:
   ```bash
   python scripts/paper_trading/specialist_4h_monitor_1h.py
   ```
   → "No position — skip"

2. Stop loss: editar portfolio.json com entry_price=73000, btc_held=0.5.
   Se preço atual ~$70,000 → drawdown -4.1% > 2% → STOP_LOSS

3. Take profit: editar portfolio.json com entry_price=69000, btc_held=0.5.
   Se preço atual ~$70,500 → profit +2.2% > 1.5% → TAKE_PROFIT

4. Trailing stop: editar portfolio.json com entry_price=69500, max_price_since_entry=71000.
   Se preço atual ~$70,000 → profit_from_entry=2.2% > 0.5% (ativado), dd_from_max=-1.4% > 1% → TRAILING_STOP

5. Race condition: rodar specialist e monitor simultaneamente, confirmar portfolio.json íntegro

6. signals.csv: nova linha tem coluna source="monitor_1h"

7. Crontab: `crontab -l` mostra 4 entries
