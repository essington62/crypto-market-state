# PROMPT — DQ Dashboard no terminal do cron
## Projeto: crypto_v2 — env: crypto_market_state
## Data: 2026-03-24

---

## Contexto

Leia o CLAUDE.md antes de qualquer ação.

O cron diário (`scripts/cron/daily_update.sh`) roda todas as ingestions e pipelines,
depois chama `dq_daily_update.py` — mas redireciona o output para o log:

```bash
${PYTHON} ${PROJECT}/scripts/cron/dq_daily_update.py >> ${LOG_DIR}/daily_update.log 2>&1
```

Problema: nada aparece no terminal ao final da carga. O report bonito está em
`logs/last_quality_report.txt`, gerado pelo `dq_deep_l1.py` que roda separadamente.

O que queremos: ao final do cron, imprimir no terminal um **display consolidado** com:
1. Status DQ — o que está FRESH / STALE / MISSING
2. Status paper trading — sinal atual, portfólio, última trade

---

## PARTE 1 — Reescrever `dq_daily_update.py`

Substituir o conteúdo atual (que conecta a banco de dados FRED via `get_db_connection`)
por um script que:
- **Não depende de banco de dados** — lê direto dos parquets e do state do paper trading
- **Imprime no terminal** (stdout, não logger) um display formatado
- **Também salva** em `logs/last_quality_report.txt` e `logs/last_update_status.json`
  (mantendo compatibilidade com o que `dq_deep_l1.py` já gera)

### Estrutura do novo `dq_daily_update.py`

```python
#!/usr/bin/env python3
"""
dq_daily_update.py
==================
Display consolidado ao final do cron diário.
Imprime no terminal:
  - DQ report (FRESH / STALE / MISSING por asset)
  - Paper trading status (sinal, portfólio, última trade)

Não depende de banco de dados — lê parquets diretamente.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state"
LOG_DIR = os.path.join(BASE, "logs")
STATE_DIR = os.path.join(BASE, "scripts/paper_trading/state")


# ─────────────────────────────────────────────────────────────────
# SEÇÃO 1 — DQ REPORT
# ─────────────────────────────────────────────────────────────────

# Reutilizar a lógica do dq_deep_l1.py:
# Importar e chamar dq_deep_l1.main() OU copiar FILES_TO_CHECK aqui.
#
# ABORDAGEM RECOMENDADA: importar e chamar
#   import importlib.util
#   spec = importlib.util.spec_from_file_location(
#       "dq_deep_l1",
#       os.path.join(BASE, "scripts/cron/dq_deep_l1.py")
#   )
#   mod = importlib.util.module_from_spec(spec)
#   spec.loader.exec_module(mod)
#   mod.main()   # imprime + salva last_quality_report.txt + last_update_status.json
#
# Assim dq_daily_update.py chama dq_deep_l1 internamente sem duplicar código.


# ─────────────────────────────────────────────────────────────────
# SEÇÃO 2 — PAPER TRADING STATUS
# ─────────────────────────────────────────────────────────────────

def print_paper_trading_status():
    """Lê equity_curve.csv e trade_log.csv e imprime status atual."""
    eq_path    = os.path.join(STATE_DIR, "equity_curve.csv")
    trade_path = os.path.join(STATE_DIR, "trade_log.csv")

    print()
    print("=" * 62)
    print("  PAPER TRADING — R11 STATUS")
    print("=" * 62)

    # Equity curve
    if not os.path.exists(eq_path):
        print("  ⚠  equity_curve.csv não encontrado")
    else:
        eq = pd.read_csv(eq_path, parse_dates=["date"])
        if eq.empty:
            print("  ⚠  equity_curve.csv vazio")
        else:
            last = eq.iloc[-1]
            first = eq.iloc[0]
            total_ret = (last["portfolio_value"] / first["portfolio_value"] - 1) * 100
            n_days = len(eq)

            regime_icon = "🟢" if last["regime"] == "Bull" else "🔴"

            print(f"  Data:        {last['date'].strftime('%Y-%m-%d')}")
            print(f"  Portfólio:   ${last['portfolio_value']:,.2f}   ({total_ret:+.2f}% desde início)")
            print(f"  Posição:     {last['position_pct']*100:.1f}%")
            print(f"  Sinal:       {regime_icon} {last['regime']}   p_bull={last['p_bull']:.4f}")
            print(f"  Dias ativos: {n_days}")

    # Última trade
    if os.path.exists(trade_path):
        trades = pd.read_csv(trade_path, parse_dates=["date"])
        real_trades = trades[trades["note"].str.contains("INITIAL", na=False) == False]
        if not real_trades.empty:
            t = real_trades.iloc[-1]
            side_icon = "📈 BUY " if t["side"] == "BUY" else "📉 SELL"
            print(f"  Última trade: {side_icon} {t['btc_qty']:.5f} BTC @ ${t['fill_price']:,.2f}  ({t['date'].strftime('%Y-%m-%d')})")
        # Mostrar todas as trades reais (exceto INITIAL)
        if len(real_trades) > 0:
            print(f"  Total trades: {len(real_trades)}")

    print("=" * 62)
    print()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    # 1. DQ Report — chama dq_deep_l1 internamente
    # (imprime tabela e salva arquivos)
    ...

    # 2. Paper Trading
    print_paper_trading_status()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[dq_daily_update] ERRO: {e}", file=sys.stderr)
        sys.exit(0)   # nunca quebrar o cron
```

### Detalhes de implementação

**Importação de dq_deep_l1:**

```python
import importlib.util, sys

def run_dq_report():
    script = os.path.join(BASE, "scripts/cron/dq_deep_l1.py")
    spec = importlib.util.spec_from_file_location("dq_deep_l1", script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()   # already prints table + saves files

run_dq_report()
```

**Filtragem de trades reais** (excluir INITIAL_POSITION):
```python
real_trades = trades[~trades["note"].str.startswith("INITIAL", na=False)]
```

---

## PARTE 2 — Ajustar `daily_update.sh`

### 2.1 — Remover redirecionamento de log no dq

Localizar a linha:
```bash
${PYTHON} ${PROJECT}/scripts/cron/dq_daily_update.py \
    >> ${LOG_DIR}/daily_update.log 2>&1
```

Substituir por (usa `tee` para imprimir no terminal E salvar no log):
```bash
echo "[$(date '+%H:%M:%S')] Running DQ display..."
${PYTHON} ${PROJECT}/scripts/cron/dq_daily_update.py \
    2>&1 | tee -a ${LOG_DIR}/daily_update.log
```

### 2.2 — Remover o summary final genérico (opcional)

O bloco atual:
```bash
echo ""
echo "========================================"
echo "Daily Update COMPLETE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
```

Pode ser simplificado — o display do dq_daily_update.py já serve de summary.
Manter apenas uma linha de timestamp de conclusão.

---

## PARTE 3 — Investigar compra inicial de 10-mar

No `trade_log.csv` a primeira entrada é:
```
2026-03-12, BUY, 1.0 BTC @ $60,230.71, INITIAL_POSITION (reconstructed)
```

Mas o paper trading pode ter iniciado antes (10-mar). Verificar:

1. Checar se existe qualquer backup ou snapshot anterior em `state/`:
   ```bash
   ls -la scripts/paper_trading/state/
   cat scripts/paper_trading/state/equity_curve.csv.bak
   ```

2. Verificar o paper trading script principal para entender como a posição inicial
   foi "reconstruída" e se o histórico de 10-12 mar foi perdido.

3. Se o histórico foi perdido: adicionar manualmente a entrada de 10-mar no equity_curve.csv
   com os valores corretos de portfólio e posição, para que os cálculos de retorno
   acumulado sejam precisos desde o início real.

**Nota**: a INITIAL_POSITION (reconstructed) @ $60,230.71 parece ser um preço
retroativo — verificar se é o preço de 10-mar ou de 12-mar.

---

## PARTE 4 — Exemplo de output esperado

Ao final do cron, o terminal deve mostrar:

```
[08:32:15] Running DQ display...
============================================================
DATA QUALITY REPORT — 2026-03-24 08:32:16 UTC
============================================================
Asset                      Last Date     Days Ago  Status      NaN
------------------------------------------------------------
BTC spot daily             2026-03-24    0         FRESH       0
BTC spot 4h                2026-03-23    1         FRESH       0
...
DGS2                       2026-03-21    3         STALE       0
...
------------------------------------------------------------
Summary: 31/33 FRESH | 2 STALE | 0 MISSING
============================================================

⚠ WARNING: files need attention:
  - DGS2: Last update 3 days ago (threshold: 2)

==============================================================
  PAPER TRADING — R11 STATUS
==============================================================
  Data:        2026-03-24
  Portfólio:   $83,399.56   (+3.85% desde início)
  Posição:     75.0%
  Sinal:       🟢 Bull   p_bull=0.8968
  Dias ativos: 10
  Última trade: 📈 BUY  0.88031 BTC @ $71,080.12  (2026-03-23)
  Total trades: 3
==============================================================
```

---

## Constraints

- `dq_daily_update.py` NUNCA deve fazer `sys.exit(1)` — sempre `sys.exit(0)`
  para não quebrar o cron
- Não duplicar a lista `FILES_TO_CHECK` — importar `dq_deep_l1` como módulo
- `tee -a` no shell (append, não overwrite) preserva o log histórico
- Não alterar `dq_deep_l1.py` — ele está correto e funcional
- Validar que `equity_curve.csv` existe antes de ler (pode não existir em ambiente novo)
