#!/usr/bin/env python3
"""
dq_daily_update.py
==================
Display consolidado ao final do cron diário.
Imprime no terminal (stdout):
  - DQ report (FRESH / STALE / MISSING por asset)  — via dq_deep_l1.main()
  - Paper trading status (sinal, portfólio, última trade)

Não depende de banco de dados — lê parquets e CSVs diretamente.
"""

import importlib.util
import os
import sys

import pandas as pd

BASE = "/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state"
STATE_DIR = os.path.join(BASE, "scripts/paper_trading/state")


# ─────────────────────────────────────────────────────────────────
# SEÇÃO 1 — DQ REPORT (delega para dq_deep_l1)
# ─────────────────────────────────────────────────────────────────

def run_dq_report():
    """Carrega dq_deep_l1.py como módulo e chama main().
    Esse módulo já imprime a tabela no terminal e salva
    last_quality_report.txt e last_update_status.json."""
    script = os.path.join(BASE, "scripts/cron/dq_deep_l1.py")
    spec = importlib.util.spec_from_file_location("dq_deep_l1", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


# ─────────────────────────────────────────────────────────────────
# SEÇÃO 2 — PAPER TRADING STATUS
# ─────────────────────────────────────────────────────────────────

def print_paper_trading_status():
    """Lê equity_curve.csv e trade_log.csv e imprime status atual."""
    eq_path = os.path.join(STATE_DIR, "equity_curve.csv")
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
            print(f"  Posição:     {last['position_pct'] * 100:.1f}%")
            print(f"  Sinal:       {regime_icon} {last['regime']}   p_bull={last['p_bull']:.4f}")
            print(f"  Dias ativos: {n_days}")

    # Última trade
    if os.path.exists(trade_path):
        trades = pd.read_csv(trade_path, parse_dates=["date"])
        real_trades = trades[~trades["note"].str.startswith("INITIAL", na=False)]
        if not real_trades.empty:
            t = real_trades.iloc[-1]
            side_icon = "📈 BUY " if t["side"] == "BUY" else "📉 SELL"
            print(
                f"  Última trade: {side_icon} {t['btc_qty']:.5f} BTC"
                f" @ ${t['fill_price']:,.2f}  ({t['date'].strftime('%Y-%m-%d')})"
            )
            print(f"  Total trades: {len(real_trades)}")

    print("=" * 62)
    print()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    # 1. DQ Report — imprime tabela e salva arquivos
    run_dq_report()

    # 2. Paper Trading
    print_paper_trading_status()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[dq_daily_update] ERRO: {e}", file=sys.stderr)
        sys.exit(0)  # nunca quebrar o cron
