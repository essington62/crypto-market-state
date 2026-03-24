# PROMPT — signal_log.csv no paper trader
## Projeto: crypto_v2 — env: crypto_market_state
## Arquivo alvo: scripts/paper_trading/r11_paper_trader.py

---

## Objetivo

Adicionar `signal_log.csv` que persiste a previsão diária completa do R11,
independentemente de haver trade. Hoje só `p_bull` e `regime` ficam no
`equity_curve.csv`; `p_bear`, `state`, `confidence` e `state_duration` se perdem.

---

## PARTE 1 — Constante e função em `r11_paper_trader.py`

### 1.1 — Adicionar constante junto às outras (linha ~49)

```python
SIGNAL_LOG = STATE_DIR / "signal_log.csv"
```

### 1.2 — Adicionar função `_append_signal_log()`

```python
_SIGNAL_LOG_FIELDS = [
    "date", "regime", "state", "p_bull", "p_bear",
    "confidence", "btc_price", "position_pct_before",
    "target_pos", "trade_executed", "state_duration",
]

def _append_signal_log(
    date_str: str,
    signal: dict,
    btc_price: float,
    position_pct_before: float,
    target_pos: float,
    trade_executed: bool,
    equity_df: pd.DataFrame,
) -> None:
    """Persiste previsão diária completa. Idempotente — pula se date já existe."""
    # state_duration: dias consecutivos no regime atual (incluindo hoje)
    if len(equity_df) > 0:
        same = (equity_df["regime"] == signal["regime"]).values
        # contar da direita até quebrar a sequência
        duration = int(np.sum(np.cumprod(same[::-1]))) + 1
    else:
        duration = 1

    row = {
        "date":                date_str,
        "regime":              signal["regime"],
        "state":               signal["state"],
        "p_bull":              signal["p_bull"],
        "p_bear":              signal["p_bear"],
        "confidence":          round(max(signal["p_bull"], signal["p_bear"]), 4),
        "btc_price":           round(btc_price, 2),
        "position_pct_before": round(position_pct_before, 4),
        "target_pos":          round(target_pos, 4),
        "trade_executed":      int(trade_executed),
        "state_duration":      duration,
    }

    write_header = not SIGNAL_LOG.exists() or SIGNAL_LOG.stat().st_size == 0
    # Idempotência
    if SIGNAL_LOG.exists() and SIGNAL_LOG.stat().st_size > 0:
        existing = pd.read_csv(SIGNAL_LOG)
        if date_str in existing["date"].values:
            return

    with open(SIGNAL_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SIGNAL_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
```

---

## PARTE 2 — Chamar `_append_signal_log()` na função principal

No fluxo de `run_paper_trader()`, localizar o bloco após o step 5 (target_pos calculado)
e antes do step 6 (execute order). Adicionar a chamada:

```python
# Após: target_pos = _target_position(signal["regime"])
# Após: delta = abs(target_pos - current_pos)
# Linha ~399 (após a linha que define `traded = False`)

# Chamar ANTES do bloco `if not status_only` de execução de ordem,
# mas DENTRO do bloco `if not status_only` geral:

prev_df_for_log = load_equity_curve()   # já existe esta função no arquivo
_append_signal_log(
    date_str             = date_str,
    signal               = signal,
    btc_price            = portfolio.get("btc_price", 0.0),
    position_pct_before  = current_pos,
    target_pos           = target_pos,
    trade_executed       = False,       # ainda não executou — será atualizado abaixo
    equity_df            = prev_df_for_log,
)
```

**Nota**: `trade_executed=False` é passado aqui pois a ordem ainda não aconteceu.
Após o bloco de execução, se `traded=True`, **atualizar** a última linha do signal_log:

```python
# Após: traded = True (nos dois branches de BUY/SELL)
# Adicionar logo após cada `traded = True`:
if SIGNAL_LOG.exists():
    _df_sl = pd.read_csv(SIGNAL_LOG)
    if len(_df_sl) > 0 and _df_sl.iloc[-1]["date"] == date_str:
        _df_sl.loc[_df_sl.index[-1], "trade_executed"] = 1
        _df_sl.to_csv(SIGNAL_LOG, index=False)
```

---

## PARTE 3 — Verificar campo `btc_price` no portfolio dict

O `executor.get_portfolio_value()` retorna um dict. Verificar se já tem `btc_price`.
Se não tiver, adicionar em `r11_order_executor.py`:

```python
# No método get_portfolio_value(), garantir que o retorno inclua:
"btc_price": btc_last_price,   # preço BTC no momento da consulta
```

Se o campo não existir, usar fallback em `_append_signal_log`:
```python
btc_price = portfolio.get("btc_price", portfolio.get("btc_held", 0) and
            portfolio["total_usdt"] / portfolio["btc_held"] if portfolio.get("btc_held", 0) > 0 else 0)
```

---

## PARTE 4 — Output esperado (signal_log.csv)

```
date,regime,state,p_bull,p_bear,confidence,btc_price,position_pct_before,target_pos,trade_executed,state_duration
2026-03-12,Bull,1,1.0000,0.0001,1.0000,70307.61,0.75,0.75,0,1
2026-03-13,Bull,1,1.0000,0.0001,1.0000,72100.00,0.7589,0.75,0,2
2026-03-16,Bear,0,0.0001,0.9999,0.9999,73952.99,0.7589,0.0,1,1
...
2026-03-24,Bull,1,0.8968,0.1032,0.8968,71008.45,0.7499,0.75,0,2
```

---

## Constraints

- Idempotente: se `date` já existe no signal_log, não duplicar
- `status_only=True`: NÃO gravar no signal_log (apenas leitura/display)
- Não alterar `equity_curve.csv` — os dois arquivos coexistem
- `state_duration` deve ser 1 no primeiro dia de um novo regime
