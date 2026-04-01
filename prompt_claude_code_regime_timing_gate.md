# TASK: Control Layer — Regime Gate + Timing Gate

## Contexto
O specialist 4h entrou em trade quando R11 = Bear (prob_bull=0.087) e tomou stop loss (-2.7%).
Análise técnica no momento da entrada: RSI 4h=57.5, Bollinger %B=0.59, volume_z=-1.67.
Todos os indicadores apontavam contra a entrada. Faltam gates de proteção.

## Objetivo
Implementar dois gates na control layer que bloqueiam trades em contexto desfavorável:
1. Regime gate: nunca entrar quando R11 indica Bear
2. Timing gate: nunca entrar em topo local (RSI alto + BB alto + volume fraco)

Sem alterar o modelo LightGBM (AB). Gates são determinísticos, pós-modelo.

## Passo 1 — Inspecionar implementação atual
1. Ler `scripts/paper_trading/specialist_4h_paper_trader.py` — encontrar:
   - Função `apply_control_layer()` atual (tem stop_loss + stop_gain + stubs de gates)
   - Como context é montado (quais campos disponíveis)
   - Como config.json é lido
2. Ler `scripts/paper_trading/state/specialist_4h/config.json` — estrutura atual
3. Verificar se rsi_4h e bollinger %B já são calculados em algum lugar do paper trader
   - Se NÃO: precisam ser calculados no paper trader antes de chamar apply_control_layer

## Passo 2 — Calcular indicadores técnicos (se não existirem)
No paper trader, ANTES de chamar apply_control_layer, calcular:

```python
def calculate_timing_indicators(ohlcv_buffer: pd.DataFrame) -> dict:
    """
    Calcula RSI, Bollinger %B e volume_zscore para timing gate.
    Usa o buffer de candles 4h que o paper trader já mantém.
    """
    close = ohlcv_buffer['close']
    volume = ohlcv_buffer['volume']
    
    # RSI 14 períodos
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - 100 / (1 + rs)).iloc[-1]
    
    # Bollinger %B (20 períodos, 2 std)
    bb_mid = close.rolling(20).mean().iloc[-1]
    bb_std = close.rolling(20).std().iloc[-1]
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pct_b = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    
    # Volume z-score (50 períodos)
    vol_mean = volume.rolling(50).mean().iloc[-1]
    vol_std = volume.rolling(50).std().iloc[-1]
    volume_z = (volume.iloc[-1] - vol_mean) / vol_std if vol_std > 0 else 0.0
    
    return {
        "rsi_4h": rsi if not pd.isna(rsi) else 50.0,
        "bb_pct_b": bb_pct_b if not pd.isna(bb_pct_b) else 0.5,
        "volume_zscore_timing": volume_z if not pd.isna(volume_z) else 0.0,
    }
```

Adicionar esses campos ao context dict antes de chamar apply_control_layer.

## Passo 3 — Atualizar config.json
Adicionar configuração dos gates:

```json
{
  "control_layer": {
    "entry_threshold": 0.38,
    "stop_loss": { "enabled": true, "pct": 0.02, "cooldown_candles": 1 },
    "stop_gain": { "enabled": true, "pct": 0.015 },
    "regime_gate": {
      "enabled": true,
      "min_prob_bull": 0.30,
      "max_entropy": 0.85
    },
    "timing_gate": {
      "enabled": true,
      "min_score": 0.30,
      "weights": {
        "rsi": 0.4,
        "bollinger": 0.3,
        "volume": 0.3
      },
      "rsi_threshold": 40,
      "bb_threshold": 0.30
    },
    "macro_gate": { "enabled": false },
    "derivatives_gate": { "enabled": false }
  }
}
```

## Passo 4 — Implementar _apply_regime_gate()
```python
def _apply_regime_gate(allocation: float, context: dict, config: dict) -> tuple[float, bool]:
    """
    Regime gate: bloqueia NOVA entrada quando R11 indica Bear.
    
    Lógica:
    - prob_bull < min_prob_bull → BLOQUEIA (regime Bear/incerto)
    - entropy > max_entropy → BLOQUEIA (HMM sem convicção)
    - Ambos precisam passar para permitir entrada
    
    NÃO afeta saída: se já está em posição, não força venda.
    Regime gate só bloqueia NOVAS entradas.
    """
    prob_bull = context.get("r11_prob_bull", 0.0)
    entropy = context.get("r11_entropy", 1.0)
    min_prob = config.get("min_prob_bull", 0.30)
    max_entropy = config.get("max_entropy", 0.85)
    
    # Se já está em posição, não interferir (deixar stop loss/gain cuidar)
    if context.get("position_btc", 0) > 0:
        return allocation, False
    
    # Gate: bloquear nova entrada se regime desfavorável
    regime_ok = (prob_bull >= min_prob) and (entropy <= max_entropy)
    
    if not regime_ok:
        return 0.0, True  # gate triggered
    
    return allocation, False
```

## Passo 5 — Implementar _apply_timing_gate()
```python
def _apply_timing_gate(allocation: float, context: dict, config: dict) -> tuple[float, float]:
    """
    Timing gate: bloqueia entrada quando indicadores técnicos desfavoráveis.
    
    Score de timing (0 a 1):
    - RSI baixo = bom (oversold, espaço para subir)
    - Bollinger %B baixo = bom (perto do suporte)
    - Volume alto = bom (confirmação de movimento)
    
    Se score < min_score: bloqueia entrada
    Se score >= min_score: permite
    
    NÃO afeta saída: se já está em posição, não interfere.
    """
    # Se já está em posição, não interferir
    if context.get("position_btc", 0) > 0:
        return allocation, 1.0
    
    # Se allocation já é 0 (regime gate bloqueou), skip
    if allocation <= 0:
        return 0.0, 0.0
    
    rsi = context.get("rsi_4h", 50.0)
    bb = context.get("bb_pct_b", 0.5)
    vol_z = context.get("volume_zscore_timing", 0.0)
    
    weights = config.get("weights", {"rsi": 0.4, "bollinger": 0.3, "volume": 0.3})
    rsi_thresh = config.get("rsi_threshold", 40)
    bb_thresh = config.get("bb_threshold", 0.30)
    min_score = config.get("min_score", 0.30)
    
    # Calcular score
    rsi_component = max(0, (rsi_thresh - rsi) / rsi_thresh)
    bb_component = max(0, (bb_thresh - bb) / bb_thresh)
    vol_component = max(0, min(vol_z / 2, 1))
    
    timing_score = (
        weights["rsi"] * rsi_component +
        weights["bollinger"] * bb_component +
        weights["volume"] * vol_component
    )
    
    if timing_score < min_score:
        return 0.0, timing_score  # bloqueado por timing
    
    return allocation, timing_score
```

## Passo 6 — Atualizar apply_control_layer()
Ordem de execução:
```python
def apply_control_layer(allocation_raw, context, config):
    allocation = allocation_raw
    gate_log = {
        "stop_loss_triggered": False,
        "stop_gain_triggered": False,
        "regime_gate_triggered": False,
        "timing_gate_score": 1.0,
        "macro_gate_multiplier": 1.0,
        "derivatives_gate_multiplier": 1.0,
    }
    cl = config.get("control_layer", {})
    
    # 1. Stop gain (se já em posição e em lucro, sair)
    sg = cl.get("stop_gain", {})
    if sg.get("enabled", False):
        allocation, triggered = _apply_stop_gain(allocation, context, sg)
        gate_log["stop_gain_triggered"] = triggered
        if triggered:
            return allocation, gate_log
    
    # 2. Stop loss (se já em posição e em perda, sair)
    sl = cl.get("stop_loss", {})
    if sl.get("enabled", False):
        allocation, triggered = _apply_stop_loss(allocation, context, sl)
        gate_log["stop_loss_triggered"] = triggered
        if triggered:
            return allocation, gate_log
    
    # 3. Regime gate (hard filter para novas entradas)
    rg = cl.get("regime_gate", {})
    if rg.get("enabled", False):
        allocation, triggered = _apply_regime_gate(allocation, context, rg)
        gate_log["regime_gate_triggered"] = triggered
    
    # 4. Timing gate (refinamento para novas entradas)
    tg = cl.get("timing_gate", {})
    if tg.get("enabled", False):
        allocation, timing_score = _apply_timing_gate(allocation, context, tg)
        gate_log["timing_gate_score"] = timing_score
    
    # 5. Macro gate (stub)
    # 6. Derivatives gate (stub)
    
    return allocation, gate_log
```

## Passo 7 — Atualizar signals.csv
Adicionar colunas:
```
regime_gate_triggered, timing_gate_score, rsi_4h, bb_pct_b
```
Manter backward compatibility com _migrate_signals_csv().

## Passo 8 — Atualizar dashboards
Em `apps/specialist_4h_dashboard.py`:
- Regime gate: enabled ✓, min_prob_bull=0.30, max_entropy=0.85
- Timing gate: enabled ✓, min_score=0.30, último timing_score

Em `/Users/brown/Documents/MLGeral/crypto_v2/crypto-trading-dashboard/`:
- Atualizar components/signals.py para mostrar novos gates
- Mostrar timing_score e regime_gate no histórico

## Passo 9 — Node Kedro para backtesting (dual-path)
ALÉM do paper trader, criar node para backtesting:

Arquivo: `src/crypto_mkt_state/pipelines/modeling/control_layer/nodes.py`

```python
def apply_control_layer_batch(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Versão batch da control layer para backtesting via Kedro.
    Mesma lógica do paper trader mas opera no DataFrame inteiro.
    Requer colunas: allocation_raw, r11_prob_bull, r11_entropy,
                    rsi_4h, bb_percent_b, volume_zscore
    """
    df = df.copy()
    cl = params.get("control_layer", {})
    
    # Regime gate
    rg = cl.get("regime_gate", {})
    if rg.get("enabled", False):
        df["regime_gate"] = (
            (df["r11_prob_bull"] >= rg.get("min_prob_bull", 0.30)) &
            (df["r11_entropy"] <= rg.get("max_entropy", 0.85))
        )
        df["allocation_after_regime"] = np.where(
            df["regime_gate"], df["allocation_raw"], 0.0
        )
    else:
        df["allocation_after_regime"] = df["allocation_raw"]
    
    # Timing gate
    tg = cl.get("timing_gate", {})
    if tg.get("enabled", False):
        w = tg.get("weights", {"rsi": 0.4, "bollinger": 0.3, "volume": 0.3})
        rsi_t = tg.get("rsi_threshold", 40)
        bb_t = tg.get("bb_threshold", 0.30)
        
        df["timing_score"] = (
            w["rsi"] * np.maximum(0, (rsi_t - df["rsi_4h"].fillna(50)) / rsi_t) +
            w["bollinger"] * np.maximum(0, (bb_t - df["bb_percent_b"].fillna(0.5)) / bb_t) +
            w["volume"] * np.maximum(0, np.minimum(df["volume_zscore"].fillna(0) / 2, 1))
        )
        df["timing_gate"] = df["timing_score"] >= tg.get("min_score", 0.30)
        df["allocation_final"] = np.where(
            df["timing_gate"], df["allocation_after_regime"], 0.0
        )
    else:
        df["allocation_final"] = df["allocation_after_regime"]
    
    # NaN safety
    df["allocation_final"] = df["allocation_final"].fillna(0.0)
    
    return df
```

Registrar pipeline: `modeling.control_layer`
Catalog: `specialist_4h_controlled_signals` (parquet em data/05_model_output/)

## Restrições
- NÃO alterar o modelo LightGBM AB
- NÃO alterar lógica de cálculo de allocation_raw
- Gates NUNCA aumentam alocação, só reduzem ou zeram
- Gates de regime e timing só afetam NOVAS entradas (position_btc == 0)
- Se já em posição, deixar stop loss/gain cuidar da saída
- NaN em qualquer indicador → bloqueia por segurança
- Todos os thresholds configuráveis via config.json e parameters.yml
- Manter backward compatibility de signals.csv

## Validação
1. Replay do trade perdedor:
   Simular context: r11_prob_bull=0.087, r11_entropy=0.426, rsi_4h=57.5, bb_pct_b=0.59, volume_z=-1.67
   - Regime gate: 0.087 < 0.30 → BLOQUEADO ✓
   - (timing gate nem executa porque regime já zerou)
   - allocation_final = 0.0 ✓

2. Simular contexto favorável:
   context: r11_prob_bull=0.85, r11_entropy=0.15, rsi_4h=32, bb_pct_b=0.15, volume_z=1.5
   - Regime gate: 0.85 > 0.30, 0.15 < 0.85 → PASSA ✓
   - Timing: 0.4*(8/40) + 0.3*(0.15/0.3) + 0.3*(0.75) = 0.08+0.15+0.225 = 0.455 > 0.30 → PASSA ✓
   - allocation_final mantém allocation_raw ✓

3. Posição aberta NÃO é afetada pelos gates (só stop loss/gain)

4. signals.csv tem novas colunas: regime_gate_triggered, timing_gate_score, rsi_4h, bb_pct_b

5. config.json tem parâmetros dos dois gates
