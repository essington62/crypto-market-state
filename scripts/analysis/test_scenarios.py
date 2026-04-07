import pandas as pd
import numpy as np

analysis = pd.read_csv('/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/06_reports/cross_analysis_l1_l2_l3.csv')

print('=' * 70)
print('TESTE DE CENARIOS: ONDE ESTA O EDGE?')
print('=' * 70)

n = len(analysis)

# CENARIO 0: Buy & Hold
ret_total = (analysis['btc_price'].iloc[-1] / analysis['btc_price'].iloc[0] - 1) * 100
print(f'\n-- CENARIO 0: BUY & HOLD --')
print(f'Retorno total: {ret_total:+.2f}%')

# CENARIO 1: Entrar SEMPRE
print(f'\n-- CENARIO 1: ENTRAR SEMPRE --')
print(f'Ret medio 4h: {analysis["ret_4h"].mean():+.3f}%')
print(f'Ret medio 12h: {analysis["ret_12h"].mean():+.3f}%')
print(f'Win rate 4h: {(analysis["ret_4h"]>0).mean()*100:.0f}%')
print(f'Trades: {n}')

# CENARIO 2: So RSI < 40
c2 = analysis[analysis['rsi_1h'] < 40]
print(f'\n-- CENARIO 2: SO RSI < 40 --')
print(f'Trades: {len(c2)} ({len(c2)/n*100:.0f}%)')
if len(c2) > 0:
    print(f'Ret 4h: {c2["ret_4h"].mean():+.3f}% | 12h: {c2["ret_12h"].mean():+.3f}% | 24h: {c2["ret_24h"].mean():+.3f}%')
    print(f'Win 4h: {(c2["ret_4h"]>0).mean()*100:.0f}% | Win 12h: {(c2["ret_12h"]>0).mean()*100:.0f}%')

# CENARIO 3: RSI < 40 E BB < 0.40
c3 = analysis[(analysis['rsi_1h'] < 40) & (analysis['bb_1h'] < 0.40)]
print(f'\n-- CENARIO 3: RSI < 40 + BB < 0.40 --')
print(f'Trades: {len(c3)} ({len(c3)/n*100:.0f}%)')
if len(c3) > 0:
    print(f'Ret 4h: {c3["ret_4h"].mean():+.3f}% | 12h: {c3["ret_12h"].mean():+.3f}% | 24h: {c3["ret_24h"].mean():+.3f}%')
    print(f'Win 4h: {(c3["ret_4h"]>0).mean()*100:.0f}% | Win 12h: {(c3["ret_12h"]>0).mean()*100:.0f}%')

# CENARIO 4: RSI < 45 + momentum up
analysis['prev_ret'] = analysis['ret_4h'].shift(1)
c4 = analysis[(analysis['rsi_1h'] < 45) & (analysis['prev_ret'] > 0)]
print(f'\n-- CENARIO 4: RSI < 45 + MOMENTUM UP (prev ret > 0) --')
print(f'Trades: {len(c4)} ({len(c4)/n*100:.0f}%)')
if len(c4) > 0:
    print(f'Ret 4h: {c4["ret_4h"].mean():+.3f}% | 12h: {c4["ret_12h"].mean():+.3f}% | 24h: {c4["ret_24h"].mean():+.3f}%')
    print(f'Win 4h: {(c4["ret_4h"]>0).mean()*100:.0f}% | Win 12h: {(c4["ret_12h"]>0).mean()*100:.0f}%')

# CENARIO 5: BB < 0.30
c5 = analysis[analysis['bb_1h'] < 0.30]
print(f'\n-- CENARIO 5: SO BB < 0.30 --')
print(f'Trades: {len(c5)} ({len(c5)/n*100:.0f}%)')
if len(c5) > 0:
    print(f'Ret 4h: {c5["ret_4h"].mean():+.3f}% | 12h: {c5["ret_12h"].mean():+.3f}% | 24h: {c5["ret_24h"].mean():+.3f}%')
    print(f'Win 4h: {(c5["ret_4h"]>0).mean()*100:.0f}% | Win 12h: {(c5["ret_12h"]>0).mean()*100:.0f}%')

# CENARIO 6: BB < 0.30 + RSI subindo
analysis['rsi_prev'] = analysis['rsi_1h'].shift(1)
c6 = analysis[(analysis['bb_1h'] < 0.30) & (analysis['rsi_1h'] > analysis['rsi_prev'])]
print(f'\n-- CENARIO 6: BB < 0.30 + RSI SUBINDO (reversal confirm) --')
print(f'Trades: {len(c6)} ({len(c6)/n*100:.0f}%)')
if len(c6) > 0:
    print(f'Ret 4h: {c6["ret_4h"].mean():+.3f}% | 12h: {c6["ret_12h"].mean():+.3f}% | 24h: {c6["ret_24h"].mean():+.3f}%')
    print(f'Win 4h: {(c6["ret_4h"]>0).mean()*100:.0f}% | Win 12h: {(c6["ret_12h"]>0).mean()*100:.0f}%')

# CENARIO 7: Alloc MID + RSI < 50
c7 = analysis[(analysis['alloc_raw'] >= 0.50) & (analysis['alloc_raw'] <= 0.54) & (analysis['rsi_1h'] < 50)]
print(f'\n-- CENARIO 7: ALLOC MID (0.50-0.54) + RSI < 50 --')
print(f'Trades: {len(c7)} ({len(c7)/n*100:.0f}%)')
if len(c7) > 0:
    print(f'Ret 4h: {c7["ret_4h"].mean():+.3f}% | 12h: {c7["ret_12h"].mean():+.3f}% | 24h: {c7["ret_24h"].mean():+.3f}%')
    print(f'Win 4h: {(c7["ret_4h"]>0).mean()*100:.0f}% | Win 12h: {(c7["ret_12h"]>0).mean()*100:.0f}%')

# CENARIO 8: RSI < 35 e nao em queda livre
c8 = analysis[(analysis['rsi_1h'] < 35) & (analysis['prev_ret'] > -1.0)]
print(f'\n-- CENARIO 8: RSI < 35 + NAO QUEDA LIVRE (prev > -1%) --')
print(f'Trades: {len(c8)} ({len(c8)/n*100:.0f}%)')
if len(c8) > 0:
    print(f'Ret 4h: {c8["ret_4h"].mean():+.3f}% | 12h: {c8["ret_12h"].mean():+.3f}% | 24h: {c8["ret_24h"].mean():+.3f}%')
    print(f'Win 4h: {(c8["ret_4h"]>0).mean()*100:.0f}% | Win 12h: {(c8["ret_12h"]>0).mean()*100:.0f}%')

# CENARIO 9: Evitar topo (RSI <= 60)
c9 = analysis[analysis['rsi_1h'] <= 60]
print(f'\n-- CENARIO 9: EVITAR TOPO (RSI <= 60) --')
print(f'Trades: {len(c9)} ({len(c9)/n*100:.0f}%)')
if len(c9) > 0:
    print(f'Ret 4h: {c9["ret_4h"].mean():+.3f}% | 12h: {c9["ret_12h"].mean():+.3f}% | 24h: {c9["ret_24h"].mean():+.3f}%')
    print(f'Win 4h: {(c9["ret_4h"]>0).mean()*100:.0f}% | Win 12h: {(c9["ret_12h"]>0).mean()*100:.0f}%')

# CENARIO 10: RSI 30-45 (oversold mas nao extremo)
c10 = analysis[(analysis['rsi_1h'] >= 30) & (analysis['rsi_1h'] <= 45)]
print(f'\n-- CENARIO 10: RSI 30-45 (oversold moderado) --')
print(f'Trades: {len(c10)} ({len(c10)/n*100:.0f}%)')
if len(c10) > 0:
    print(f'Ret 4h: {c10["ret_4h"].mean():+.3f}% | 12h: {c10["ret_12h"].mean():+.3f}% | 24h: {c10["ret_24h"].mean():+.3f}%')
    print(f'Win 4h: {(c10["ret_4h"]>0).mean()*100:.0f}% | Win 12h: {(c10["ret_12h"]>0).mean()*100:.0f}%')

# RANKING FINAL
print(f'\n{"=" * 70}')
print('RANKING DOS CENARIOS (por ret_12h)')
print(f'{"=" * 70}')

scenarios = [
    ('0-Buy&Hold', 1, ret_total/13, ret_total),
    ('1-Sempre', n, analysis['ret_4h'].mean(), analysis['ret_12h'].mean()),
    ('2-RSI<40', len(c2), c2['ret_4h'].mean() if len(c2)>0 else 0, c2['ret_12h'].mean() if len(c2)>0 else 0),
    ('3-RSI<40+BB<0.4', len(c3), c3['ret_4h'].mean() if len(c3)>0 else 0, c3['ret_12h'].mean() if len(c3)>0 else 0),
    ('4-RSI<45+MomUp', len(c4), c4['ret_4h'].mean() if len(c4)>0 else 0, c4['ret_12h'].mean() if len(c4)>0 else 0),
    ('5-BB<0.30', len(c5), c5['ret_4h'].mean() if len(c5)>0 else 0, c5['ret_12h'].mean() if len(c5)>0 else 0),
    ('6-BB<0.3+RSIup', len(c6), c6['ret_4h'].mean() if len(c6)>0 else 0, c6['ret_12h'].mean() if len(c6)>0 else 0),
    ('7-AllocMid+RSI<50', len(c7), c7['ret_4h'].mean() if len(c7)>0 else 0, c7['ret_12h'].mean() if len(c7)>0 else 0),
    ('8-RSI<35+estavel', len(c8), c8['ret_4h'].mean() if len(c8)>0 else 0, c8['ret_12h'].mean() if len(c8)>0 else 0),
    ('9-Evitar topo', len(c9), c9['ret_4h'].mean() if len(c9)>0 else 0, c9['ret_12h'].mean() if len(c9)>0 else 0),
    ('10-RSI 30-45', len(c10), c10['ret_4h'].mean() if len(c10)>0 else 0, c10['ret_12h'].mean() if len(c10)>0 else 0),
]
scenarios.sort(key=lambda x: x[3], reverse=True)
for name, trades, r4, r12 in scenarios:
    print(f'  {name:25s} | trades={trades:3d} | ret_4h={r4:+.3f}% | ret_12h={r12:+.3f}%')
