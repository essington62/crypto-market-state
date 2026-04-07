import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

analysis = pd.read_csv('/Users/brown/Documents/MLGeral/crypto_v2/crypto-market-state/data/06_reports/cross_analysis_l1_l2_l3.csv')
analysis['prev_ret'] = analysis['ret_4h'].shift(1)

# Target: retorno 12h positivo
analysis['target'] = (analysis['ret_12h'] > 0).astype(int)

# Features
analysis['z_score_bb'] = (analysis['bb_1h'] - 0.5) / 0.25
analysis['rsi_norm'] = (analysis['rsi_1h'] - 50) / 20

features_all = {
    'alloc_raw': analysis['alloc_raw'],
    'rsi_1h': analysis['rsi_1h'],
    'bb_1h': analysis['bb_1h'],
    'rsi_norm': analysis['rsi_norm'],
    'z_score_bb': analysis['z_score_bb'],
    'prev_ret': analysis['prev_ret'],
}

X = pd.DataFrame(features_all).dropna()
y = analysis.loc[X.index, 'target']
ret_12h = analysis.loc[X.index, 'ret_12h']
ret_4h = analysis.loc[X.index, 'ret_4h']

# Split temporal (60/40)
split = int(len(X) * 0.6)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
ret_test_12h = ret_12h.iloc[split:]
ret_test_4h = ret_4h.iloc[split:]

# Escalar
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 1. Logistic Regression
lr = LogisticRegression(C=0.1, random_state=42)
lr.fit(X_train_s, y_train)
y_prob = lr.predict_proba(X_test_s)[:, 1]

auc = roc_auc_score(y_test, y_prob)
print('=' * 60)
print('LOGISTIC REGRESSION - Sideways mean reversion')
print('=' * 60)
print(f'AUC: {auc:.3f}')
print(f'Samples: train={len(X_train)}, test={len(X_test)}')

# Coeficientes
print(f'\nCoeficientes:')
for feat, coef in zip(X.columns, lr.coef_[0]):
    direction = 'BULLISH' if coef > 0 else 'BEARISH'
    bar = '+' * int(abs(coef) * 10)
    print(f'  {feat:15s}: {coef:+.3f} ({direction}) {bar}')

print(f'  intercept      : {lr.intercept_[0]:+.3f}')

# Threshold por percentil
print('\nThreshold analysis:')
for q in [0.4, 0.5, 0.6, 0.7]:
    threshold = np.quantile(y_prob, q)
    entries = y_prob >= threshold
    if entries.sum() > 0:
        ret_e = ret_test_12h[entries]
        win = (ret_e > 0).mean() * 100
        avg_ret = ret_e.mean() * 100
        print(f'  Q{int(q*100)} thr={threshold:.3f}: trades={entries.sum():3d} win_12h={win:.0f}% ret_12h={avg_ret:+.3f}%')

# 2. Regras puras
print(f'\n{"=" * 60}')
print('REGRAS PURAS - Mean Reversion')
print(f'{"=" * 60}')

test_data = analysis.iloc[split:].copy()
n_test = len(test_data)

rules = []

# Regra A: BB < 0.30
ra = test_data[test_data['bb_1h'] < 0.30]
if len(ra) > 0:
    rules.append(('A: BB<0.30', len(ra), (ra['ret_12h']>0).mean()*100, ra['ret_12h'].mean()*100))
    print(f'\nRegra A (BB < 0.30): trades={len(ra)} win_12h={(ra["ret_12h"]>0).mean()*100:.0f}% ret_12h={ra["ret_12h"].mean()*100:+.3f}%')

# Regra B: RSI < 45 + BB < 0.50
rb = test_data[(test_data['rsi_1h'] < 45) & (test_data['bb_1h'] < 0.50)]
if len(rb) > 0:
    rules.append(('B: RSI<45+BB<0.5', len(rb), (rb['ret_12h']>0).mean()*100, rb['ret_12h'].mean()*100))
    print(f'Regra B (RSI<45+BB<0.50): trades={len(rb)} win_12h={(rb["ret_12h"]>0).mean()*100:.0f}% ret_12h={rb["ret_12h"].mean()*100:+.3f}%')

# Regra C7 (baseline)
c7 = test_data[(test_data['alloc_raw'] >= 0.50) & (test_data['alloc_raw'] <= 0.54) & (test_data['rsi_1h'] < 50)]
if len(c7) > 0:
    rules.append(('C7: alloc mid+RSI<50', len(c7), (c7['ret_12h']>0).mean()*100, c7['ret_12h'].mean()*100))
    print(f'Regra C7 (alloc mid+RSI<50): trades={len(c7)} win_12h={(c7["ret_12h"]>0).mean()*100:.0f}% ret_12h={c7["ret_12h"].mean()*100:+.3f}%')

# Regra D: BB < 0.20 (forte desvio)
rd = test_data[test_data['bb_1h'] < 0.20]
if len(rd) > 0:
    rules.append(('D: BB<0.20', len(rd), (rd['ret_12h']>0).mean()*100, rd['ret_12h'].mean()*100))
    print(f'Regra D (BB < 0.20): trades={len(rd)} win_12h={(rd["ret_12h"]>0).mean()*100:.0f}% ret_12h={rd["ret_12h"].mean()*100:+.3f}%')

# Regra E: RSI < 40 + BB < 0.40 + alloc > 0.50
re = test_data[(test_data['rsi_1h'] < 40) & (test_data['bb_1h'] < 0.40) & (test_data['alloc_raw'] > 0.50)]
if len(re) > 0:
    rules.append(('E: RSI<40+BB<0.4+alloc>0.5', len(re), (re['ret_12h']>0).mean()*100, re['ret_12h'].mean()*100))
    print(f'Regra E (RSI<40+BB<0.40+alloc>0.50): trades={len(re)} win_12h={(re["ret_12h"]>0).mean()*100:.0f}% ret_12h={re["ret_12h"].mean()*100:+.3f}%')

# Regra F: BB < 0.35 + RSI subindo
test_data['rsi_prev'] = test_data['rsi_1h'].shift(1)
rf = test_data[(test_data['bb_1h'] < 0.35) & (test_data['rsi_1h'] > test_data['rsi_prev'])]
if len(rf) > 0:
    rules.append(('F: BB<0.35+RSI subindo', len(rf), (rf['ret_12h']>0).mean()*100, rf['ret_12h'].mean()*100))
    print(f'Regra F (BB<0.35+RSI subindo): trades={len(rf)} win_12h={(rf["ret_12h"]>0).mean()*100:.0f}% ret_12h={rf["ret_12h"].mean()*100:+.3f}%')

# Regra G: preço abaixo da MA (alloc qualquer) + RSI < 55
rg = test_data[(test_data['bb_1h'] < 0.50) & (test_data['rsi_1h'] < 55)]
if len(rg) > 0:
    rules.append(('G: BB<0.50+RSI<55', len(rg), (rg['ret_12h']>0).mean()*100, rg['ret_12h'].mean()*100))
    print(f'Regra G (BB<0.50+RSI<55): trades={len(rg)} win_12h={(rg["ret_12h"]>0).mean()*100:.0f}% ret_12h={rg["ret_12h"].mean()*100:+.3f}%')

# 3. LR nos mesmos filtros do C7
print(f'\n{"=" * 60}')
print('LR + C7 COMBINADO')
print(f'{"=" * 60}')

c7_mask = (test_data['alloc_raw'] >= 0.50) & (test_data['alloc_raw'] <= 0.54) & (test_data['rsi_1h'] < 50)
c7_idx = test_data[c7_mask].index
lr_c7 = [i for i in range(len(X_test)) if X_test.index[i] in c7_idx]
if len(lr_c7) > 0:
    lr_probs_c7 = y_prob[lr_c7]
    ret_c7 = ret_test_12h.iloc[lr_c7]
    median_prob = np.median(lr_probs_c7)
    above = lr_probs_c7 >= median_prob
    if above.sum() > 0:
        ret_above = ret_c7[above]
        print(f'C7 trades onde LR prob > mediana ({median_prob:.3f}):')
        print(f'  trades={above.sum()} win_12h={(ret_above>0).mean()*100:.0f}% ret_12h={ret_above.mean()*100:+.3f}%')

# RANKING
print(f'\n{"=" * 60}')
print('RANKING (por ret_12h)')
print(f'{"=" * 60}')
rules.sort(key=lambda x: x[3], reverse=True)
for name, trades, win, ret in rules:
    print(f'  {name:30s} | trades={trades:3d} | win={win:.0f}% | ret_12h={ret:+.3f}%')
