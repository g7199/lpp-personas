"""Development-half test of engagement-graded sharpening (Zaller 1992: response stability grows with political awareness):
each persona's distribution is sharpened with a temperature that depends on its sampled engagement level, T_e, instead of one T.
Scores every (T_high, T_mid, T_low) on the CEK development half with the selection criterion (mean of T2 under party-crosstab
and S2 under topline calibration, relative to BASE+OT), next to the uniform-T configurations, in one evaluation path.
Usage: python dev_engsharp.py MODEL"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import os, sys, re, itertools, json, numpy as np, pandas as pd
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3.8-27b'
src = open(os.path.join(HERE, 'var_analyze.py')).read()
pre = src[:src.index('rows = []')]; post_defs = src[src.index('def ot_map'):src.index('for v in list(loaded):')]
sys.argv = ['x', MODEL, 'cal', '100']; g = {}; exec(pre, g)
from variants import rng_for
SEL = json.load(open(RES + 'selected_cfg.json'))
R = RES
P0, pid0 = g['load'](R + f'var_{MODEL}_cal_n100_BASE.csv'); PL, pidL = g['load'](R + f'var_{MODEL}_cal_n100_{SEL["persona"]}.csv')
sids = sorted(pd.read_csv(R + f'var_{MODEL}_cal_n100_{SEL["persona"]}.csv').sid.unique())
p_eng = {'high': 0.25, 'mid': 0.35, 'low': 0.40}
eng = np.array([rng_for(s, 'lpp').choice(list(p_eng), p=list(p_eng.values())) for s in sids])
def sharp(P, T): Q = P ** (1 / T); return Q / Q.sum(-1, keepdims=True)
def sharp_e(P, Ts): Q = P.copy(); [Q.__setitem__(eng == e, sharp(P[eng == e], Ts[e])) for e in Ts]; return Q
res = {}
for otp in ('1', '0'):
    os.environ['OTP'] = otp; g2 = dict(g); g2['rng'] = np.random.default_rng(0); exec(post_defs, g2)
    ev = lambda name, P, pid: g2['evaluate'](name, P, pid)
    rows = [ev('BASE+OT', g2['ot'](P0, pid0), pid0)]
    for T in (0.1, 0.15, 0.25): rows.append(ev(f'uniform T={T}', g2['ot'](sharp(PL, T), pidL), pidL))
    for th, tm, tl in itertools.product((0.05, 0.1, 0.15), (0.1, 0.15, 0.25), (0.15, 0.25, 0.35, 0.5)):
        if not (th <= tm <= tl): continue
        rows.append(ev(f'graded {th}/{tm}/{tl}', g2['ot'](sharp_e(PL, {'high': th, 'mid': tm, 'low': tl}), pidL), pidL))
    res[otp] = pd.DataFrame(rows).set_index('variant')
FL = float(re.search(r'corrF ([0-9.]+)', [l for l in open(R + 'va_cal_otp.log') if 'floors (human' in l][0]).group(1))
def T2(D, m): b = D.loc['BASE+OT']; return (D.loc[m, 'corrF'] - FL) / (b.corrF - FL) + abs(1 - D.loc[m, 'retest_ratio']) / abs(1 - b.retest_ratio)
def S2(D, m): b = D.loc['BASE+OT']; return D.loc[m, 'ED_party'] / b.ED_party + T2(D, m)
O, Pp = res['1'], res['0']
S = pd.DataFrame({m: dict(score=(T2(O, m) + S2(Pp, m)) / 2, corrF_otp=O.loc[m, 'corrF'], rel=O.loc[m, 'retest_ratio'], corrF_pool=Pp.loc[m, 'corrF']) for m in O.index if m != 'BASE+OT'}).T.sort_values('score')
print(S.round(3).head(12).to_string()); print(S.loc[[i for i in S.index if i.startswith('uniform')]].round(3).to_string())
S.to_csv(R + f'dev_engsharp_{MODEL}.csv')
