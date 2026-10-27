"""Reliability-structure frontier without calibration (CEK test half): every persona type under the temperature/sharpening grid,
answers drawn from the uncalibrated distributions. Tests whether the matched-reliability result depends on calibration re-injecting
noise. LPP uses all available latent draws (one per half-sample). Usage: python nocal_check.py MODEL"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import os, sys, re, json, numpy as np, pandas as pd
MODEL = sys.argv[1]; os.environ['OTP'] = '1'; sys.argv = ['x', 'cg', MODEL]
exec(open(os.path.join(HERE, 'final_eval.py')).read().split("sids = list(He.index)")[0])
sids = list(He.index); Y = He.to_numpy(); py = pe.to_numpy()
HRT = np.mean([m.loc[sids, [COL[k], COL[k].replace('_w1', '_w2')]].astype(float).dropna().corr().iloc[0, 1] for k in KEYS])
def retest(P):
    x = np.arange(1, 8); E = P @ x; V = P @ x ** 2 - E ** 2; return np.mean([E[:, j].var() / (E[:, j].var() + V[:, j].mean()) for j in range(J)])
DB = SEL['diverse_best'].split('+')[0]; TYPES = ['BASE', 'DIVERSE', DB, 'ARGYLE', 'BACKSTORY', 'ANTHOLOGY', SEL['persona']]
seeds = [SEL['persona']] + [f"{SEL['persona']}_seed{k}" for k in range(2, 7)]
rng = np.random.default_rng(0); n = len(sids); h = n // 2; rows = []
for v in TYPES:
    Ps = [loadP(files[k], sids) for k in (seeds if v == SEL['persona'] else [v]) if k in files]
    for T in (3.0, 2.0, 1.5, 1.0, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.01):
        r = []
        for rep in range(60):
            P = temp(Ps[rep % len(Ps)], T) if T != 1.0 else Ps[rep % len(Ps)]; idx = rng.choice(n, h, replace=False)
            U = rng.random((h, J, 1)); X = np.minimum((U > np.cumsum(P[idx], -1)).sum(-1) + 1.0, 7)
            r.append(dict(corrF=metrics(X, py[idx], Y[idx], py[idx])['corrF'], rel=retest(P[idx]) / HRT))
        d = pd.DataFrame(r).mean(); rows.append(dict(persona=v, T=T, corrF=d.corrF, rel=d.rel))
D = pd.DataFrame(rows); D.to_csv(R + f'nocal_{MODEL}.csv', index=False)
def at(v, rt=0.9):
    d = D[D.persona == v].sort_values('rel'); x, y = d.rel.to_numpy(), d.corrF.to_numpy()
    if x.max() < rt: return None
    i = int(np.argmax(x >= rt)); return y[0] if i == 0 else y[i - 1] + (y[i] - y[i - 1]) * (rt - x[i - 1]) / (x[i] - x[i - 1])
print({v: (None if at(v) is None else round(at(v), 3), round(D[D.persona == v].rel.max(), 3)) for v in TYPES})
