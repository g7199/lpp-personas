"""Human 4x5 effect matrix for the Hopkins-Mummolo replications (per sample and pooled), control-SD units, bootstrap within arm."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, json, numpy as np, pandas as pd
from hm_common import load, FR, OUT, summ, loo_rms, OFFM
w = load(); rng = np.random.default_rng(0); res = {}
def mat(df, SD):
    c = df[df.arm == 'control'][OUT].mean()
    return np.stack([(df[df.arm == f][OUT].mean() - c).to_numpy() / SD for f in FR])
for s in ['gfk', 'mt', 'pooled']:
    df = w if s == 'pooled' else w[w['sample'] == s]; SD = df[df.arm == 'control'][OUT].std().to_numpy(); T = mat(df, SD)
    arms = {a: g for a, g in df.groupby('arm')}; B = []
    for _ in range(2000):
        bd = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in arms.values()]); Tb = mat(bd, SD); B.append({**summ(Tb), 'loo': loo_rms(Tb), 'T': Tb})
    BB = pd.DataFrame([{k: v for k, v in b.items() if k != 'T'} for b in B]); Ts = np.stack([b['T'] for b in B])
    se = Ts.std(0); z = T / se; from scipy.stats import norm; p = 2 * norm.sf(np.abs(z))
    po = p[OFFM]; o = np.sort(po); m = len(o); bh = (o <= 0.05 * np.arange(1, m + 1) / m); nd = int(np.max(np.where(bh)[0]) + 1) if bh.any() else 0
    s0 = summ(T); res[s] = dict(T=T.round(4).tolist(), SE=se.round(4).tolist(), SD=SD.round(4).tolist(), n=df.arm.value_counts().to_dict(), loo=loo_rms(T), n_declared_off=nd, MDE80=float(2.8 * np.median(se[OFFM])),
                            **s0, **{f'{k}_lo': float(BB[k].quantile(.025)) for k in ['direct', 'off', 'rho', 'loo']}, **{f'{k}_hi': float(BB[k].quantile(.975)) for k in ['direct', 'off', 'rho', 'loo']})
    print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[s].items() if k not in ('T', 'SE', 'SD')})
    print(pd.DataFrame(T, index=FR, columns=OUT).round(3))
json.dump(res, open(RES + 'hm_human.json', 'w'), indent=1)
