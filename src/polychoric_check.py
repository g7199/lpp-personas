"""Robustness of the correlation error to the correlation measure: polychoric (ordinal) instead of Pearson correlations, and the
item-count-normalized error corr_F / sqrt(#pairs). CEK test half, party-crosstab calibration, Qwen and Gemma. Simulated samples:
10 independent answers per persona from the post-processed distributions. Usage: python polychoric_check.py MODEL"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import os, sys, json, numpy as np, pandas as pd
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize_scalar
from multiprocessing import Pool
MODEL = sys.argv[1]; os.environ['OTP'] = '1'; sys.argv = ['x', 'cg', MODEL]
exec(open(os.path.join(HERE, 'final_eval.py')).read().split("sids = list(He.index)")[0])
sids = list(He.index); SELb = SEL['baselines']; TS = SEL['sharp_T']
def thr(x, K=7):
    c = np.cumsum(np.bincount(x.astype(int) - 1, minlength=K)[:K]) / len(x); c = np.clip(c[:-1], 1e-6, 1 - 1e-6)
    return np.concatenate([[-8.0], norm.ppf(c), [8.0]])
def poly_pair(args):
    x, y = args; a, b = thr(x), thr(y); T = np.zeros((7, 7)); np.add.at(T, (x.astype(int) - 1, y.astype(int) - 1), 1)
    grid = np.array([[i, j] for i in a for j in b])
    def nll(r):
        F = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf(grid).reshape(8, 8)
        Pc = F[1:, 1:] - F[:-1, 1:] - F[1:, :-1] + F[:-1, :-1]; return -(T * np.log(np.clip(Pc, 1e-12, None))).sum()
    return minimize_scalar(nll, bounds=(-0.99, 0.99), method='bounded', options={'xatol': 1e-3}).x
def polymat(X, pool):
    J = X.shape[1]; pairs = [(i, j) for i in range(J) for j in range(i + 1, J)]
    r = pool.map(poly_pair, [(X[:, i], X[:, j]) for i, j in pairs]); C = np.eye(J)
    for (i, j), v in zip(pairs, r): C[i, j] = C[j, i] = v
    return C
rng = np.random.default_rng(5)
def draws(P, k=10):
    Q = np.repeat(P, k, axis=0); U = rng.random(Q.shape[:2] + (1,)); return np.minimum((U > np.cumsum(Q, -1)).sum(-1) + 1.0, 7)
P0 = {v: loadP(files[v], sids) for v in ('BASE', 'DIVERSE', 'ARGYLE', 'BACKSTORY', 'ANTHOLOGY', SEL['persona']) if v in files}
def postp(v, spec):
    P = P0[v]; s = spec[len(v) + 1:]
    if s == 'OT': return ot(P)
    T = float(s.replace('temp', '').replace('sharp', '').replace('+OT', '')); return ot(temp(P, T))
M = {'LPP': postp(SEL['persona'], SEL['method'])}
for b, spec in SELb.items():
    if b in P0: M[b] = postp(b, spec)
iu = np.triu_indices(J, 1); npair = len(iu[0])
with Pool(30) as pool:
    Ch = polymat(He.to_numpy(), pool); Cd = polymat(Hc.to_numpy(), pool)
    out = {'human floor (dev vs test)': (np.sqrt(((Cd - Ch) ** 2)[iu].sum()), np.sqrt(((np.corrcoef(Hc.to_numpy().T) - np.corrcoef(He.to_numpy().T)) ** 2)[iu].sum()))}
    for nm, P in M.items():
        X = draws(P); Cp = polymat(X, pool)
        out[nm] = (np.sqrt(((Cp - Ch) ** 2)[iu].sum()), np.sqrt(((np.corrcoef(X.T) - np.corrcoef(He.to_numpy().T)) ** 2)[iu].sum()))
D = pd.DataFrame(out, index=['corrF_polychoric', 'corrF_pearson']).T; D['pearson_normalized'] = D.corrF_pearson / np.sqrt(npair)
D['rank_poly'] = D.corrF_polychoric.rank(); D['rank_pearson'] = D.corrF_pearson.rank()
print(D.round(3).to_string()); D.to_csv(R + f'polychoric_{MODEL}.csv')
