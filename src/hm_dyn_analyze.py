"""Dynamic test: Hopkins-Mummolo framing effects for persona populations. For each variant and post-processing
(raw; sharp(T)+OT), the OT map is fit once on the simulated CONTROL arm (to the human CAL-half control marginals) and then applied
unchanged to every framed arm, so post-processing cannot manufacture or remove treatment effects by itself.
Effect matrix = (mean answer in framed arm - mean in control) / human control SD, oriented + = conservative (hm_common.SIGN).
Human matrix and CIs: hm_human.json (hm_human.py) (pooled GfK+MTurk). Bootstrap over simulated personas within arm.
Usage: python hm_dyn_analyze.py MODEL V1,V2,..."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, json, numpy as np, pandas as pd
from hm_common import FR, OUT, SIGN, summ, loo_rms, OFFM, load
MODEL, VS = sys.argv[1], sys.argv[2].split(','); R = RES
SEL = json.load(open(R + 'selected_cfg.json')); TS = SEL['sharp_T']
HJ = json.load(open(RES + 'hm_human.json'))['pooled']; TH = np.array(HJ['T']); SD = np.array(HJ['SD'])
w = load(); c = w[w.arm == 'control'].dropna(subset=['pid_7', 'ideo_4', 'age_5', 'race_4', 'educ_5', 'female'] + [f'Y_{o}' for o in OUT]).copy()
sp = np.where(np.random.default_rng(7).permutation(len(c)) % 2 == 0, 'cal', 'eval'); Hc = np.stack([8 - c[f'Y_{o}'].to_numpy()[sp == 'cal'] for o in OUT], 1)
MC = np.stack([np.bincount(Hc[:, j].astype(int) - 1, minlength=7) / len(Hc) for j in range(5)])
def ot_map(pL, pH):
    M = np.zeros((7, 7)); a = pL.copy(); b = pH.copy(); i = j = 0
    while i < 7 and j < 7:
        t = min(a[i], b[j]); M[i, j] += t; a[i] -= t; b[j] -= t
        if a[i] <= 1e-12: i += 1
        if j < 7 and b[j] <= 1e-12: j += 1
    rs = M.sum(1, keepdims=True); return np.where(rs > 0, M / np.maximum(rs, 1e-12), np.eye(7))
def sharp(P, T): Q = (P / P.max(-1, keepdims=True)) ** (1 / T); return Q / Q.sum(-1, keepdims=True)
def arm(cond, v):
    d = pd.read_csv(R + f'hmstat_{MODEL}_{v}_{cond}.csv'); d['p'] = [np.array(json.loads(p)) for p in d.p]
    return np.stack([np.stack(list(g.set_index('key').loc[OUT, 'p'])) for _, g in d.groupby('sid')])  # (n, 5, 7) answer-index order
x = np.arange(1, 8); sign = np.array([SIGN[o] for o in OUT])
def E(P): return (8 - P @ x) * sign  # expected answer in data coding (7 = increased a great deal), oriented
rng = np.random.default_rng(0); rows = []
print('HUMAN', {k: round(HJ[k], 3) for k in ['direct', 'off', 'rho']}, 'rho CI', round(HJ['rho_lo'], 3), round(HJ['rho_hi'], 3))
for v in VS:
    A0 = {cnd: arm(cnd, v) for cnd in ['control'] + FR}
    for post in ['raw', f'sharp{TS}+OT', 'OT', f"temp{SEL['temp_T']}+OT"]:
        if post == 'raw': A = A0
        else:
            A1 = {k: (sharp(a, TS) if post.startswith('sharp') else sharp(a, SEL['temp_T']) if post.startswith('temp') else a) for k, a in A0.items()}
            M = np.stack([ot_map(A1['control'][:, j].mean(0), MC[j]) for j in range(5)]); A = {k: np.einsum('njk,jkl->njl', a, M) for k, a in A1.items()}
        EA = {k: E(a) for k, a in A.items()}
        def mat(EA): return np.stack([(EA[f].mean(0) - EA['control'].mean(0)) / SD for f in FR])
        T = mat(EA); BB = pd.DataFrame([summ(mat({k: a[rng.integers(0, len(a), len(a))] for k, a in EA.items()})) for _ in range(1000)])
        s = summ(T); rows.append(dict(variant=v, post=post, direct=s['direct'], off=s['off'], rho=s['rho'], rho_lo=BB.rho.quantile(.025), rho_hi=BB.rho.quantile(.975),
                                      direct_gain=s['direct'] / HJ['direct'], off_gain=s['off'] / HJ['off'], r_cells=np.corrcoef(T.ravel(), TH.ravel())[0, 1],
                                      rmse=np.sqrt(((T - TH) ** 2).mean()), rmse_off=np.sqrt(((T - TH)[OFFM] ** 2).mean()), loo=loo_rms(T)))
        if post != 'OT': print(v, post); print(pd.DataFrame(T, index=FR, columns=OUT).round(3))
D = pd.DataFrame(rows); D.to_csv(R + f'hmdyn_{MODEL}.csv', index=False); pd.set_option('display.width', 220); print(D.round(3).to_string(index=False))
print('human matrix'); print(pd.DataFrame(TH, index=FR, columns=OUT).round(3))
