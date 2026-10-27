"""Paired bootstrap of the framing-effect error (RMSE of the 4x5 effect matrix vs. humans, and mean direct effect) between two persona
variants on the same simulated respondents. Persons are resampled within arm with the same indices for both variants.
Usage: python hm_dyn_boot.py MODEL VA VB"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2] + ',' + sys.argv[3]] + sys.argv[4:]
src = open(os.path.join(HERE, 'hm_dyn_analyze.py')).read(); src = src[:src.index('rng = np.random.default_rng(0)')]
exec(src)
VA, VB = VS; rng = np.random.default_rng(1)
def post_arms(v, post):
    A0 = {cnd: arm(cnd, v) for cnd in ['control'] + FR}
    if post == 'raw': return A0
    A1 = {k: (sharp(a, TS) if post.startswith('sharp') else sharp(a, SEL['temp_T']) if post.startswith('temp') else a) for k, a in A0.items()}
    M = np.stack([ot_map(A1['control'][:, j].mean(0), MC[j]) for j in range(5)]); return {k: np.einsum('njk,jkl->njl', a, M) for k, a in A1.items()}
def mat(EA, idx): return np.stack([(EA[f][idx[f]].mean(0) - EA['control'][idx['control']].mean(0)) / SD for f in FR])
for post in ['raw', f'sharp{TS}+OT', f"temp{SEL['temp_T']}+OT"]:
    EA, EB = ({k: E(a) for k, a in post_arms(v, post).items()} for v in (VA, VB))
    full = {k: np.arange(len(a)) for k, a in EA.items()}
    def stat(idx): r = lambda T: np.sqrt(((T - TH) ** 2).mean()); return r(mat(EA, idx)) - r(mat(EB, idx))
    d0 = stat(full); bs = np.array([stat({k: rng.integers(0, len(a), len(a)) for k, a in EA.items()}) for _ in range(1000)])
    print(f'{post}: RMSE({VA}) - RMSE({VB}) = {d0:+.3f}  95% CI [{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}]  P(>0) {np.mean(bs > 0):.3f}')
