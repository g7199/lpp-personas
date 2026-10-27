"""Final evaluation on the test halves. Usage: python final_eval.py DATASET(cg|gss) MODEL [NREP]
Target humans = EVAL half. Simulated personas = the same EVAL respondents' demographics (+ seeded latents). Aggregate information
allowed to the simulator = the CAL half's per-item marginals only (used by OT and the reweighting baseline; temperature T* fixed on CAL).
Uncertainty: 200 random half-samples of respondents (paired across methods and with humans), one sampled answer per persona per draw;
intervals are 2.5/97.5 percentiles of half-samples (conservative). Floor = same metric between a human CAL half-sample and the human
EVAL half-sample (an independent human sample of the same size)."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, os, re, glob, json, numpy as np, pandas as pd, pyreadr
DS, MODEL = sys.argv[1], sys.argv[2]; NREP = int(sys.argv[3]) if len(sys.argv) > 3 else 200
R = RES; SEL = json.load(open(R + 'selected_cfg.json'))
if DS == 'cg':
    from variants import ITEMS
    KEYS = [k for k, *_ in ITEMS]; DOMOF = {k: d for k, d, *_ in ITEMS}; KK = {k: 7 for k in KEYS}
    src = open(os.path.join(HERE, 'cek_items.py')).read(); g = {}; exec(src[src.index('COL = '):src.index('PID = ')], g); COL = g['COL']
    FLIP = {'wall_main', 'flat_main', 'climate_3', 'climate_4', 'amtrak_3', 'vets_3', 'flat_3', 'flat_4'}
    m = pyreadr.read_r(DATA + 'cg/study_1_mturk_cleaned.rds')[None].set_index('subject_id'); split = json.load(open(RES + 'exp2_split.json'))
    def human(ids):
        H = m.loc[ids, [COL[k] for k in KEYS]].astype(float); H.columns = KEYS
        for k in FLIP: H[k] = 8 - H[k]
        H = H.dropna(); return H, m.loc[H.index, 'pid_7'].astype(float) - 1  # pid on 0..6
    Hc, pc = human(split['cal']); He, pe = human(split['eval'])
    files = {f.split('_n311_')[1][:-4]: f for f in glob.glob(R + f'var_{MODEL}_eval_n311_*.csv') if not f.endswith('_para.csv')}
elif DS == 'hm':
    from hm_common import load as hload, OUT
    KEYS = OUT; DOMOF = {o: o for o in OUT}; KK = {o: 7 for o in OUT}
    w = hload(); c = w[w.arm == 'control'].dropna(subset=['pid_7', 'ideo_4', 'age_5', 'race_4', 'educ_5', 'female'] + [f'Y_{o}' for o in OUT]).copy()
    c.index = c.anon_ID.astype(str); H = pd.DataFrame({o: 8 - c[f'Y_{o}'].astype(float) for o in OUT}); pid = c.pid_7.astype(float) - 1
    sp = np.where(np.random.default_rng(7).permutation(len(c)) % 2 == 0, 'cal', 'eval'); Hc, pc, He, pe = H[sp == 'cal'], pid[sp == 'cal'], H[sp == 'eval'], pid[sp == 'eval']
    files = {f.split(f'hmstat_{MODEL}_')[1][:-len('_control.csv')]: f for f in glob.glob(R + f'hmstat_{MODEL}_*_control.csv')}
else:
    from gss_common import load, KEYS, DOMOF, K as KK
    d = load(); A_, B_ = ('eval', 'cal') if os.environ.get('DEVTARGET') == '1' else ('cal', 'eval')  # DEVTARGET=1: score on the CAL half (development use)
    Hc = d[d.split == A_][KEYS].astype(float); pc = d.loc[Hc.index, 'pid7'].astype(float); He = d[d.split == B_][KEYS].astype(float); pe = d.loc[He.index, 'pid7'].astype(float)
    if os.environ.get('CALYEAR'):  # external calibration: tabulations from another GSS wave instead of the 2024 development half
        dx = load(int(os.environ['CALYEAR'])); Hc = dx[KEYS].astype(float); pc = dx['pid7'].astype(float)
    files = {f.split(f'gss_{MODEL}_')[1][:-4]: f for f in glob.glob(R + f'gss_{MODEL}_*.csv')}
J = len(KEYS); SD = He.to_numpy().std(0)
pairs = [(a, b) for a in range(J) for b in range(a + 1, J)]; CR = [(a, b) for a, b in pairs if DOMOF[KEYS[a]] != DOMOF[KEYS[b]]]; WR = [(a, b) for a, b in pairs if DOMOF[KEYS[a]] == DOMOF[KEYS[b]]]
def party(p): return np.where(p <= 2, 'D', np.where((p >= 4) & (p <= 6), 'R', 'I'))
def edist(X, Y):
    X, Y = X / SD, Y / SD; d = lambda A, B: np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)); n, k = len(X), len(Y)
    return 2 * d(X, Y).mean() - d(X, X).sum() / (n * (n - 1)) - d(Y, Y).sum() / (k * (k - 1))
def metrics(X, px, Y, py):
    with np.errstate(invalid='ignore', divide='ignore'): Cx, Cy = np.corrcoef(X.T), np.corrcoef(Y.T)  # a constant simulated item gives NaN entries; they are skipped below
    tx, ty = party(px), party(py)
    def comp(Z, p, t, C): return dict(cross=np.nanmean([abs(C[a, b]) for a, b in CR]), within=np.nanmean([abs(C[a, b]) for a, b in WR]) if WR else np.nan,
                                      wpsd=np.mean([(Z[t == q] / SD).std(0).mean() for q in 'DR']), gap=np.mean(np.abs(Z[t == 'R'].mean(0) - Z[t == 'D'].mean(0)) / SD),
                                      pidr2=np.nanmean([np.corrcoef(p, Z[:, j])[0, 1] ** 2 if Z[:, j].std() > 0 else np.nan for j in range(J)]))
    a, b = comp(X, px, tx, Cx), comp(Y, py, ty, Cy)
    return dict(ED=edist(X, Y), ED_party=np.mean([edist(X[tx == q], Y[ty == q]) for q in 'DR']), corrF=np.sqrt(np.nansum(((Cx - Cy) ** 2)[np.triu_indices(J, 1)])),
                bias=np.mean(np.abs(X.mean(0) - Y.mean(0)) / SD), **{f'{k}_ratio': a[k] / b[k] for k in a})
def loadP(f, sids):
    t = pd.read_csv(f); Pm = {(r.sid, r.key): np.array(json.loads(r.p)) for r in t.itertuples()}
    P = np.zeros((len(sids), J, 7))
    for i, s in enumerate(sids):
        for j, k in enumerate(KEYS): q = Pm[(s, k)][:KK[k]]; P[i, j, :KK[k]] = q / q.sum()
    return P
def marg(H): return np.stack([np.bincount(np.clip(np.round(H[:, j]), 1, KK[k]).astype(int) - 1, minlength=7)[:7] / len(H) for j, k in enumerate(KEYS)])
MC = marg(Hc.to_numpy())  # CAL aggregate marginals: the only human answer information given to any method
def ot_map(pL, pH):
    M = np.zeros((7, 7)); a = pL.copy(); b = pH.copy(); i = j = 0
    while i < 7 and j < 7:
        t = min(a[i], b[j]); M[i, j] += t; a[i] -= t; b[j] -= t
        if a[i] <= 1e-12: i += 1
        if j < 7 and b[j] <= 1e-12: j += 1
    rs = M.sum(1, keepdims=True); return np.where(rs > 0, M / np.maximum(rs, 1e-12), np.eye(7))
OTP = os.environ.get('OTP') == '1'  # party-conditional OT (per item x party group, to the CAL half's published-crosstab marginals)
def _ot(P, MCq): return np.einsum('njk,jkl->njl', P, np.stack([ot_map(P[:, j].mean(0) / P[:, j].mean(0).sum(), MCq[j]) for j in range(J)]))
def ot(P):
    if isinstance(P, Mix): return Mix(ot(x) for x in P)
    if not OTP: return _ot(P, MC)
    Q = np.empty_like(P); g = party(pe.to_numpy()); gc = party(pc.to_numpy())
    for q in 'DIR':
        if (g == q).any(): Q[g == q] = _ot(P[g == q], marg(Hc.to_numpy()[gc == q]))
    return Q
class Mix(list): """LPP over several independent latent draws: one draw is used per half-sample (rep % n), so seed variance enters every interval"""
def temp(P, T):
    if isinstance(P, Mix): return Mix(temp(x, T) for x in P)
    Q = (P / np.maximum(P.max(-1, keepdims=True), 1e-300)) ** (1 / T); return Q / Q.sum(-1, keepdims=True)
def reweight(P, iters=3000, lr=0.1, tau=0.05):
    """Population-aligned reweighting baseline (aggregate-only; cf. Hu et al. 2025): simplex weights over personas minimizing
    sum_j KL(CAL marginal_j || weighted simulated marginal_j) + tau * KL(w || uniform), by exponentiated gradient."""
    n = len(P); w = np.ones(n) / n
    for _ in range(iters):
        Q = np.einsum('n,njk->jk', w, P) + 1e-4; grad = -np.einsum('jk,njk->n', MC / Q, P) / J + tau * (np.log(w * n + 1e-300) + 1)
        g = np.clip(grad - grad.mean(), -5, 5); w = w * np.exp(-lr * g); w = np.maximum(w, 1e-12); w /= w.sum()
    return w
sids = list(He.index); Y = He.to_numpy(); py = pe.to_numpy()
HRT = None
if DS == 'cg':  # human test-retest (EVAL control respondents with wave 2)
    HRT = np.mean([m.loc[sids, [COL[k], COL[k].replace('_w1', '_w2')]].astype(float).dropna().corr().iloc[0, 1] for k in KEYS]); print('human retest', round(HRT, 3))
elif DS == 'hm':  # HM follow-up wave: control respondents re-asked the same five items (no treatment in between)
    ce = c[sp == 'eval']; HRT = float(np.mean([ce[[f'Y_{o}', f'Y_{o}_w2']].astype(float).dropna().corr().iloc[0, 1] for o in OUT])); print('human retest (HM wave 2)', round(HRT, 3))
elif DS == 'gss':  # GSS 2006-08 panel, same 12 items (2-year interval: includes true change, so a lower bound on reliability)
    HRT = float(np.mean(list(json.load(open(R + 'gss_panel_retest.json')).values()))); print('human retest (GSS panel 06-08)', round(HRT, 3))
def retest(P):
    x = np.arange(1, 8); E = P @ x; V = P @ x ** 2 - E ** 2; return np.mean([E[:, j].var() / (E[:, j].var() + V[:, j].mean()) for j in range(J)])  # padded options carry zero mass
TS = SEL['sharp_T']; METH = {}
SEEDK = [SEL['persona']] + [f"{SEL['persona']}_seed{k}" for k in range(2, 7)]; NSEED = int(os.environ.get('NSEED', 6))
for v, f in sorted(files.items()):
    P = loadP(f, sids)
    if v == SEL['persona'] and NSEED > 1 and all(k in files for k in SEEDK[:NSEED]): P = Mix([P] + [loadP(files[k], sids) for k in SEEDK[1:NSEED]]); print('LPP averaged over', NSEED, 'latent draws'); METH[f'{v}_seed1+sharp{TS}+OT'] = (ot(temp(P[0], TS)), None)
    METH[v] = (P, None); METH[v + '+OT'] = (ot(P), None); METH[f'{v}+sharp{TS}+OT'] = (ot(temp(P, TS)), None)
for b, spec in SEL['baselines'].items():  # each baseline with its own CAL-best post-processing
    if b not in files: continue
    P = METH[b][0]; post = spec[len(b) + 1:]
    if post.startswith('temp'): METH[spec] = (ot(temp(P, float(post[4:].replace('+OT', '')))), None)
    elif post.startswith('sharp'): METH[spec] = (ot(temp(P, float(post[5:].replace('+OT', '')))), None)
if SEL.get('diverse_best'):  # budget-matched diversity baseline: best of five wordings x post-processing, selected on development data
    b = SEL['diverse_best'].split('+')[0]
    if b in files:
        P = METH[b][0] if b in METH else loadP(files[b], sids); post = SEL['diverse_best'][len(b) + 1:]
        METH['DIVBEST'] = (ot(temp(P, float(re.sub(r'[a-z+A-Z]', '', post.replace('+OT', ''))))) if post != 'OT' else ot(P), None)
FRONT = ['BASE', 'DIVERSE', 'IDEO1D', SEL.get('diverse_best', 'DIVERSE').split('+')[0], 'ARGYLE', 'BACKSTORY', 'ANTHOLOGY', SEL['persona'], 'LPP_k1.6', 'CONV_B_abs', 'CONV_B_seed2', 'CONV_B_engLow', 'CONV_B_engHigh', SEL.get('gpt', {}).get('persona', ''), 'VALUES_LPP_k1.6', 'CONV_B_novalues', 'ENG_VALUES']
for v in FRONT:  # reliability-structure frontier: every persona type under the full post-processing grid
    if v not in files: continue
    for T in (3.0, 2.0, 1.5, 0.5, 0.35, 0.25, 0.1, 0.05, 0.01):
        key = f'{v}+temp{T}+OT' if T > 1 else f'{v}+sharp{T}+OT'
        if key not in METH: METH[key] = (ot(temp(METH[v][0], T)), None)
for _nm, _key, _cpl in (('OFFSET', 'offset_best', True), ('OFFSETU', 'offsetu_best', False)):  # non-LLM controls: demographic distributions tilted by LPP's latents (offsets.py)
    if DS != 'cg' or 'BASE' not in files or not SEL.get(_key): continue
    from offsets import tilt; _b = float(re.match(r'OFFSETU?_b([0-9.]+)', SEL[_key]).group(1)); PO = tilt(METH['BASE'][0], sids, py.astype(float), _b, _cpl)
    METH[_nm] = (PO, None); METH[_nm + '+OT'] = (ot(PO), None)
    for T in (3.0, 2.0, 1.5, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.01): METH[f'{_nm}+temp{T}+OT' if T > 1 else f'{_nm}+sharp{T}+OT'] = (ot(temp(PO, T)), None)
if 'BASE' in files: METH['BASE+OT+reweight'] = (METH['BASE+OT'][0], reweight(METH['BASE+OT'][0]))
# non-LLM control using exactly the calibration information: each simulated person answers every item independently from the reference
# marginal of their party group (party crosstabs) or from the pooled topline -> dependence only through party
_g = party(py); _gc = party(pc.to_numpy())
METH['AGG_SAMPLER'] = (np.stack([marg(Hc.to_numpy()[_gc == q]) if OTP else MC for q in _g]), None)
rng = np.random.default_rng(0); n = len(sids); h = n // 2; Xc = Hc.to_numpy(); pcc = pc.to_numpy()
rows = []
for rep in range(NREP):
    idx = rng.choice(n, h, replace=False); cidx = rng.choice(len(Xc), h, replace=False)
    rows.append(dict(rep=rep, method='HUMAN_FLOOR', retest_ratio=1.0, **metrics(Xc[cidx], pcc[cidx], Y[idx], py[idx])))
    for name, (P, w) in METH.items():
        if isinstance(P, Mix): P = P[rep % len(P)]
        ii = idx if w is None else rng.choice(n, h, replace=True, p=w)  # reweighting: draw personas by weight
        U = rng.random((h, J, 1)); X = np.minimum((U > np.cumsum(P[ii], -1)).sum(-1) + 1.0, np.array([KK[k] for k in KEYS]))
        rows.append(dict(rep=rep, method=name, retest_ratio=retest(P[ii]) / HRT if HRT else np.nan, **metrics(X, py[ii], Y[idx], py[idx])))
D = pd.DataFrame(rows); D['rel'] = np.abs(1 - D.retest_ratio)
fl = D[D.method == 'HUMAN_FLOOR'].set_index('rep').corrF; bb = D[D.method == 'BASE+OT'].set_index('rep')
D['S'] = [r.ED_party / bb.ED_party[r.rep] + (r.corrF - fl[r.rep]) / (bb.corrF[r.rep] - fl[r.rep]) for r in D.itertuples()]
D['S2'] = D.S + (np.abs(1 - D.retest_ratio) / np.abs(1 - D.rep.map(bb.retest_ratio)) if HRT else 0)
D['T2'] = [(r.corrF - fl[r.rep]) / (bb.corrF[r.rep] - fl[r.rep]) + (r.rel / bb.rel[r.rep] if HRT else 0) for r in D.itertuples()]
TAG = ('_otp' if OTP else '') + ('_dev' if os.environ.get('DEVTARGET') == '1' else '') + (f"_cal{os.environ['CALYEAR']}" if os.environ.get('CALYEAR') else ''); D.to_csv(R + f'final_{DS}_{MODEL}{TAG}_reps.csv', index=False)
S = D.groupby('method').agg(['mean', lambda x: x.quantile(.025), lambda x: x.quantile(.975)]); S.columns = [f'{a}_{["m", "lo", "hi"][i % 3]}' for i, (a, _) in enumerate(S.columns)]
S = S.drop(columns=[c for c in S.columns if c.startswith('rep')]).sort_values('S2_m'); S.to_csv(R + f'final_{DS}_{MODEL}{TAG}_summary.csv')
pd.set_option('display.width', 250); print(S[[c for c in S.columns if c.endswith('_m')]].round(3).to_string())
ref = SEL['method']; comps = ['AGG_SAMPLER', 'DIVBEST', 'BASE+OT', 'BASE+OT+reweight', SEL['v1_method']] + list(SEL['baselines'].values())
P0 = SEL['persona']; comps += [f'{P0}+OT', P0, f'BASE+sharp{TS}+OT', f'ENG_VALUES+sharp{TS}+OT', f'LPP_k1.6+sharp{TS}+OT', f'CONV_B_novalues+sharp{TS}+OT', f'VALUES_LPP_k1.6+sharp{TS}+OT', 'VALUES_LPP_k1.6+sharp0.05+OT', f'CONV_B_abs+sharp{TS}+OT', f'CONV_B_placebo+sharp{TS}+OT', f'CONV_B_seed2+sharp{TS}+OT']
for other in comps:
    if other not in METH or ref not in METH: continue
    a, b = D[D.method == ref].set_index('rep'), D[D.method == other].set_index('rep')
    print(f'{ref} vs {other}: ' + '  '.join(f'{k} {(a[k] - b[k]).mean():+.3f} [{(a[k] - b[k]).quantile(.025):+.3f},{(a[k] - b[k]).quantile(.975):+.3f}] P<0 {((a[k] - b[k]) < 0).mean():.3f}' for k in ['S2', 'T2', 'ED_party', 'corrF', 'rel']))
