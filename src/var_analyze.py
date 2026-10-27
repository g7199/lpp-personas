"""Method-search analysis (DEV = CAL half). For each persona variant (and post-hoc readout baselines applied to BASE/LPP), compare the
simulated population (one draw per persona per replicate, 20 replicates) with human CAL respondents on the 14 items.
Primary: energy distance (ED) on human-SD-standardized items, with a human floor (ED between a random 100-respondent human
subsample and the other human half). Components: cross-domain |r| ratio, within-domain |r| ratio, within-party SD ratio,
party-gap ratio, party-ID R^2, marginal W1, mean |bias|.
Usage: python var_analyze.py MODEL SPLIT N"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, glob, json, numpy as np, pandas as pd, pyreadr
from variants import ITEMS
MODEL, SPLIT, N = sys.argv[1], sys.argv[2], int(sys.argv[3]); NREP = int(sys.argv[4]) if len(sys.argv) > 4 else 20
KEYS = [k for k, *_ in ITEMS]; DOMOF = {k: d for k, d, *_ in ITEMS}
src = open(os.path.join(HERE, 'cek_items.py')).read(); g = {}; exec(src[src.index('COL = '):src.index('PID = ')], g); COL = g['COL']
FLIP = {'wall_main', 'flat_main', 'climate_3', 'climate_4', 'amtrak_3', 'vets_3', 'flat_3', 'flat_4'}
m = pyreadr.read_r(DATA + 'cg/study_1_mturk_cleaned.rds')[None].set_index('subject_id')
split = json.load(open(RES + 'exp2_split.json'))
def human(ids):
    H = m.loc[ids, [COL[k] for k in KEYS]].astype(float); H.columns = KEYS
    for k in FLIP: H[k] = 8 - H[k]
    ok = H.notna().all(1); return H[ok].to_numpy(), m.loc[ids, 'pid_7'].astype(float)[ok].to_numpy()
H, Hp = human(split[SPLIT]); Ho, Hop = human(split['eval' if SPLIT == 'cal' else 'cal']); SD = H.std(0)
pairs = [(a, b) for a in range(len(KEYS)) for b in range(a + 1, len(KEYS))]; CR = [(a, b) for a, b in pairs if DOMOF[KEYS[a]] != DOMOF[KEYS[b]]]; WR = [(a, b) for a, b in pairs if DOMOF[KEYS[a]] == DOMOF[KEYS[b]]]
def party(p): return np.where(p <= 3, 'D', np.where(p >= 5, 'R', 'I'))
def comps(X, pid):
    C = np.corrcoef(X.T); pt = party(pid)
    return dict(cross=np.mean([abs(C[a, b]) for a, b in CR]), within=np.mean([abs(C[a, b]) for a, b in WR]), wpsd=np.mean([X[pt == q].std(0).mean() for q in 'DR']),
                gap=np.mean(np.abs(X[pt == 'R'].mean(0) - X[pt == 'D'].mean(0))), pidr2=np.mean([np.corrcoef(pid, X[:, j])[0, 1] ** 2 for j in range(X.shape[1])]))
def ed(X, Y):
    X, Y = X / SD, Y / SD
    d = lambda A, B: np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    dxy = d(X, Y).mean(); dxx = d(X, X); dyy = d(Y, Y); n, k = len(X), len(Y)
    return 2 * dxy - dxx.sum() / (n * (n - 1)) - dyy.sum() / (k * (k - 1))
def w1(X): return np.mean([np.abs(np.cumsum(np.bincount(X[:, j].astype(int), minlength=8)[1:] / len(X)) - np.cumsum(np.bincount(np.clip(np.round(H[:, j]), 1, 7).astype(int), minlength=8)[1:] / len(H))).sum() for j in range(X.shape[1])])
hc = comps(H, Hp); rng = np.random.default_rng(0)
HRT = np.mean([m.loc[split[SPLIT], [COL[k], COL[k].replace('_w1', '_w2')]].astype(float).dropna().corr().iloc[0, 1] for k in KEYS])
def retest(P):
    x = np.arange(1, 8); E = P @ x; V = P @ x ** 2 - E ** 2; return np.mean([E[:, j].var() / (E[:, j].var() + V[:, j].mean()) for j in range(P.shape[1])])
floor = np.mean([ed(Ho[rng.choice(len(Ho), N, replace=False)], H) for _ in range(20)])
corrF_floor = np.mean([np.sqrt(((np.corrcoef(Ho[i].T) - np.corrcoef(H.T)) ** 2).sum() / 2) for i in [rng.choice(len(Ho), N, replace=False) for _ in range(20)]])
hsub = [comps(Ho[i], Hop[i]) for i in [rng.choice(len(Ho), N, replace=False) for _ in range(20)]]
print(f"HUMAN {SPLIT} retest {HRT:.3f} n={len(H)}: cross {hc['cross']:.3f} within {hc['within']:.3f} wpSD {hc['wpsd']:.3f} gap {hc['gap']:.3f} pidR2 {hc['pidr2']:.3f} | floors (human n={N} subsample of other half): ED {floor:.4f} corrF {corrF_floor:.3f}")
def load(f):
    d = pd.read_csv(f); sids = sorted(d.sid.unique()); Pm = {(r.sid, r.key): np.array(json.loads(r.p)) for r in d.itertuples()}
    P = np.stack([np.stack([Pm[(s, k)] / Pm[(s, k)].sum() for k in KEYS]) for s in sids])
    return P, m.loc[sids, 'pid_7'].astype(float).to_numpy()
# NOTE: simulated answers are indices in the label order shown to the model; human data were mapped to the same label order (8 - x for FLIP items)
def evaluate(name, P, pid):
    reps = []
    for _ in range(NREP):
        U = rng.random(P.shape[:2] + (1,)); X = (U > np.cumsum(P, -1)).sum(-1) + 1.0; X = np.minimum(X, 7)
        c = comps(X, pid); pt = party(pid); hpt = party(Hp)
        corrF = np.sqrt(((np.corrcoef(X.T) - np.corrcoef(H.T)) ** 2).sum() / 2)  # Frobenius error of the correlation matrix (upper triangle)
        edP = np.mean([ed(X[pt == q], H[hpt == q]) for q in 'DR'])  # party-conditional joint distance
        reps.append(dict(ED=ed(X, H), corrF=corrF, ED_party=edP, W1=w1(X), bias=np.mean(np.abs(X.mean(0) - H.mean(0)) / SD), **c))
    R = pd.DataFrame(reps).mean()
    return dict(variant=name, retest_ratio=retest(P) / HRT, ED=R.ED, ED_party=R.ED_party, corrF=R.corrF, cross_ratio=R.cross / hc['cross'], within_ratio=R.within / hc['within'], wpsd_ratio=R.wpsd / hc['wpsd'], gap_ratio=R.gap / hc['gap'], pidr2=R.pidr2, W1=R.W1, bias=R.bias)
rows = []
files = sorted(glob.glob(f'{RES}var_{MODEL}_{SPLIT}_n{N}_*.csv'))
import os as _os
if _os.environ.get('VA_ONLY'): files = [f for f in files if f.split(f'_n{N}_')[1].replace('.csv', '') in _os.environ['VA_ONLY'].split(',')]  # restricted family run
loaded = {f.split(f'_n{N}_')[1].replace('.csv', ''): load(f) for f in files}
for v, (P, pid) in loaded.items(): rows.append(evaluate(v, P, pid))
def ot_map(pL, pH):  # monotone transport matrix moving pooled simulated marginal pL to human marginal pH (row-stochastic)
    M = np.zeros((7, 7)); a = pL.copy(); b = pH.copy(); i = j = 0
    while i < 7 and j < 7:
        t = min(a[i], b[j]); M[i, j] += t; a[i] -= t; b[j] -= t
        if a[i] <= 1e-12: i += 1
        if j < 7 and b[j] <= 1e-12: j += 1
    rs = M.sum(1, keepdims=True); return np.where(rs > 0, M / np.maximum(rs, 1e-12), np.eye(7))
HM_REF = H  # marginals used for OT: the human DEV half itself in development; at final evaluation the CAL half is used and scored on EVAL
def ot_pooled(P): return np.einsum('njk,jkl->njl', P, np.stack([ot_map(P[:, j].mean(0), np.bincount(np.clip(np.round(HM_REF[:, j]), 1, 7).astype(int) - 1, minlength=7) / len(HM_REF)) for j in range(P.shape[1])]))
import os
OTP = os.environ.get('OTP') == '1'  # party-conditional OT: per (item, party group) map to the human group's marginal (published crosstabs)
HP_REF = party(Hp)
def ot(P, pid=None):
    if not OTP or pid is None: return ot_pooled(P)
    g = party(pid); Q = np.empty_like(P)
    for q in 'DIR':
        Hq = HM_REF[HP_REF == q]; idx = g == q
        if idx.sum() == 0: continue
        Q[idx] = np.einsum('njk,jkl->njl', P[idx], np.stack([ot_map(P[idx][:, j].mean(0), np.bincount(np.clip(np.round(Hq[:, j]), 1, 7).astype(int) - 1, minlength=7) / len(Hq)) for j in range(P.shape[1])]))
    return Q
for v in list(loaded):
    P, pid = loaded[v]; rows.append(evaluate(f'{v}+OT', ot(P, pid), pid))
for v in list(loaded):  # sharpening (T<1) + OT: heterogeneity from persons, not from response noise
    P, pid = loaded[v]
    for T in (0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.7):
        Q = P ** (1 / T); Q = Q / Q.sum(-1, keepdims=True); rows.append(evaluate(f'{v}+sharp{T}+OT', ot(Q, pid), pid))
for base in [b for b in ('BASE', 'LPP', 'DIVERSE', 'ARGYLE', 'BACKSTORY', 'ANTHOLOGY', 'VALUES_LPP_k1.6', 'VALUES_LPP') if b in loaded]:  # post-hoc readout baselines (tuned on the same DEV metric)
    P, pid = loaded[base]
    for T in (1.5, 2.0, 3.0, 4.0):
        Q = P ** (1 / T); Q = Q / Q.sum(-1, keepdims=True); rows.append(evaluate(f'{base}+temp{T}', Q, pid)); rows.append(evaluate(f'{base}+temp{T}+OT', ot(Q, pid), pid))
    for a in (0.2, 0.4, 0.6): rows.append(evaluate(f'{base}+unif{a}', (1 - a) * P + a / 7, pid))
R = pd.DataFrame(rows).sort_values('ED'); R.insert(0, 'model', MODEL); R.to_csv(f'{RES}var_summary_{MODEL}_{SPLIT}' + _os.environ.get('VA_TAG', '') + ('_otp' if OTP else '') + '.csv', index=False)
pd.set_option('display.width', 250); print(R.round(3).to_string(index=False))
hs = pd.DataFrame(hsub); print('human n=100 subsample ratios (sampling variability reference): cross %.2f±%.2f  gap %.2f±%.2f  wpSD %.2f±%.2f' % ((hs.cross / hc['cross']).mean(), (hs.cross / hc['cross']).std(), (hs.gap / hc['gap']).mean(), (hs.gap / hc['gap']).std(), (hs.wpsd / hc['wpsd']).mean(), (hs.wpsd / hc['wpsd']).std()))
