"""Static-fidelity transfer to the Hopkins-Mummolo control arm (GfK probability panel + MTurk, 1,284 complete controls; 5 federal
spending items, one per domain). Persona settings frozen from the CEK development choice; next-token probabilities over the 7 options.
Usage: python hm_static_run.py MODEL DEVICE V1,V2,...   (also runs framed conditions with suffix @frame, e.g. BASE@crime, for the dynamic test)"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, os, json, zlib, time, numpy as np, pandas as pd
from llm import Scorer
from hm_common import TEMPL, OUT, load, context, qblock
from variants import ENG, VALUES, VLEV, MORAL, bins
MODEL, DEV, VS = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
R = RES
SEL = json.load(open(R + 'selected_cfg.json')) if os.path.exists(R + 'selected_cfg.json') else {}
DTXT = {'crime': 'crime and policing', 'terror': 'national security and terrorism', 'poor': 'government help for the poor', 'health': 'health care policy', 'stimulus': 'government spending to boost the economy'}
HV = {'BASE': {}, 'DIVERSE': {'diverse': True}, 'LPP': {'eng': True, 'dev': True}, 'ENG_VALUES': {'eng': True, 'values': True}, 'VALUES_LPP': {'values': True, 'eng': True, 'dev': True},
      'LPPH': {'eng': True, 'dev': True, 'doms': ['crime', 'terror', 'poor']}, 'ENG_MORAL': {'eng': True, 'moral': True}, 'ENG_VALUES_MORAL': {'eng': True, 'values': True, 'moral': True},
      'VALUES_LPP_k1.6': {'values': True, 'eng': True, 'dev': True, 'dev_k': 1.6}, 'LPP_k1.6': {'eng': True, 'dev': True, 'dev_k': 1.6},
      'CONV_B': {'values': True, 'eng': True, 'dev': True, 'k_by_eng': {'high': None, 'mid': 1.6, 'low': 0.6}}}
if SEL.get('cfg'): HV['SELECTED'] = SEL['cfg']
from variants import DIVERSE_TEXTS, VARIANTS as _V; HV.update({k: {'diverse_text': t} for k, t in DIVERSE_TEXTS.items()}); HV.update({k: _V[k] for k in ('CONV_B_novalues', 'CONV_B_seed2', 'CONV_B_seed3', 'CONV_B_seed4', 'CONV_B_seed5', 'CONV_B_seed6', 'CONV_B_engLow', 'CONV_B_engHigh')})
def persona(r, cfg):
    s = TEMPL[0](r); sid = str(r.anon_ID); g = np.random.default_rng(zlib.crc32(f"{sid}|lpp{cfg.get('salt', '')}".encode()))
    p_eng = cfg.get('p_eng', {'high': 0.25, 'mid': 0.35, 'low': 0.40}); e = g.choice(list(p_eng), p=list(p_eng.values())); z = dict(zip(OUT, g.normal(0, 1, len(OUT))))
    if cfg.get('eng'): s += " " + ENG[e]
    if cfg.get('values'):
        gv = np.random.default_rng(zlib.crc32(f"{sid}|values{cfg.get('salt', '')}".encode())); s += " In your life, " + "; ".join(f"{VALUES[v][0]} is {VLEV[gv.integers(0, 3)]} to you" for v in VALUES) + "."
    if cfg.get('moral'):
        gm = np.random.default_rng(zlib.crc32(f'{sid}|moral'.encode())); s += " When you decide what is right, " + "; ".join(f"{MORAL[v]} matters {['a little', 'somewhat', 'a lot'][gm.integers(0, 3)]} to you" for v in MORAL) + "."
    if cfg.get('diverse_text'): s += cfg['diverse_text']
    if cfg.get('diverse'): s += " Keep in mind that real people often hold mixed views that do not follow a single party line; answer as this specific individual, not as a typical member of a group."
    kk = cfg.get('k_by_eng', {}).get(e, cfg.get('dev_k', 1.0))  # Converse coupling
    if cfg.get('dev') and kk is not None:
        B = bins(kk); s += "\nYour views on specific issues:\n" + "\n".join(f"- On {DTXT[d]}, your views are {next(t for lo, hi, t in B if lo <= z[d] < hi)} most people who share your general politics." for d in cfg.get('doms', OUT))
    return s
w = load()
def people(cond):  # control arm: all complete controls; framed arms: up to 400 complete respondents per arm (seeded)
    c = w[w.arm == cond].dropna(subset=['pid_7', 'ideo_4', 'age_5', 'race_4', 'educ_5', 'female'] + [f'Y_{o}' for o in OUT])
    if cond == 'control' and os.environ.get('EVAL_ONLY') == '1': return c[np.where(np.random.default_rng(7).permutation(len(c)) % 2 == 0, 'cal', 'eval') == 'eval']  # test half only (same split as final_eval.py)
    return c if cond == 'control' else c.iloc[np.random.default_rng(1).choice(len(c), min(400, len(c)), replace=False)]
sc = Scorer(MODEL, device=DEV); DIG = [str(i) for i in range(1, 8)]
for vv in VS:
    v, cond = (vv.split('@') + ['control'])[:2]; out = R + f'hmstat_{MODEL}_{v}_{cond}.csv'
    if os.path.exists(out): print('skip', vv); continue
    rows = []; t0 = time.time()
    B = int(os.environ.get('BS', 5)); rs = [r for _, r in people(cond).iterrows()]; k = max(1, B // len(OUT))
    for i in range(0, len(rs), k):  # several personas per forward pass
        chunk = rs[i:i + k]; Pr, mass = sc.probs([sc.chat(context(persona(r, HV[v]), cond) + qblock(o) + "\nAnswer with a single number from 1 to 7.") for r in chunk for o in OUT], DIG, bs=B)
        for j, r in enumerate(chunk):
            for t, o in enumerate(OUT): rows.append(dict(model=MODEL, variant=v, cond=cond, sid=str(r.anon_ID), key=o, p=json.dumps([round(x, 5) for x in Pr[j * len(OUT) + t]]), mass=mass[j * len(OUT) + t]))
    pd.DataFrame(rows).to_csv(out, index=False); print('done', vv, round(time.time() - t0), flush=True)
