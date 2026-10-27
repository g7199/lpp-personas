"""Run persona variants on GSS 2024 (all 856 complete-case respondents; next-token probabilities over the K printed options).
Usage: python gss_run.py MODEL DEVICE V1,V2,...   Variant settings mirror variants.py; the selected configuration is read from selected_cfg.json."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, os, json, time, numpy as np, pandas as pd
from llm import Scorer
from gss_common import load, persona, question, ITEMS, K
MODEL, DEV, VS = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
R = RES
SEL = json.load(open(R + 'selected_cfg.json')) if os.path.exists(R + 'selected_cfg.json') else {}
GV = {'BASE': {}, 'DIVERSE': {'diverse': True}, 'ENG': {'eng': True}, 'LPP': {'eng': True, 'dev': True}, 'DEV': {'dev': True},
      'LPPH': {'eng': True, 'dev': True, 'doms': ['econ', 'crime', 'moral']}, 'VALUES': {'values': True}, 'ENG_VALUES': {'eng': True, 'values': True},
      'VALUES_LPP': {'values': True, 'eng': True, 'dev': True}, 'ENG_MORAL': {'eng': True, 'moral': True}, 'ENG_VALUES_MORAL': {'eng': True, 'values': True, 'moral': True},
      'LPP_k0.6': {'eng': True, 'dev': True, 'dev_k': 0.6}, 'VALUES_LPP_k1.6': {'values': True, 'eng': True, 'dev': True, 'dev_k': 1.6}, 'LPP_k1.6': {'eng': True, 'dev': True, 'dev_k': 1.6},
      'VALUES_LPPH_k1.6': {'values': True, 'eng': True, 'dev': True, 'dev_k': 1.6, 'doms': ['econ', 'crime', 'moral']}}
GV['CONV_B'] = {'values': True, 'eng': True, 'dev': True, 'k_by_eng': {'high': None, 'mid': 1.6, 'low': 0.6}}
if SEL.get('gss_cfg'): GV['SELECTED'] = SEL['gss_cfg']
from variants import DIVERSE_TEXTS, VARIANTS as _V; GV.update({k: {'diverse_text': t} for k, t in DIVERSE_TEXTS.items()}); GV.update({k: _V[k] for k in ('CONV_B_novalues', 'CONV_B_seed2', 'CONV_B_seed3', 'CONV_B_seed4', 'CONV_B_seed5', 'CONV_B_seed6', 'CONV_B_engLow', 'CONV_B_engHigh')})
d = load(); d = d[d.split == 'eval'] if os.environ.get('EVAL_ONLY') == '1' else d; sc = Scorer(MODEL, device=DEV); DIG = [str(i) for i in range(1, 8)]
for v in VS:
    out = R + f'gss_{MODEL}_{v}.csv'
    if os.path.exists(out): print('skip', v); continue
    rows = []; t0 = time.time()
    B = int(os.environ.get('BS', 6)); rs = [r for _, r in d.iterrows()]; k = max(1, B // len(ITEMS))
    for i in range(0, len(rs), k):  # several personas per forward pass
        chunk = rs[i:i + k]; Pr, mass = sc.probs([sc.chat(persona(r, GV[v]) + "\n\nNow answer the survey question as this person would.\n" + question(it)) for r in chunk for it in ITEMS], DIG, bs=B)
        for j, r in enumerate(chunk):
            for t, it in enumerate(ITEMS):
                p, ms = Pr[j * len(ITEMS) + t], mass[j * len(ITEMS) + t]
                q = np.array(p[:K[it[0]]]); rows.append(dict(model=MODEL, variant=v, sid=r.sid, key=it[0], p=json.dumps([round(x, 5) for x in (q / q.sum()).tolist()]), mass=ms * q.sum()))
    pd.DataFrame(rows).to_csv(out, index=False); json.dump(dict(cfg=GV[v], example=persona(d.iloc[0], GV[v])), open(out.replace('.csv', '.json'), 'w'), indent=1)
    print('done', v, round(time.time() - t0), flush=True)
