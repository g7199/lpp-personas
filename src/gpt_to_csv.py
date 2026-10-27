"""Convert API-model sampled answers (gpt_variants.py cache) to the open-model per-item probability format (one-hot), so that the same
analysis code (var_analyze.py / final_eval.py, incl. OT as a stochastic map) applies. Usage: python gpt_to_csv.py MODEL-EFFORT SPLIT N"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, json, pandas as pd
from variants import ITEMS
ME, SPLIT, N = sys.argv[1], sys.argv[2], int(sys.argv[3]); R = RES
rec = {}
for l in open(R + f'cvar_{ME}_{SPLIT}_n{N}.jsonl'):
    d = json.loads(l)
    if d.get('ans'): rec[(d['variant'], d['sid'])] = d['ans']
for v in sorted({k[0] for k in rec}):
    rows = [dict(model=ME, variant=v, sid=s, key=it[0], p=json.dumps([1.0 if a == i + 1 else 0.0 for i in range(7)]), mass=1.0) for (vv, s), ans in rec.items() if vv == v for it, a in zip(ITEMS, ans)]
    n = len({r['sid'] for r in rows}); pd.DataFrame(rows).to_csv(R + f'var_{ME}_{SPLIT}_n{N}_{v}.csv', index=False); print(v, n, 'personas')
