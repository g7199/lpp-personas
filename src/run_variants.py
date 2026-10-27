"""Run persona variants with an open model (per-item next-token probability readout). Loads the model once.
Usage: python run_variants.py MODEL DEVICE SPLIT(cal|eval) N V1,V2,..."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, os, json, time, numpy as np, pandas as pd, pyreadr
from llm import Scorer
from variants import build, VARIANTS, ITEMS, scale
MODEL, DEV, SPLIT, N, VS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5].split(',')
m = pyreadr.read_r(DATA + 'cg/study_1_mturk_cleaned.rds')[None].set_index('subject_id', drop=False)
ids = json.load(open(RES + 'exp2_split.json'))[SPLIT]
P = m.loc[ids].dropna(subset=['pid_7', 'ideo_5', 'age_5', 'race_4', 'educ_5', 'female']); P = P.iloc[np.random.default_rng(2).choice(len(P), N, replace=False)]
BSF = f'{RES}backstory_{MODEL}_{SPLIT}.json'
BS = json.load(open(BSF)) if os.path.exists(BSF) else {}
PARA = __import__("paraphrase").PARA if os.environ.get("PARA") == "1" else None  # perturbed-prompt retest: reworded questions
def q(per, it): k, _, qt, lab = it; qt = PARA[k] if PARA else qt; return per + "\n\nNow answer the survey question as this person would.\n" + f"{qt}\n{scale(lab)}\nAnswer with a single number from 1 to 7."
sc = Scorer(MODEL, device=DEV); DIG = [str(i) for i in range(1, 8)]
for v in VS:
    out = f'{RES}var_{MODEL}_{SPLIT}_n{N}_{v}' + ('_para' if PARA else '') + '.csv'
    if os.path.exists(out): print('skip', v); continue
    cfg = VARIANTS[v]
    if cfg.get('backstory') and not BS: print('no backstories, skip', v); continue
    rows = []; t0 = time.time()
    B = int(os.environ.get('BS', 7)); rs = [r for _, r in P.iterrows()]; k = max(1, B // len(ITEMS))
    for i in range(0, len(rs), k):  # several personas per forward pass
        chunk = rs[i:i + k]; Pr, mass = sc.probs([sc.chat(q(build(r, cfg, BS.get(r.subject_id)), it)) for r in chunk for it in ITEMS], DIG, bs=B)
        for j, r in enumerate(chunk):
            for t, it in enumerate(ITEMS): rows.append(dict(model=MODEL, variant=v, sid=r.subject_id, key=it[0], p=json.dumps([round(x, 5) for x in Pr[j * len(ITEMS) + t]]), mass=mass[j * len(ITEMS) + t]))
    pd.DataFrame(rows).to_csv(out, index=False); json.dump(dict(cfg=cfg, example=build(P.iloc[0], cfg, BS.get(P.iloc[0].subject_id))), open(out.replace('.csv', '.json'), 'w'), indent=1)
    print('done', v, round(time.time() - t0), flush=True)
