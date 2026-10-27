"""Backstory baseline (Anthology-style, demographic-conditioned): the model writes a short first-person life narrative for each
persona (sampled, T=1, seeded), which is then prepended to the persona. Usage: python gen_backstory.py MODEL DEVICE SPLIT N"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, os, json, zlib, numpy as np, pyreadr
from llm import Scorer
from variants import demo
MODEL, DEV, SPLIT, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
m = pyreadr.read_r(DATA + 'cg/study_1_mturk_cleaned.rds')[None].set_index('subject_id', drop=False)
ids = json.load(open(RES + 'exp2_split.json'))[SPLIT]
P = m.loc[ids].dropna(subset=['pid_7', 'ideo_5', 'age_5', 'race_4', 'educ_5', 'female']); P = P.iloc[np.random.default_rng(2).choice(len(P), N, replace=False)]
sc = Scorer(MODEL, device=DEV); out = {}
GB = int(os.environ.get('GBS', 1))  # GBS>1: personas generated in batches (seeded per batch), for slow models
rs = [r for _, r in P.iterrows()]
def prompt(r): return sc.chat(demo(r) + "\n\nTell me about yourself: write a short first-person story (about 120 words) about your life — where you grew up, your family, your work, and what matters to you. Do not discuss specific political issues.")
for i in range(0, len(rs), GB):
    ch = rs[i:i + GB]; seed = zlib.crc32(ch[0].subject_id.encode()) % 2**31
    for r, o in zip(ch, sc.generate([prompt(r) for r in ch], max_new_tokens=220, bs=GB, temperature=1.0, seed=seed)): out[r.subject_id] = o
    if GB > 1: print('backstories', i + len(ch), '/', len(rs), flush=True)
json.dump(out, open(f'{RES}backstory_{MODEL}_{SPLIT}.json', 'w'), indent=1); print('done', len(out))
