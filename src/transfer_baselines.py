"""Argyle backstory, life-story backstory and Anthology baselines on the transfer surveys (GSS, HM control arm), test half only,
built exactly as on CEK: Argyle = first-person covariate sentences; life-story = model-written ~120-word story conditioned on the
profile, prepended to the demographic persona; Anthology = the model's own 400-story pool (anthology_pool_{MODEL}_400_us.json) with
its demographic-questionnaire answers, matched one-to-one to respondents by maximum-weight bipartite matching.
Usage: python transfer_baselines.py DS(gss|hm) MODEL DEVICE V1,V2,..."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, os, json, zlib, time, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from llm import Scorer
DS, MODEL, DEV, VS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(',')
R = RES
STORY = "\n\nTell me about yourself: write a short first-person story (about 120 words) about your life — where you grew up, your family, your work, and what matters to you. Do not discuss specific political issues."
PIDTXT = {0: 'a strong Democrat', 1: 'a Democrat, though not a strong one', 2: 'an independent who leans toward the Democrats', 3: 'an independent',
          4: 'an independent who leans toward the Republicans', 5: 'a Republican, though not a strong one', 6: 'a strong Republican'}
if DS == 'gss':
    from gss_common import load, demo, question, ITEMS, K
    d = load(); P = d[d.split == 'eval']; KEYS = [it[0] for it in ITEMS]
    sid = lambda r: r.sid
    def prompts(per): return [per + "\n\nNow answer the survey question as this person would.\n" + question(it) for it in ITEMS]
    def kk(key): return K[key]
    def argyle(r):
        race = {1: 'White', 2: 'Black', 3: 'of another race'}[int(r.race)]; pid = int(r.partyid)
        ideo = {1: 'extremely liberal', 2: 'liberal', 3: 'slightly liberal', 4: 'moderate', 5: 'slightly conservative', 6: 'conservative', 7: 'extremely conservative'}[int(r.polviews)]
        edu = {0: 'less than high school', 1: 'high school', 2: 'a junior college degree', 3: "a bachelor's degree", 4: 'a graduate degree'}[int(r.degree)]
        return (f"Racially, I am {race}. I am {'a woman' if int(r.sex) == 2 else 'a man'}. In terms of my age, I am {int(r.age)} years old. In terms of education, my highest level is {edu}. "
                f"Ideologically, I describe myself as {ideo}. Politically, I am {PIDTXT.get(pid, 'a supporter of another party')}. Below are my answers to an opinion survey.")
    def attrs(r):  # respondent attributes -> option index of the Anthology demographic questionnaire (same categories as on CEK)
        a = int(r.age); age = 0 if a < 30 else 1 if a < 40 else 2 if a < 50 else 3 if a < 60 else 4
        pid = int(r.partyid); ideo = int(r.polviews)
        return dict(age=age, gender=int(int(r.sex) == 2), race={1: 0, 2: 1, 3: 3}[int(r.race)], educ={0: 0, 1: 1, 2: 2, 3: 3, 4: 4}[int(r.degree)],
                    pid=pid if pid <= 6 else 3, ideo=0 if ideo <= 3 else 1 if ideo == 4 else 2)
    out_name = lambda v: R + f'gss_{MODEL}_{v}.csv'
else:
    from hm_common import load, OUT, TEMPL, context, qblock, AGE, EDU, IDEO
    w = load(); c = w[w.arm == 'control'].dropna(subset=['pid_7', 'ideo_4', 'age_5', 'race_4', 'educ_5', 'female'] + [f'Y_{o}' for o in OUT])
    sp = np.where(np.random.default_rng(7).permutation(len(c)) % 2 == 0, 'cal', 'eval'); P = c[sp == 'eval']; KEYS = OUT  # identical split to final_eval.py
    sid = lambda r: str(r.anon_ID)
    def prompts(per): return [context(per, 'control') + qblock(o) + "\nAnswer with a single number from 1 to 7." for o in OUT]
    def kk(key): return 7
    demo = TEMPL[0]
    def argyle(r):
        race = {'White': 'White', 'Black': 'Black', 'Hispanic': 'Hispanic', 'Race: Other': 'of another race'}[r.race_4]
        ideo = {'Liberal': 'liberal', 'Moderate': 'moderate', 'Conservative': 'conservative', 'Ideology: Other': 'not really any of the usual labels'}[r.ideo_4]
        return (f"Racially, I am {race}. I am {'a woman' if r.female == 1 else 'a man'}. In terms of my age, I am {AGE[r.age_5]} years old. In terms of education, my highest level is {EDU[r.educ_5]}. "
                f"Ideologically, I describe myself as {ideo}. Politically, I am {PIDTXT[int(r.pid_7) - 1]}. Below are my answers to an opinion survey.")
    def attrs(r):
        return dict(age=['18 - 29', '30 - 39', '40 - 49', '50 - 59', 'More than 60'].index(r.age_5), gender=int(r.female == 1),
                    race=['White', 'Black', 'Hispanic', 'Race: Other'].index(r.race_4), educ=['Less than High School', 'High School', 'Some College', 'College', 'Graduate School'].index(r.educ_5),
                    pid=int(r.pid_7) - 1, ideo={'Liberal': 0, 'Moderate': 1, 'Conservative': 2, 'Ideology: Other': 4}[r.ideo_4])
    out_name = lambda v: R + f'hmstat_{MODEL}_{v}_control.csv'
sc = Scorer(MODEL, device=DEV); DIG = [str(i) for i in range(1, 8)]
def score(name, personas):
    out = out_name(name)
    if os.path.exists(out): print('skip', name); return
    rows = []; t0 = time.time(); BS = int(os.environ.get('BS', 6)); rs = [r for _, r in P.iterrows()]; per = max(1, BS // len(KEYS))
    for i in range(0, len(rs), per):  # several personas per forward pass
        chunk = rs[i:i + per]; Pr, mass = sc.probs([sc.chat(q) for r in chunk for q in prompts(personas(r))], DIG, bs=BS)
        for j, r in enumerate(chunk):
            for t, key in enumerate(KEYS):
                p, ms = Pr[j * len(KEYS) + t], mass[j * len(KEYS) + t]
                q = np.array(p[:kk(key)]); rows.append(dict(model=MODEL, variant=name, cond='control', sid=sid(r), key=key, p=json.dumps([round(x, 5) for x in (q / q.sum()).tolist()]), mass=ms * q.sum()))
    pd.DataFrame(rows).to_csv(out, index=False); print('done', name, len(P), round(time.time() - t0), flush=True)
for v in VS:
    if v == 'ARGYLE':
        score('ARGYLE', argyle)
    elif v == 'BACKSTORY':
        bf = R + f'backstory_{MODEL}_{DS}_eval.json'
        if os.path.exists(bf): BS = json.load(open(bf))
        else:
            BS = {}; t0 = time.time(); rs = list(P.iterrows())
            TB = int(os.environ.get('TBS', 12))
            for i in range(0, len(rs), TB):  # batched sampling (T=1), seeded per batch
                chunk = rs[i:i + TB]
                outs = sc.generate([sc.chat(demo(r) + STORY) for _, r in chunk], max_new_tokens=220, bs=TB, temperature=1.0, seed=zlib.crc32(f'{DS}{i}'.encode()) % 2**31)
                for (_, r), o in zip(chunk, outs): BS[sid(r)] = o
            json.dump(BS, open(bf, 'w'), indent=1); print('stories', len(BS), round(time.time() - t0), flush=True)
        score('BACKSTORY', lambda r: demo(r) + "\n\nHere is some background about you, in your own words:\n" + BS[sid(r)].strip())
    elif v == 'ANTHOLOGY':
        pool = json.load(open(R + f'anthology_pool_{MODEL}_400_us.json')); D = {k: np.array(x) for k, x in json.load(open(R + f'anthology_demo_{MODEL}_400_us.json')).items()}
        A = [attrs(r) for _, r in P.iterrows()]; W = np.zeros((len(A), len(pool)))
        for i, a in enumerate(A):
            for k, idx in a.items(): W[i] += np.log(D[k][:, idx] + 1e-6)
        cap = int(np.ceil(len(A) / len(pool))); Wc = np.tile(W, (1, cap))  # each story usable at most `cap` times when respondents outnumber the pool
        ri, ci = linear_sum_assignment(-Wc); match = {sid(P.iloc[i]): int(j) % len(pool) for i, j in zip(ri, ci)}; assert len(match) == len(A)
        json.dump(dict(match=match, cap=cap, mean_logw=float(Wc[ri, ci].mean())), open(R + f'anthology_match_{MODEL}_{DS}_eval.json', 'w'))
        score('ANTHOLOGY', lambda r: "Here is a person describing themselves:\n\n" + pool[match[sid(r)]].strip() + "\n\nYou are this person.")
