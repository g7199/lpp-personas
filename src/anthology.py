"""Anthology baseline (Moon et al. 2024, arXiv 2407.06576), adapted: (1) unconditional backstory pool from "Tell me about yourself."
(T=1, seeded; NOTE: original uses a base model, we use the same instruct model as all other conditions); (2) demographic survey on each
backstory (next-token probabilities); (3) maximum-weight matching (Hungarian) of target respondents to backstories by
sum of log P(backstory trait = respondent trait); (4) survey items answered conditioned on the matched backstory only.
Usage: python anthology.py MODEL DEVICE SPLIT N POOL"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, os, json, zlib, time, numpy as np, pandas as pd, pyreadr
from scipy.optimize import linear_sum_assignment
from llm import Scorer
from variants import ITEMS, scale
MODEL, DEV, SPLIT, N, POOL = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
R = RES
m = pyreadr.read_r(DATA + 'cg/study_1_mturk_cleaned.rds')[None].set_index('subject_id', drop=False)
ids = json.load(open(RES + 'exp2_split.json'))[SPLIT]
P = m.loc[ids].dropna(subset=['pid_7', 'ideo_5', 'age_5', 'race_4', 'educ_5', 'female']); P = P.iloc[np.random.default_rng(2).choice(len(P), N, replace=False)]
sc = Scorer(MODEL, device=DEV); DIG = [str(i) for i in range(1, 8)]
pf = R + f'anthology_pool_{MODEL}_{POOL}_us.json'
import re; BAD = re.compile(r"\b(AI|A\.I\.|language model|AI assistant|virtual assistant|chatbot|Qwen|Gemma|Google|Alibaba|DeepMind|OpenAI|I don't have personal|as a model)\b", re.I)
if os.path.exists(pf): pool = json.load(open(pf))
else:  # Moon et al.'s 'Tell me about yourself' elicitation, adapted to instruction-tuned models (both models answer the bare question, and even a raw-text
      # continuation, as an AI): the model voices a randomly chosen U.S. adult; AI self-descriptions are rejected and redrawn
    QA = 'Imagine a randomly chosen adult living in the United States is asked in an interview: "Tell me about yourself." Write their answer in the first person, in their own voice, as a single paragraph of about 150 words. Write only the answer.'
    pool, i = [], 0
    while len(pool) < POOL:
        AB = int(os.environ.get('ABS', 8)); out = sc.generate([sc.chat(QA)] * AB, max_new_tokens=220, bs=AB, temperature=1.0, seed=zlib.crc32(f'anthraw{i}'.encode()) % 2**31); i += 1
        for b in out:
            b = b.strip()
            if len(b.split()) >= 30 and not BAD.search(b) and len(pool) < POOL: pool.append(b)
        print('pool', len(pool), 'batches', i, flush=True)
        assert i < 400, 'too many rejected backstories'
    json.dump(pool, open(pf, 'w'), indent=1)
print('pool', len(pool), flush=True)
DQ = {  # attribute: (question, options, function mapping a respondent row to option index)
 'age': ("What is your age?", ['18-29', '30-39', '40-49', '50-59', '60 or older'], lambda r: ['18 - 29', '30 - 39', '40 - 49', '50 - 59', '60+'].index(r.age_5)),
 'gender': ("What is your gender?", ['Male', 'Female'], lambda r: int(r.female == 1)),
 'race': ("What is your race or ethnicity?", ['White', 'Black', 'Hispanic', 'Other'], lambda r: ['White', 'Black', 'Hispanic', 'Other'].index(r.race_4)),
 'educ': ("What is the highest level of education you have completed?", ['Less than high school', 'High school', 'Some college', 'College degree', 'Graduate school'], lambda r: ['Less than High School', 'High School', 'Some College', 'College', 'Graduate School'].index(r.educ_5)),
 'pid': ("Generally speaking, do you think of yourself as a Democrat, a Republican, or an independent?", ['Strong Democrat', 'Not very strong Democrat', 'Independent, leaning Democrat', 'Independent', 'Independent, leaning Republican', 'Not very strong Republican', 'Strong Republican'], lambda r: int(r.pid_7) - 1),
 'ideo': ("When it comes to politics, how do you describe yourself?", ['Liberal', 'Moderate', 'Conservative', 'Libertarian', 'Other / have not thought much about it'], lambda r: ['Liberal', 'Moderate', 'Conservative', 'Libertarian', 'Other'].index(r.ideo_5)),
}
df = R + f'anthology_demo_{MODEL}_{POOL}_us.json'
if os.path.exists(df): D = {k: np.array(v) for k, v in json.load(open(df)).items()}
else:
    D = {}
    for k, (q, opts, _) in DQ.items():
        prompts = [sc.chat(f"Here is a person describing themselves:\n\n{b}\n\nBased on this description, answer the following question as this person.\n{q}\n" + "\n".join(f"{i + 1} = {o}" for i, o in enumerate(opts)) + f"\nAnswer with a single number from 1 to {len(opts)}.") for b in pool]
        D[k] = np.array(sc.probs(prompts, [str(i + 1) for i in range(len(opts))], bs=8)[0])
    json.dump({k: v.tolist() for k, v in D.items()}, open(df, 'w'))
W = np.zeros((N, len(pool)))
for i, (_, r) in enumerate(P.iterrows()):
    for k, (_, _, f) in DQ.items(): W[i] += np.log(D[k][:, f(r)] + 1e-6)
ri, ci = linear_sum_assignment(-W); match = {P.iloc[i].subject_id: int(c) for i, c in zip(ri, ci)}
json.dump(dict(match=match, mean_logw=float(W[ri, ci].mean())), open(R + f'anthology_match_{MODEL}_{SPLIT}_n{N}.json', 'w'))
PARA = __import__('paraphrase').PARA if os.environ.get('PARA') == '1' else None
rows = []; t0 = time.time()
for _, r in P.iterrows():
    per = "Here is a person describing themselves:\n\n" + pool[match[r.subject_id]].strip() + "\n\nYou are this person."
    Pr, mass = sc.probs([sc.chat(per + "\n\nNow answer the survey question as this person would.\n" + f"{PARA[it[0]] if PARA else it[2]}\n{scale(it[3])}\nAnswer with a single number from 1 to 7.") for it in ITEMS], DIG, bs=7)
    for it, p, ms in zip(ITEMS, Pr, mass): rows.append(dict(model=MODEL, variant='ANTHOLOGY', sid=r.subject_id, key=it[0], p=json.dumps([round(x, 5) for x in p]), mass=ms))
pd.DataFrame(rows).to_csv(R + f'var_{MODEL}_{SPLIT}_n{N}_ANTHOLOGY' + ('_para' if PARA else '') + '.csv', index=False); print('DONE', round(time.time() - t0))
