"""API-model persona variants, queried through the OpenAI Codex CLI (`codex exec`) (joint 14-item JSON answer; one sampled answer set per persona). Personas and latents
are identical to the open-model runs (variants.build). Rate limits: exponential back-off, never a model switch.
Usage: python gpt_variants.py MODEL EFFORT SPLIT N WORKERS V1,V2,...   -> results/cvar_{MODEL}-{EFFORT}_{SPLIT}_n{N}.jsonl"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, os, re, json, time, subprocess, tempfile, threading, numpy as np, pyreadr
from concurrent.futures import ThreadPoolExecutor
from variants import build, VARIANTS, ITEMS, scale
MODEL, EFFORT, SPLIT, N, W, VS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6].split(',')
R = RES
if os.path.exists(R + 'selected_cfg.json'): VARIANTS['SELECTED'] = json.load(open(R + 'selected_cfg.json'))['cfg']
m = pyreadr.read_r(DATA + 'cg/study_1_mturk_cleaned.rds')[None].set_index('subject_id', drop=False)
ids = json.load(open(RES + 'exp2_split.json'))[SPLIT]
P = m.loc[ids].dropna(subset=['pid_7', 'ideo_5', 'age_5', 'race_4', 'educ_5', 'female']); P = P.iloc[np.random.default_rng(2).choice(len(P), N, replace=False)]
def joint_prompt(per):
    qs = "\n\n".join(f"Q{i + 1}. {q}\n{scale(lab)}" for i, (_, _, q, lab) in enumerate(ITEMS))
    return per + "\n\nNow answer the following survey questions as this person would.\n\n" + qs + "\n\nAnswer each question with a single number from 1 to 7. Respond with only a JSON object of the form {" + ", ".join(f'"Q{i + 1}": n' for i in range(len(ITEMS))) + "}. Do not run any commands or read any files."
CACHE = R + f'cvar_{MODEL}-{EFFORT}_{SPLIT}_n{N}.jsonl'; LOCK = threading.Lock()
def codex(p):
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'o.txt')
        try: r = subprocess.run(['codex', 'exec', '-m', MODEL, '-c', f'model_reasoning_effort="{EFFORT}"', '-s', 'read-only', '--ephemeral', '--skip-git-repo-check', '--ignore-user-config', '-C', td, '-o', out, p],
                                capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired: return '', -9
        return (open(out).read() if os.path.exists(out) else ''), (429 if '429' in (r.stdout + r.stderr) else r.returncode)
def job(j):
    v, sid, p = j
    for a in range(12):
        txt, rc = codex(p); ans = None
        if rc == 429: time.sleep(min(1800, 120 * 2 ** min(a, 4))); continue  # rate limited: back off, never switch model
        mm = re.search(r'\{[^{}]*\}', txt)
        try: d = json.loads(mm.group()); ans = [int(d[f'Q{i + 1}']) for i in range(len(ITEMS))]; ans = ans if all(1 <= x <= 7 for x in ans) else None
        except Exception: ans = None
        rec = dict(variant=v, sid=sid, attempt=a, rc=rc, ans=ans, raw=txt[:300])
        with LOCK: open(CACHE, 'a').write(json.dumps(rec) + '\n')
        if ans: return rec
        time.sleep(5)
    return rec
done = set()
if os.path.exists(CACHE):
    for l in open(CACHE):
        d = json.loads(l)
        if d.get('ans'): done.add((d['variant'], d['sid']))
jobs = [(v, r.subject_id, joint_prompt(build(r, VARIANTS[v.split('#')[0]])))  # 'V#2' = independent second draw (test-retest)
        for v in VS for _, r in P.iterrows() if (v, r.subject_id) not in done]
print('jobs', len(jobs), flush=True); t0 = time.time(); n = 0
with ThreadPoolExecutor(W) as ex:
    for rec in ex.map(job, jobs):
        n += 1
        if n % 50 == 0: print(n, round(time.time() - t0), flush=True)
json.dump(dict(model=MODEL, effort=EFFORT, split=SPLIT, N=N, variants=VS, cfgs={v: VARIANTS[v.split('#')[0]] for v in VS}), open(CACHE.replace('.jsonl', '_meta.json'), 'w'), indent=1)
print('DONE', round(time.time() - t0), flush=True)
