"""Budget-matched diversity baseline: 5 wordings of the diversity instruction x the same 12 post-processing options, selected on the
CEK development half by the same criterion as LPP (mean of T2 under party-crosstab and S2 under topline calibration, both relative to
BASE+OT). Runs the development analysis restricted to this family (VA_ONLY) and writes SEL['diverse_best']."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import os, re, json, subprocess, pandas as pd
R = RES; FAM = ['BASE', 'DIVERSE', 'DIVERSE2', 'DIVERSE3', 'DIVERSE4', 'DIVERSE5']
for otp in ('1', '0'):
    env = dict(os.environ, OTP=otp, VA_ONLY=','.join(FAM), VA_TAG='_div')
    subprocess.run([sys.executable, '-W', 'ignore', os.path.join(HERE, 'var_analyze.py'), 'qwen3.8-27b', 'cal', '100'], env=env, check=True, capture_output=True)
FL = float(re.search(r'corrF ([0-9.]+)', [l for l in open(R + 'va_cal_otp.log') if 'floors (human' in l][0]).group(1))
O, P = [pd.read_csv(R + f'var_summary_qwen3.8-27b_cal_div{t}.csv').set_index('variant') for t in ('_otp', '')]
def T2(D, m): b = D.loc['BASE+OT']; return (D.loc[m, 'corrF'] - FL) / (b.corrF - FL) + abs(1 - D.loc[m, 'retest_ratio']) / abs(1 - b.retest_ratio)
def S2(D, m): b = D.loc['BASE+OT']; return D.loc[m, 'ED_party'] / b.ED_party + T2(D, m)
sc = {}
for v in FAM[1:]:
    for m in O.index:
        if m in P.index and re.match(rf'^{v}\+((temp|sharp)[0-9.]+\+)?OT$', m): sc[m] = (T2(O, m) + S2(P, m)) / 2
S = pd.Series(sc).sort_values(); best = S.index[0]
print(S.groupby(lambda m: m.split('+')[0]).min().round(3).to_string()); print('best', best, round(S.iloc[0], 3))
p = R + 'selected_cfg.json'; d = json.load(open(p)); d['diverse_best'] = best; d['diverse_family_scores'] = S.groupby(lambda m: m.split('+')[0]).min().round(4).to_dict(); json.dump(d, open(p, 'w'), indent=1)
