"""All result numbers quoted in the paper text, as LaTeX macros generated from the final_eval half-sample files (no hand-copied numbers).
Usage in LaTeX: \\N{cg-qwen-otp-lpp-corr}. Keys: {ds}-{model}-{reg}-{what}; see below. Also churn macros.
Writes paper/macros_res.tex and prints every key."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import json, os, re, glob, numpy as np, pandas as pd
R = RES; P = PAPER; SEL = json.load(open(R + 'selected_cfg.json')); O = SEL['method']
BL = SEL['baselines']; NAMES = {'divbest': 'DIVBEST', 'rw': 'BASE+OT+reweight', 'base': BL['BASE'], 'div': BL['DIVERSE'], 'argyle': BL['ARGYLE'], 'story': BL['BACKSTORY'], 'anth': BL['ANTHOLOGY']}
SHP = {k: f'{v}+sharp0.05+OT' for k, v in [('base', 'BASE'), ('div', 'DIVERSE'), ('argyle', 'ARGYLE'), ('story', 'BACKSTORY')]}
MOD = {'qwen': 'qwen3.8-27b', 'gemma': 'gemma4-26b', 'gpt': 'gpt-6.1-sol-low'}
mac = {}
f2 = lambda x: f'{x:.2f}'; f3 = lambda x: f'{x:.3f}'; neg = lambda s: s.replace('-', '$-$') if s.startswith('-') else s
def rng(v, f=f2):
    v = [x for x in v if np.isfinite(x)]
    if not v: return '--'
    a, b = f(min(v)), f(max(v)); return a if a == b else f'{a}--{b}'
for ds in ('cg', 'gss', 'hm'):
    for mk, m in MOD.items():
        for reg, tag in (('otp', '_otp'), ('pool', '')):
            f = R + f'final_{ds}_{m}{tag}_reps.csv'
            if not os.path.exists(f): continue
            D = pd.read_csv(f); M = D.groupby('method').mean(numeric_only=True); have = set(M.index); S = f'{ds}-{mk}-{reg}'
            ours = O if O in have else SEL.get('gpt', {}).get('method') if mk == 'gpt' else None
            if ours not in have: continue
            hasrel = M.retest_ratio.notna().any()
            mac[f'{S}-floor-corr'] = f2(M.loc['HUMAN_FLOOR', 'corrF']); mac[f'{S}-floor-cross'] = f2(M.loc['HUMAN_FLOOR', 'cross_ratio'])
            for k, c in (('corr', 'corrF'), ('rel', 'retest_ratio'), ('cross', 'cross_ratio'), ('within', 'within_ratio'), ('gap', 'gap_ratio'), ('wpsd', 'wpsd_ratio'), ('edp', 'ED_party')):
                if c in M and np.isfinite(M.loc[ours, c]): mac[f'{S}-lpp-{k}'] = (f3 if k == 'edp' else f2)(M.loc[ours, c])
            a = D[D.method == ours].set_index('rep')
            sel = {k: v for k, v in NAMES.items() if v in have}; shp = {k: v for k, v in SHP.items() if v in have}
            for grp, G in (('sel', sel), ('shp', shp)):
                if not G: continue
                mac[f'{S}-{grp}-corr'] = rng([M.loc[v, 'corrF'] for v in G.values()])
                if hasrel: mac[f'{S}-{grp}-rel'] = rng([M.loc[v, 'retest_ratio'] for v in G.values()])
                mac[f'{S}-{grp}-n'] = str(len(G)); mac[f'{S}-{grp}-nworse'] = str(sum(M.loc[v, 'corrF'] > M.loc[ours, 'corrF'] for v in G.values()))
                for k, v in G.items():
                    b = D[D.method == v].set_index('rep'); d = (a.corrF - b.corrF).dropna()
                    mac[f'{S}-{grp}-{k}-corr'] = f2(M.loc[v, 'corrF'])
                    if hasrel: mac[f'{S}-{grp}-{k}-rel'] = f2(M.loc[v, 'retest_ratio'])
                    mac[f'{S}-{grp}-{k}-cross'] = f2(M.loc[v, 'cross_ratio'])
                    mac[f'{S}-{grp}-{k}-d'] = neg(f2(d.mean())); mac[f'{S}-{grp}-{k}-ci'] = f'$[{f2(d.quantile(.025))}, {f2(d.quantile(.975))}]$'
                    mac[f'{S}-{grp}-{k}-pwin'] = f'{100 * (d < 0).mean():.0f}\\%'
                    db = (a.ED_party - b.ED_party).dropna(); mac[f'{S}-{grp}-{k}-edp'] = f3(M.loc[v, 'ED_party']); mac[f'{S}-{grp}-{k}-edpci'] = f'$[{f3(db.quantile(.025))}, {f3(db.quantile(.975))}]$'
                dd = [(a.corrF - D[D.method == v].set_index('rep').corrF).dropna() for v in G.values()]
                mac[f'{S}-{grp}-drange'] = neg(f2(min(d.mean() for d in dd))) + ' to ' + ('+' if max(d.mean() for d in dd) > 0 else '') + neg(f2(max(d.mean() for d in dd)))
                mac[f'{S}-{grp}-nsig'] = str(sum((d.quantile(.025) > 0) or (d.quantile(.975) < 0) for d in dd))
                if hasrel: mac[f'{S}-{grp}-relx'] = rng([M.loc[ours, 'retest_ratio'] / M.loc[v, 'retest_ratio'] for v in G.values()], lambda x: f'{x:.1f}')
                ds_ = [((a.corrF - D[D.method == v].set_index('rep').corrF) < 0).mean() for v in G.values()]
                mac[f'{S}-{grp}-pwinmin'] = f'{100 * min(ds_):.0f}\\%'
                mac[f'{S}-{grp}-pwinphrase'] = 'all' if min(ds_) == 1 else 'more than 99\\% of' if min(ds_) >= .995 else f'at least {100 * min(ds_):.0f}\\% of'
            for k, v in (('abl-k16', f"VALUES_LPP_k1.6+sharp{SEL['sharp_T']}+OT"), ('abl-novals', f"CONV_B_novalues+sharp{SEL['sharp_T']}+OT"), ('abl-nodev', f"ENG_VALUES+sharp{SEL['sharp_T']}+OT"),
                         ('abl-nosharp', f"{SEL['persona']}+OT"), ('agg', 'AGG_SAMPLER'), ('abl-abs', f"CONV_B_abs+sharp{SEL['sharp_T']}+OT"), ('abl-placebo', f"CONV_B_placebo+sharp{SEL['sharp_T']}+OT"), ('abl-seed2', f"CONV_B_seed2+sharp{SEL['sharp_T']}+OT"), ('abl-englow', f"CONV_B_engLow+sharp{SEL['sharp_T']}+OT"), ('abl-enghigh', f"CONV_B_engHigh+sharp{SEL['sharp_T']}+OT"), ('abl-demsharp', f"BASE+sharp{SEL['sharp_T']}+OT"), ('demo', 'BASE+OT')):
                if v in have:
                    mac[f'{S}-{k}-corr'] = f2(M.loc[v, 'corrF']); mac[f'{S}-{k}-cross'] = f2(M.loc[v, 'cross_ratio'])
                    dk = (a.corrF - D[D.method == v].set_index('rep').corrF).dropna(); mac[f'{S}-{k}-pwin'] = f'{100 * (dk < 0).mean():.1f}\\%'; mac[f'{S}-{k}-gap'] = f2(M.loc[v, 'gap_ratio'])
                    bv = D[D.method == v].set_index('rep')
                    for col, tag in (('cross_ratio', 'dcross'), ('gap_ratio', 'dgap'), ('corrF', 'dcorr')):
                        dd2 = (a[col] - bv[col]).dropna()
                        if len(dd2): mac[f'{S}-{k}-{tag}'] = neg(f2(dd2.mean())); mac[f'{S}-{k}-{tag}-ci'] = f'$[{f2(dd2.quantile(.025))}, {f2(dd2.quantile(.975))}]$'
                    if hasrel: mac[f'{S}-{k}-rel'] = f2(M.loc[v, 'retest_ratio'])
# settings (open models) where LPP has the lowest corr_F of all methods (selected + sharpened baselines)
for reg in ('otp', 'pool'):
    low = [f'{ds}-{mk}' for ds in ('cg', 'gss', 'hm') for mk in ('qwen', 'gemma') if f'{ds}-{mk}-{reg}-sel-n' in mac and mac[f'{ds}-{mk}-{reg}-sel-nworse'] == mac[f'{ds}-{mk}-{reg}-sel-n'] and mac.get(f'{ds}-{mk}-{reg}-shp-nworse') == mac.get(f'{ds}-{mk}-{reg}-shp-n')]
    mac[f'nlowest-{reg}'] = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'all nine'}[len(low)]; print(reg, 'lowest in', low)
# reliability of selected baselines on CEK (true short-term human reference), across the two open models
v = [float(x) for k, x in mac.items() if re.match(r'cg-(qwen|gemma)-otp-sel-(base|div|argyle|story|anth)-rel$', k)]
if v: mac['all-sel-rel-pct'] = f'{100 * min(v):.0f}--{100 * max(v):.0f}\\%'
vq = [float(x) for k, x in mac.items() if re.match(r'cg-qwen-otp-sel-(base|div|argyle|story|anth)-rel$', k)]
if vq: mac['qwen-sel-rel-pct'] = f'{100 * min(vq):.0f}--{100 * max(vq):.0f}\\%'
C = pd.read_csv(R + 'churn_qwen3.8-27b.csv').set_index('method'); fit = [m for m in BL.values() if m in C.index]
mac.update({'churn-human': f"{100 * C.churn.iloc[0]:.1f}\\%", 'churn-sel': f"{100 * C.loc[fit, 'churn'].min():.1f}--{100 * C.loc[fit, 'churn'].max():.1f}\\%",
            'churn-sel-ratio': f"{C.loc[fit, 'ratio'].min():.1f}--{C.loc[fit, 'ratio'].max():.1f}", 'churn-lpp': f"{100 * C.loc[f'{SEL['persona']}+sharp{SEL['sharp_T']}+OT', 'churn']:.1f}\\%",
            'churn-lpp-ratio': f"{C.loc[f'{SEL['persona']}+sharp{SEL['sharp_T']}+OT', 'ratio']:.2f}"})

# framing experiment (Qwen): effect-matrix summaries and paired-bootstrap differences
H = pd.read_csv(R + 'hmdyn_qwen3.8-27b.csv').set_index(['variant', 'post']); TS = SEL['sharp_T']; TT = SEL['temp_T']; V = SEL['persona']
bo = {l.split(':')[0]: l for l in open(R + 'hmdyn_boot_qwen.txt') if 'RMSE(' in l}
def bci(k): l = bo[k]; m = re.search(r'= \+?([-0-9.]+)\s+95% CI \[\+?([-0-9.]+), \+?([-0-9.]+)\]', l); return f'{float(m.group(1)):.2f} [{float(m.group(2)):.2f}, {float(m.group(3)):.2f}]'
dyn = dict(HMDYNhuman=f"{H.loc[('BASE', 'raw'), 'direct'] / H.loc[('BASE', 'raw'), 'direct_gain']:.2f}", HMDYNbaser=f"{H.loc[('BASE', 'raw'), 'r_cells']:.2f}",
           HMDYNbasedirect=f"{H.loc[('BASE', 'raw'), 'direct']:.2f}", HMDYNlppdirect=rng([H.loc[(V, p), 'direct'] for p in ('raw', f'sharp{TS}+OT', f'temp{TT}+OT')]),
           HMDYNlppr=rng([H.loc[(V, p), 'r_cells'] for p in ('raw', f'sharp{TS}+OT', f'temp{TT}+OT')]),
           HMDYNbaseraw=f"{H.loc[('BASE', 'raw'), 'rmse']:.2f}", HMDYNlppraw=f"{H.loc[(V, 'raw'), 'rmse']:.2f}", HMDYNbasesharp=f"{H.loc[('BASE', f'sharp{TS}+OT'), 'rmse']:.2f}",
           HMDYNlppsharp=f"{H.loc[(V, f'sharp{TS}+OT'), 'rmse']:.2f}", HMDYNbasetemp=f"{H.loc[('BASE', f'temp{TT}+OT'), 'rmse']:.2f}", HMDYNlpptemp=f"{H.loc[(V, f'temp{TT}+OT'), 'rmse']:.2f}",
           HMDYNdraw=bci('raw'), HMDYNdsharp=bci(f'sharp{TS}+OT'), HMDYNdtemp=bci(f'temp{TT}+OT'))

G = pd.read_csv(R + 'hmdyn_gemma4-26b.csv').set_index(['variant', 'post']); bg = {l.split(':')[0]: l for l in open(R + 'hmdyn_boot_gemma.txt') if 'RMSE(' in l}
def bcig(k): l = bg[k]; m = re.search(r'= \+?([-0-9.]+)\s+95% CI \[\+?([-0-9.]+), \+?([-0-9.]+)\]', l); return f'{float(m.group(1)):.2f} [{float(m.group(2)):.2f}, {float(m.group(3)):.2f}]'
dyn.update(HMDYNGbasedirect=f"{G.loc[('BASE', 'raw'), 'direct']:.2f}", HMDYNGlppdirect=f"{G.loc[(V, 'raw'), 'direct']:.2f}", HMDYNGbaseraw=f"{G.loc[('BASE', 'raw'), 'rmse']:.2f}",
           HMDYNGlppraw=f"{G.loc[(V, 'raw'), 'rmse']:.2f}", HMDYNGbasesharp=f"{G.loc[('BASE', f'sharp{TS}+OT'), 'rmse']:.2f}", HMDYNGlppsharp=f"{G.loc[(V, f'sharp{TS}+OT'), 'rmse']:.2f}",
           HMDYNGdraw=bcig('raw'), HMDYNGdsharp=bcig(f'sharp{TS}+OT'))
s = ['\\makeatletter', '\\newcommand{\\N}[1]{\\ifcsname N@#1\\endcsname\\csname N@#1\\endcsname\\else\\textbf{\\textcolor{red}{??#1}}\\fi}']
s += [f'\\expandafter\\def\\csname N@{k}\\endcsname{{{v}}}' for k, v in mac.items()] + ['\\makeatother'] + [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in dyn.items()]
open(P + 'macros_res.tex', 'w').write('\n'.join(s) + '\n')
used = set(re.findall(r'\\N\{([^}]+)\}', ''.join(open(f).read() for f in glob.glob(P + 'sec_*.tex'))))
print(len(mac), 'macros;', 'MISSING:', sorted(used - set(mac)))
