"""Persona-builder variants for the method search. Every builder maps (respondent row) -> persona text; latents are seeded by
respondent id (+ variant salt) and never use the respondent's own answers. Items/human coding are shared with cek_items.py."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import zlib, json, numpy as np
src = open(os.path.join(HERE, 'cek_items.py')).read(); _g = {'np': np, 'zlib': zlib}
exec(src[src.index('DOM = '):src.index('def latents')], _g)
DOM, DOMTXT, ITEMS, demo, ENG = _g['DOM'], _g['DOMTXT'], _g['ITEMS'], _g['demo'], _g['ENG']
def scale(lab): return "\n".join(f"{i + 1} = {l}" for i, l in enumerate(lab)) if isinstance(lab, list) else f"1 = {lab[0]}\n7 = {lab[1]}"
DIVERSE = " Keep in mind that real people often hold mixed views that do not follow a single party line; answer as this specific individual, not as a typical member of a group."
VALUES = {'tradition': ('respecting traditions and customs', ), 'selfdir': ('thinking for yourself and making your own choices', ), 'security': ('safety and stability for yourself and your family', ),
          'benevolence': ('helping the people close to you', ), 'achievement': ('personal success and ambition', ), 'universalism': ('fairness and equal opportunity for everyone', )}
VLEV = ['not very important', 'somewhat important', 'very important']
ABS = {'climate': ('taking action on climate change', ), 'wall': ('stricter rules for banks and Wall Street', ), 'amtrak': ('more government spending on transportation and infrastructure', ),
       'vets': ("the government's own veterans' health system", ), 'flat': ('raising taxes on the wealthy', )}
def rng_for(sid, salt): return np.random.default_rng(zlib.crc32(f'{sid}|{salt}'.encode()))
def bins(k):
    t1, t2 = 0.4 * k, 1.2 * k
    return [(-np.inf, -t2, 'noticeably more liberal than'), (-t2, -t1, 'somewhat more liberal than'), (-t1, t1, 'about the same as'), (t1, t2, 'somewhat more conservative than'), (t2, np.inf, 'noticeably more conservative than')]
def build(r, cfg, backstory=None):
    """cfg keys: eng (bool), p_eng (dict), dev (bool), dev_k (float bin scale), doms (list), dev_style ('rel'|'abs'), values (bool), diverse (bool), backstory (bool)."""
    s = demo(r); g = rng_for(r.subject_id, 'lpp' + cfg.get('salt', ''))  # same random stream for engagement and deviations as cek_items.latents
    p_eng = cfg.get('p_eng', {'high': 0.25, 'mid': 0.35, 'low': 0.40})
    e = g.choice(list(p_eng), p=list(p_eng.values())); z = dict(zip(DOM, g.normal(0, 1, len(DOM))))
    if cfg.get('backstory') and backstory: s += "\n\nHere is some background about you, in your own words:\n" + backstory.strip()
    if cfg.get('eng'): s += " " + ENG[e]
    if cfg.get('values'):
        gv = rng_for(r.subject_id, 'values' + cfg.get('salt', '')); s += " In your life, " + "; ".join(f"{VALUES[v][0]} is {VLEV[gv.integers(0, 3)]} to you" for v in VALUES) + "."
    kk = cfg.get('k_by_eng', {}).get(e, cfg.get('dev_k', 1.0))  # Converse coupling: deviation scale depends on the sampled engagement level
    if cfg.get('dev') and kk is not None:
        doms = cfg.get('doms', DOM); B = bins(kk)
        if cfg.get('dev_style', 'rel') == 'rel':
            lines = [f"- On {DOMTXT[d]}, your views are {next(t for lo, hi, t in B if lo <= z[d] < hi)} most people who share your general politics." for d in doms]
        else:
            lev = lambda x: next(t for lo, hi, t in [(-np.inf, -1.2 * kk, 'strongly favor'), (-1.2 * kk, -0.4 * kk, 'somewhat favor'), (-0.4 * kk, 0.4 * kk, 'feel neutral about'), (0.4 * kk, 1.2 * kk, 'are somewhat skeptical of'), (1.2 * kk, np.inf, 'are strongly skeptical of')] if lo <= x < hi)
            lines = [f"- You {lev(z[d])} {ABS[d][0]}." for d in doms]
        s += "\nYour views on specific issues:\n" + "\n".join(lines)
    if cfg.get('diverse'): s += DIVERSE
    return s
VARIANTS = {
 'BASE': {}, 'DIVERSE': {'diverse': True}, 'ENG': {'eng': True}, 'DEV': {'dev': True}, 'LPP': {'eng': True, 'dev': True},
 'LPP_k0.6': {'eng': True, 'dev': True, 'dev_k': 0.6}, 'LPP_k1.6': {'eng': True, 'dev': True, 'dev_k': 1.6}, 'LPP_k2.5': {'eng': True, 'dev': True, 'dev_k': 2.5},
 'LPP_englow': {'eng': True, 'dev': True, 'p_eng': {'high': 0.15, 'mid': 0.35, 'low': 0.50}}, 'LPP_enghigh': {'eng': True, 'dev': True, 'p_eng': {'high': 0.4, 'mid': 0.4, 'low': 0.2}},
 'LPP_abs': {'eng': True, 'dev': True, 'dev_style': 'abs'}, 'VALUES': {'values': True}, 'ENG_VALUES': {'eng': True, 'values': True},
 'LPP_DIVERSE': {'eng': True, 'dev': True, 'diverse': True}, 'BACKSTORY': {'backstory': True}, 'BACKSTORY_LPP': {'backstory': True, 'eng': True, 'dev': True},
 'LPPH': {'eng': True, 'dev': True, 'doms': ['climate', 'wall', 'amtrak']}, 'LPPH_k1.6': {'eng': True, 'dev': True, 'dev_k': 1.6, 'doms': ['climate', 'wall', 'amtrak']},
}
# --- baselines added after the novelty check ---
AGE1 = {'18 - 24': 'in my early twenties', '25 - 34': 'in my late twenties or early thirties', '35 - 44': 'in my late thirties or early forties', '45 - 54': 'in my late forties or early fifties', '55 - 64': 'in my late fifties or early sixties', '65 or over': '65 or older'}
PID1 = {1: 'a strong Democrat', 2: 'a Democrat, though not a strong one', 3: 'an independent who leans toward the Democrats', 4: 'an independent', 5: 'an independent who leans toward the Republicans', 6: 'a Republican, though not a strong one', 7: 'a strong Republican'}
def argyle(r):  # first-person backstory in the style of Argyle et al. (2023), built only from the available survey fields
    age = AGE1.get(r.age_5, r.age_5.replace(' - ', ' to ') + ' years old')
    ideo = {'Liberal': 'liberal', 'Moderate': 'moderate', 'Conservative': 'conservative', 'Libertarian': 'libertarian', 'Other': 'not really any of the usual labels'}[r.ideo_5]
    return (f"Racially, I am {r.race_4}. I am {'a woman' if r.female == 1 else 'a man'}. In terms of my age, I am {age}. "
            f"In terms of education, my highest level is {str(r.educ_5).lower()}. Ideologically, I describe myself as {ideo}. Politically, I am {PID1[int(r.pid_7)]}. "
            "Below are my answers to an opinion survey.")
VARIANTS.update({'DEV_abs': {'dev': True, 'dev_style': 'abs'}, 'ARGYLE': {'argyle': True}, 'ARGYLE_LPP': {'argyle': True, 'eng': True, 'dev': True}})
_build0 = build
def build(r, cfg, backstory=None):
    if cfg.get('argyle'):
        s = _build0(r, {**cfg, 'argyle': False}, backstory); return s.replace(demo(r), argyle(r), 1)
    if cfg.get('anthology'): return backstory  # Anthology: the matched backstory alone is the persona
    return _build0(r, cfg, backstory)
# --- wave 3: values x deviations combinations (values lowered corrF; deviations fixed party structure) ---
VARIANTS.update({'VALUES_LPP': {'values': True, 'eng': True, 'dev': True}, 'VALUES_LPP_k0.6': {'values': True, 'eng': True, 'dev': True, 'dev_k': 0.6},
                 'VALUES_LPP_k1.6': {'values': True, 'eng': True, 'dev': True, 'dev_k': 1.6}, 'VALUES_LPPH': {'values': True, 'eng': True, 'dev': True, 'doms': ['climate', 'wall', 'amtrak']}})
# --- wave 4: alternative cross-cutting latents (moral foundations; Graham et al. 2009) and values + weak issue deviations ---
MORAL = {'care': 'protecting people from harm and caring for those who are vulnerable', 'fairness': 'fairness and making sure people are treated equally',
         'loyalty': 'loyalty to your group, community and country', 'authority': 'respect for authority, order and tradition',
         'sanctity': 'purity, decency and living by moral or religious standards', 'liberty': 'personal freedom and not being told what to do'}
_build1 = build
def build(r, cfg, backstory=None):
    s = _build1(r, {k: v for k, v in cfg.items() if k != 'moral'}, backstory)
    if cfg.get('moral'):
        gm = rng_for(r.subject_id, 'moral'); txt = " When you decide what is right, " + "; ".join(f"{MORAL[v]} matters {['a little', 'somewhat', 'a lot'][gm.integers(0, 3)]} to you" for v in MORAL) + "."
        i = s.find("\nYour views on specific issues:"); s = s + txt if i < 0 else s[:i] + txt + s[i:]
    return s
VARIANTS.update({'MORAL': {'moral': True}, 'ENG_MORAL': {'eng': True, 'moral': True}, 'ENG_VALUES_k2.5': {'eng': True, 'values': True, 'dev': True, 'dev_k': 2.5},
                 'ENG_VALUES_englow': {'eng': True, 'values': True, 'p_eng': {'high': 0.15, 'mid': 0.35, 'low': 0.50}}, 'ENG_VALUES_MORAL': {'eng': True, 'values': True, 'moral': True}})
# --- wave 5 (GPT-6.1-Sol per-model CAL grid, fixed in advance as secondary): stronger issue deviations ---
VARIANTS.update({'LPP_k0.4': {'eng': True, 'dev': True, 'dev_k': 0.4}, 'LPP_k0.25': {'eng': True, 'dev': True, 'dev_k': 0.25},
                 'VALUES_LPP_k0.4': {'values': True, 'eng': True, 'dev': True, 'dev_k': 0.4}, 'VALUES_LPP_k0.25': {'values': True, 'eng': True, 'dev': True, 'dev_k': 0.25}})
# held-out-domain version of the selected persona
VARIANTS.update({'VALUES_LPPH_k1.6': {'values': True, 'eng': True, 'dev': True, 'dev_k': 1.6, 'doms': ['climate', 'wall', 'amtrak']}})
VARIANTS.update({'LPP_k0.1': {'eng': True, 'dev': True, 'dev_k': 0.1}, 'VALUES_LPP_k0.1': {'values': True, 'eng': True, 'dev': True, 'dev_k': 0.1}})
# --- engagement-coupled deviations (Converse 1964: constraint concentrated among the engaged) ---
VARIANTS.update({'CONV_A': {'values': True, 'eng': True, 'dev': True, 'k_by_eng': {'high': 2.5, 'mid': 1.6, 'low': 0.6}},
                 'CONV_B': {'values': True, 'eng': True, 'dev': True, 'k_by_eng': {'high': None, 'mid': 1.6, 'low': 0.6}},
                 'CONV_C': {'values': True, 'eng': True, 'dev': True, 'k_by_eng': {'high': None, 'mid': 1.0, 'low': 0.4}},
                 'CONV_D': {'values': True, 'eng': True, 'dev': True, 'k_by_eng': {'high': None, 'mid': 2.5, 'low': 1.0}},
                 'CONV_B_novalues': {'eng': True, 'dev': True, 'k_by_eng': {'high': None, 'mid': 1.6, 'low': 0.6}}})
# per-model scale of the engagement-coupled deviations (GPT selects its overall deviation scale on the dev half, as for k)
VARIANTS.update({f'CONV_B_s{s}': {'values': True, 'eng': True, 'dev': True, 'k_by_eng': {'high': None, 'mid': s, 'low': round(s * 0.375, 4)}} for s in (0.1, 0.25, 0.6)})
# API-model variants: LPP components combined with the diversity instruction (selected on the dev half)
VARIANTS.update({f'{k}_div': dict(VARIANTS[k], diverse=True) for k in ('CONV_B_s0.1', 'CONV_B_s0.25', 'CONV_B', 'ENG_VALUES')})
# --- reviewer controls: absolute (not party-relative) deviations, placebo (non-political) attributes, second latent seed ---
PLACEBO_A = {'spending time outdoors': 0, 'cooking at home': 0, 'watching sports': 0, 'reading books': 0, 'listening to music': 0, 'spending time with pets': 0}
PLACEBO_L = [('your favorite season', ['spring', 'summer', 'autumn', 'winter']), ('your usual breakfast', ['eggs and toast', 'cereal', 'just coffee', 'fruit and yogurt']),
             ('the kind of vacation you prefer', ['the beach', 'the mountains', 'a big city', 'staying home']), ('the music you listen to most', ['country', 'pop', 'classic rock', 'hip-hop']),
             ('your morning routine', ['getting up early', 'sleeping in when you can', 'exercising first thing', 'reading the news over coffee'])]
_build2 = build
def build(r, cfg, backstory=None):
    if not cfg.get('placebo'): return _build2(r, cfg, backstory)
    s = _build2(r, {k: v for k, v in cfg.items() if k not in ('placebo', 'values', 'dev')}, backstory); gp = rng_for(r.subject_id, 'placebo')
    s += " In your free time, " + "; ".join(f"{a} is {VLEV[gp.integers(0, 3)]} to you" for a in PLACEBO_A) + "."
    s += "\nSome details about you:\n" + "\n".join(f"- When it comes to {k}, you prefer {o[gp.integers(0, len(o))]}." for k, o in PLACEBO_L)
    return s
VARIANTS.update({'CONV_B_abs': dict(VARIANTS['CONV_B'], dev_style='abs'), 'CONV_B_placebo': {'eng': True, 'placebo': True}, 'CONV_B_seed2': dict(VARIANTS['CONV_B'], salt='#2')})
# --- sensitivity of the engagement prior (high/mid/low): fewer engaged vs. more engaged than the default (.25/.35/.40) ---
VARIANTS.update({'CONV_B_engLow': dict(VARIANTS['CONV_B'], p_eng={'high': 0.15, 'mid': 0.35, 'low': 0.50}),
                 'CONV_B_engHigh': dict(VARIANTS['CONV_B'], p_eng={'high': 0.40, 'mid': 0.40, 'low': 0.20})})
# --- budget-matched baseline: alternative wordings of the diversity instruction (selected on the development half like everything else) ---
DIVERSE_TEXTS = {
 'DIVERSE2': " Remember that this person's opinions are their own and may differ from what most people in their party think.",
 'DIVERSE3': " People who share a party often disagree on specific issues. Answer based on this person's individual circumstances and values, not on party stereotypes.",
 'DIVERSE4': " Avoid stereotypes: this person holds their own combination of views, which need not be consistent across issues.",
 'DIVERSE5': " Think about this person's own life and experiences when answering; real people's opinions are often inconsistent across issues and only loosely tied to party.",
}
_build3 = build
def build(r, cfg, backstory=None):
    if cfg.get('diverse_text'): return _build3(r, {k: v for k, v in cfg.items() if k != 'diverse_text'}, backstory) + cfg['diverse_text']
    return _build3(r, cfg, backstory)
VARIANTS.update({k: {'diverse_text': t} for k, t in DIVERSE_TEXTS.items()})
# --- candidate refinement: latent predispositions on top of a model-written life story (richer between-person heterogeneity) ---
VARIANTS.update({'CONV_B_story': dict(VARIANTS['CONV_B'], backstory=True)})
# --- reviewer controls: more latent seeds; a one-dimensional latent (a continuous ideology score varying within party) ---
VARIANTS.update({f'CONV_B_seed{k}': dict(VARIANTS['CONV_B'], salt=f'#{k}') for k in (3, 4, 5, 6)})
_build4 = build
def build(r, cfg, backstory=None):
    if not cfg.get('ideo1d'): return _build4(r, cfg, backstory)
    g = rng_for(r.subject_id, 'ideo1d'); x = int(np.clip(round(g.normal(50 + 10 * (int(r.pid_7) - 4), 15)), 0, 100))
    return _build4(r, {k: v for k, v in cfg.items() if k != 'ideo1d'}, backstory) + f" On a scale from 0 (very liberal) to 100 (very conservative), your overall political position is {x}."
VARIANTS.update({'IDEO1D': {'ideo1d': True}})
