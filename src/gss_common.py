"""GSS 2024 transfer benchmark (probability sample, different item formats). 12 items co-asked on one ballot pair,
5 issue domains. Personas are built from the same respondents' demographics; human answers are coded in the printed option order.
CAL/EVAL halves (seeded) are used only for the OT marginal map (fit on CAL, scored on EVAL); persona-latent settings are frozen
from the CEK development choice (selected_cfg.json) and are NOT tuned on GSS."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import zlib, json, numpy as np, pandas as pd
from variants import ENG, VALUES, VLEV, MORAL, bins
GSS = DATA + 'gss/gss_sub.parquet'
CONF = "I am going to name some institutions in this country. As far as the people running these institutions are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?"
WRONG = ['Always wrong', 'Almost always wrong', 'Wrong only sometimes', 'Not wrong at all']
# (item, domain, question, options in printed order, map from GSS code -> option index 1..K)
ITEMS = [
 ('eqwlth', 'econ', "Some people think that the government in Washington ought to reduce the income differences between the rich and the poor, perhaps by raising the taxes of wealthy families or by giving income assistance to the poor. Others think that the government should not concern itself with reducing this income difference between the rich and the poor. Where would you place yourself on this scale?",
  ['The government should reduce income differences', '2', '3', '4', '5', '6', 'The government should not concern itself with reducing income differences'], {i: i for i in range(1, 8)}),
 ('helppoor', 'econ', "Some people think that the government in Washington should do everything possible to improve the standard of living of all poor Americans. Other people think it is not the government's responsibility, and that each person should take care of himself. Where would you place yourself on this scale?",
  ['The government should help', '2', 'Agree with both', '4', 'People should take care of themselves'], {i: i for i in range(1, 6)}),
 ('cappun', 'crime', "Do you favor or oppose the death penalty for persons convicted of murder?", ['Favor', 'Oppose'], {1: 1, 2: 2}),
 ('gunlaw', 'crime', "Would you favor or oppose a law which would require a person to obtain a police permit before he or she could buy a gun?", ['Favor', 'Oppose'], {1: 1, 2: 2}),
 ('abany', 'moral', "Please tell me whether or not you think it should be possible for a pregnant woman to obtain a legal abortion if the woman wants it for any reason.", ['Yes', 'No'], {1: 1, 2: 2}),
 ('homosex', 'moral', "What about sexual relations between two adults of the same sex: do you think it is always wrong, almost always wrong, wrong only sometimes, or not wrong at all?", WRONG, {i: i for i in range(1, 5)}),
 ('xmarsex', 'moral', "What is your opinion about a married person having sexual relations with someone other than the marriage partner: is it always wrong, almost always wrong, wrong only sometimes, or not wrong at all?", WRONG, {i: i for i in range(1, 5)}),
 ('colath', 'liberty', "There are always some people whose ideas are considered bad or dangerous by other people. For instance, somebody who is against all churches and religion. Should such a person be allowed to teach in a college or university, or not?", ['Yes, allowed to teach', 'Not allowed'], {4: 1, 5: 2}),
 ('pornlaw', 'liberty', "Which of these statements comes closest to your feelings about pornography laws?", ['There should be laws against the distribution of pornography whatever the age', 'There should be laws against the distribution of pornography to persons under 18', 'There should be no laws forbidding the distribution of pornography'], {1: 1, 2: 2, 3: 3}),
 ('confed', 'trust', CONF + " The executive branch of the federal government.", ['A great deal', 'Only some', 'Hardly any'], {1: 1, 2: 2, 3: 3}),
 ('conpress', 'trust', CONF + " The press.", ['A great deal', 'Only some', 'Hardly any'], {1: 1, 2: 2, 3: 3}),
 ('conarmy', 'trust', CONF + " The military.", ['A great deal', 'Only some', 'Hardly any'], {1: 1, 2: 2, 3: 3}),
]
KEYS = [i[0] for i in ITEMS]; K = {i[0]: len(i[3]) for i in ITEMS}; DOMOF = {i[0]: i[1] for i in ITEMS}
DOM = ['econ', 'crime', 'moral', 'liberty', 'trust']
DTXT = {'econ': 'government help for the poor and reducing income inequality', 'crime': 'crime, guns and the death penalty', 'moral': 'abortion and sexual morality',
        'liberty': 'free speech and censorship', 'trust': 'trust in national institutions such as the government, the press and the military'}
PID = {0: 'a strong Democrat', 1: 'a not very strong Democrat', 2: 'an independent who leans Democratic', 3: 'an independent', 4: 'an independent who leans Republican', 5: 'a not very strong Republican', 6: 'a strong Republican', 7: 'a supporter of a party other than the Democrats or Republicans'}
IDEO = {1: 'extremely liberal', 2: 'liberal', 3: 'slightly liberal', 4: 'moderate, middle of the road', 5: 'slightly conservative', 6: 'conservative', 7: 'extremely conservative'}
DEG = {0: 'less than high school', 1: 'high school', 2: 'a junior college degree', 3: "a bachelor's degree", 4: 'a graduate degree'}
RACE = {1: 'White', 2: 'Black', 3: ''}
def load(year=2024):
    d = pd.read_parquet(GSS); d = d[d.year == year]
    d = d[d[['age', 'sex', 'race', 'degree', 'partyid', 'polviews'] + KEYS].notna().all(axis=1)].copy(); d['sid'] = 'g' + d.id.astype(int).astype(str)
    for k, _, _, _, mp in ITEMS: d[k] = d[k].astype(int).map(mp)
    d = d.dropna(subset=KEYS); perm = np.random.default_rng(7).permutation(len(d)); d['split'] = np.where(np.arange(len(d))[np.argsort(perm)] % 2 == 0, 'cal', 'eval')
    d['pid7'] = d.partyid.astype(int); return d.set_index('sid', drop=False)
def demo(r):
    race = RACE[int(r.race)]; who = f"{int(r.age)}-year-old {race + ' ' if race else ''}{'woman' if int(r.sex) == 2 else 'man'}"
    return (f"You are a {who} living in the United States. Your highest level of education is {DEG[int(r.degree)]}. Generally speaking, you think of yourself as {PID[int(r.partyid)]}, "
            f"and when it comes to politics you think of yourself as {IDEO[int(r.polviews)]}.")
def persona(r, cfg):
    s = demo(r); g = np.random.default_rng(zlib.crc32(f"{r.sid}|lpp{cfg.get('salt', '')}".encode()))
    p_eng = cfg.get('p_eng', {'high': 0.25, 'mid': 0.35, 'low': 0.40}); e = g.choice(list(p_eng), p=list(p_eng.values())); z = dict(zip(DOM, g.normal(0, 1, len(DOM))))
    if cfg.get('eng'): s += " " + ENG[e]
    if cfg.get('values'):
        gv = np.random.default_rng(zlib.crc32(f"{r.sid}|values{cfg.get('salt', '')}".encode())); s += " In your life, " + "; ".join(f"{VALUES[v][0]} is {VLEV[gv.integers(0, 3)]} to you" for v in VALUES) + "."
    if cfg.get('moral'):
        gm = np.random.default_rng(zlib.crc32(f'{r.sid}|moral'.encode())); s += " When you decide what is right, " + "; ".join(f"{MORAL[v]} matters {['a little', 'somewhat', 'a lot'][gm.integers(0, 3)]} to you" for v in MORAL) + "."
    if cfg.get('diverse_text'): s += cfg['diverse_text']
    if cfg.get('diverse'): s += " Keep in mind that real people often hold mixed views that do not follow a single party line; answer as this specific individual, not as a typical member of a group."
    kk = cfg.get('k_by_eng', {}).get(e, cfg.get('dev_k', 1.0))  # Converse coupling
    if cfg.get('dev') and kk is not None:
        B = bins(kk)
        s += "\nYour views on specific issues:\n" + "\n".join(f"- On {DTXT[d]}, your views are {next(t for lo, hi, t in B if lo <= z[d] < hi)} most people who share your general politics." for d in cfg.get('doms', DOM))
    return s
def question(it):
    k, _, q, opts, _ = it; return q + "\n" + "\n".join(f"{i + 1} = {o}" for i, o in enumerate(opts) if o != str(i + 1)) + f"\nAnswer with a single number from 1 to {len(opts)}."
