"""HM benchmark: Hopkins & Mummolo (2017) 'Assessing the Breadth of Framing Effects', single-frame replications (GfK, MTurk)
from Coppock (2022) Persuasion in Parallel archive (doi:10.7910/DVN/I9GSKI, frame_breath_topic.rds). Frame texts and item wordings
from the original TESS 092 questionnaire (osf.io/4kjtn). 4 conservative frames (crime, terror, health, stimulus) x 5 spending items.
Outcomes are oriented so + = conservative direction (crime/terror spending up; poor/health/stimulus spending down)."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import numpy as np, pandas as pd, pyreadr
FR = ['crime', 'terror', 'health', 'stimulus']; OUT = ['crime', 'terror', 'poor', 'health', 'stimulus']
ZMAP = {'crime': 'crime', 'terror': 'terror', 'healthcare': 'health', 'stimulus': 'stimulus', 'control': 'control'}
SIGN = {'crime': 1, 'terror': 1, 'poor': -1, 'health': -1, 'stimulus': -1}
DIAG = [(i, OUT.index(f)) for i, f in enumerate(FR)]
OFFM = np.ones((4, 5), bool)
for i, j in DIAG: OFFM[i, j] = False
INTRO = "The argument below was recently made by a U.S. Senator.  Please take a moment to read the argument carefully and then tell us what you think."
FRAMES = {
 'terror': "The September 11th attacks and the news that al-Qaeda was planning new attacks on U.S. soil show how vulnerable America still is to terrorists.  Innocent people can be killed while traveling to visit family or going to work.  Across the country, we have to do everything we can to reduce the threat of terrorism.  We have to stop terrorists before they act.  This means conducting more frequent searches of suspicious people boarding planes, trains, subways, and buses.",
 'crime': "America is very vulnerable to violent crime, with forty-two Americans murdered every single day on average.  Innocent people can be killed in their front yards.  Across the country, we have to do everything we can to reduce the threat of violent crime.   We have to stop violent criminals before they act.  This means cracking down on the smaller offenses that all too often lead to violent crime, and making sure that convicted criminals always serve out their full sentences.",
 'health': "Health care is one of the most complicated issues we face.  It involves 1 of every 6 dollars spent here in the United States.  The health care system includes millions of doctors and nurses and thousands of hospitals and clinics.  Together, they regularly make decisions that can mean life or death.  The government in Washington can't even balance its own budget.  How can we trust it to run something as complicated as the health care system?",
 'stimulus': "With a recession as deep as this one, there are more than 10 million unemployed Americans, and it's going to take years for our economy to recover.  In February 2009, the government in Washington made things worse by passing an $800 billion stimulus package, which is more than $2,500 for every person living in this country.  Now, it looks like a lot of that money didn't help the economy.  Unemployment is still very high.  The money went to pork-barrel projects and federal bureaucrats rather than creating jobs for unemployed Americans.  The government in Washington can't even balance its own budget.  How can we trust it to spend so much taxpayer money?",
}
QTXT = {'crime': "Should federal spending on dealing with crime be increased, decreased, or kept the same?",
        'terror': "Should federal spending on the war on terrorism be increased, decreased, or kept the same?",
        'poor': "Should federal spending on aid to the poor be increased, decreased, or kept the same?",
        'health': "Should federal spending on health care be increased, decreased, or kept the same?",
        'stimulus': "Should federal spending to stimulate the economy be increased, decreased, or kept the same?"}
OPTS = ["Increased a great deal", "Increased a moderate amount", "Increased a little", "Kept about the same", "Decreased a little", "Decreased a moderate amount", "Decreased a lot"]  # human order (polarity randomized in original; data coded 7 = increased a great deal)
def load():
    d = pyreadr.read_r(DATA + 'hm/frame_breath_topic.rds')[None]
    w = d[d['sample'].isin(['gfk', 'mt'])].drop_duplicates('anon_ID').copy(); w['arm'] = w.Z_Hopkins.map(ZMAP)
    for o in OUT: w[o] = w[f'Y_{o}'] * SIGN[o]
    return w
def summ(T):
    T = np.asarray(T); dr = np.mean([T[i, j] for i, j in DIAG]); off = T[OFFM].mean()
    return dict(direct=dr, off=off, rho=off / dr, off_abs=np.abs(T[OFFM]).mean(), poor_col=T[:, 2].mean())
def loo_rms(T):  # leave-own-out contrast on off cells: T[k,j] - mean of other frames' effects on j
    T = np.asarray(T); c = [T[k, j] - np.mean([T[m, j] for m in range(4) if m != k]) for k in range(4) for j in range(5) if OFFM[k, j]]
    return float(np.sqrt(np.mean(np.square(c))))
PID = {1: 'a strong Democrat', 2: 'a not very strong Democrat', 3: 'an independent who leans Democratic', 4: 'an independent', 5: 'an independent who leans Republican', 6: 'a not very strong Republican', 7: 'a strong Republican'}
IDEO = {'Liberal': 'liberal', 'Moderate': 'moderate', 'Conservative': 'conservative', 'Ideology: Other': "haven't thought much about it"}
AGE = {'18 - 29': '18-29', '30 - 39': '30-39', '40 - 49': '40-49', '50 - 59': '50-59', 'More than 60': '60+'}
EDU = {'Less than High School': 'less than high school', 'High School': 'high school', 'Some College': 'some college', 'College': 'a college degree', 'Graduate School': 'graduate school'}
RACE = {'White': 'White', 'Black': 'Black', 'Hispanic': 'Hispanic', 'Race: Other': 'multiracial or other-race'}
TEMPL = [  # same two persona templates as the CEK experiments, mapped to this dataset's covariate codings
 lambda r: f"You are a {AGE[r.age_5]} year-old {RACE[r.race_4]} {'woman' if r.female == 1 else 'man'} living in the United States. Your highest level of education is {EDU[r.educ_5]}. Generally speaking, you think of yourself as {PID[int(r.pid_7)]}, and when it comes to politics you think of yourself as {IDEO[r.ideo_4]}.",
 lambda r: f"Imagine you are this survey participant. Age: {AGE[r.age_5]}. Gender: {'female' if r.female == 1 else 'male'}. Race/ethnicity: {RACE[r.race_4]}. Education: {EDU[r.educ_5]}. Party identification: {PID[int(r.pid_7)]}. Ideology: {IDEO[r.ideo_4]}. You are answering an online opinion survey.",
]
CONDS = ['control'] + FR
def personas(N=100):
    w = load(); c = w[(w.arm == 'control')].dropna(subset=['pid_7', 'ideo_4', 'age_5', 'race_4', 'educ_5', 'female'])
    return c.iloc[np.random.default_rng(0).choice(len(c), N, replace=False)]
def context(per, cond):
    return per + (f"\n\n{INTRO}\n\n{FRAMES[cond]}" if cond != 'control' else "") + "\n\nNow, we would like to ask your views on a few issues in the news. Answer as this person would.\n"
def qblock(o): return QTXT[o] + "\n" + "\n".join(f"{i + 1} = {l}" for i, l in enumerate(OPTS))
def to_data(x): return 8 - x  # answer index 1..7 (1 = increased a great deal) -> data coding (7 = increased a great deal)
