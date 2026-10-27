"""Non-LLM control for the issue deviations: the demographic persona's answer distributions are tilted directly by the same
sampled latents that LPP states in text (engagement level, engagement-coupled deviation bins per domain), with no prompting.
Bin level l in {-2..2} (liberal..conservative relative to the party); item direction s_j = sign of the Republican-minus-Democrat
difference in the demographic persona's own expected answers (model output only, no human answers).
P'(k) ~ P(k) exp(beta * l * s_j * (k - 4) / 3). coupled=False: persistent persona x domain shifts with one bin scale for all personas
(no engagement coupling), the theory-free variant."""
import numpy as np
from variants import rng_for, bins, DOM, ITEMS, VARIANTS
CFG = VARIANTS['CONV_B']; DOMOF = [d for _, d, *_ in ITEMS]
def levels(sids, coupled=True):
    L = np.zeros((len(sids), len(DOM)))
    for i, s in enumerate(sids):
        g = rng_for(s, 'lpp'); p = CFG.get('p_eng', {'high': 0.25, 'mid': 0.35, 'low': 0.40})
        e = g.choice(list(p), p=list(p.values())); z = g.normal(0, 1, len(DOM)); kk = CFG['k_by_eng'][e] if coupled else 1.0  # uncoupled: one scale for everyone
        if kk is None: continue
        B = bins(kk); L[i] = [next(t for t, (lo, hi, _) in enumerate(B) if lo <= x < hi) - 2 for x in z]
    return L
def tilt(P, sids, pid, beta, coupled=True):
    x = np.arange(1, 8); E = P @ x; R, D = pid >= 5, pid <= 3; s = np.sign(E[R].mean(0) - E[D].mean(0))
    L = levels(sids, coupled); Lj = L[:, [DOM.index(d) for d in DOMOF]]
    Q = P * np.exp(beta * (Lj * s)[:, :, None] * (x - 4) / 3); Q = Q * (P > 0); return Q / Q.sum(-1, keepdims=True)
