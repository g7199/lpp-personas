"""CEK data: MTurk control arm split 50/50 (seed 0) into CAL (calibration; may be used by methods) and EVAL (held-out
static target; its covariates also define evaluation personas). Human split-half reference band for static metrics.
Static metrics on 5 main DVs (CEK coded 1-7): mean W1 of marginals, corr-matrix RMSE, |mean |r| - human|,
conditional (party-3 group means) RMSE."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import json, numpy as np, pandas as pd, pyreadr
from scipy.stats import wasserstein_distance
ISS = ['climate', 'wall', 'amtrak', 'vets', 'flat']; MAIN = [f'dv_{k}_main_w1' for k in ISS]
def party3(p): return np.where(p <= 3, 'D', np.where(p >= 5, 'R', 'I'))
def static_metrics(X, Y, gX, gY):
    """X: sim (n x 5) responses, Y: human (m x 5); gX, gY party labels. Returns dict."""
    w1 = np.mean([wasserstein_distance(X[:, j], Y[:, j]) for j in range(5)])
    iu = np.triu_indices(5, 1); CX = np.corrcoef(X.T)[iu]; CY = np.corrcoef(Y.T)[iu]
    cond = np.sqrt(np.mean([(X[gX == g].mean(0) - Y[gY == g].mean(0)) ** 2 for g in ['D', 'I', 'R'] if (gX == g).sum() > 2 and (gY == g).sum() > 2]))
    return dict(w1=w1, corr_rmse=float(np.sqrt(np.mean((CX - CY) ** 2))), absr_err=float(abs(np.abs(CX).mean() - np.abs(CY).mean())), cond_rmse=float(cond),
                absr=float(np.abs(CX).mean()), sd_ratio=float(X.std(0).mean() / Y.std(0).mean()))
if __name__ == '__main__':
    m = pyreadr.read_r(DATA + 'cg/study_1_mturk_cleaned.rds')[None]
    c = m[m.Z == 'control'].dropna(subset=['pid_7', 'ideo_5', 'age_5', 'race_4', 'educ_5', 'female'] + MAIN).reset_index(drop=True)
    idx = np.random.default_rng(0).permutation(len(c)); cal, ev = c.iloc[idx[:len(c) // 2]], c.iloc[idx[len(c) // 2:]]
    json.dump(dict(cal=cal.subject_id.tolist(), eval=ev.subject_id.tolist()), open(RES + 'exp2_split.json', 'w'))
    # reference band: 1000 random half splits of the full control arm (human vs human)
    rng = np.random.default_rng(1); ref = []
    Y = c[MAIN].to_numpy(float); g = party3(c.pid_7.to_numpy())
    for _ in range(1000):
        p = rng.permutation(len(c)); a, b = p[:len(c) // 2], p[len(c) // 2:]
        ref.append(static_metrics(Y[a], Y[b], g[a], g[b]))
    R = pd.DataFrame(ref); band = R.quantile(0.95).to_dict(); band_med = R.median().to_dict()
    json.dump(dict(band95=band, median=band_med, n_cal=len(cal), n_eval=len(ev)), open(RES + 'exp2_human_band.json', 'w'), indent=1)
    print('n control', len(c), 'cal', len(cal), 'eval', len(ev)); print('human split-half 95th pct:', {k: round(v, 3) for k, v in band.items()})
    print('human eval-half static:', {k: round(v, 3) for k, v in static_metrics(ev[MAIN].to_numpy(float), cal[MAIN].to_numpy(float), party3(ev.pid_7.to_numpy()), party3(cal.pid_7.to_numpy())).items()})
