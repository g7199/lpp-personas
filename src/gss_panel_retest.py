"""Human test-retest reference for GSS: correlation between waves 1 and 2 (2006 and 2008) of the GSS 2006-10 panel for the same
12 items, coded as in gss_common.ITEMS. Writes results/gss_panel_retest.json. Usage: python gss_panel_retest.py"""
import os, json, numpy as np, pandas as pd
from paths import DATA, RES
from gss_common import ITEMS
d = pd.read_stata(os.path.join(DATA, 'gsspanel', 'GSS_panel06w123_R6a - Stata.dta'), convert_categoricals=False)
def code(x, mp): return mp.get(int(x)) if pd.notna(x) and int(x) == x else None
out = {}
for k, _, _, _, mp in ITEMS:
    a, b = d[k + '_1'].map(lambda x: code(x, mp)), d[k + '_2'].map(lambda x: code(x, mp)); ok = a.notna() & b.notna()
    out[k] = round(float(np.corrcoef(a[ok].astype(float), b[ok].astype(float))[0, 1]), 3)
json.dump(out, open(os.path.join(RES, 'gss_panel_retest.json'), 'w')); print(out, 'mean', round(np.mean(list(out.values())), 3))
