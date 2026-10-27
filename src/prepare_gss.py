"""Extract the GSS variables used here from the cumulative GSS file (gss7224_r3a.dta, 1972-2024 cross-sections) into
data/gss/gss_sub.parquet. Usage: python prepare_gss.py"""
import os, json, pyreadstat
from paths import DATA
COLS = ['year', 'id', 'age', 'sex', 'race', 'degree', 'partyid', 'polviews', 'eqwlth', 'helppoor', 'cappun', 'gunlaw', 'abany', 'homosex',
        'xmarsex', 'colath', 'pornlaw', 'confed', 'conpress', 'conarmy']
SRC = os.path.join(DATA, 'gss', 'GSS_stata', 'gss7224_r3a.dta')
df, meta = pyreadstat.read_dta(SRC, usecols=COLS, apply_value_formats=False)
df.to_parquet(os.path.join(DATA, 'gss', 'gss_sub.parquet'))
json.dump({c: {str(k): v for k, v in meta.variable_value_labels.get(c, {}).items()} for c in COLS}, open(os.path.join(DATA, 'gss', 'labels.json'), 'w'), indent=0)
print(df.shape)
