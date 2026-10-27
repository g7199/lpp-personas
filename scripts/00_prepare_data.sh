#!/bin/bash
# Raw survey files -> extracts and human reference values. Needs the files listed in data/README.md.
set -e; cd "$(dirname "$0")/../src"
python prepare_gss.py              # data/gss/gss_sub.parquet
python gss_panel_retest.py         # results/gss_panel_retest.json
python prepare_cek_split.py        # results/exp2_split.json (also shipped; regenerating gives the same split)
python hm_human.py                 # results/hm_human.json (human framing-effect matrix)
python retest_by_strength.py       # results/retest_by_strength.json
