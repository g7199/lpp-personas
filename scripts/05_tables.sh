#!/bin/bash
# Tables, text macros and Figure 3 into $LPP_TABLES (default tables/).
set -e; cd "$(dirname "$0")/../src"
python make_tables.py otp; python make_tables.py pool; python make_transfer.py otp; python make_transfer.py pool
python make_diag.py; python make_diag.py DIVERSE; python tab_gpt.py; python make_matched.py 0.9
python make_robust.py; python make_hmdyn_table.py; python fig_frontier.py; python make_macros.py
