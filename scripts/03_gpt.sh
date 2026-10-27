#!/bin/bash
# GPT-6.1-Sol on the CEK test half through the Codex CLI (one joint answer per persona; a second call for reliability).
set -e; cd "$(dirname "$0")/../src"; M=gpt-6.1-sol; E=low; W=${WORKERS:-8}
python gpt_variants.py $M $E cal 100 $W CONV_B,CONV_B_s0.1,VALUES_LPP_k0.1   # deviation scale for the API model (development half)
python gpt_to_csv.py $M-$E cal 100
python gpt_variants.py $M $E eval 311 $W BASE,DIVERSE,CONV_B_s0.1
python gpt_variants.py $M $E eval 311 $W 'BASE#2,DIVERSE#2,CONV_B_s0.1#2'
python gpt_to_csv.py $M-$E eval 311
python gpt_retest.py $M-$E eval 311 BASE,DIVERSE,CONV_B_s0.1
