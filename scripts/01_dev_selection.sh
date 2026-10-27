#!/bin/bash
# Configuration search on the CEK development half with Qwen (100 simulated respondents). Writes results/selected_cfg.json.
# The shipped selected_cfg.json is the outcome of this script; running it again overwrites that file.
set -e; cd "$(dirname "$0")/../src"; M=qwen3.8-27b; DEV=${DEV:-cuda:0}
LPP="CONV_A CONV_B CONV_C CONV_D CONV_B_novalues CONV_B_story DEV DEV_abs ENG ENG_MORAL ENG_VALUES ENG_VALUES_MORAL ENG_VALUES_englow ENG_VALUES_k2.5
     LPP LPPH LPPH_k1.6 LPP_DIVERSE LPP_abs LPP_enghigh LPP_englow LPP_k0.6 LPP_k1.6 LPP_k2.5 MORAL VALUES VALUES_LPP VALUES_LPPH VALUES_LPP_k0.6
     VALUES_LPP_k1.6 ARGYLE_LPP BACKSTORY_LPP"
BASELINES="BASE DIVERSE DIVERSE2 DIVERSE3 DIVERSE4 DIVERSE5 ARGYLE BACKSTORY IDEO1D"
python gen_backstory.py $M $DEV cal 100
python run_variants.py $M $DEV cal 100 $(echo $LPP $BASELINES | tr ' ' ',')
python anthology.py $M $DEV cal 100 400
OTP=0 python var_analyze.py $M cal 100 30
OTP=1 python var_analyze.py $M cal 100 30
python select_cfg.py $M --write    # LPP configuration and sharpening
python select_baselines.py $M      # post-processing of every baseline
python select_diverse.py           # best of the five diversity wordings
python dev_offset.py $M coupled; python dev_offset.py $M uncoupled   # LLM-free offset controls
python dev_engsharp.py $M          # engagement-graded sharpening (appendix check)
