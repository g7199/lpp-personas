#!/bin/bash
# Score every persona type on the three test sets with one open model.
# Usage: scripts/02_generate.sh qwen3.8-27b|gemma4-26b   (Qwen: one GPU, DEV=cuda:0; Gemma: two GPUs, DEV is ignored)
set -e; cd "$(dirname "$0")/../src"; M=$1; DEV=${DEV:-cuda:0}; [ "$M" = gemma4-26b ] && DEV=auto
SEEDS=CONV_B_seed2,CONV_B_seed3,CONV_B_seed4,CONV_B_seed5,CONV_B_seed6
MAIN=CONV_B,BASE,DIVERSE,DIVERSE5,ARGYLE,VALUES_LPP_k1.6
ABL=CONV_B_novalues,ENG_VALUES,CONV_B_abs,CONV_B_placebo,CONV_B_engLow,CONV_B_engHigh,IDEO1D
dyn() { for f in crime terror health stimulus; do printf "%s@%s," $1 $f; done; }
export BS=${BS:-28}
# CEK test half (311 respondents)
python gen_backstory.py $M $DEV eval 311
python run_variants.py $M $DEV eval 311 $MAIN,BACKSTORY,$SEEDS,$ABL
python anthology.py $M $DEV eval 311 400
PARA=1 python run_variants.py $M $DEV eval 311 CONV_B,BASE,DIVERSE,DIVERSE5,ARGYLE,BACKSTORY   # reworded-question retest
PARA=1 python anthology.py $M $DEV eval 311 400
# GSS 2024 test half
EVAL_ONLY=1 BS=36 python gss_run.py $M $DEV $MAIN,$SEEDS
EVAL_ONLY=1 python transfer_baselines.py gss $M $DEV ARGYLE,BACKSTORY,ANTHOLOGY
# HM control arm and the four framed arms
BS=30 python hm_static_run.py $M $DEV $MAIN,$SEEDS
BS=30 python hm_static_run.py $M $DEV $(dyn CONV_B)$(dyn BASE | sed 's/,$//')
python transfer_baselines.py hm $M $DEV ARGYLE,BACKSTORY,ANTHOLOGY
