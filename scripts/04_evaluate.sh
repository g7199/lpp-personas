#!/bin/bash
# Metrics over 200 paired half-samples for every survey, model and calibration regime, plus the robustness checks. CPU only.
set -e; cd "$(dirname "$0")/../src"; export CUDA_VISIBLE_DEVICES=
for M in qwen3.8-27b gemma4-26b; do
  for D in cg gss hm; do OTP=1 python -W ignore final_eval.py $D $M 200; OTP=0 python -W ignore final_eval.py $D $M 200; done
  CALYEAR=2022 OTP=1 python -W ignore final_eval.py gss $M 200      # GSS calibrated with the 2022 wave
  python nocal_check.py $M; python para_eval.py $M; python polychoric_check.py $M; python churn.py $M
  python hm_dyn_analyze.py $M BASE,CONV_B
done
OTP=1 python -W ignore final_eval.py cg gpt-6.1-sol-low 200; OTP=0 python -W ignore final_eval.py cg gpt-6.1-sol-low 200
python hm_dyn_boot.py qwen3.8-27b BASE CONV_B > "${LPP_RESULTS:-../results}/hmdyn_boot_qwen.txt"
python hm_dyn_boot.py gemma4-26b BASE CONV_B > "${LPP_RESULTS:-../results}/hmdyn_boot_gemma.txt"
