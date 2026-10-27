# Latent-Predisposition Personas

Code for *Persons, Not Noise: Latent Predispositions for Heterogeneous and Reliable LLM-Simulated Publics*.

LLM personas built from a demographic profile tend to answer every issue along party lines. Raising the temperature or adding
backstories breaks that up, but mostly by making each simulated person answer at random. LPP instead gives every persona a few
latent predispositions drawn from a fixed prior (political engagement, value priorities, and issue-specific deviations from
its own party), sharpens its answer distribution, and calibrates the population to aggregate toplines and party crosstabs.
The paper compares this against six baseline persona types on three U.S. surveys and two open models, at matched test-retest
reliability.

Everything needed to go from the raw survey files to the tables and figures in the paper is here. Model outputs are not
included (see below).

## Setup

Python 3.12. The versions in `requirements.txt` are the ones we ran; newer `transformers` releases should work but the Gemma
loader touches its internals, so pin if something breaks.

```
pip install -r requirements.txt
```

Hardware: Qwen3.8-27B runs in 4-bit NF4 on one 24 GB GPU. Gemma-4-26B-A4B needs two 24 GB GPUs (FP8 experts, about 26 GB in
total) and is loaded by `src/gemma26.py`, because the FP8 checkpoint stores its experts in a layout that `transformers` does not
read directly. Scoring the 311 CEK test personas on 14 items takes about 25 minutes with Qwen and 6 minutes with Gemma.

The GPT-6.1-Sol runs went through the OpenAI Codex CLI (`codex exec`), so `src/gpt_variants.py` expects that CLI to be installed
and logged in. Nothing else talks to an API.

Paths are set in `src/paths.py` and can be overridden:

| variable | default | contents |
|---|---|---|
| `LPP_DATA` | `data/` | raw survey files |
| `LPP_RESULTS` | `results/` | model outputs, evaluation files |
| `LPP_TABLES` | `tables/` | LaTeX tables, macros, figures |
| `LPP_MODEL_QWEN3_8_27B`, `LPP_MODEL_GEMMA4_26B` | Hugging Face ids | local model directories, if you have them |

## Data

None of the survey data is redistributed here. `data/README.md` lists where to download each file and where to put it.
`results/exp2_split.json` (the development/test split of the CEK respondents) and `results/selected_cfg.json` (the configuration
chosen on the development half) are included, so the test-set numbers can be reproduced without redoing the selection.

## Running it

`scripts/` has one shell script per stage, in order. Each is short; read it before running, since the model stages take hours.

1. `00_prepare_data.sh` builds the GSS extract, the GSS panel retest values, the CEK split and the human framing-effect matrix.
2. `01_dev_selection.sh` runs the configuration search on the CEK development half (Qwen only) and writes
   `results/selected_cfg.json`. Skip it to use the shipped file.
3. `02_generate.sh MODEL` scores every persona type on the three test sets with one model (`qwen3.8-27b` or `gemma4-26b`).
4. `03_gpt.sh` runs the GPT-6.1-Sol conditions.
5. `04_evaluate.sh` computes all metrics over the 200 paired half-samples, plus the robustness checks.
6. `05_tables.sh` writes the tables, the macros used in the text, and Figure 3 into `tables/`.

The persona configurations are named in `src/variants.py`. `CONV_B` is LPP as reported, `CONV_B_seed2` to `CONV_B_seed6` are the
other latent draws, `VALUES_LPP_k1.6` is the uncoupled variant, and `BASE`, `DIVERSE`, `DIVERSE5`, `ARGYLE`, `BACKSTORY`,
`ANTHOLOGY` are the baselines. A suffix such as `+sharp0.15+OT` in the evaluation files means sharpening with tau = 0.15 followed
by calibration; `+OT` alone is calibration only.

## Notes

- Latents are seeded by respondent id, so re-running a model reproduces the same persona texts. Answer probabilities can still
  differ in the last digits across GPUs and library versions.
- Reliability for the open models is computed from the answer distributions, not from two sampled answers; `src/final_eval.py`
  has the formula next to the code.
- `results/selected_cfg.json` records the post-processing chosen for each baseline as well. Changing it changes what the
  "selected" rows in the tables mean.

## License

MIT for the code (see `LICENSE`). The survey data keep their own terms.
