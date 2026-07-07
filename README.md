# Knowing When to Ask: Multi-Model Disagreement as a Triage Signal for Biomedical Concept-Mapping Quality

Code, data, prompts, and analysis for the IEEE BIBM 2026 paper on treating
LLM-based OMOP/SSSOM mapping-quality assessment as a **selective-prediction**
(abstain-and-route) problem.

If you use this benchmark or code, please cite the paper (see [Citation](#citation)).

---

## What this repository contains

Four LLM judges (Claude Opus 4.7, GPT-5.5, DeepSeek V4 Pro, Llama 4 Maverick)
grade biomedical concept mappings on the five-tier SSSOM scale
(`exact`, `close`, `narrow`, `broad`, `related`). On a 220-row expert-graded
benchmark we report a scoped, preliminary set of findings:

- A **single strong model** (Claude, 80.9% accuracy) performs **at least as well
  as** a four-model jury (77.3%); on semantic cost-matrix κ the two are
  statistically indistinguishable (0.840 vs. 0.835), stable across five cost
  matrices. Majority voting does not improve on the best single model.
- **Multi-model disagreement** (vote margin) is a more *discriminative*
  abstention signal than verbalized confidence (risk–coverage AURC 0.043 vs.
  0.106). A held-out check shows the operating-point gain does **not** yet
  generalize at this sample size — we report this negative result rather than
  feature an in-sample operating point.
- The `close` tier collapses for models (jury 0.19) and is also where two human
  curators agree least (0.20 on a 34-row overlap), suggesting substantial
  ambiguity in that tier.

The practical hypothesis (to be confirmed at larger scale): **label with the
best single model, and use ensemble disagreement to flag rows for curator
review.**

The repo reproduces every number in the paper from cached model outputs, and
re-runs the live pipeline if you have API access.

---

## Repository layout

```
IEEE-BIBM/
├── README.md                     ← you are here
├── RUNBOOK.md                    ← operational notes
│
├── run_benchmark_v4.py           ← live jury runner (verbalized-confidence)
├── run_benchmark_live.py         ← earlier label-only runner
├── sample_holdout.py             ← builds the OHDSI external probe
├── analyze_bibm_v2.py            ← results JSON → metrics/figures/stats
├── analyze_results.py            ← lighter analyzer (label-only runs)
├── baseline_sapbert.py           ← SapBERT cosine + percentile-threshold baseline
├── baseline_majority.py          ← majority-class floor
│
├── extract_for_grading.py        ← builds the blind second-curator grading file + hidden key
├── integrate_graded.py           ← computes inter-curator agreement + merges new rows
├── csv_to_xlsx.py                ← converts a grading CSV to a dropdown-validated .xlsx
│
├── prompts/
│   └── v4_verbalized_confidence.txt   ← the exact judge prompt
│
├── data/
│   ├── curator_spreadsheet_v2.csv     ← 220-row primary benchmark (two curators)
│   ├── curator_spreadsheet.csv        ← original 50-row benchmark (curator 1)
│   ├── holdout_sample.csv             ← 80-row OHDSI Athena external probe
│   ├── answer_key.csv                 ← hidden reference labels for the grading study
│   └── mappings.json                  ← source mapping pool the samples draw from
│
├── results/
│   ├── benchmark_results_v5.json      ← raw judge outputs, 220-row benchmark (main run)
│   ├── stats_v5.json                  ← analyzer output (cost-κ, AURC, calibration)
│   ├── baseline_sapbert_v2.json       ← SapBERT on 220 rows
│   ├── baseline_majority_v2.json      ← majority-class on 220 rows
│   └── holdout_results.json           ← raw judge outputs, OHDSI probe
│
└── examples/
    ├── motivating_example_v3.py       ← single-row motivating example (imported by runner)
    ├── motivating_example_v3_output.json
    └── motivating_example_v3_run.txt
```

> **Paths take arguments.** No script hard-codes a location; each takes explicit
> `--csv` / `--in` / `--out` flags. Adjust the commands below if your local
> layout differs.

---

## Requirements

- **Python 3.10+**
- Analysis only (no API): `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- SapBERT baseline: `torch`, `transformers` (CPU is fine)
- Second-curator xlsx: `openpyxl`
- Live jury: an [OpenRouter](https://openrouter.ai) API key

```bash
pip install numpy pandas scikit-learn matplotlib torch transformers openpyxl
```

> **NumPy 2.x note:** `analyze_bibm_v2.py` uses `np.trapezoid` (the NumPy ≥2.0
> name for `np.trapz`). On NumPy 1.x it falls back automatically.

---

## Quick start: reproduce the paper from cached results (no API key)

```bash
# 1. Primary-benchmark metrics (cost-κ, AURC, calibration, per-tier)
python analyze_bibm_v2.py --in results/benchmark_results_v5.json --out results/stats_v5.json

# 2. SapBERT baseline
python baseline_sapbert.py --csv data/curator_spreadsheet_v2.csv --out results/baseline_sapbert_v2.json

# 3. Majority-class floor
python baseline_majority.py --in results/benchmark_results_v5.json --out results/baseline_majority_v2.json

# 4. (optional) External probe metrics
python analyze_bibm_v2.py --in results/holdout_results.json --out results/holdout_stats.json
```

Expected headline numbers (220-row benchmark):

| Method            | Accuracy | cost-κ |
|-------------------|:--------:|:------:|
| Claude (single)   | 0.809    | 0.840  |
| GPT               | 0.791    | 0.815  |
| LLM jury          | 0.773    | 0.835  |
| Llama             | 0.664    | 0.709  |
| DeepSeek*         | 0.650    | 0.792  |
| SapBERT           | 0.673    | —      |
| Majority class    | 0.427    | 0.000  |

\* DeepSeek was run under a 500-token budget that truncated its reasoning on
35/220 rows (counted as errors); its figures understate its capability and it
should be read as a handicapped participant.

Vote-margin risk–coverage AURC **0.043** vs. mean-confidence **0.106**.
Per-tier `close` accuracy: jury **0.19**, Claude alone **0.42**.

---

## Reproducing the live jury run (requires API access)

```bash
export OPENROUTER_API_KEY="sk-or-..."

# 220-row primary benchmark
python run_benchmark_v4.py \
    --csv data/curator_spreadsheet_v2.csv \
    --per-tier 100 --seed 20260630 \
    --out results/benchmark_results_v5.json

# 80-row OHDSI external probe
python run_benchmark_v4.py \
    --csv data/holdout_sample.csv \
    --per-tier 40 --seed 20260704 \
    --out results/holdout_results.json
```

`--per-tier 100` ≥ the largest tier count (94 exact), so all 220 rows are drawn.

### Judges (OpenRouter slugs, as of July 2026)

| Judge            | Slug                            |
|------------------|---------------------------------|
| Claude Opus 4.7  | `anthropic/claude-opus-4.7`     |
| GPT-5.5          | `openai/gpt-5.5`                |
| DeepSeek V4 Pro  | `deepseek/deepseek-v4-pro`      |
| Llama 4 Maverick | `meta-llama/llama-4-maverick`   |

Inference: `temperature = 0`, `max_tokens = 500`,
`reasoning.effort = "minimal"` where exposed; two-field `LABEL` / `CONFIDENCE`
response.

---

## The two-curator study (inter-curator agreement)

The 220-row benchmark merges the original 50 rows (curator 1) with 170 new rows
graded by a **second, independent curator**, who also re-graded a 34-row overlap
blind to enable an inter-curator agreement estimate.

```bash
# 1. Build the blind grading file + hidden answer key
python extract_for_grading.py \
    --pool data/mappings.json \
    --original data/curator_spreadsheet.csv \
    --exclude data/holdout_sample.csv \
    --n-new 170 --n-overlap 35 --seed 20260705 \
    --out to_grade.csv --key data/answer_key.csv

# 2. (optional) Convert to a dropdown-validated spreadsheet for the grader
python csv_to_xlsx.py --in to_grade.csv --out to_grade.xlsx

# 3. After grading, compute inter-curator agreement + merge new rows
python integrate_graded.py \
    --graded to_grade_filled.csv \
    --key data/answer_key.csv \
    --original data/curator_spreadsheet.csv \
    --out data/curator_spreadsheet_v2.csv
```

Reported inter-curator agreement (34-row overlap): raw 0.706, Cohen's κ 0.617
(unweighted), 0.757 (cost-weighted); `close`-tier agreement 1/5.

---

## Seeds

| Artifact                          | Seed       |
|-----------------------------------|------------|
| Primary benchmark sampling        | `20260630` |
| Second-curator sampling           | `20260705` |
| OHDSI external sampling           | `20260704` |
| Bootstrap resampling (CIs)        | `20260702` |

---

## Metrics reported

Total and committed accuracy; Cohen's κ (unweighted, quadratic, semantic
cost-matrix); cost-matrix sensitivity across five weightings; inter-judge
Fleiss' κ; risk–coverage AURC (a **discrimination** measure — risk = 1 −
selective accuracy); ECE and Brier before/after Platt; per-tier accuracy;
inter-curator agreement; and a held-out (70/30) threshold-selection check.
Headline accuracy figures carry 95% bootstrap CIs.

---

## Limitations (see paper §VII)

This is a preliminary study. The Claude-vs-jury accuracy difference rests on few
discordant pairs and is not robust to perturbation; the operating-point triage
gain does not survive a held-out split at N=220; the panel mixes strong and
weak judges (and DeepSeek is handicapped by its token budget); the inter-curator
overlap is 34 rows; and the OHDSI probe's gold is heuristic. Confirming the
findings needs a larger benchmark, a matched-strength jury, non-truncated
reasoning models, a continuous disagreement signal, and a larger dual-curated
overlap.

---

## License and data governance

Graded SSSOM labels are not protected health information. Source concept
mappings are redistributed under the originating datasets' public releases
(primary source dataset; OHDSI Athena vocabulary license). Consult those before
redistributing. Code released under the MIT License (see `LICENSE`).

---

## Citation

```bibtex
@inproceedings{TODO_bibm2026_triage_signal,
  title     = {Knowing When to Ask: Multi-Model Disagreement as a Triage Signal
               for Biomedical Concept-Mapping Quality},
  author    = {TODO},
  booktitle = {2026 IEEE International Conference on Bioinformatics and
               Biomedicine (BIBM)},
  year      = {2026}
}
```

---

## Contact

TODO — add contact / issue-tracker guidance after acceptance.
