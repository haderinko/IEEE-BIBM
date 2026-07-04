# Knowing When to Ask: An LLM Jury as a Calibrated Triage Layer for Biomedical Concept-Mapping Quality

Code, data, prompts, and analysis for the IEEE BIBM 2026 paper on treating
LLM-based OMOP/SSSOM mapping-quality assessment as a selective-prediction
(abstain-and-route) problem rather than a raw-accuracy contest.

If you use this benchmark or code, please cite the paper (see [Citation](#citation)).

---

## What this repository contains

A four-judge LLM "jury" grades biomedical concept mappings on the five-tier
SSSOM scale (`exact`, `close`, `narrow`, `broad`, `related`), reports a
verbalized confidence per row, and uses vote-margin uncertainty to decide when
to **abstain and route a row to a human curator**. The headline result: at 90%
coverage the jury reaches 77.8% selective accuracy, a 7.8-point gain over
committing on every row.

The repo lets you reproduce every number in the paper from the raw model
outputs, and re-run the live jury against the models if you have API access.

---

## Repository layout

```
IEEE-BIBM/
├── README.md                     ← you are here
├── RUNBOOK.md                    ← operational notes for running the pipeline
│
├── run_benchmark_v4.py           ← live jury runner, verbalized-confidence variant (paper's v4)
├── run_benchmark_live.py         ← earlier live runner (label-only, pre-confidence)
├── sample_holdout.py             ← builds the OHDSI external benchmark from the mapping pool
├── analyze_bibm_v2.py            ← turns a results JSON into the paper's metrics/stats
├── analyze_results.py            ← lighter analyzer for the label-only runs
├── baseline_sapbert.py           ← SapBERT cosine + percentile-threshold baseline
├── baseline_majority.py          ← majority-class floor
│
├── prompts/
│   └── v4_verbalized_confidence.txt   ← the exact judge prompt (LABEL + CONFIDENCE)
│
├── data/
│   ├── curator_spreadsheet.csv        ← 50-row primary benchmark (expert SSSOM grades)
│   ├── holdout_sample.csv             ← 80-row OHDSI Athena external benchmark
│   └── mappings.json                  ← source mapping pool the holdout is sampled from
│
├── results/
│   ├── benchmark_results_v4.json      ← raw judge outputs, primary benchmark (paper's main run)
│   ├── holdout_results.json           ← raw judge outputs, OHDSI external benchmark
│   ├── baseline_sapbert.json          ← SapBERT predictions + metrics
│   └── baseline_majority.json         ← majority-class metrics
│
└── examples/
    ├── motivating_example_v3.py       ← the single-row motivating example (paper intro)
    ├── motivating_example_v3_output.json
    └── motivating_example_v3_run.txt
```

> **Paths take arguments.** None of the scripts hard-code file locations; every
> one takes explicit `--csv` / `--in` / `--out` / `--mappings` flags. The commands
> below use the foldered paths above — adjust if your local layout differs.

---

## Requirements

- **Python 3.10+**
- Analysis only (no API calls): `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- SapBERT baseline: `torch`, `transformers` (CPU is fine; the model is small)
- Live jury: an [OpenRouter](https://openrouter.ai) API key

```bash
pip install numpy pandas scikit-learn matplotlib torch transformers
```

The majority-class baseline uses only the Python standard library.

---

## Quick start: reproduce the paper from cached results (no API key needed)

Every metric in the paper regenerates from the raw judge outputs already in
`results/`, so you do **not** need model access to check our numbers.

```bash
# 1. Primary-benchmark metrics: main comparison, risk-coverage / calibration /
#    per-tier stats, aggregation ablation, confusion matrix
python analyze_bibm_v2.py --in results/benchmark_results_v4.json --out results/stats_bibm_v2.json

# 2. SapBERT baseline (accuracy, unweighted/quadratic/cost kappa, per-tier)
python baseline_sapbert.py --csv data/curator_spreadsheet.csv --out results/baseline_sapbert.json

# 3. Majority-class floor
python baseline_majority.py --in results/benchmark_results_v4.json --out results/baseline_majority.json

# 4. (optional) External-benchmark metrics
python analyze_bibm_v2.py --in results/holdout_results.json --out results/holdout_stats.json
```

Expected headline numbers (primary 50-row benchmark):

| Method            | Accuracy | cost-κ |
|-------------------|:--------:|:------:|
| **LLM jury**      | 0.700    | 0.795  |
| Best single judge | 0.700    | 0.775  |
| SapBERT           | 0.500    | 0.431  |
| Majority class    | 0.260    | 0.000  |

Jury selective accuracy at 90% coverage: **0.778**.
Vote-margin AURC **0.052** vs. mean-confidence AURC **0.207**.

Per-tier: the ordinal middle collapses — jury `close` accuracy is **0.11** on
the primary benchmark and **0.10** on the 80-row external benchmark.

---

## Reproducing the live jury run (requires API access)

This re-queries the four judges. Same-seed reruns were label-identical in our
tests, but temperature-zero determinism is not guaranteed across providers (see
[Determinism](#a-note-on-determinism)).

```bash
export OPENROUTER_API_KEY="sk-or-..."

# Primary benchmark (50 rows, all five tiers)
python run_benchmark_v4.py \
    --csv data/curator_spreadsheet.csv \
    --per-tier 10 --seed 20260630 \
    --out results/benchmark_results_v4.json

# OHDSI external benchmark (80 rows, exact + close)
python run_benchmark_v4.py \
    --csv data/holdout_sample.csv \
    --per-tier 40 --seed 20260704 \
    --out results/holdout_results.json
```

(Check `--per-tier`/`--seed` against `RUNBOOK.md`; the primary run stratifies
across five tiers to 50 rows total, the holdout to 80.)

### Rebuilding the external benchmark from scratch

```bash
python sample_holdout.py \
    --mappings data/mappings.json \
    --exclude  data/curator_spreadsheet.csv \
    --per-tier 40 --seed 20260704 \
    --out      data/holdout_sample.csv
```

This samples rows from the OHDSI Athena mapping pool that do **not** appear in
the primary benchmark, stratified by tier.

### Judges and access

Accessed via OpenRouter using these public model slugs (as of July 3, 2026):

| Judge            | OpenRouter slug                 |
|------------------|---------------------------------|
| Claude Opus 4.7  | `anthropic/claude-opus-4.7`     |
| GPT-5.5          | `openai/gpt-5.5`                |
| DeepSeek V4 Pro  | `deepseek/deepseek-v4-pro`      |
| Llama 4 Maverick | `meta-llama/llama-4-maverick`   |

### Inference settings (identical across judges)

- `temperature = 0`
- `max_tokens = 500`
- `reasoning.effort = "minimal"` (where the provider exposes it)
- Two-field structured response: `LABEL` and `CONFIDENCE` (integer 0–100)

> **Heads-up on DeepSeek:** under the 500-token budget with minimal reasoning,
> DeepSeek V4 Pro returns empty responses on some hard rows (7 of 50 on the
> primary benchmark, 13 of 80 on the external set), counted as `UNKNOWN` misses.
> Raising `max_tokens` to ~2000 recovers them but multiplies cost. We keep the
> tight budget because the failure is itself a reported finding.

---

## Seeds

| Artifact                          | Seed       |
|-----------------------------------|------------|
| Primary benchmark sampling        | `20260630` |
| OHDSI external benchmark sampling | `20260704` |
| Bootstrap resampling (95% CIs)    | `20260702` |

---

## The two benchmarks

**Primary (`data/curator_spreadsheet.csv`, N=50).** Concept mappings sampled
from a public multimodal biomedical dataset, stratified across all five SSSOM
tiers (7 exact, 9 close, 10 narrow, 13 broad, 11 related). Each row was graded
on the SSSOM scale by a domain-expert curator with a per-row confidence.

CSV columns: `row_id`, `source_label`, `target_concept_name`,
`target_omop_concept_id`, `curator_sssom_label`, `curator_confidence`.

**External (`data/holdout_sample.csv`, N=80).** Built from the OHDSI Athena
public vocabulary release via `sample_holdout.py`. Gold tiers are derived from
`CONCEPT_RELATIONSHIP.relationship_id` by a fixed rule
(`Maps to` → `exact`, `Concept replaces` → `close`); other relationship types
had no defensible one-to-one SSSOM mapping and were dropped, so only two tiers
appear. **These labels are heuristic, not curated** — the external benchmark
tests replication of a failure mode on independent public data, not a second
curator's opinion.

---

## Output JSON schema

Each row in a results JSON carries: `row_id`, `source_term`, `target_label`,
`target_omop_id`, `gold_label`, `gold_confidence`, `judges[]` (each with
`judge_name`, `model`, `label`, `raw_response`, `confidence`,
`verbalized_confidence`), `vote_counts`, `jury_label`, `jury_abstained`,
`jury_correct`, and `unknown_count`. The top-level `config` block records the
CSV path, `per_tier`, `seed`, `prompt_version`, and the judge roster.

---

## Metrics reported

Total accuracy (ABSTAIN/UNKNOWN as misses) and committed accuracy; Cohen's κ
unweighted, quadratic-weighted, and under the directional SSSOM cost matrix;
inter-judge Fleiss' κ; risk–coverage AURC; ECE and Brier before/after Platt
scaling; per-tier accuracy and the jury confusion matrix. Headline metrics
carry 95% bootstrap CIs (n=1000).

---

## A note on determinism

Temperature-zero decoding does **not** guarantee bit-exact reproducibility
across API providers because of non-deterministic kernels in the underlying
inference stacks. In our runs, same-seed reruns produced label-identical
results across all 50 primary-benchmark rows, but you may see occasional
single-row differences, especially on the hardest (`close`) rows and on the
reasoning-model judge. This does not affect the paper's qualitative findings.

---

## License and data governance

The graded SSSOM labels are not protected health information. The source
concept mappings are redistributed under the terms of the originating datasets
(the primary source dataset's public release, and the OHDSI Athena vocabulary
license). Consult those licenses before redistributing the CSVs.

Code is released under the MIT License (see `LICENSE`).

---

## Citation

```bibtex
@inproceedings{TODO_bibm2026_llm_jury,
  title     = {Knowing When to Ask: An LLM Jury as a Calibrated Triage Layer
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
