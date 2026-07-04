# Runbook — get real numbers into the Rice abstract tonight

**Target:** by the time you go to bed, you have a stratified live run of the
AI-READI SSSOM benchmark, a stats summary, and an abstract with real
accuracy + κ numbers in place of "results forthcoming."

**Budget:** ~90 min of active work plus ~30–60 min of unattended API time.

**Skipping for tonight** (kept for the BIBM full paper, July 5):
second-curator regrade, bootstrap CIs, McNemar test, directional weighted κ
with a custom SSSOM cost matrix. None of these are abstract-critical;
unweighted κ + quadratic-weighted κ (ordinal-aware) cover the Rice claim.

---

## Phase 0 — Sanity check  (10 min)

1. **Confirm the CSV is in hand and inspect its header.**
   ```bash
   head -1 /path/to/ai_readi_mappings.csv
   wc -l /path/to/ai_readi_mappings.csv
   ```
   Open `run_benchmark_live.py` and edit the six `COL_*` constants near the
   top so they match your actual column names. The defaults assume canonical
   SSSOM (`subject_label`, `object_label`, `object_id`, `predicate_id`,
   `confidence`, `subject_id`).

2. **Check the predicate format.** Default mapping expects
   `skos:exactMatch` etc. If your file stores tiers as plain strings
   (`exact`, `close`, …) the loader already handles that; if it uses a
   different convention (e.g. `semapv:LexicalMatching`), extend
   `PREDICATE_TO_TIER` in `run_benchmark_live.py`.

3. **API keys.** Export whatever you have. Missing ones are silently
   dropped — a 3-judge run still produces a valid jury vote.
   ```bash
   export ANTHROPIC_API_KEY=...
   export OPENAI_API_KEY=...
   export DEEPSEEK_API_KEY=...
   export TOGETHER_API_KEY=...
   ```

4. **Place the two scripts next to `motivating_example_v3.py`** —
   `run_benchmark_live.py` imports from it directly.

---

## Phase 1 — Smoke test on 5 rows  (5 min)

Catch CSV/key/prompt problems before you spend money on a 250-row run.

```bash
python run_benchmark_live.py \
    --csv /path/to/ai_readi_mappings.csv \
    --per-tier 1 \
    --limit 5 \
    --out smoke.json
```

Expected: 5 rows complete, every active judge returns a valid SSSOM label
on at least most rows. Skim `smoke.json` — confirm `gold_label` looks
right and `judges[*].label` is one of the five tiers (or `UNKNOWN` rarely).
If `UNKNOWN` is showing up often, check the prompt is returning bare
labels (not "The mapping is exact." which the v3 normaliser still
catches, but excessive prose suggests a token limit issue).

Run the analyzer on the smoke output to make sure it works end-to-end:

```bash
python analyze_results.py --in smoke.json
```

---

## Phase 2 — Full live run  (30–60 min unattended)

```bash
python run_benchmark_live.py \
    --csv /path/to/ai_readi_mappings.csv \
    --per-tier 50 \
    --seed 20260630 \
    --out benchmark_results.json
```

50 per tier = up to 250 rows. The script reports population vs. drawn
counts per tier — if `narrow`, `broad`, or `related` are sparse in the
file, you'll see "under-represented" flags. That's fine for the abstract;
report whatever was available.

Results stream to disk after every row, so an API blip mid-run doesn't
cost you progress — restart and only the missed rows redo. (If a
restart is needed, just rerun the command; the existing file will be
overwritten — copy `benchmark_results.json` to `benchmark_results_partial.json`
first if you want to keep the partial.)

Rate-of-progress check every 10 rows. ~2–4 sec per row with four serial
judge calls is normal; budget 30–45 min for the full 250-row run.

---

## Phase 3 — Analyze  (5 min)

```bash
python analyze_results.py \
    --in benchmark_results.json \
    --out stats.json
```

Read the `FOR ABSTRACT` block at the bottom of the output. It gives you,
verbatim:

- N rows evaluated
- Best single judge name + accuracy + unweighted κ + quadratic-weighted κ
- Jury accuracy + unweighted κ + quadratic-weighted κ
- Inter-judge Fleiss κ
- Number of jury abstentions
- H1 verdict (jury > / ≈ / < best single judge by quadratic κ)

Save `stats.json` — it's the source of truth for both the Rice abstract
tonight and the BIBM paper next week.

---

## Phase 4 — Plug numbers into the abstract  (10 min)

Take the second paragraph of the abstract I drafted and swap the
"results forthcoming" sentence for one of the three templates below,
matching the H1 verdict the analyzer printed. Round κ to two decimals,
accuracy to whole percentages.

### Template A — H1 confirmed (jury > best single judge)

> On a stratified sample of {N} mappings drawn equally across all five
> SSSOM tiers, the strongest single judge ({best_judge}) reproduced the
> expert label on {best_acc}% of rows (Cohen's κ = {best_kappa_u},
> quadratic-weighted κ = {best_kappa_q}). The four-judge majority vote
> raised accuracy to {jury_acc}% (κ = {jury_kappa_u},
> quadratic-weighted κ = {jury_kappa_q}) and abstained on
> {n_abstain} rows where judges disagreed, routing those to human
> review. Inter-judge agreement (Fleiss κ = {fleiss}) confirms the
> task is non-trivial.

### Template B — H1 marginal (jury ≈ best single judge)

> On a stratified sample of {N} mappings drawn equally across all five
> SSSOM tiers, the strongest single judge ({best_judge}) and the
> four-judge majority vote reached comparable agreement with the expert
> labels ({best_judge} accuracy {best_acc}%, κ = {best_kappa_u}; jury
> accuracy {jury_acc}%, κ = {jury_kappa_u}). The practical payoff of the
> jury was not raw accuracy but selective abstention: on
> {n_abstain} rows judges split and the jury declined to commit,
> a built-in triage signal for curators.

### Template C — H1 negated (best single judge > jury)

> On a stratified sample of {N} mappings drawn equally across all five
> SSSOM tiers, the strongest single judge ({best_judge}) outperformed
> the four-judge majority vote (accuracy {best_acc}% vs. {jury_acc}%;
> κ {best_kappa_u} vs. {jury_kappa_u}). Naïve majority voting did not
> improve calibration in this domain; the contribution of multiple
> judges shifted instead to disagreement-based triage, with the jury
> declining to commit on {n_abstain} rows where curator review is
> most warranted.

All three are defensible benchmark findings. C is the most interesting
because it complicates the LLM-jury narrative — don't be disappointed
if that's what you get.

### Other small edits to the rest of the abstract

If you had fewer than 4 judges (some API key missing), change
"four frontier LLM judges (Claude, GPT, DeepSeek, and Llama)" to match
the actual count and names — analyzer prints them.

If `per-tier` had to drop below 50 for sparse tiers, change "stratified
across all five SSSOM tiers" to "stratified, where supply allowed,
across all five SSSOM tiers."

If `jury abstentions` exceeded ~15% of rows, mention it explicitly in
the abstract rather than burying it — high abstention rates are a
finding, not a bug.

---

## Phase 5 — Reproducibility receipts  (5 min)

Keep alongside the abstract for the supervisor and the BIBM paper:

- `stats.json` — the headline numbers and per-tier breakdown
- `benchmark_results.json` — per-row judge calls with logprobs
- `--seed 20260630` — the seed used (change it in the runbook if you
  re-roll; keep both)
- Hash the CSV so future versions don't break the comparison:
  ```bash
  sha256sum /path/to/ai_readi_mappings.csv
  ```

That's enough for a reviewer to ask "show me row AIR-00123" and get an
answer.
