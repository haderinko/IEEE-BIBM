#!/usr/bin/env python3
"""
run_benchmark_live.py
─────────────────────
Read the AI-READI SSSOM mapping table from CSV, stratify a sample across all
five SSSOM tiers, run the four-judge protocol live against the LLM APIs, and
write per-row results to JSON.

Designed as a thin driver on top of motivating_example_v3.py — reuses its
JUDGES dict, PROMPT_TEMPLATE, normalize(), majority_vote(), and JudgeResponse.

Usage
-----
    # Place this file in the same directory as motivating_example_v3.py.
    # Set API keys for whichever judges you want to include:
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    export DEEPSEEK_API_KEY=...
    export TOGETHER_API_KEY=...

    python run_benchmark_live.py \
        --csv /path/to/ai_readi_mappings.csv \
        --per-tier 50 \
        --out benchmark_results.json

Missing API keys cause that judge to be silently dropped. A run with 3 judges
is still valid (jury votes still resolve; the analysis script reports N_judges).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Optional

# Pull in everything the existing pipeline already implements.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motivating_example_v3 import (  # noqa: E402
    ABSTAIN,
    JUDGES,
    JudgeResponse,
    MODEL_VERSIONS,
    PROMPT_TEMPLATE,
    SSSOM_LABELS,
    UNKNOWN,
    _to_confidence,
    majority_vote,
    normalize,
)


# ─────────────────────────────────────────────────────────────────────────────
# OPENROUTER OVERRIDE
# Routes all four judges through https://openrouter.ai/api/v1 using one
# OPENROUTER_API_KEY env var. Overrides the JUDGES list, MODEL_VERSIONS dict,
# and JUDGE_KEY_ENV mapping imported from motivating_example_v3.
#
# If any model slug returns a 404, the slug has rotated — visit
# https://openrouter.ai/models, copy the current slug, and edit the dict.
# ─────────────────────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter model slugs (provider/model). Update if any 404.
MODEL_VERSIONS = {
    "Claude":   "anthropic/claude-opus-4.7",
    "GPT":      "openai/gpt-5.5",
    "DeepSeek": "deepseek/deepseek-v4-pro",
    "Llama":    "meta-llama/llama-4-maverick",
}


def _query_openrouter(model_slug: str, prompt: str) -> tuple[str, Optional[float]]:
    """Single OpenAI-compatible call routed through OpenRouter.

    Two tuning notes:
    1. max_tokens=500 (not 20). OpenAI's reasoning-model family (GPT-5.x)
       and DeepSeek V4 count internal reasoning tokens against this budget;
       a tight budget leaves zero room for the visible one-word label.
    2. reasoning.effort=minimal passed via extra_body. SSSOM grading is a
       single-label classification — elaborate chain-of-thought is wasted
       compute and inflates latency. OpenRouter forwards this hint to any
       provider that supports it; providers that don't, ignore it.
    3. logprobs intentionally NOT requested — reasoning models reject it,
       and JudgeResponse.logprob defaults to None, which the analyzer
       handles gracefully. Confidence analysis is a BIBM-paper extension."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )
    resp = client.chat.completions.create(
        model=model_slug,
        max_tokens=500,
        temperature=0.0,
        top_p=1.0,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"reasoning": {"effort": "minimal"}},
    )
    text = resp.choices[0].message.content
    return text, None


# Rebuild JUDGES so every callable goes through OpenRouter.
JUDGES = [
    ("Claude",   lambda p: _query_openrouter(MODEL_VERSIONS["Claude"],   p)),
    ("GPT",      lambda p: _query_openrouter(MODEL_VERSIONS["GPT"],      p)),
    ("DeepSeek", lambda p: _query_openrouter(MODEL_VERSIONS["DeepSeek"], p)),
    ("Llama",    lambda p: _query_openrouter(MODEL_VERSIONS["Llama"],    p)),
]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURE FOR YOUR CSV
# Edit these three blocks if your column names don't match.
# ─────────────────────────────────────────────────────────────────────────────

# Column names in the AI-READI mapping CSV.
#
# Defaults below are configured for the project's curator_spreadsheet.csv
# (50-row stratified sample with AI-READI curator labels copied across).
# To run against the canonical AI-READI v3 published SSSOM table instead,
# swap to these:
#   COL_SUBJECT_LABEL  = "subject_label"
#   COL_OBJECT_LABEL   = "object_label"
#   COL_OBJECT_ID      = "object_id"          # e.g. "OMOP:3004410"
#   COL_PREDICATE      = "predicate_id"       # e.g. "skos:exactMatch"
#   COL_CONFIDENCE     = "confidence"
#   COL_SUBJECT_ID     = "subject_id"
#   OBJECT_ID_PREFIX   = "OMOP:"
COL_SUBJECT_LABEL  = "source_label"
COL_OBJECT_LABEL   = "target_concept_name"
COL_OBJECT_ID      = "target_omop_concept_id"
COL_PREDICATE      = "curator_sssom_label"
COL_CONFIDENCE     = "curator_confidence"
COL_SUBJECT_ID     = "row_id"

# Map the SSSOM predicate IRI/CURIE to one of the five tier labels we score on.
# If your table uses a different convention (e.g. plain strings "exact" already,
# or sssom:semapv:* predicates), extend this dict.
PREDICATE_TO_TIER = {
    "skos:exactmatch":   "exact",
    "skos:closematch":   "close",
    "skos:narrowmatch":  "narrow",
    "skos:broadmatch":   "broad",
    "skos:relatedmatch": "related",
    # Plain-string fallback in case the CSV stores the tier directly:
    "exact":   "exact",
    "close":   "close",
    "narrow":  "narrow",
    "broad":   "broad",
    "related": "related",
}

# Strip this prefix from object_id before storing (cosmetic only).
# curator_spreadsheet.csv has bare integer IDs; the canonical v3 table uses "OMOP:".
OBJECT_ID_PREFIX = ""


# ─────────────────────────────────────────────────────────────────────────────
# CSV LOADER + STRATIFIED SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

def load_rows(csv_path: str) -> list[dict]:
    """Read the SSSOM CSV and produce row dicts in the shape the protocol expects."""
    rows: list[dict] = []
    skipped = Counter()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader):
            pred = (raw.get(COL_PREDICATE) or "").strip().lower()
            tier = PREDICATE_TO_TIER.get(pred)
            if tier is None:
                skipped["unrecognised_predicate"] += 1
                continue

            subj = (raw.get(COL_SUBJECT_LABEL) or "").strip()
            obj_label = (raw.get(COL_OBJECT_LABEL) or "").strip()
            obj_id_raw = (raw.get(COL_OBJECT_ID) or "").strip()
            if not subj or not obj_label:
                skipped["missing_text"] += 1
                continue

            # Parse OMOP id if present; record raw string if not numeric.
            obj_id = obj_id_raw
            if obj_id.lower().startswith(OBJECT_ID_PREFIX.lower()):
                obj_id = obj_id[len(OBJECT_ID_PREFIX):]
            try:
                target_omop_id: object = int(obj_id)
            except (TypeError, ValueError):
                target_omop_id = obj_id or None

            # Confidence: accept 0–1 or 0–100; store raw float if parseable.
            conf_raw = (raw.get(COL_CONFIDENCE) or "").strip()
            try:
                conf_val: Optional[float] = float(conf_raw) if conf_raw else None
            except ValueError:
                conf_val = None

            row_id = (raw.get(COL_SUBJECT_ID) or "").strip() or f"AIR-{i:05d}"

            rows.append({
                "row_id":           row_id,
                "source_term":      subj,
                "target_label":     obj_label,
                "target_omop_id":   target_omop_id,
                "gold_sssom_label": tier,
                "gold_confidence":  conf_val,
            })

    if skipped:
        print(f"  Skipped {sum(skipped.values())} rows during load: {dict(skipped)}",
              file=sys.stderr)
    return rows


def stratified_sample(rows: list[dict], per_tier: int, seed: int) -> list[dict]:
    """Return up to per_tier rows from each SSSOM tier, deterministic by seed."""
    rng = random.Random(seed)
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tier[r["gold_sssom_label"]].append(r)

    sample: list[dict] = []
    print("\n  Natural distribution in file vs. sample drawn:")
    print(f"    {'tier':<10s}  {'population':>12s}   {'sampled':>8s}")
    for tier in SSSOM_LABELS:
        pop = by_tier.get(tier, [])
        rng.shuffle(pop)
        drawn = pop[:per_tier]
        sample.extend(drawn)
        flag = "" if len(drawn) >= per_tier else "  ← under-represented"
        print(f"    {tier:<10s}  {len(pop):>12d}   {len(drawn):>8d}{flag}")
    rng.shuffle(sample)
    return sample


# ─────────────────────────────────────────────────────────────────────────────
# JUDGE WRAPPER WITH GRACEFUL FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_KEY_ENV = {name: "OPENROUTER_API_KEY" for name, _ in JUDGES}


def active_judges() -> list[tuple[str, callable]]:
    """Return the JUDGES list, filtered to those whose API key is present."""
    keep = []
    for name, fn in JUDGES:
        env = JUDGE_KEY_ENV.get(name)
        if env and os.environ.get(env):
            keep.append((name, fn))
        else:
            print(f"  ⚠  Skipping {name}: {env} not set in environment.",
                  file=sys.stderr)
    if len(keep) < 2:
        sys.exit("\nERROR: fewer than 2 judges available; jury vote needs ≥2. Exiting.")
    return keep


def query_judge_safe(query_fn, prompt: str, retries: int = 2,
                     backoff_s: float = 2.0) -> tuple[str, Optional[float]]:
    """Call the judge with simple retry-on-exception. Returns ("", None) on failure."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            return query_fn(prompt)
        except Exception as e:                       # broad on purpose; API SDKs vary
            last_err = e
            time.sleep(backoff_s * (attempt + 1))
    print(f"    judge call failed after {retries+1} attempts: {last_err}",
          file=sys.stderr)
    return "", None


# ─────────────────────────────────────────────────────────────────────────────
# RUN ONE ROW (mirrors run_row() in v3, but uses the filtered judge list)
# ─────────────────────────────────────────────────────────────────────────────

def run_row_live(row: dict, judges: list) -> dict:
    prompt = PROMPT_TEMPLATE.format(**{
        "source_term":    row["source_term"],
        "target_label":   row["target_label"],
        "target_omop_id": row["target_omop_id"],
    })

    judge_records = []
    responses: list[JudgeResponse] = []
    for name, fn in judges:
        raw, logprob = query_judge_safe(fn, prompt)
        resp = JudgeResponse(
            judge_name=name,
            model=MODEL_VERSIONS[name],
            label=normalize(raw),
            raw_response=raw,
            logprob=logprob,
            confidence=_to_confidence(logprob),
        )
        responses.append(resp)
        judge_records.append(asdict(resp))

    jury = majority_vote(responses)
    gold = row["gold_sssom_label"]
    vote_counts = Counter(r.label for r in responses if r.label != UNKNOWN)

    return {
        "row_id":           row["row_id"],
        "source_term":      row["source_term"],
        "target_label":     row["target_label"],
        "target_omop_id":   row["target_omop_id"],
        "gold_label":       gold,
        "gold_confidence":  row.get("gold_confidence"),
        "judges":           judge_records,
        "vote_counts":      dict(vote_counts),
        "jury_label":       jury,
        "jury_abstained":   jury == ABSTAIN,
        "jury_correct":     jury == gold,
        "unknown_count":    sum(1 for r in responses if r.label == UNKNOWN),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Path to AI-READI SSSOM mapping CSV")
    p.add_argument("--per-tier", type=int, default=50,
                   help="Rows per SSSOM tier to sample (default 50 → max 250 total)")
    p.add_argument("--seed", type=int, default=20260630,
                   help="Random seed for stratified sampling (default 20260630)")
    p.add_argument("--out", default="benchmark_results.json",
                   help="Where to write per-row results JSON (default benchmark_results.json)")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on rows after stratification (for a smoke test)")
    args = p.parse_args()

    print("\n══════════════════════════════════════════════════════════════════════")
    print("  AI-READI SSSOM BENCHMARK — live run")
    print("══════════════════════════════════════════════════════════════════════")
    print(f"  CSV:        {args.csv}")
    print(f"  Per tier:   {args.per_tier}")
    print(f"  Seed:       {args.seed}")

    print("\n  Loading rows…")
    all_rows = load_rows(args.csv)
    print(f"  Loaded {len(all_rows)} rows with a recognised SSSOM tier.")

    sample = stratified_sample(all_rows, args.per_tier, args.seed)
    if args.limit:
        sample = sample[:args.limit]
    print(f"\n  Drew {len(sample)} rows for evaluation.")

    judges = active_judges()
    print(f"\n  Active judges ({len(judges)}): {', '.join(n for n, _ in judges)}")

    print("\n  Running…")
    out_rows = []
    t0 = time.time()
    for i, row in enumerate(sample, start=1):
        rec = run_row_live(row, judges)
        out_rows.append(rec)
        # Stream results to disk after every row so a crash doesn't lose progress.
        with open(args.out, "w") as f:
            json.dump({
                "config": {
                    "csv":      os.path.abspath(args.csv),
                    "per_tier": args.per_tier,
                    "seed":     args.seed,
                    "judges":   [{"name": n, "model": MODEL_VERSIONS[n]} for n, _ in judges],
                },
                "rows": out_rows,
            }, f, indent=2)
        if i % 10 == 0 or i == len(sample):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            eta = (len(sample) - i) / rate if rate else 0
            print(f"    [{i:>4d}/{len(sample)}]  elapsed {elapsed:>6.0f}s  "
                  f"rate {rate:>4.2f} row/s  ETA {eta:>5.0f}s")

    print(f"\n  Done. Wrote {len(out_rows)} rows → {args.out}\n")
