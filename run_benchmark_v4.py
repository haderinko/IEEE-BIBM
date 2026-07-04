#!/usr/bin/env python3
"""
run_benchmark_v4.py — verbalized-confidence variant of the SSSOM jury protocol.

Same 50-row live pipeline as run_benchmark_live.py, but each judge is asked
for TWO fields per row: the SSSOM tier label AND a self-reported confidence
score in [0, 100]. The confidence is what BIBM contribution C2 calibrates.

Runs from the SAME directory as motivating_example_v3.py, using the same
OpenRouter routing and env var (OPENROUTER_API_KEY).

Usage
-----
    python run_benchmark_v4.py \
        --csv curator_spreadsheet.csv \
        --per-tier 15 \
        --seed 20260630 \
        --out benchmark_results_v4.json

Same seed 20260630 → same 50-row sample as your prior run, so results are
directly comparable to `stats_bibm.json` from v1 analyzer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motivating_example_v3 import (  # noqa: E402
    ABSTAIN,
    JudgeResponse,
    MODEL_VERSIONS as _V3_MODELS,   # unused; we redefine below
    SSSOM_LABELS,
    UNKNOWN,
    _to_confidence,
    majority_vote,
    normalize,
)


# ─────────────────────────────────────────────────────────────────────
# CSV column names — configured for curator_spreadsheet.csv
# ─────────────────────────────────────────────────────────────────────
COL_SUBJECT_LABEL = "source_label"
COL_OBJECT_LABEL  = "target_concept_name"
COL_OBJECT_ID     = "target_omop_concept_id"
COL_PREDICATE     = "curator_sssom_label"
COL_CONFIDENCE    = "curator_confidence"
COL_SUBJECT_ID    = "row_id"
OBJECT_ID_PREFIX  = ""

PREDICATE_TO_TIER = {
    "exact":   "exact",  "close":   "close", "narrow": "narrow",
    "broad":   "broad",  "related": "related",
    "skos:exactmatch":   "exact",  "skos:closematch":   "close",
    "skos:narrowmatch":  "narrow", "skos:broadmatch":   "broad",
    "skos:relatedmatch": "related",
}


# ─────────────────────────────────────────────────────────────────────
# OPENROUTER — same as v3 override
# ─────────────────────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_VERSIONS = {
    "Claude":   "anthropic/claude-opus-4.7",
    "GPT":      "openai/gpt-5.5",
    "DeepSeek": "deepseek/deepseek-v4-pro",
    "Llama":    "meta-llama/llama-4-maverick",
}


# ─────────────────────────────────────────────────────────────────────
# NEW PROMPT — asks for label + verbalized confidence
# ─────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE_V4 = """You are grading the semantic quality of a biomedical concept mapping.

Source term: "{source_term}"
Target concept: "{target_label}" (OMOP concept_id: {target_omop_id})

Rate the mapping on the SSSOM five-tier scale (pick EXACTLY ONE):
  exact   — identical concepts
  close   — near-identical, minor specificity gap
  narrow  — target is more specific than source (target ⊂ source)
  broad   — target is more general than source (source ⊂ target)
  related — connected but not equivalent

Also report how confident you are (0-100 integer).

Respond in EXACTLY this format, no extra text:
LABEL: <tier>
CONFIDENCE: <integer 0-100>
"""


# ─────────────────────────────────────────────────────────────────────
# Judge query — same OpenRouter mechanics as v3
# ─────────────────────────────────────────────────────────────────────
def _query_openrouter(model_slug: str, prompt: str) -> tuple[str, Optional[float]]:
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
    return resp.choices[0].message.content, None


JUDGES = [
    ("Claude",   lambda p: _query_openrouter(MODEL_VERSIONS["Claude"],   p)),
    ("GPT",      lambda p: _query_openrouter(MODEL_VERSIONS["GPT"],      p)),
    ("DeepSeek", lambda p: _query_openrouter(MODEL_VERSIONS["DeepSeek"], p)),
    ("Llama",    lambda p: _query_openrouter(MODEL_VERSIONS["Llama"],    p)),
]
JUDGE_KEY_ENV = {name: "OPENROUTER_API_KEY" for name, _ in JUDGES}


# ─────────────────────────────────────────────────────────────────────
# Parse LABEL + CONFIDENCE from response — robust to formatting drift
# ─────────────────────────────────────────────────────────────────────
_LABEL_RE = re.compile(r"label\s*[:\-]?\s*([a-zA-Z]+)", re.IGNORECASE)
_CONF_RE  = re.compile(r"confidence\s*[:\-]?\s*(\d{1,3})",  re.IGNORECASE)


def parse_label_and_confidence(text: str) -> tuple[str, Optional[float]]:
    """Return (normalized_label, confidence_0_to_100_or_None) from a judge response.

    Robust to:
      LABEL: narrow / CONFIDENCE: 80   (canonical)
      narrow, 80                        (bare)
      "The label is narrow with confidence 80"  (chatty)
      Missing CONFIDENCE                 (returns None for that slot)
    """
    if not text:
        return UNKNOWN, None

    # Try the structured extraction first.
    label = None
    conf = None
    m = _LABEL_RE.search(text)
    if m:
        label = normalize(m.group(1))
    m = _CONF_RE.search(text)
    if m:
        try:
            v = int(m.group(1))
            if 0 <= v <= 100:
                conf = float(v)
        except ValueError:
            pass

    # Fallback: run the v3 normalizer on the whole response to catch bare labels.
    if not label or label == UNKNOWN:
        label = normalize(text)

    # Fallback confidence: any 2-3 digit number in [0, 100] not already claimed.
    if conf is None:
        for tok in re.findall(r"\b(\d{1,3})\b", text):
            v = int(tok)
            if 0 <= v <= 100:
                conf = float(v)
                break

    return label or UNKNOWN, conf


# ─────────────────────────────────────────────────────────────────────
# CSV loader + stratified sampler — copied from run_benchmark_live.py
# ─────────────────────────────────────────────────────────────────────
def load_rows(csv_path: str) -> list[dict]:
    rows = []
    skipped = Counter()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, raw in enumerate(csv.DictReader(f)):
            pred = (raw.get(COL_PREDICATE) or "").strip().lower()
            tier = PREDICATE_TO_TIER.get(pred)
            if tier is None:
                skipped["unrecognised_predicate"] += 1
                continue
            subj = (raw.get(COL_SUBJECT_LABEL) or "").strip()
            obj_label = (raw.get(COL_OBJECT_LABEL) or "").strip()
            if not subj or not obj_label:
                skipped["missing_text"] += 1
                continue
            obj_id_raw = (raw.get(COL_OBJECT_ID) or "").strip()
            obj_id = obj_id_raw[len(OBJECT_ID_PREFIX):] if OBJECT_ID_PREFIX and obj_id_raw.lower().startswith(OBJECT_ID_PREFIX.lower()) else obj_id_raw
            try:
                target_omop_id = int(obj_id)
            except (TypeError, ValueError):
                target_omop_id = obj_id or None
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
        print(f"  Skipped {sum(skipped.values())} rows: {dict(skipped)}", file=sys.stderr)
    return rows


def stratified_sample(rows, per_tier, seed):
    rng = random.Random(seed)
    by_tier = defaultdict(list)
    for r in rows:
        by_tier[r["gold_sssom_label"]].append(r)
    print("\n  Sample distribution:")
    print(f"    {'tier':<10s}  {'pop':>6s}  {'drawn':>6s}")
    sample = []
    for tier in SSSOM_LABELS:
        pop = by_tier.get(tier, [])
        rng.shuffle(pop)
        drawn = pop[:per_tier]
        sample.extend(drawn)
        print(f"    {tier:<10s}  {len(pop):>6d}  {len(drawn):>6d}")
    rng.shuffle(sample)
    return sample


def active_judges():
    keep = []
    for name, fn in JUDGES:
        env = JUDGE_KEY_ENV[name]
        if os.environ.get(env):
            keep.append((name, fn))
        else:
            print(f"  ⚠  Skipping {name}: {env} not set", file=sys.stderr)
    if len(keep) < 2:
        sys.exit("ERROR: need ≥ 2 active judges.")
    return keep


def query_safe(fn, prompt, retries=2, backoff=2.0):
    last = None
    for i in range(retries + 1):
        try:
            return fn(prompt)
        except Exception as e:
            last = e
            time.sleep(backoff * (i + 1))
    print(f"    query failed after {retries+1} attempts: {last}", file=sys.stderr)
    return "", None


def run_row_v4(row, judges):
    prompt = PROMPT_TEMPLATE_V4.format(
        source_term=row["source_term"],
        target_label=row["target_label"],
        target_omop_id=row["target_omop_id"],
    )
    judge_records = []
    responses = []
    for name, fn in judges:
        raw, _lp = query_safe(fn, prompt)
        label, verb_conf = parse_label_and_confidence(raw)
        resp = JudgeResponse(
            judge_name=name,
            model=MODEL_VERSIONS[name],
            label=label,
            raw_response=raw,
            logprob=None,
            confidence=verb_conf,   # verbalized confidence in [0, 100]
        )
        responses.append(resp)
        rec = asdict(resp)
        rec["verbalized_confidence"] = verb_conf  # explicit key for clarity
        judge_records.append(rec)

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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--per-tier", type=int, default=15)
    p.add_argument("--seed", type=int, default=20260630)
    p.add_argument("--out", default="benchmark_results_v4.json")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    print("\n══════════════════════════════════════════════════════════════════")
    print("  AI-READI SSSOM BENCHMARK v4 — verbalized confidence live run")
    print("══════════════════════════════════════════════════════════════════")
    print(f"  CSV:      {args.csv}")
    print(f"  Per tier: {args.per_tier}")
    print(f"  Seed:     {args.seed}")

    all_rows = load_rows(args.csv)
    print(f"\n  Loaded {len(all_rows)} rows.")
    sample = stratified_sample(all_rows, args.per_tier, args.seed)
    if args.limit:
        sample = sample[:args.limit]
    print(f"\n  Sample: {len(sample)} rows.")

    judges = active_judges()
    print(f"  Active judges ({len(judges)}): {', '.join(n for n, _ in judges)}")

    print("\n  Running…")
    out_rows = []
    t0 = time.time()
    for i, row in enumerate(sample, 1):
        rec = run_row_v4(row, judges)
        out_rows.append(rec)
        with open(args.out, "w") as f:
            json.dump({
                "config": {
                    "csv":      os.path.abspath(args.csv),
                    "per_tier": args.per_tier,
                    "seed":     args.seed,
                    "prompt_version": "v4-verbalized-confidence",
                    "judges":   [{"name": n, "model": MODEL_VERSIONS[n]} for n, _ in judges],
                },
                "rows": out_rows,
            }, f, indent=2)
        if i % 5 == 0 or i == len(sample):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            eta = (len(sample) - i) / rate if rate else 0
            print(f"    [{i:>4d}/{len(sample)}]  elapsed {elapsed:>5.0f}s  ETA {eta:>4.0f}s")

    print(f"\n  Done. Wrote {len(out_rows)} rows → {args.out}\n")
