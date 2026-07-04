#!/usr/bin/env python3
"""
sample_holdout.py — build a held-out generalization sample from AI-READI v3.

Draws ~100 rows from the AI-READI v3 published OMOP mapping table
(mappings.json) that do NOT appear in the primary benchmark
(curator_spreadsheet.csv), stratified across the tiers the v3 table
actually contains: exactMatch, closeMatch, relatedMatch.

The v3 published table has essentially no narrowMatch / broadMatch rows
(0 and 0 respectively per our profile), so the held-out sample cannot
mirror the primary benchmark's 5-tier stratification. We are honest
about this in §VI.H of the paper: the held-out sample tests whether
the protocol generalizes to a *natural* AI-READI distribution, not to
a matched-stratification distribution.

Sample composition (defaults):
    40 exact
    40 close
    3  related     (all available; v3 has only 3)
    = 83 rows total

Adjust with --n-exact / --n-close if you want to shift the balance.

Usage
-----
    python sample_holdout.py \
        --mappings mappings.json \
        --exclude  curator_spreadsheet.csv \
        --out      holdout_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter

PRED_TO_TIER = {
    "skos:exactMatch":   "exact",
    "skos:closeMatch":   "close",
    "skos:narrowMatch":  "narrow",
    "skos:broadMatch":   "broad",
    "skos:relatedMatch": "related",
}


def load_mappings(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data:
        pred = (r.get("PREDICATE_ID") or "").strip()
        tier = PRED_TO_TIER.get(pred)
        if tier is None:
            continue
        target_id = r.get("TARGET_CONCEPT_ID")
        target_name = (r.get("TARGET_CONCEPT_NAME") or "").strip()
        source_name = (r.get("SRC_CD_DESCRIPTION") or "").strip()
        if not source_name or not target_name or target_id in (None, ""):
            continue
        rows.append({
            "row_id":                  f"AIRV3-{r.get('SRC_CODE_ID', '?')}-{target_id}",
            "source_code":             r.get("SRC_CODE") or "",
            "source_label":            source_name,
            "target_omop_concept_id":  target_id,
            "target_concept_name":     target_name,
            "target_vocabulary":       r.get("TARGET_VOCABULARY_ID") or "",
            "source_form":             r.get("FORM_NAME") or "",
            "curator_sssom_label":     tier,
            "curator_confidence":      int(round((r.get("CONFIDENCE") or 1.0) * 100)),
            "curator_notes":           f"v3 published; predicate={pred}",
        })
    return rows


def load_exclude_ids(path: str) -> set:
    """Read primary benchmark to build a set of (source_label, target_omop_id)
    pairs to exclude from the held-out draw."""
    if not path:
        return set()
    excluded = set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (
                (r.get("source_label") or "").strip().lower(),
                (r.get("target_omop_concept_id") or "").strip(),
            )
            excluded.add(key)
    return excluded


def draw(rows: list[dict], n_by_tier: dict, seed: int, excluded: set) -> list[dict]:
    """Draw n_by_tier[tier] rows per tier without replacement, skipping any
    row whose (source, target) pair is in the excluded set."""
    rng = random.Random(seed)
    by_tier = {}
    for r in rows:
        key = ((r["source_label"] or "").strip().lower(), str(r["target_omop_concept_id"]))
        if key in excluded:
            continue
        by_tier.setdefault(r["curator_sssom_label"], []).append(r)

    sample = []
    print("  Held-out sample composition:")
    print(f"    {'tier':<10s}  {'pop (post-exclude)':>18s}  {'wanted':>8s}  {'drawn':>8s}")
    for tier, wanted in n_by_tier.items():
        pop = by_tier.get(tier, [])
        rng.shuffle(pop)
        drawn = pop[:wanted]
        if len(drawn) < wanted:
            print(f"    {tier:<10s}  {len(pop):>18d}  {wanted:>8d}  {len(drawn):>8d}  ← under-represented")
        else:
            print(f"    {tier:<10s}  {len(pop):>18d}  {wanted:>8d}  {len(drawn):>8d}")
        sample.extend(drawn)
    rng.shuffle(sample)
    return sample


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        sys.exit("ERROR: no rows drawn; check --mappings and --exclude paths.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mappings", required=True,
                   help="Path to AI-READI v3 mappings.json (downloaded from ai-readi-docs repo).")
    p.add_argument("--exclude", default="curator_spreadsheet.csv",
                   help="Primary benchmark CSV; rows matching (source, target) here are excluded.")
    p.add_argument("--out", default="holdout_sample.csv",
                   help="Output CSV path (default: holdout_sample.csv).")
    p.add_argument("--n-exact", type=int, default=40)
    p.add_argument("--n-close", type=int, default=40)
    p.add_argument("--n-related", type=int, default=3,
                   help="Only 3 related rows exist in v3; use --n-related 3 or drop tier.")
    p.add_argument("--seed", type=int, default=20260704,
                   help="Random seed for the held-out draw (default: 20260704).")
    args = p.parse_args()

    print(f"Loading mappings from {args.mappings}…")
    all_rows = load_mappings(args.mappings)
    print(f"  Loaded {len(all_rows)} SSSOM-graded rows from v3.")
    tier_pop = Counter(r["curator_sssom_label"] for r in all_rows)
    for tier in ["exact", "close", "narrow", "broad", "related"]:
        print(f"    {tier:<10s}  {tier_pop.get(tier, 0):>5d}")

    print(f"\nLoading exclusion set from {args.exclude}…")
    excluded = load_exclude_ids(args.exclude)
    print(f"  Excluding {len(excluded)} (source, target) pairs.")

    n_by_tier = {
        "exact":   args.n_exact,
        "close":   args.n_close,
        "related": args.n_related,
    }
    print(f"\nDrawing held-out sample (seed={args.seed})…")
    sample = draw(all_rows, n_by_tier, args.seed, excluded)

    write_csv(sample, args.out)
    print(f"\nWrote {len(sample)} rows to {args.out}\n")
    print("Next step: run the jury on it")
    print(f"    python run_benchmark_v4.py --csv {args.out} \\")
    print(f"        --per-tier 40 --seed 20260704 --out holdout_results.json")
    print(f"    python analyze_bibm_v2.py --in holdout_results.json --out holdout_stats.json")
