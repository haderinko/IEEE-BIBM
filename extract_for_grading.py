#!/usr/bin/env python3
"""
extract_for_grading.py -- build a BLIND grading file for a second curator,
structured to yield inter-rater agreement, plus a hidden answer key.

What it produces
----------------
1. to_grade.csv    -- the file you hand to curator 2. Two blank columns
                      (curator_sssom_label, curator_confidence) to fill.
                      NO existing labels visible -> independent grading.
                      Contains ~N_NEW new rows sampled from the pool PLUS
                      ~N_OVERLAP rows re-used from your ORIGINAL 50 (curator 1's
                      labels hidden), shuffled together so the grader can't tell
                      which are overlap rows.

2. answer_key.csv  -- DO NOT give this to the grader. Hidden reference for every
                      row: the pool's pre-existing PREDICATE_ID-derived label for
                      new rows, and curator 1's label for overlap rows, tagged by
                      origin. Used later to compute inter-rater agreement and fold
                      new rows into the benchmark.

Usage
-----
    python extract_for_grading.py \
        --pool data/mappings.json \
        --original data/curator_spreadsheet.csv \
        --exclude data/holdout_sample.csv \
        --n-new 170 --n-overlap 35 --seed 20260705 \
        --out to_grade.csv --key answer_key.csv

Then curator 2 fills to_grade.csv. Delete the two '#' comment lines before
re-importing. Send the graded file back to compute inter-rater kappa and re-run.
"""

import argparse
import csv
import json
import random
import sys


# ---- Field mapping for THIS pool (AI-READI mappings.json) -----------------
F_SOURCE = "SRC_CD_DESCRIPTION"
F_TARGET = "TARGET_CONCEPT_NAME"
F_OMOP   = "TARGET_CONCEPT_ID"
F_VOCAB  = "TARGET_VOCABULARY_ID"
F_ROWID  = "SRC_CODE"
F_ROWID2 = "FIELD_ID_NUM"
F_PRED   = "PREDICATE_ID"
F_CONF   = "CONFIDENCE"

SKOS_TO_TIER = {
    "skos:exactmatch":   "exact",
    "skos:closematch":   "close",
    "skos:narrowmatch":  "narrow",
    "skos:broadmatch":   "broad",
    "skos:relatedmatch": "related",
}

# Field names in your ORIGINAL curator_spreadsheet.csv.
ORIG_SOURCE = "source_label"
ORIG_TARGET = "target_concept_name"
ORIG_OMOP   = "target_omop_concept_id"
ORIG_VOCAB  = "target_vocabulary"
ORIG_ROWID  = "row_id"
ORIG_LABEL  = "curator_sssom_label"
ORIG_CONF   = "curator_confidence"


def norm(s):
    return str(s if s is not None else "").strip()


def pair_key(src, tgt):
    return (norm(src).lower(), norm(tgt).lower())


def load_pool(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"ERROR: expected a JSON list in {path}")
    return data


def read_csv_skip_comments(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--n-new", type=int, default=170)
    ap.add_argument("--n-overlap", type=int, default=35)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--out", default="to_grade.csv")
    ap.add_argument("--key", default="answer_key.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    original = read_csv_skip_comments(args.original)
    print(f"Loaded {len(original)} rows from {args.original}")
    used_pairs = {pair_key(r.get(ORIG_SOURCE), r.get(ORIG_TARGET)) for r in original}

    for p in args.exclude:
        try:
            for r in read_csv_skip_comments(p):
                used_pairs.add(pair_key(r.get("source_label"), r.get("target_concept_name")))
        except FileNotFoundError:
            print(f"  (exclude file not found, skipping: {p})")

    pool = load_pool(args.pool)
    print(f"Loaded {len(pool)} rows from {args.pool}")

    candidates = []
    seen = set()
    dropped = 0
    for i, m in enumerate(pool):
        if not isinstance(m, dict):
            continue
        src, tgt = norm(m.get(F_SOURCE)), norm(m.get(F_TARGET))
        if not src or not tgt:
            dropped += 1
            continue
        k = pair_key(src, tgt)
        if k in used_pairs or k in seen:
            continue
        seen.add(k)
        rid = norm(m.get(F_ROWID)) or f"G{norm(m.get(F_ROWID2)) or i}"
        pred = norm(m.get(F_PRED)).lower()
        candidates.append({
            "row_id": rid,
            "source_label": src,
            "target_concept_name": tgt,
            "target_omop_concept_id": norm(m.get(F_OMOP)),
            "target_vocabulary": norm(m.get(F_VOCAB)),
            "_ref_label": SKOS_TO_TIER.get(pred, ""),
            "_ref_conf": norm(m.get(F_CONF)),
            "_origin": "new",
        })
    if dropped:
        print(f"  ({dropped} pool rows dropped for missing source/target)")
    print(f"{len(candidates)} unique new candidates available")
    if len(candidates) < args.n_new:
        print(f"  WARNING: only {len(candidates)} new rows available (< {args.n_new})")
    rng.shuffle(candidates)
    new_rows = candidates[:args.n_new]

    orig_pool = list(original)
    rng.shuffle(orig_pool)
    overlap_src = orig_pool[:args.n_overlap]
    overlap_rows = []
    for r in overlap_src:
        overlap_rows.append({
            "row_id": norm(r.get(ORIG_ROWID)),
            "source_label": norm(r.get(ORIG_SOURCE)),
            "target_concept_name": norm(r.get(ORIG_TARGET)),
            "target_omop_concept_id": norm(r.get(ORIG_OMOP)),
            "target_vocabulary": norm(r.get(ORIG_VOCAB)),
            "_ref_label": norm(r.get(ORIG_LABEL)),
            "_ref_conf": norm(r.get(ORIG_CONF)),
            "_origin": "overlap",
        })
    print(f"{len(overlap_rows)} overlap rows drawn from the original {len(original)}")

    all_rows = new_rows + overlap_rows
    rng.shuffle(all_rows)

    grade_fields = ["row_id", "source_label", "target_concept_name",
                    "target_omop_concept_id", "target_vocabulary",
                    "curator_sssom_label", "curator_confidence"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        f.write("# SSSOM grading. Fill curator_sssom_label with ONE of: exact, close, narrow, broad, related. "
                "curator_confidence = your certainty 0-100.\n")
        f.write("# exact=identical | close=near-identical, minor specificity gap | "
                "narrow=target more specific than source | broad=target more general than source | "
                "related=connected but not equivalent\n")
        w = csv.DictWriter(f, fieldnames=grade_fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({
                "row_id": r["row_id"],
                "source_label": r["source_label"],
                "target_concept_name": r["target_concept_name"],
                "target_omop_concept_id": r["target_omop_concept_id"],
                "target_vocabulary": r["target_vocabulary"],
                "curator_sssom_label": "",
                "curator_confidence": "",
            })

    key_fields = ["row_id", "source_label", "target_concept_name", "origin",
                  "reference_label", "reference_confidence"]
    with open(args.key, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=key_fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({
                "row_id": r["row_id"],
                "source_label": r["source_label"],
                "target_concept_name": r["target_concept_name"],
                "origin": r["_origin"],
                "reference_label": r["_ref_label"],
                "reference_confidence": r["_ref_conf"],
            })

    print(f"\nWrote {len(all_rows)} rows to {args.out}  (grader fills last 2 columns)")
    print(f"Wrote hidden key to {args.key}  (DO NOT send this to the grader)")
    print(f"  new rows: {len(new_rows)}   overlap rows: {len(overlap_rows)}")
    print("Before re-importing the graded file: delete the two '#' comment lines.")


if __name__ == "__main__":
    main()
