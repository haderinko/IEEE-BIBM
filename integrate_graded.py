#!/usr/bin/env python3
"""
integrate_graded.py -- process curator 2's graded file.

Does three things once curator 2 returns to_grade.csv (with the two label
columns filled in):

  1. INTER-RATER AGREEMENT on the 35 overlap rows: curator 2's labels vs
     curator 1's labels (from answer_key.csv). Reports unweighted Cohen's kappa,
     semantic cost-weighted kappa, raw agreement %, and PER-TIER agreement --
     specifically whether the two humans disagree most on `close` (which would
     make the ordinal-middle collapse a task property, not an AI failure).

  2. MERGE the 170 new rows (curator 2's grades) into the primary benchmark,
     writing curator_spreadsheet_v2.csv (~220 rows) in the same schema
     run_benchmark_v4.py expects.

  3. Also reports curator2-vs-pool-reference agreement on the new rows as a
     secondary sanity check (how often curator 2 matched the dataset's own
     PREDICATE_ID label). Informative, not a ground-truth claim.

Usage
-----
    python integrate_graded.py \
        --graded to_grade.csv \
        --key answer_key.csv \
        --original data/curator_spreadsheet.csv \
        --out data/curator_spreadsheet_v2.csv

Notes
-----
- Delete the two '#' comment lines from the graded CSV first (or leave them;
  this script skips lines starting with '#').
- Rows curator 2 left blank or deleted are dropped with a report, so skipping
  ambiguous rows is fine.
"""

import argparse
import csv
import sys
from collections import defaultdict, Counter


SSSOM = ["exact", "close", "narrow", "broad", "related"]

# Semantic SSSOM cost matrix C[gold][pred] -- same as paper Table I.
COST = {
    "exact":   {"exact": 0, "close": 1, "narrow": 4, "broad": 4, "related": 9},
    "close":   {"exact": 1, "close": 0, "narrow": 2, "broad": 2, "related": 4},
    "narrow":  {"exact": 4, "close": 2, "narrow": 0, "broad": 9, "related": 4},
    "broad":   {"exact": 4, "close": 2, "narrow": 9, "broad": 0, "related": 4},
    "related": {"exact": 9, "close": 4, "narrow": 4, "broad": 4, "related": 0},
}


def norm_label(s):
    return str(s or "").strip().lower()


def read_csv_skip_comments(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def cohen_kappa(a, b, weighted=False):
    """Cohen's kappa between two label lists. weighted=True uses COST matrix
    (normalized to its max), else unweighted 0/1 disagreement."""
    n = len(a)
    if n == 0:
        return float("nan")
    idx = {l: i for i, l in enumerate(SSSOM)}
    k = len(SSSOM)
    O = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        if x in idx and y in idx:
            O[idx[x]][idx[y]] += 1
    total = sum(sum(row) for row in O)
    if total == 0:
        return float("nan")
    row = [sum(O[i]) for i in range(k)]
    col = [sum(O[i][j] for i in range(k)) for j in range(k)]
    E = [[row[i] * col[j] / total for j in range(k)] for i in range(k)]
    if weighted:
        mx = max(COST[g][p] for g in SSSOM for p in SSSOM)
        w = [[COST[SSSOM[i]][SSSOM[j]] / mx for j in range(k)] for i in range(k)]
    else:
        w = [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * O[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * E[i][j] for i in range(k) for j in range(k))
    return 1.0 - num / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", required=True, help="curator 2's filled to_grade.csv")
    ap.add_argument("--key", required=True, help="answer_key.csv from extraction")
    ap.add_argument("--original", required=True, help="original curator_spreadsheet.csv")
    ap.add_argument("--out", default="data/curator_spreadsheet_v2.csv")
    args = ap.parse_args()

    graded = read_csv_skip_comments(args.graded)
    key = read_csv_skip_comments(args.key)
    original = read_csv_skip_comments(args.original)

    # Index key by row_id.
    key_by_id = {norm_label(r["row_id"]): r for r in key}

    # Collect curator 2's grades, split by origin.
    overlap_pairs = []   # (curator1_label, curator2_label)
    new_graded = []      # rows curator 2 graded that are 'new'
    blank = 0
    for g in graded:
        c2 = norm_label(g.get("curator_sssom_label"))
        rid = norm_label(g.get("row_id"))
        if c2 not in SSSOM:
            blank += 1
            continue
        k = key_by_id.get(rid)
        if not k:
            continue
        origin = k["origin"]
        ref = norm_label(k["reference_label"])
        if origin == "overlap":
            if ref in SSSOM:
                overlap_pairs.append((ref, c2))   # (curator1, curator2)
        else:  # new
            new_graded.append({
                "row_id": g["row_id"],
                "source_label": g["source_label"],
                "target_concept_name": g["target_concept_name"],
                "target_omop_concept_id": g.get("target_omop_concept_id", ""),
                "target_vocabulary": g.get("target_vocabulary", ""),
                "curator_sssom_label": c2,
                "curator_confidence": g.get("curator_confidence", ""),
                "_ref": ref,   # pool's PREDICATE_ID label, for secondary check
            })

    print(f"Graded rows read: {len(graded)}  (blank/skipped: {blank})")
    print(f"Overlap rows usable for inter-rater: {len(overlap_pairs)}")
    print(f"New rows graded: {len(new_graded)}")

    # ---- 1. INTER-RATER AGREEMENT (curator 1 vs curator 2) ----
    print("\n" + "=" * 60)
    print("INTER-RATER AGREEMENT  (curator 1 vs curator 2, overlap rows)")
    print("=" * 60)
    if overlap_pairs:
        c1 = [p[0] for p in overlap_pairs]
        c2 = [p[1] for p in overlap_pairs]
        raw = sum(1 for a, b in overlap_pairs if a == b) / len(overlap_pairs)
        k_unw = cohen_kappa(c1, c2, weighted=False)
        k_cost = cohen_kappa(c1, c2, weighted=True)
        print(f"  n overlap rows:        {len(overlap_pairs)}")
        print(f"  raw agreement:         {raw:.3f}")
        print(f"  Cohen's kappa (unw.):  {k_unw:.3f}")
        print(f"  Cohen's kappa (cost):  {k_cost:.3f}")
        print("\n  Per-tier agreement (curator 1's tier -> how often curator 2 matched):")
        by_tier_tot = Counter(c1)
        by_tier_agree = Counter(a for a, b in overlap_pairs if a == b)
        for t in SSSOM:
            tot = by_tier_tot.get(t, 0)
            if tot:
                print(f"    {t:8s}: {by_tier_agree.get(t,0)}/{tot} = {by_tier_agree.get(t,0)/tot:.2f}")
        # Highlight the close-tier finding.
        close_tot = by_tier_tot.get("close", 0)
        if close_tot:
            close_agree = by_tier_agree.get("close", 0) / close_tot
            print(f"\n  >> close-tier human agreement: {close_agree:.2f}")
            print("     (Low here = humans ALSO struggle on close -> ordinal-middle")
            print("      collapse is a task property, not just an LLM failure.)")
    else:
        print("  No usable overlap pairs -- check that overlap rows were graded.")

    # ---- 2. SECONDARY: curator 2 vs pool reference on new rows ----
    print("\n" + "=" * 60)
    print("SECONDARY: curator 2 vs pool PREDICATE_ID label (new rows)")
    print("=" * 60)
    ref_pairs = [(r["_ref"], r["curator_sssom_label"]) for r in new_graded if r["_ref"] in SSSOM]
    if ref_pairs:
        raw = sum(1 for a, b in ref_pairs if a == b) / len(ref_pairs)
        print(f"  n rows with a pool label: {len(ref_pairs)}")
        print(f"  raw agreement:            {raw:.3f}")
        print(f"  Cohen's kappa (unw.):     {cohen_kappa([p[0] for p in ref_pairs],[p[1] for p in ref_pairs]):.3f}")
        print("  (Informational: pool labels are heuristic, not ground truth.)")
    else:
        print("  No pool-labeled new rows to compare.")

    # ---- 3. MERGE new rows into the benchmark ----
    out_fields = ["row_id", "source_label", "target_concept_name",
                  "target_omop_concept_id", "target_vocabulary",
                  "curator_sssom_label", "curator_confidence"]
    merged = []
    for r in original:
        merged.append({k: r.get(k, "") for k in out_fields})
    for r in new_graded:
        merged.append({k: r.get(k, "") for k in out_fields})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in merged:
            w.writerow(r)

    print("\n" + "=" * 60)
    print(f"MERGED BENCHMARK written to {args.out}")
    print(f"  original rows: {len(original)}")
    print(f"  new rows added: {len(new_graded)}")
    print(f"  total: {len(merged)}")
    print("=" * 60)
    print("\nNext: re-run the jury on the merged file:")
    print(f"  python run_benchmark_v4.py --csv {args.out} --per-tier 50 --seed 20260630 --out results/benchmark_results_v5.json")
    print("  (adjust --per-tier so the stratified sample covers all merged rows)")


if __name__ == "__main__":
    main()
