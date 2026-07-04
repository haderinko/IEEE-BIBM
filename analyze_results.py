#!/usr/bin/env python3
"""
analyze_results.py
──────────────────
Read benchmark_results.json (output of run_benchmark_live.py) and compute
the headline statistics for the Rice abstract and BIBM paper:

  - Per-judge accuracy
  - Jury (majority-vote) accuracy, with ABSTAIN/UNKNOWN handling
  - Cohen's κ (unweighted) for each judge vs gold and for jury vs gold
  - Quadratic-weighted κ (ordinal-aware) for the same comparisons
  - Per-tier accuracy breakdown
  - Inter-judge Fleiss κ (agreement among the judges themselves)

The "FOR ABSTRACT" block at the end prints sentence-ready numbers.

Usage
-----
    pip install scikit-learn numpy        # if not already installed
    python analyze_results.py --in benchmark_results.json
    python analyze_results.py --in benchmark_results.json --out stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


SSSOM_LABELS = ["exact", "close", "narrow", "broad", "related"]
UNKNOWN = "UNKNOWN"
ABSTAIN = "ABSTAIN"


# ─────────────────────────────────────────────────────────────────────────────
# Fleiss' κ — implemented in numpy so we don't add a statsmodels dependency
# ─────────────────────────────────────────────────────────────────────────────

def fleiss_kappa(judge_labels_per_row: list[list[str]],
                 categories: list[str]) -> float:
    """Fleiss' κ across N rows × R raters with K categories.

    judge_labels_per_row[i] is the list of labels assigned to row i.
    UNKNOWN labels are dropped for the row in question; rows with fewer than
    2 remaining ratings are excluded (Fleiss requires constant rater count,
    so we restrict to rows where every judge produced a categorical label).
    """
    K = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    # Keep only rows where every judge produced a category in the set.
    clean = []
    for labels in judge_labels_per_row:
        if all(l in cat_idx for l in labels):
            clean.append([cat_idx[l] for l in labels])
    if not clean:
        return float("nan")

    N = len(clean)
    R = len(clean[0])
    if any(len(row) != R for row in clean):
        return float("nan")

    # n[i][k] = number of raters who assigned category k to row i.
    n = np.zeros((N, K), dtype=int)
    for i, row in enumerate(clean):
        for r_idx in row:
            n[i, r_idx] += 1

    P_i = (np.sum(n * n, axis=1) - R) / (R * (R - 1))
    P_bar = P_i.mean()
    p_j = n.sum(axis=0) / (N * R)
    P_e = (p_j * p_j).sum()
    if P_e == 1.0:
        return 1.0
    return float((P_bar - P_e) / (1.0 - P_e))


# ─────────────────────────────────────────────────────────────────────────────
# Per-judge metrics
# ─────────────────────────────────────────────────────────────────────────────

def metrics_for_predictions(y_true: list[str], y_pred: list[str]) -> dict:
    """Accuracy + unweighted κ + quadratic-weighted κ on paired label lists.

    Rows where the prediction is UNKNOWN or ABSTAIN are reported as a separate
    coverage count; the κ statistics are computed on the committed subset only
    (sklearn can't handle missing predictions in cohen_kappa_score)."""
    pairs = [(t, p) for t, p in zip(y_true, y_pred)
             if p not in (UNKNOWN, ABSTAIN)]
    n_total = len(y_true)
    n_committed = len(pairs)
    if n_committed == 0:
        return {
            "n_total":         n_total,
            "n_committed":     0,
            "coverage":        0.0,
            "accuracy_total":  0.0,
            "accuracy_committed": None,
            "kappa_unweighted":  None,
            "kappa_quadratic":   None,
        }
    yt = [t for t, _ in pairs]
    yp = [p for _, p in pairs]
    correct = sum(1 for t, p in pairs if t == p)
    accuracy_committed = correct / n_committed
    accuracy_total = correct / n_total  # treats UNKNOWN/ABSTAIN as misses

    # Cohen's κ — both label sets must use the same category space.
    kappa_u = cohen_kappa_score(yt, yp, labels=SSSOM_LABELS)
    kappa_q = cohen_kappa_score(yt, yp, labels=SSSOM_LABELS, weights="quadratic")

    return {
        "n_total":            n_total,
        "n_committed":        n_committed,
        "coverage":           n_committed / n_total,
        "accuracy_total":     accuracy_total,
        "accuracy_committed": accuracy_committed,
        "kappa_unweighted":   float(kappa_u),
        "kappa_quadratic":    float(kappa_q),
    }


def per_tier_accuracy(y_true: list[str], y_pred: list[str]) -> dict:
    by_tier = defaultdict(lambda: {"n": 0, "correct": 0})
    for t, p in zip(y_true, y_pred):
        d = by_tier[t]
        d["n"] += 1
        if p == t:
            d["correct"] += 1
    return {tier: {"n": d["n"],
                   "correct": d["correct"],
                   "accuracy": (d["correct"] / d["n"]) if d["n"] else None}
            for tier, d in sorted(by_tier.items(),
                                  key=lambda kv: SSSOM_LABELS.index(kv[0]))}


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze(data: dict) -> dict:
    rows = data["rows"]
    judges_meta = data["config"]["judges"]
    judge_names = [j["name"] for j in judges_meta]

    y_true = [r["gold_label"] for r in rows]
    jury_pred = [r["jury_label"] for r in rows]
    per_judge_pred = {name: [] for name in judge_names}
    for r in rows:
        seen = {j["judge_name"]: j["label"] for j in r["judges"]}
        for name in judge_names:
            per_judge_pred[name].append(seen.get(name, UNKNOWN))

    # Per-judge stats
    per_judge_stats = {}
    for name in judge_names:
        ms = metrics_for_predictions(y_true, per_judge_pred[name])
        ms["per_tier"] = per_tier_accuracy(y_true, per_judge_pred[name])
        per_judge_stats[name] = ms

    # Jury stats
    jury_stats = metrics_for_predictions(y_true, jury_pred)
    jury_stats["per_tier"] = per_tier_accuracy(y_true, jury_pred)
    jury_stats["n_abstained"] = sum(1 for p in jury_pred if p == ABSTAIN)

    # Inter-judge agreement on rows where every judge committed.
    per_row_labels = [
        [next((j["label"] for j in r["judges"] if j["judge_name"] == name), UNKNOWN)
         for name in judge_names]
        for r in rows
    ]
    fleiss = fleiss_kappa(per_row_labels, SSSOM_LABELS)

    # Identify the best single judge by quadratic-weighted κ (ordinal scale).
    rankable = [(n, s["kappa_quadratic"]) for n, s in per_judge_stats.items()
                if s["kappa_quadratic"] is not None]
    rankable.sort(key=lambda x: x[1], reverse=True)
    best_judge = rankable[0][0] if rankable else None

    return {
        "n_rows":         len(rows),
        "judge_names":    judge_names,
        "per_judge":      per_judge_stats,
        "jury":           jury_stats,
        "fleiss_kappa_judges": fleiss,
        "best_judge_by_quadratic_kappa": best_judge,
        "config":         data["config"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v, n=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  -  "
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def print_report(stats: dict) -> None:
    n = stats["n_rows"]
    print("\n══════════════════════════════════════════════════════════════════════")
    print(f"  AI-READI SSSOM BENCHMARK — analysis ({n} rows, "
          f"{len(stats['judge_names'])} judges)")
    print("══════════════════════════════════════════════════════════════════════")

    # Headline table
    print("\n  HEADLINE TABLE")
    print(f"    {'who':<10s}  {'acc':>6s}  {'κ (unw.)':>9s}  {'κ (quad.)':>10s}  "
          f"{'committed':>10s}  {'coverage':>9s}")
    print(f"    {'-'*10}  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*9}")
    for name in stats["judge_names"]:
        s = stats["per_judge"][name]
        print(f"    {name:<10s}  {fmt(s['accuracy_committed']):>6s}  "
              f"{fmt(s['kappa_unweighted']):>9s}  {fmt(s['kappa_quadratic']):>10s}  "
              f"{s['n_committed']:>10d}  {fmt(s['coverage']):>9s}")
    j = stats["jury"]
    print(f"    {'-'*10}")
    print(f"    {'JURY':<10s}  {fmt(j['accuracy_committed']):>6s}  "
          f"{fmt(j['kappa_unweighted']):>9s}  {fmt(j['kappa_quadratic']):>10s}  "
          f"{j['n_committed']:>10d}  {fmt(j['coverage']):>9s}")
    print(f"    Jury abstained on {j['n_abstained']} of {n} rows "
          f"(tied votes → human review).")

    print(f"\n  Inter-judge Fleiss κ: {fmt(stats['fleiss_kappa_judges'])}  "
          f"(agreement among judges, gold ignored)")

    # Per-tier breakdown for the jury and the best single judge.
    print("\n  PER-TIER ACCURACY (jury vs best single judge)")
    best = stats["best_judge_by_quadratic_kappa"]
    print(f"    Best single judge by quadratic κ: {best}")
    print(f"    {'tier':<10s}  {'n':>5s}  {'jury':>6s}  {best+' acc':>14s}")
    print(f"    {'-'*10}  {'-'*5}  {'-'*6}  {'-'*14}")
    jt = stats["jury"]["per_tier"]
    bt = stats["per_judge"][best]["per_tier"] if best else {}
    for tier in SSSOM_LABELS:
        jrow = jt.get(tier, {"n": 0, "accuracy": None})
        brow = bt.get(tier, {"accuracy": None})
        print(f"    {tier:<10s}  {jrow['n']:>5d}  {fmt(jrow['accuracy']):>6s}  "
              f"{fmt(brow['accuracy']):>14s}")

    # ── FOR ABSTRACT ────────────────────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  FOR ABSTRACT (sentence-ready)")
    print("──────────────────────────────────────────────────────────────────────")
    if best is None:
        print("  (no rankable judges — check the run)")
        return
    best_stats = stats["per_judge"][best]
    print(f"  N rows evaluated:           {n}")
    print(f"  Judges:                     {', '.join(stats['judge_names'])}")
    print(f"  Best single judge:          {best}  "
          f"(acc {fmt(best_stats['accuracy_committed'])}, "
          f"κ {fmt(best_stats['kappa_unweighted'])}, "
          f"quad-κ {fmt(best_stats['kappa_quadratic'])})")
    print(f"  Jury (majority vote):       acc {fmt(j['accuracy_committed'])}, "
          f"κ {fmt(j['kappa_unweighted'])}, quad-κ {fmt(j['kappa_quadratic'])}")
    print(f"  Inter-judge Fleiss κ:       {fmt(stats['fleiss_kappa_judges'])}")
    print(f"  Jury abstentions:           {j['n_abstained']} of {n} "
          f"({100*j['n_abstained']/n:.1f}%)")
    # H1 outcome
    if j["kappa_quadratic"] is not None and best_stats["kappa_quadratic"] is not None:
        delta_kappa = j["kappa_quadratic"] - best_stats["kappa_quadratic"]
        verdict = ("jury > best single judge" if delta_kappa > 0.01
                   else "jury ≈ best single judge" if abs(delta_kappa) <= 0.01
                   else "jury < best single judge")
        print(f"  H1 (quad-κ jury vs best):   Δ = {delta_kappa:+.3f}  →  {verdict}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="benchmark_results.json",
                   help="Per-row results JSON from run_benchmark_live.py")
    p.add_argument("--out", default=None,
                   help="Optional path to write the stats dict as JSON")
    args = p.parse_args()

    with open(args.inp) as f:
        data = json.load(f)
    stats = analyze(data)
    print_report(stats)

    if args.out:
        # Strip non-serialisable bits, keep the analysis.
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"  Stats JSON saved to: {args.out}\n")
