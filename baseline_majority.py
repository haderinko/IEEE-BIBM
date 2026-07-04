#!/usr/bin/env python3
"""
baseline_majority.py -- majority-class floor for the SSSOM benchmark.

The trivial baseline that always predicts the most frequent gold tier.
Reviewers use this as a sanity anchor: it is the accuracy a model gets
for free by ignoring the input entirely. Reports the same metric family
as the other methods in Table II so the row drops straight in.

Reads the gold labels from the SAME v4 results JSON the rest of the
analysis uses, so there is no separate data dependency and no API cost.

Usage
-----
    python baseline_majority.py --in benchmark_results_v4.json
    # or point at any results JSON that has rows[].gold_label
"""

import argparse
import json
from collections import Counter


SSSOM_LABELS = ["exact", "close", "narrow", "broad", "related"]

# Directional SSSOM cost matrix C[gold][pred] -- identical to paper Table I.
COST = {
    "exact":   {"exact": 0, "close": 1, "narrow": 4, "broad": 4, "related": 9},
    "close":   {"exact": 1, "close": 0, "narrow": 2, "broad": 2, "related": 4},
    "narrow":  {"exact": 4, "close": 2, "narrow": 0, "broad": 9, "related": 4},
    "broad":   {"exact": 4, "close": 2, "narrow": 9, "broad": 0, "related": 4},
    "related": {"exact": 9, "close": 4, "narrow": 4, "broad": 4, "related": 0},
}


def cohen_kappa(gold, pred, weights=None):
    """Cohen's kappa. weights=None -> unweighted; 'quadratic' or a cost
    matrix dict -> weighted. Weighted kappa = 1 - sum(w*O)/sum(w*E)."""
    n = len(gold)
    labels = SSSOM_LABELS
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)

    # Confusion (observed) and expected-under-independence matrices.
    O = [[0.0] * k for _ in range(k)]
    for g, p in zip(gold, pred):
        O[idx[g]][idx[p]] += 1
    row = [sum(O[i]) for i in range(k)]
    col = [sum(O[i][j] for i in range(k)) for j in range(k)]
    E = [[row[i] * col[j] / n for j in range(k)] for i in range(k)]

    if weights is None:  # unweighted
        w = [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    elif weights == "quadratic":
        w = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    else:  # cost-matrix dict, normalized to its own max so kappa is in a sane range
        mx = max(COST[g][p] for g in labels for p in labels)
        w = [[COST[labels[i]][labels[j]] / mx for j in range(k)] for i in range(k)]

    num = sum(w[i][j] * O[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * E[i][j] for i in range(k) for j in range(k))
    return 1.0 - num / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True,
                    help="results JSON with rows[].gold_label")
    ap.add_argument("--out", dest="outfile", default="baseline_majority.json")
    args = ap.parse_args()

    with open(args.infile) as f:
        data = json.load(f)

    gold = [r["gold_label"] for r in data["rows"]]
    n = len(gold)

    # Majority tier and the constant prediction vector.
    counts = Counter(gold)
    majority_tier = counts.most_common(1)[0][0]
    pred = [majority_tier] * n

    accuracy = sum(1 for g, p in zip(gold, pred) if g == p) / n

    k_unw = cohen_kappa(gold, pred, weights=None)
    k_quad = cohen_kappa(gold, pred, weights="quadratic")
    k_cost = cohen_kappa(gold, pred, weights="cost")

    # Mean expected cost of the constant prediction.
    mean_cost = sum(COST[g][majority_tier] for g in gold) / n

    result = {
        "baseline": "majority_class",
        "n_rows": n,
        "majority_tier": majority_tier,
        "tier_counts": dict(counts),
        "accuracy": round(accuracy, 3),
        "kappa_unweighted": round(k_unw, 3),
        "kappa_quadratic": round(k_quad, 3),
        "kappa_cost": round(k_cost, 3),
        "mean_cost": round(mean_cost, 3),
    }

    with open(args.outfile, "w") as f:
        json.dump(result, f, indent=2)

    print("Majority-class baseline")
    print(f"  Majority tier: {majority_tier} ({counts[majority_tier]}/{n})")
    print(f"  Accuracy:      {accuracy:.3f}")
    print(f"  kappa (unw.):  {k_unw:.3f}")
    print(f"  kappa (quad.): {k_quad:.3f}")
    print(f"  kappa (cost):  {k_cost:.3f}")
    print(f"  Mean cost:     {mean_cost:.3f}")
    print(f"Saved to: {args.outfile}")


if __name__ == "__main__":
    main()
