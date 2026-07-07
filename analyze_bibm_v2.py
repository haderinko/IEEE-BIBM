#!/usr/bin/env python3
"""
analyze_bibm_v2.py — extended analysis with calibration + risk-coverage.

Adds four things on top of analyze_bibm.py:

  1. Directional cost matrix κ (using the approved C matrix in Table I).
  2. Calibration analysis on verbalized confidences:
        - reliability diagram data (10 bins)
        - Expected Calibration Error (ECE)
        - Brier score
        - Platt-scaled recalibration on a held-out 20% split
  3. Risk-coverage sweep + AURC (area under risk-coverage curve).
        - Three uncertainty signals: vote margin, vote entropy, mean confidence.
        - For each: sweep threshold τ, compute (coverage, selective_accuracy).
  4. Aggregation ablation:
        - Majority vote (current baseline)
        - Confidence-weighted vote (each judge weighted by verbalized conf)
        - Cost-sensitive ordinal vote (minimize expected C[gold, pred])

Usage
-----
    pip install scikit-learn scipy numpy
    python analyze_bibm_v2.py --in benchmark_results_v4.json --out stats_bibm_v2.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

SSSOM_LABELS = ["exact", "close", "narrow", "broad", "related"]
UNKNOWN, ABSTAIN = "UNKNOWN", "ABSTAIN"
LABEL_TO_IDX = {l: i for i, l in enumerate(SSSOM_LABELS)}

# ─────────────────────────────────────────────────────────────────────
# Directional SSSOM cost matrix (Table I)
# Rows = gold, cols = predicted.
# ─────────────────────────────────────────────────────────────────────
COST_MATRIX = np.array([
    # exact close narrow broad related
    [    0,    1,     4,    4,       9],   # gold = exact
    [    1,    0,     2,    2,       4],   # gold = close
    [    4,    2,     0,    9,       4],   # gold = narrow
    [    4,    2,     9,    0,       4],   # gold = broad
    [    9,    4,     4,    4,       0],   # gold = related
], dtype=float)


# ─────────────────────────────────────────────────────────────────────
# Weighted κ against arbitrary cost matrix
# ─────────────────────────────────────────────────────────────────────
def kappa_cost_matrix(y_true, y_pred, C):
    """Weighted Cohen's κ with an arbitrary cost matrix C[gold, pred]."""
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p not in (UNKNOWN, ABSTAIN)]
    if len(pairs) < 2:
        return float("nan")
    N = len(pairs)
    K = len(SSSOM_LABELS)
    # Confusion matrix (raw counts)
    O = np.zeros((K, K))
    for t, p in pairs:
        O[LABEL_TO_IDX[t], LABEL_TO_IDX[p]] += 1
    # Row/col marginals
    row = O.sum(axis=1)
    col = O.sum(axis=0)
    # Expected under independence
    E = np.outer(row, col) / N
    # Weighted disagreement
    obs = (O * C).sum() / N
    exp = (E * C).sum() / N
    if exp == 0:
        return 1.0 - obs / max(1e-12, C.max())
    return float(1.0 - obs / exp)


# ─────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────
def reliability_bins(confs, correct, n_bins=10):
    """Bin (confidence, correctness) pairs. Returns list of dicts per bin."""
    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs >= lo) & (confs < hi if hi < 1.0 else confs <= hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "mean_conf": None, "acc": None})
        else:
            bins.append({
                "lo": float(lo), "hi": float(hi), "n": n,
                "mean_conf": float(confs[mask].mean()),
                "acc":       float(correct[mask].mean()),
            })
    return bins


def expected_calibration_error(confs, correct, n_bins=10):
    """ECE = weighted sum of |conf - acc| over bins."""
    if len(confs) == 0:
        return float("nan")
    bins = reliability_bins(confs, correct, n_bins)
    N = len(confs)
    ece = 0.0
    for b in bins:
        if b["n"] > 0:
            ece += (b["n"] / N) * abs(b["mean_conf"] - b["acc"])
    return float(ece)


def brier_score(confs, correct):
    """Brier = mean squared error between confidence and 0/1 correctness."""
    if len(confs) == 0:
        return float("nan")
    return float(np.mean((confs - correct) ** 2))


def platt_recalibrate(confs_train, correct_train, confs_test):
    """Fit Platt scaling (logistic on confidence) and apply to test set."""
    if len(confs_train) < 5:
        return confs_test.copy()
    X = confs_train.reshape(-1, 1)
    y = correct_train.astype(int)
    if y.min() == y.max():
        # No variance in y — return uncalibrated
        return confs_test.copy()
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf.predict_proba(confs_test.reshape(-1, 1))[:, 1]


# ─────────────────────────────────────────────────────────────────────
# Risk-coverage sweep
# ─────────────────────────────────────────────────────────────────────
def risk_coverage_curve(uncertainties, correct, n_points=51):
    """Sweep abstention threshold τ; return per-point (coverage, selective_acc, tau).

    Rows with uncertainty > τ are abstained (routed to human).
    Selective accuracy computed on committed rows only.
    """
    u = np.asarray(uncertainties, dtype=float)
    c = np.asarray(correct, dtype=float)
    n = len(u)
    taus = np.linspace(0, 1, n_points)
    out = []
    for tau in taus:
        commit_mask = u <= tau
        n_commit = int(commit_mask.sum())
        coverage = n_commit / n
        sel_acc = float(c[commit_mask].mean()) if n_commit > 0 else float("nan")
        out.append({
            "tau":          float(tau),
            "coverage":     float(coverage),
            "n_committed":  n_commit,
            "sel_accuracy": sel_acc,
        })
    return out


def aurc(rc_points):
    """Area under risk-coverage curve. Risk = 1 - selective_accuracy.
       Integrate risk over coverage from 0 to 1 via trapezoidal rule."""
    pts = [p for p in rc_points if not np.isnan(p["sel_accuracy"])]
    if len(pts) < 2:
        return float("nan")
    pts.sort(key=lambda p: p["coverage"])
    cov = np.array([p["coverage"] for p in pts])
    risk = 1.0 - np.array([p["sel_accuracy"] for p in pts])
    return float(np.trapezoid(risk, cov))


# ─────────────────────────────────────────────────────────────────────
# Uncertainty signals per row
# ─────────────────────────────────────────────────────────────────────
def vote_entropy(labels):
    """Shannon entropy of the vote distribution (normalized to [0, 1])."""
    labels = [l for l in labels if l not in (UNKNOWN, ABSTAIN)]
    if not labels:
        return 1.0
    counts = Counter(labels)
    ps = np.array(list(counts.values()), dtype=float) / len(labels)
    max_ent = np.log(len(SSSOM_LABELS))
    return float(-np.sum(ps * np.log(ps + 1e-12)) / max_ent) if max_ent > 0 else 0.0


def vote_margin(labels):
    """1 - (fraction agreeing on top label). Higher = less certain."""
    labels = [l for l in labels if l not in (UNKNOWN, ABSTAIN)]
    if not labels:
        return 1.0
    counts = Counter(labels)
    top = counts.most_common(1)[0][1]
    return 1.0 - top / len(labels)


def mean_conf(confs):
    """Mean of verbalized confidences (drop None). Return uncertainty = 1 - mean/100."""
    vals = [c for c in confs if c is not None]
    if not vals:
        return 1.0
    return 1.0 - float(np.mean(vals)) / 100.0


# ─────────────────────────────────────────────────────────────────────
# Aggregation ablations
# ─────────────────────────────────────────────────────────────────────
def agg_majority(labels, confs):
    labs = [l for l in labels if l not in (UNKNOWN, ABSTAIN)]
    if not labs:
        return ABSTAIN
    counts = Counter(labs)
    ranked = counts.most_common()
    top_count = ranked[0][1]
    top_labels = [l for l, c in ranked if c == top_count]
    return ABSTAIN if len(top_labels) > 1 else top_labels[0]


def agg_confidence_weighted(labels, confs):
    weights = defaultdict(float)
    for l, c in zip(labels, confs):
        if l in (UNKNOWN, ABSTAIN):
            continue
        w = (c or 50.0) / 100.0
        weights[l] += w
    if not weights:
        return ABSTAIN
    max_w = max(weights.values())
    tops = [l for l, w in weights.items() if w == max_w]
    return ABSTAIN if len(tops) > 1 else tops[0]


def agg_cost_sensitive(labels, confs, C=COST_MATRIX):
    """For each candidate label c, compute expected cost given the judges'
       stated label distribution (weighted by confidence). Pick argmin."""
    weights = defaultdict(float)
    for l, c in zip(labels, confs):
        if l in (UNKNOWN, ABSTAIN):
            continue
        w = (c or 50.0) / 100.0
        weights[l] += w
    total = sum(weights.values())
    if total == 0:
        return ABSTAIN
    p = np.zeros(len(SSSOM_LABELS))
    for l, w in weights.items():
        p[LABEL_TO_IDX[l]] = w / total
    exp_costs = C.T @ p        # exp_costs[j] = sum_i p_i * C[i, j]
    best = int(np.argmin(exp_costs))
    return SSSOM_LABELS[best]


# ─────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────
def load(path):
    with open(path) as f:
        return json.load(f)


def extract_per_judge(rows, judge_names):
    per = {}
    for name in judge_names:
        labels, confs = [], []
        for r in rows:
            j = next((x for x in r["judges"] if x["judge_name"] == name), None)
            labels.append(j["label"] if j else UNKNOWN)
            c = j.get("verbalized_confidence") if j else None
            if c is None and j:
                c = j.get("confidence")
            confs.append(c)
        per[name] = {"labels": labels, "confs": confs}
    return per


# ─────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────
def analyze(data):
    rows = data["rows"]
    judge_names = [j["name"] for j in data["config"]["judges"]]
    y_true = [r["gold_label"] for r in rows]
    jury_pred = [r["jury_label"] for r in rows]
    per_judge = extract_per_judge(rows, judge_names)

    N = len(rows)
    correct_arr = {}
    for name, d in per_judge.items():
        correct_arr[name] = np.array([1.0 if p == t else 0.0
                                      for p, t in zip(d["labels"], y_true)])
    jury_correct = np.array([1.0 if p == t else 0.0
                             for p, t in zip(jury_pred, y_true)])
    correct_arr["jury"] = jury_correct

    # ── 1. Directional cost-matrix κ ──────────────────────────────────
    cost_kappa = {"jury": kappa_cost_matrix(y_true, jury_pred, COST_MATRIX)}
    for name in judge_names:
        cost_kappa[name] = kappa_cost_matrix(y_true, per_judge[name]["labels"], COST_MATRIX)

    # ── 2. Calibration ────────────────────────────────────────────────
    calibration = {}
    rng = np.random.default_rng(20260702)
    for name in judge_names:
        confs = np.array([c / 100.0 if c is not None else np.nan
                          for c in per_judge[name]["confs"]])
        mask = ~np.isnan(confs)
        confs_v = confs[mask]
        cor_v = correct_arr[name][mask]
        if len(confs_v) < 5:
            calibration[name] = {"note": "insufficient confidence data"}
            continue
        # Held-out 20% for Platt
        idx = np.arange(len(confs_v))
        rng.shuffle(idx)
        n_test = max(1, int(0.2 * len(confs_v)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        raw_ece = expected_calibration_error(confs_v, cor_v)
        raw_brier = brier_score(confs_v, cor_v)
        recal = platt_recalibrate(confs_v[train_idx], cor_v[train_idx], confs_v[test_idx])
        recal_ece = expected_calibration_error(recal, cor_v[test_idx])
        recal_brier = brier_score(recal, cor_v[test_idx])
        calibration[name] = {
            "n_with_confidence": int(mask.sum()),
            "reliability_bins":  reliability_bins(confs_v, cor_v),
            "raw_ece":           raw_ece,
            "raw_brier":         raw_brier,
            "n_test":            n_test,
            "recal_ece":         recal_ece,
            "recal_brier":       recal_brier,
        }

    # ── 3. Risk-coverage sweep ────────────────────────────────────────
    # Uncertainty signals per row
    u_margin = np.array([vote_margin([per_judge[n]["labels"][i] for n in judge_names])
                         for i in range(N)])
    u_entropy = np.array([vote_entropy([per_judge[n]["labels"][i] for n in judge_names])
                          for i in range(N)])
    u_meanconf = np.array([mean_conf([per_judge[n]["confs"][i] for n in judge_names])
                           for i in range(N)])

    rc = {}
    for u_name, u in [("vote_margin", u_margin),
                      ("vote_entropy", u_entropy),
                      ("mean_confidence", u_meanconf)]:
        rc_points = risk_coverage_curve(u, jury_correct)
        rc[u_name] = {"points": rc_points, "aurc": aurc(rc_points)}

    # Also compute risk-coverage for the best single judge using its own confidence.
    best_by_acc = max(judge_names, key=lambda n: correct_arr[n].mean())
    if any(c is not None for c in per_judge[best_by_acc]["confs"]):
        best_confs = np.array([c / 100.0 if c is not None else 0.5
                               for c in per_judge[best_by_acc]["confs"]])
        u_best = 1.0 - best_confs
        rc_best = risk_coverage_curve(u_best, correct_arr[best_by_acc])
        rc[f"{best_by_acc}_own_confidence"] = {"points": rc_best, "aurc": aurc(rc_best)}

    # ── 4. Aggregation ablation ───────────────────────────────────────
    agg_results = {}
    for agg_name, agg_fn in [("majority", agg_majority),
                             ("confidence_weighted", agg_confidence_weighted),
                             ("cost_sensitive_ordinal", agg_cost_sensitive)]:
        preds = []
        for i in range(N):
            labels_i = [per_judge[n]["labels"][i] for n in judge_names]
            confs_i  = [per_judge[n]["confs"][i]  for n in judge_names]
            preds.append(agg_fn(labels_i, confs_i))
        n_abstain = sum(1 for p in preds if p == ABSTAIN)
        n_correct_all = sum(1 for p, t in zip(preds, y_true) if p == t)
        committed = [(p, t) for p, t in zip(preds, y_true) if p not in (ABSTAIN, UNKNOWN)]
        n_correct_com = sum(1 for p, t in committed if p == t)
        agg_results[agg_name] = {
            "acc_all":       n_correct_all / N,
            "acc_committed": n_correct_com / max(1, len(committed)),
            "n_abstain":     n_abstain,
            "kappa_unw":     float(cohen_kappa_score([t for _, t in committed],
                                                     [p for p, _ in committed],
                                                     labels=SSSOM_LABELS)) if committed else float("nan"),
            "kappa_quad":    float(cohen_kappa_score([t for _, t in committed],
                                                     [p for p, _ in committed],
                                                     labels=SSSOM_LABELS,
                                                     weights="quadratic")) if committed else float("nan"),
            "kappa_cost":    kappa_cost_matrix(y_true, preds, COST_MATRIX),
        }

    return {
        "n_rows":               N,
        "judge_names":          judge_names,
        "cost_matrix":          COST_MATRIX.tolist(),
        "kappa_cost_matrix":    cost_kappa,
        "calibration":          calibration,
        "risk_coverage":        rc,
        "aggregation_ablation": agg_results,
        "best_by_accuracy":     best_by_acc,
    }


# ─────────────────────────────────────────────────────────────────────
# Reporter
# ─────────────────────────────────────────────────────────────────────
def fmt(v, n=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def print_report(s):
    print("\n" + "═" * 78)
    print(f"  AI-READI SSSOM BENCHMARK v4 — extended analysis "
          f"({s['n_rows']} rows, {len(s['judge_names'])} judges)")
    print("═" * 78)

    print("\n  DIRECTIONAL COST-MATRIX κ (Table I weighting)")
    for who in ["jury"] + s["judge_names"]:
        print(f"    {who:<10s}  cost-κ = {fmt(s['kappa_cost_matrix'][who])}")

    print("\n  CALIBRATION")
    print(f"    {'judge':<10s}  {'n_conf':>8s}  {'ECE raw':>10s}  {'ECE recal':>12s}"
          f"  {'Brier raw':>10s}  {'Brier recal':>13s}")
    for name in s["judge_names"]:
        c = s["calibration"].get(name, {})
        if "note" in c:
            print(f"    {name:<10s}  ({c['note']})")
            continue
        print(f"    {name:<10s}  {c['n_with_confidence']:>8d}  "
              f"{fmt(c['raw_ece']):>10s}  {fmt(c['recal_ece']):>12s}  "
              f"{fmt(c['raw_brier']):>10s}  {fmt(c['recal_brier']):>13s}")

    print("\n  RISK-COVERAGE (AURC — lower is better)")
    for name, d in s["risk_coverage"].items():
        print(f"    {name:<28s}  AURC = {fmt(d['aurc'])}")

    print("\n  AGGREGATION ABLATION")
    print(f"    {'method':<22s}  {'acc_all':>8s}  {'acc_com':>8s}  "
          f"{'κ':>6s}  {'quad-κ':>7s}  {'cost-κ':>7s}  {'abst':>5s}")
    for m, r in s["aggregation_ablation"].items():
        print(f"    {m:<22s}  {fmt(r['acc_all']):>8s}  {fmt(r['acc_committed']):>8s}  "
              f"{fmt(r['kappa_unw'],2):>6s}  {fmt(r['kappa_quad'],2):>7s}  "
              f"{fmt(r['kappa_cost'],2):>7s}  {r['n_abstain']:>5d}")

    print("\n  FOR PAPER (sentence-ready)")
    j = s["kappa_cost_matrix"]["jury"]
    best = s["best_by_accuracy"]
    b = s["kappa_cost_matrix"][best]
    print(f"    Directional cost-matrix κ: jury = {fmt(j)}, best single ({best}) = {fmt(b)}")
    best_rc = min(s["risk_coverage"].items(), key=lambda x: x[1]["aurc"] if not np.isnan(x[1]["aurc"]) else 1e9)
    print(f"    Best risk-coverage signal: {best_rc[0]}, AURC = {fmt(best_rc[1]['aurc'])}")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="benchmark_results_v4.json")
    p.add_argument("--out", default="stats_bibm_v2.json")
    args = p.parse_args()

    data = load(args.inp)
    s = analyze(data)
    print_report(s)
    with open(args.out, "w") as f:
        json.dump(s, f, indent=2, default=str)
    print(f"  Full stats saved to: {args.out}\n")
