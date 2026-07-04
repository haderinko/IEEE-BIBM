#!/usr/bin/env python3
"""
baseline_sapbert.py — SapBERT cosine similarity baseline for §VI.

For each row (source_term, target_concept_name), encode both with SapBERT
(a biomedical concept-embedding model), compute cosine similarity, and
map the similarity to an SSSOM tier via percentile thresholds calibrated
on a small labeled subset.

This gives a "retrieval-style" baseline: the LLM jury is doing a
classification task, and SapBERT is what a straightforward embedding
approach would produce.

Usage
-----
    pip install torch transformers scikit-learn
    python baseline_sapbert.py --csv curator_spreadsheet.csv \
                               --out baseline_sapbert.json

First run downloads ~500 MB (SapBERT weights). Later runs use the cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import cohen_kappa_score

SSSOM_LABELS = ["exact", "close", "narrow", "broad", "related"]
MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# ─────────────────────────────────────────────────────────────────────
# CSV loader — same schema as curator_spreadsheet.csv
# ─────────────────────────────────────────────────────────────────────
COL_SUBJECT = "source_label"
COL_OBJECT  = "target_concept_name"
COL_PRED    = "curator_sssom_label"
COL_ROW_ID  = "row_id"


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            tier = (r.get(COL_PRED) or "").strip().lower()
            if tier not in SSSOM_LABELS:
                continue
            rows.append({
                "row_id":       (r.get(COL_ROW_ID) or f"R{i:04d}").strip(),
                "source":       (r.get(COL_SUBJECT) or "").strip(),
                "target":       (r.get(COL_OBJECT) or "").strip(),
                "gold":         tier,
            })
    return rows


# ─────────────────────────────────────────────────────────────────────
# Encode with SapBERT (CLS pooling)
# ─────────────────────────────────────────────────────────────────────
def encode(model, tokenizer, texts, device, batch_size=16):
    import torch
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, return_tensors="pt",
                         max_length=48)
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            out = model(**toks)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_emb.append(cls)
    return np.vstack(all_emb)


# ─────────────────────────────────────────────────────────────────────
# Percentile-based tier assignment from cosine similarity
#
# We calibrate thresholds by looking at the distribution of cosines within
# each gold tier and assigning bin edges at the tier-median cosines.
# Simple, defensible, no hyperparameter tuning.
# ─────────────────────────────────────────────────────────────────────
def calibrate_and_predict(cos, gold, tiers=SSSOM_LABELS):
    """Compute tier-median cosines from gold; use midpoints as thresholds."""
    tier_median = {}
    for t in tiers:
        vals = [c for c, g in zip(cos, gold) if g == t]
        tier_median[t] = float(np.median(vals)) if vals else float("nan")
    # Order tiers by empirical median cosine (high → low)
    ordered = sorted([t for t in tiers if not np.isnan(tier_median[t])],
                     key=lambda t: -tier_median[t])
    # Thresholds are the midpoints between consecutive tier medians
    medians = [tier_median[t] for t in ordered]
    thresholds = [(medians[i] + medians[i+1]) / 2 for i in range(len(medians) - 1)]

    preds = []
    for c in cos:
        assigned = ordered[-1]  # lowest-cosine tier as default
        for i, thr in enumerate(thresholds):
            if c >= thr:
                assigned = ordered[i]
                break
        preds.append(assigned)
    return preds, {"tier_medians": tier_median, "thresholds": thresholds,
                   "order": ordered}


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="baseline_sapbert.json")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        sys.exit("Install: pip install torch transformers")

    rows = load_rows(args.csv)
    print(f"Loaded {len(rows)} rows.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    sources = [r["source"] for r in rows]
    targets = [r["target"] for r in rows]

    print("Encoding sources...")
    src_emb = encode(model, tokenizer, sources, device)
    print("Encoding targets...")
    tgt_emb = encode(model, tokenizer, targets, device)

    # Cosine similarity, pairwise (row-wise)
    src_n = src_emb / (np.linalg.norm(src_emb, axis=1, keepdims=True) + 1e-12)
    tgt_n = tgt_emb / (np.linalg.norm(tgt_emb, axis=1, keepdims=True) + 1e-12)
    cos = np.sum(src_n * tgt_n, axis=1)

    gold = [r["gold"] for r in rows]
    preds, meta = calibrate_and_predict(cos.tolist(), gold)

    acc = float(np.mean([p == g for p, g in zip(preds, gold)]))
    k_unw = float(cohen_kappa_score(gold, preds, labels=SSSOM_LABELS))
    k_quad = float(cohen_kappa_score(gold, preds, labels=SSSOM_LABELS,
                                     weights="quadratic"))

    print(f"\nBaseline: SapBERT cosine + percentile thresholds")
    print(f"  Accuracy:   {acc:.3f}")
    print(f"  κ (unw.):   {k_unw:.3f}")
    print(f"  κ (quad.):  {k_quad:.3f}")
    print(f"  Thresholds: {meta['thresholds']}")
    print(f"  Order:      {meta['order']}")

    out = {
        "model": args.model,
        "n_rows": len(rows),
        "accuracy": acc,
        "kappa_unweighted": k_unw,
        "kappa_quadratic":  k_quad,
        "meta": meta,
        "rows": [
            {"row_id": r["row_id"], "gold": r["gold"], "pred": p, "cosine": float(c)}
            for r, p, c in zip(rows, preds, cos)
        ],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to: {args.out}\n")


if __name__ == "__main__":
    main()
