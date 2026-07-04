#!/usr/bin/env python3
"""
motivating_example.py  (v3 — confidence logging + bias check)
─────────────────────────────────────────────────────────────
AI-READI SSSOM Benchmark — pilot demonstration with multi-row support.

Runs the LLM-jury protocol on one or more rows from AI-READI's OMOP mapping
table, collects independent ratings, aggregates by majority vote (with tie =
abstain), and compares both single-judge and jury outputs to the expert
gold label.

New in v3 (inspired by Ahmed et al., MSR'25):
  - Per-judge confidence logging. Captures the log-probability of each
    judge's chosen label, which downstream analysis can use for
    confidence-based sample selection (Ahmed et al., Section IV-C).
  - Judge label-distribution sanity check. Across all rows in a run,
    tracks each judge's marginal distribution over SSSOM labels. Flags
    judges that appear collapsed to a small subset of the label space
    (e.g., the Llama-3 "only picks Strongly-Agree/Strongly-Disagree"
    failure mode documented in Ahmed et al., Section IV-A).
  - Multi-row scaffold. A run can now process a list of rows; per-row
    records and an aggregate run-summary are both written to disk.

Carried forward from v2:
  - Aggregation: ties route to ABSTAIN (human review), not arbitrary pick.
  - Normalisation: word-boundary matching; "unrelated" is not "related".
  - Reproducibility: temperature=0, top_p=1.0, pinned model versions logged
    per row.
  - UNKNOWN policy: excluded from majority vote, counted separately.

Out of scope for this script (handled in the analysis layer downstream):
  - Weighted Cohen's κ with directional SSSOM cost matrix (judge ↔ expert)
  - Fleiss' κ and Krippendorff's α for inter-judge agreement
  - McNemar test for paired jury-vs-best-judge comparison
  - Bootstrap 95% CIs on all headline metrics, stratified by SSSOM tier
  - Second-curator regrade for the human ceiling

Usage
-----
    python motivating_example.py --demo     # No API keys; uses recorded responses
    python motivating_example.py --live     # Real API calls; requires API keys
    python motivating_example.py --test     # Self-tests for normalize + majority vote

Required environment variables for --live:
    ANTHROPIC_API_KEY        (Claude)
    OPENAI_API_KEY           (GPT)
    DEEPSEEK_API_KEY         (DeepSeek)
    TOGETHER_API_KEY         (Llama via Together AI)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. THE ROWS UNDER TEST
#    Each row carries the source term, OMOP target, and the expert
#    curator's SSSOM label (hidden from judges; used only for comparison).
#
#    The HbA1c row is the canonical motivating example. Additional rows
#    cover the other four SSSOM tiers so the distribution check has a
#    realistic label space to verify against.
# ─────────────────────────────────────────────────────────────────────────────

ROWS = [
    {
        "row_id":           "AIR-001",
        "source_term":      "HbA1c",
        "target_label":     "Hemoglobin A1c/Hemoglobin.total in Blood",
        "target_omop_id":   3004410,
        "gold_sssom_label": "exact",
    },
    {
        "row_id":           "AIR-002",
        "source_term":      "BP",
        "target_label":     "Blood pressure systolic and diastolic",
        "target_omop_id":   4154790,
        "gold_sssom_label": "broad",   # source ambiguous; target adds systolic/diastolic split
    },
    {
        "row_id":           "AIR-003",
        "source_term":      "Fasting glucose 8h",
        "target_label":     "Glucose [Mass/volume] in Serum or Plasma --8 hours fasting",
        "target_omop_id":   3037110,
        "gold_sssom_label": "close",   # close but target adds Serum/Plasma + Mass/volume specifier
    },
    {
        "row_id":           "AIR-004",
        "source_term":      "Eye exam",
        "target_label":     "Retinal photography",
        "target_omop_id":   4079775,
        "gold_sssom_label": "narrow",  # target is a specific kind of eye exam
    },
    {
        "row_id":           "AIR-005",
        "source_term":      "Smoking",
        "target_label":     "History of tobacco use",
        "target_omop_id":   4055674,
        "gold_sssom_label": "related", # connected but smoking ≠ tobacco-use history
    },
]

# Backwards-compatible single-row alias for legacy callers.
ROW = ROWS[0]


# ─────────────────────────────────────────────────────────────────────────────
# 2. THE JUDGE PROMPT (identical across all four judges)
# ─────────────────────────────────────────────────────────────────────────────

SSSOM_LABELS = ["exact", "close", "narrow", "broad", "related"]

PROMPT_TEMPLATE = """You are evaluating the quality of a translation between a clinical source term and a target OMOP concept.

Rate the mapping quality on the SSSOM scale, choosing exactly ONE label:

 - exact   : source and target represent the same concept
 - close   : nearly the same; minor lexical or specificity difference
 - narrow  : the target is more specific than the source
 - broad   : the target is more general than the source
 - related : connected but not equivalent

Source term:  "{source_term}"
Target OMOP:  "{target_label}" (concept_id: {target_omop_id})

Respond with ONE word only from the set above. No explanation."""


# ─────────────────────────────────────────────────────────────────────────────
# 3. JUDGE INTERFACES — one function per LLM
#    All judges pinned to specific model versions and temperature=0 for
#    reproducibility. Versions are logged into the run record.
# ─────────────────────────────────────────────────────────────────────────────

# Pinned model identifiers — change only with a version-bump comment.
MODEL_VERSIONS = {
    "Claude":   "claude-opus-4-7",
    "GPT":      "gpt-5.4",
    "DeepSeek": "deepseek-chat-2024-12",
    "Llama":    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
}

# Generation knobs shared across all judges.
GEN_PARAMS = {
    "temperature": 0.0,
    "max_tokens":  10,
    "top_p":       1.0,
}


@dataclass
class JudgeResponse:
    judge_name:    str
    model:         str
    label:         str
    raw_response:  str
    logprob:       Optional[float] = None   # log-probability of the chosen token
    confidence:    Optional[float] = None   # probability in [0, 1] = exp(logprob)


def _to_confidence(logprob: Optional[float]) -> Optional[float]:
    """Map a log-probability (natural log) to a probability in [0, 1]."""
    if logprob is None:
        return None
    try:
        return float(math.exp(logprob))
    except (OverflowError, ValueError):
        return None


def query_claude(prompt: str) -> tuple[str, Optional[float]]:
    """Returns (text, logprob_of_first_token). Anthropic currently does not
    expose token log-probabilities in the public Messages API, so logprob
    is None for Claude. Downstream analysis should handle missing
    confidence values gracefully."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=MODEL_VERSIONS["Claude"],
        max_tokens=GEN_PARAMS["max_tokens"],
        temperature=GEN_PARAMS["temperature"],
        top_p=GEN_PARAMS["top_p"],
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text, None


def query_openai(prompt: str) -> tuple[str, Optional[float]]:
    """Returns (text, logprob_of_first_token). OpenAI exposes per-token
    log-probabilities via the `logprobs` parameter on chat completions."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=MODEL_VERSIONS["GPT"],
        max_tokens=GEN_PARAMS["max_tokens"],
        temperature=GEN_PARAMS["temperature"],
        top_p=GEN_PARAMS["top_p"],
        logprobs=True,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = resp.choices[0]
    text = choice.message.content
    logprob = None
    try:
        # The first content token's log-probability is what we care about,
        # since the chosen SSSOM label is (modulo whitespace) the first token.
        logprob = float(choice.logprobs.content[0].logprob)
    except (AttributeError, IndexError, TypeError):
        pass
    return text, logprob


def query_deepseek(prompt: str) -> tuple[str, Optional[float]]:
    """Returns (text, logprob_of_first_token). DeepSeek uses an
    OpenAI-compatible API and supports the same `logprobs` flag."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    resp = client.chat.completions.create(
        model=MODEL_VERSIONS["DeepSeek"],
        max_tokens=GEN_PARAMS["max_tokens"],
        temperature=GEN_PARAMS["temperature"],
        top_p=GEN_PARAMS["top_p"],
        logprobs=True,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = resp.choices[0]
    text = choice.message.content
    logprob = None
    try:
        logprob = float(choice.logprobs.content[0].logprob)
    except (AttributeError, IndexError, TypeError):
        pass
    return text, logprob


def query_llama(prompt: str) -> tuple[str, Optional[float]]:
    """Returns (text, logprob_of_first_token). Together AI's OpenAI-compatible
    endpoint supports `logprobs` for many models including Llama 4."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )
    resp = client.chat.completions.create(
        model=MODEL_VERSIONS["Llama"],
        max_tokens=GEN_PARAMS["max_tokens"],
        temperature=GEN_PARAMS["temperature"],
        top_p=GEN_PARAMS["top_p"],
        logprobs=True,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = resp.choices[0]
    text = choice.message.content
    logprob = None
    try:
        logprob = float(choice.logprobs.content[0].logprob)
    except (AttributeError, IndexError, TypeError):
        pass
    return text, logprob


JUDGES = [
    ("Claude",    query_claude),
    ("GPT",       query_openai),
    ("DeepSeek",  query_deepseek),
    ("Llama",     query_llama),
]


# Hard-coded responses for --demo mode (no API keys needed).
# Each entry is (label, logprob). Demo logprobs are chosen to look like
# realistic single-token confidences: high for unambiguous rows, lower for
# rows where the LLM genuinely waffles. None matches Claude's API behaviour.
DEMO_RESPONSES: dict[str, dict[str, tuple[str, Optional[float]]]] = {
    "AIR-001": {  # HbA1c — clear "exact" for most, "close" outlier
        "Claude":   ("exact",   None),
        "GPT":      ("exact",   -0.05),   # ~95% confidence
        "DeepSeek": ("close",   -0.41),   # ~66% confidence (genuine waffle)
        "Llama":    ("exact",   -0.10),   # ~90% confidence
    },
    "AIR-002": {  # BP → systolic/diastolic — should be "broad"
        "Claude":   ("broad",   None),
        "GPT":      ("broad",   -0.22),
        "DeepSeek": ("broad",   -0.36),
        "Llama":    ("exact",   -0.51),   # Llama wrong + lower confidence
    },
    "AIR-003": {  # Fasting glucose 8h — "close"
        "Claude":   ("close",   None),
        "GPT":      ("close",   -0.18),
        "DeepSeek": ("close",   -0.29),
        "Llama":    ("close",   -0.33),
    },
    "AIR-004": {  # Eye exam → Retinal photography — "narrow"
        "Claude":   ("narrow",  None),
        "GPT":      ("narrow",  -0.13),
        "DeepSeek": ("related", -0.61),   # DeepSeek dissent + low confidence
        "Llama":    ("narrow",  -0.27),
    },
    "AIR-005": {  # Smoking → tobacco history — "related"
        "Claude":   ("related", None),
        "GPT":      ("related", -0.24),
        "DeepSeek": ("related", -0.31),
        "Llama":    ("close",   -0.78),   # Llama wrong with very low confidence
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. NORMALISE FREE-FORM OUTPUT TO ONE SSSOM LABEL
#
#    v2 fixes:
#      - Word-boundary matching: "unrelated" no longer reads as "related";
#        "closely related" matches "related" only (not "close").
#      - Prefers the *first* SSSOM label found, with exact whole-string match
#        winning over substring match.
#      - Returns UNKNOWN when no SSSOM label appears, or when multiple
#        non-overlapping SSSOM labels appear in the response (ambiguous).
# ─────────────────────────────────────────────────────────────────────────────

UNKNOWN = "UNKNOWN"
ABSTAIN = "ABSTAIN"   # used by the jury when no majority emerges

_PUNCT_RE = re.compile(r"[.,!?;:\"'`]")
_LABEL_RE = {label: re.compile(rf"\b{label}\b") for label in SSSOM_LABELS}


def normalize(raw: str) -> str:
    """Map a free-form judge response to one of {exact, close, narrow, broad,
    related, UNKNOWN}. Word-boundary aware."""
    if not raw:
        return UNKNOWN
    text = _PUNCT_RE.sub("", raw.strip().lower()).strip()

    # Whole-string exact match wins.
    if text in SSSOM_LABELS:
        return text

    # Otherwise scan for SSSOM labels as whole words.
    hits = [label for label, pat in _LABEL_RE.items() if pat.search(text)]
    if len(hits) == 1:
        return hits[0]
    # Zero hits, or ambiguous multi-hit response → UNKNOWN.
    return UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAJORITY VOTE (tie = ABSTAIN → route to human review)
#
#    Policy:
#      - UNKNOWN responses are excluded from the count entirely.
#      - If a single label has strictly more votes than any other label
#        among the remaining responses, that label is the jury verdict.
#      - Otherwise (tie at the top, or all responses UNKNOWN) the jury
#        ABSTAINs. ABSTAIN is a first-class outcome; downstream the row
#        is flagged for human review.
# ─────────────────────────────────────────────────────────────────────────────

def majority_vote(responses: list[JudgeResponse]) -> str:
    counts = Counter(r.label for r in responses if r.label != UNKNOWN)
    if not counts:
        return ABSTAIN
    ranked = counts.most_common()
    top_label, top_count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == top_count:
        return ABSTAIN     # tie at the top
    return top_label


# ─────────────────────────────────────────────────────────────────────────────
# 6. RUN — per-row + multi-row
# ─────────────────────────────────────────────────────────────────────────────

# When fewer than this many of the 5 SSSOM labels appear in a judge's marginal
# across all rows, the judge is flagged as collapsed. Ahmed et al. (MSR'25)
# documented exactly this failure mode for Llama-3 on code-summarisation
# tasks, where the model only ever picked the two extreme Likert options.
MIN_DISTINCT_LABELS = 3


def run_row(row: dict, demo: bool) -> dict:
    """Run the four-judge protocol on a single row."""
    prompt = PROMPT_TEMPLATE.format(**row)

    responses: list[JudgeResponse] = []
    for name, query_fn in JUDGES:
        if demo:
            raw, logprob = DEMO_RESPONSES[row["row_id"]][name]
        else:
            raw, logprob = query_fn(prompt)
        responses.append(
            JudgeResponse(
                judge_name=name,
                model=MODEL_VERSIONS[name],
                label=normalize(raw),
                raw_response=raw,
                logprob=logprob,
                confidence=_to_confidence(logprob),
            )
        )

    jury = majority_vote(responses)
    gold = row["gold_sssom_label"]

    abstained = jury == ABSTAIN
    unknown_count = sum(1 for r in responses if r.label == UNKNOWN)

    return {
        # Identification
        "row": row,
        "prompt": prompt,

        # Per-judge
        "judges": [asdict(r) for r in responses],

        # Aggregation
        "vote_counts": dict(Counter(r.label for r in responses)),
        "jury_label": jury,
        "jury_abstained": abstained,
        "unknown_count": unknown_count,
        "routed_to_human": abstained,

        # Evaluation
        "gold_label": gold,
        "per_judge_correct": {r.judge_name: (r.label == gold) for r in responses},
        "jury_correct": (False if abstained else jury == gold),
    }


def compute_label_distributions(rows: list[dict]) -> dict:
    """Across all rows, compute each judge's marginal distribution over the
    SSSOM label space (including UNKNOWN). Flag any judge whose marginal
    covers fewer than MIN_DISTINCT_LABELS labels — the Llama-3-style
    label-collapse failure mode."""
    by_judge: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        for j in r["judges"]:
            by_judge[j["judge_name"]][j["label"]] += 1

    distributions = {}
    flagged: list[dict] = []
    for judge_name, counts in by_judge.items():
        total = sum(counts.values())
        distribution = {label: counts[label] / total for label in counts}
        labels_used = [l for l in counts if counts[l] > 0 and l != UNKNOWN]
        distributions[judge_name] = {
            "counts":       dict(counts),
            "distribution": distribution,
            "n_distinct":   len(labels_used),
            "total_rows":   total,
        }
        if len(labels_used) < MIN_DISTINCT_LABELS:
            flagged.append({
                "judge":        judge_name,
                "distinct":     len(labels_used),
                "minimum":      MIN_DISTINCT_LABELS,
                "labels_used":  sorted(labels_used),
                "reason":       "label-space collapse — judge picks too narrow a subset",
            })

    return {
        "per_judge":      distributions,
        "judges_flagged": flagged,
        "rule":           f"flag if a judge uses fewer than {MIN_DISTINCT_LABELS} of {len(SSSOM_LABELS)} SSSOM labels",
    }


def compute_confidence_summary(rows: list[dict]) -> dict:
    """Per-judge confidence summary: mean, min, max, count with logprobs.
    Confidence-by-correctness lets the analysis layer test the Ahmed et al.
    selection-by-confidence rule on this run's data."""
    by_judge: dict[str, list[tuple[Optional[float], bool]]] = defaultdict(list)
    for r in rows:
        for j in r["judges"]:
            correct = r["per_judge_correct"][j["judge_name"]]
            by_judge[j["judge_name"]].append((j["confidence"], correct))

    summary = {}
    for judge_name, pairs in by_judge.items():
        confidences = [c for c, _ in pairs if c is not None]
        if not confidences:
            summary[judge_name] = {
                "n_with_confidence":    0,
                "n_without_confidence": len(pairs),
                "note":                 "provider did not return log-probabilities",
            }
            continue
        correct_conf = [c for c, ok in pairs if ok and c is not None]
        wrong_conf   = [c for c, ok in pairs if not ok and c is not None]
        summary[judge_name] = {
            "n_with_confidence":    len(confidences),
            "n_without_confidence": len(pairs) - len(confidences),
            "mean_confidence":      sum(confidences) / len(confidences),
            "min_confidence":       min(confidences),
            "max_confidence":       max(confidences),
            "mean_when_correct":    (sum(correct_conf) / len(correct_conf)) if correct_conf else None,
            "mean_when_wrong":      (sum(wrong_conf)   / len(wrong_conf))   if wrong_conf   else None,
        }
    return summary


def run(demo: bool) -> dict:
    """Run the four-judge protocol over all rows, then compute the
    multi-row sanity checks (label distribution + confidence summary)."""
    per_row = [run_row(row, demo) for row in ROWS]

    label_distributions = compute_label_distributions(per_row)
    confidence_summary  = compute_confidence_summary(per_row)

    n_committed = sum(1 for r in per_row if not r["jury_abstained"])
    n_correct   = sum(1 for r in per_row if r["jury_correct"])

    return {
        "mode": "demo" if demo else "live",

        # Reproducibility
        "model_versions": MODEL_VERSIONS,
        "gen_params": GEN_PARAMS,

        # Per-row records (downstream metrics consume this)
        "rows": per_row,

        # Multi-row sanity checks
        "run_summary": {
            "n_rows":              len(per_row),
            "n_committed":         n_committed,
            "n_abstained":         len(per_row) - n_committed,
            "n_jury_correct":      n_correct,
            "coverage":            n_committed / len(per_row) if per_row else 0.0,
            "conditional_accuracy": (n_correct / n_committed) if n_committed else None,
            "label_distributions": label_distributions,
            "confidence_summary":  confidence_summary,
        },

        # Caveats propagate to whoever reads this file
        "caveats": {
            "single_curator": True,
            "weighted_kappa_required": True,
            "directional_narrow_broad": True,
            "demo_data": demo,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. PRETTY PRINT
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_match(label: str, gold: str) -> str:
    if label == UNKNOWN:
        return "n/a"
    return "yes" if label == gold else "no"


def _fmt_conf(c: Optional[float]) -> str:
    return f"{c:.2f}" if c is not None else "  - "


def print_report(result: dict) -> None:
    print("\n" + "═" * 78)
    print("  AI-READI SSSOM BENCHMARK — pilot demonstration  (v3)")
    print(f"  Mode: {result['mode'].upper()}    Temperature: {result['gen_params']['temperature']}")
    print("═" * 78)

    # ── Per-row results ─────────────────────────────────────────────────────
    for r in result["rows"]:
        row = r["row"]
        print(f"\n  ROW {row['row_id']}  ·  {row['source_term']!r} → {row['target_label']!r}")
        print(f"      gold: {r['gold_label']}    (hidden from judges)")

        print(f"      {'Judge':<10s}  {'Label':<10s}  {'Conf':>5s}   match")
        print(f"      {'-' * 10}  {'-' * 10}  {'-' * 5}   -----")
        for j in r["judges"]:
            print(f"      {j['judge_name']:<10s}  {j['label']:<10s}  "
                  f"{_fmt_conf(j['confidence']):>5s}   {_fmt_match(j['label'], r['gold_label'])}")

        vc = ", ".join(f"{k}={v}" for k, v in r["vote_counts"].items())
        if r["jury_abstained"]:
            print(f"      → jury: ABSTAIN  ({vc}) — routed to human review")
        else:
            verdict = "✓" if r["jury_correct"] else "✗"
            print(f"      → jury: {r['jury_label']}  ({vc})  vs gold: {verdict}")

    # ── Run-level summary ───────────────────────────────────────────────────
    s = result["run_summary"]
    print("\n" + "─" * 78)
    print("  RUN SUMMARY")
    print(f"    Rows:               {s['n_rows']}")
    print(f"    Coverage:           {s['n_committed']}/{s['n_rows']}  "
          f"({s['coverage']:.0%}) — fraction of rows where the jury committed")
    if s['conditional_accuracy'] is not None:
        print(f"    Conditional acc:    {s['n_jury_correct']}/{s['n_committed']}  "
              f"({s['conditional_accuracy']:.0%}) — accuracy on committed rows only")
    print(f"    Abstained:          {s['n_abstained']} (routed to human curator)")

    # ── Label-distribution check (Ahmed et al. inspired) ─────────────────────
    print("\n  JUDGE LABEL-DISTRIBUTION CHECK")
    print(f"    ({s['label_distributions']['rule']})")
    print(f"    {'Judge':<10s}  distribution over labels                    n_distinct")
    print(f"    {'-' * 10}  -----------------------------------------    ----------")
    for judge_name, info in s["label_distributions"]["per_judge"].items():
        dist_str = ", ".join(f"{k}={v}" for k, v in info["counts"].items())
        print(f"    {judge_name:<10s}  {dist_str:<43s}  {info['n_distinct']}")
    flagged = s["label_distributions"]["judges_flagged"]
    if flagged:
        print(f"\n    ⚠  Flagged judges (potential label-space collapse):")
        for f in flagged:
            print(f"       - {f['judge']}: used {f['distinct']} of {len(SSSOM_LABELS)} labels "
                  f"({', '.join(f['labels_used'])})")
    else:
        print(f"\n    ✓ No judges flagged for label-space collapse.")

    # ── Confidence summary ──────────────────────────────────────────────────
    print("\n  JUDGE CONFIDENCE SUMMARY")
    print(f"    {'Judge':<10s}  {'mean':>5s}  {'when correct':>13s}  {'when wrong':>11s}  notes")
    print(f"    {'-' * 10}  {'-' * 5}  {'-' * 13}  {'-' * 11}  -----")
    for judge_name, info in s["confidence_summary"].items():
        if info.get("n_with_confidence", 0) == 0:
            print(f"    {judge_name:<10s}  {'-':>5s}  {'-':>13s}  {'-':>11s}  "
                  f"{info.get('note', 'no log-probabilities available')}")
            continue
        wc = f"{info['mean_when_correct']:.2f}" if info["mean_when_correct"] is not None else "  -  "
        ww = f"{info['mean_when_wrong']:.2f}"   if info["mean_when_wrong"]   is not None else "  -  "
        print(f"    {judge_name:<10s}  {info['mean_confidence']:>5.2f}  "
              f"{wc:>13s}  {ww:>11s}")

    # ── Caveats ─────────────────────────────────────────────────────────────
    print("\n  CAVEATS (propagated to JSON output)")
    print("    - Single curator: 'gold' labels are one expert's opinion. Second-curator")
    print("      regrade of 50 stratified rows pending. Inter-curator κ will be the")
    print("      human ceiling for all jury-vs-expert metrics.")
    print("    - This script captures per-row data + multi-row sanity checks only.")
    print("      Downstream analysis computes: weighted κ with directional SSSOM cost")
    print("      matrix, Fleiss κ / Krippendorff α (inter-judge), McNemar (jury vs.")
    print("      best single judge), bootstrap 95% CIs, all stratified by SSSOM tier.")
    print("    - Ahmed et al. (MSR'25): adopted label-distribution flag rule and")
    print("      per-judge confidence logging for confidence-based sample selection.")

    print("\n" + "═" * 78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. SELF-TESTS
#    Runs with --test. Locks in the v2 fixes so they cannot regress silently.
# ─────────────────────────────────────────────────────────────────────────────

def _selftest() -> int:
    failures = []

    def expect(name, got, want):
        if got != want:
            failures.append(f"  ✗ {name}: got {got!r}, want {want!r}")
        else:
            print(f"  ✓ {name}")

    print("\n  normalize() — word-boundary matching")
    expect("plain 'exact'",                 normalize("exact"),                       "exact")
    expect("'Exact.' with punctuation",     normalize("Exact."),                      "exact")
    expect("'unrelated' is NOT 'related'",  normalize("unrelated"),                   UNKNOWN)
    expect("'closely related' is 'related'",normalize("closely related"),             "related")
    expect("'a close match'",               normalize("a close match"),               "close")
    expect("empty string → UNKNOWN",        normalize(""),                            UNKNOWN)
    expect("ambiguous 'narrow or broad'",   normalize("narrow or broad"),             UNKNOWN)
    expect("garbage → UNKNOWN",             normalize("I think it is hard to say"),   UNKNOWN)
    expect("None → UNKNOWN",                normalize(None),                          UNKNOWN)

    print("\n  majority_vote() — tie = ABSTAIN")
    def jr(name, label): return JudgeResponse(name, "test-model", label, label)
    expect("3-1 majority",
           majority_vote([jr("A","exact"), jr("B","exact"), jr("C","exact"), jr("D","close")]),
           "exact")
    expect("2-2 tie → ABSTAIN",
           majority_vote([jr("A","exact"), jr("B","exact"), jr("C","close"), jr("D","close")]),
           ABSTAIN)
    expect("1-1-1-1 tie → ABSTAIN",
           majority_vote([jr("A","exact"), jr("B","close"), jr("C","narrow"), jr("D","broad")]),
           ABSTAIN)
    expect("UNKNOWN excluded from count",
           majority_vote([jr("A","exact"), jr("B","exact"), jr("C",UNKNOWN), jr("D","close")]),
           "exact")
    expect("all UNKNOWN → ABSTAIN",
           majority_vote([jr("A",UNKNOWN), jr("B",UNKNOWN), jr("C",UNKNOWN), jr("D",UNKNOWN)]),
           ABSTAIN)

    print("\n  _to_confidence() — log-prob → probability")
    expect("None → None",                _to_confidence(None),       None)
    expect("logprob 0.0 → 1.0",          round(_to_confidence(0.0), 4), 1.0)
    expect("logprob -1.0 → ~0.3679",     round(_to_confidence(-1.0), 4), 0.3679)
    expect("logprob -0.05 → ~0.9512",    round(_to_confidence(-0.05), 4), 0.9512)

    print("\n  compute_label_distributions() — label-collapse detection")
    # Two rows, judge "Healthy" uses 3 distinct labels, judge "Collapsed" only 1.
    fake_rows = [
        {"judges": [
            {"judge_name": "Healthy",   "label": "exact"},
            {"judge_name": "Collapsed", "label": "exact"},
        ]},
        {"judges": [
            {"judge_name": "Healthy",   "label": "close"},
            {"judge_name": "Collapsed", "label": "exact"},
        ]},
        {"judges": [
            {"judge_name": "Healthy",   "label": "narrow"},
            {"judge_name": "Collapsed", "label": "exact"},
        ]},
    ]
    dist = compute_label_distributions(fake_rows)
    expect("Healthy judge uses 3 distinct labels",
           dist["per_judge"]["Healthy"]["n_distinct"], 3)
    expect("Collapsed judge uses 1 distinct label",
           dist["per_judge"]["Collapsed"]["n_distinct"], 1)
    flagged_names = {f["judge"] for f in dist["judges_flagged"]}
    expect("Collapsed judge is flagged",  "Collapsed" in flagged_names, True)
    expect("Healthy judge is NOT flagged", "Healthy"   in flagged_names, False)

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f)
        return 1
    print(f"All self-tests passed.\n")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true", help="Use real API calls (requires API keys)")
    group.add_argument("--demo", action="store_true", help="Use recorded responses (default)")
    group.add_argument("--test", action="store_true", help="Run normalize() and majority_vote() self-tests")
    args = parser.parse_args()

    if args.test:
        sys.exit(_selftest())

    demo = not args.live
    try:
        result = run(demo=demo)
    except KeyError as e:
        print(f"\nERROR: missing environment variable {e}. Set it or use --demo.\n", file=sys.stderr)
        sys.exit(1)

    print_report(result)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motivating_example_output.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Full JSON saved to: {out_path}\n")
