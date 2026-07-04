#!/usr/bin/env python3
"""
motivating_example.py  (v2 — review feedback incorporated)
──────────────────────────────────────────────────────────
AI-READI SSSOM Benchmark — single-row demonstration.

Runs the LLM-jury protocol on one row from AI-READI's OMOP mapping table,
collects four independent ratings, aggregates by majority vote (with tie =
abstain), and compares both single-judge and jury outputs to the expert
gold label.

Changes from v1:
  - Aggregation: ties now route to human review (abstain), not arbitrary pick.
  - Normalisation: word-boundary matching; "unrelated" no longer reads as
    "related", "closely related" no longer reads as "close".
  - Reproducibility: temperature pinned to 0, model versions pinned, and
    logged into the output record.
  - UNKNOWN policy: explicitly logged, excluded from majority vote, contributes
    to abstention rate as a quality-control signal.

Out of scope for this script (handled in the analysis layer downstream):
  - Weighted Cohen's κ with SSSOM cost matrix (judge ↔ expert)
  - Fleiss' κ and Krippendorff's α for inter-judge agreement
  - McNemar test for paired jury-vs-best-judge comparison
  - Bootstrap confidence intervals on all headline metrics
  - Second-curator regrade for the human ceiling

Usage
-----
    python motivating_example.py --demo     # No API keys; uses recorded responses
    python motivating_example.py --live     # Real API calls; requires API keys

Required environment variables for --live:
    ANTHROPIC_API_KEY        (Claude)
    OPENAI_API_KEY           (GPT)
    DEEPSEEK_API_KEY         (DeepSeek)
    TOGETHER_API_KEY         (Llama via Together AI)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. THE ROW UNDER TEST
#    (real OMOP concept_id 3004410 = LOINC 4548-4)
# ─────────────────────────────────────────────────────────────────────────────

ROW = {
    "source_term":      "HbA1c",
    "target_label":     "Hemoglobin A1c/Hemoglobin.total in Blood",
    "target_omop_id":   3004410,
    "gold_sssom_label": "exact",   # expert curator label (hidden from judges)
}


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
    judge_name:   str
    model:        str
    label:        str
    raw_response: str


def query_claude(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=MODEL_VERSIONS["Claude"],
        max_tokens=GEN_PARAMS["max_tokens"],
        temperature=GEN_PARAMS["temperature"],
        top_p=GEN_PARAMS["top_p"],
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def query_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=MODEL_VERSIONS["GPT"],
        max_tokens=GEN_PARAMS["max_tokens"],
        temperature=GEN_PARAMS["temperature"],
        top_p=GEN_PARAMS["top_p"],
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def query_deepseek(prompt: str) -> str:
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
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def query_llama(prompt: str) -> str:
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
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


JUDGES = [
    ("Claude",    query_claude),
    ("GPT",       query_openai),
    ("DeepSeek",  query_deepseek),
    ("Llama",     query_llama),
]


# Hard-coded responses for --demo mode (no API keys needed).
# These reflect plausible single-word LLM outputs for the HbA1c row.
DEMO_RESPONSES = {
    "Claude":    "exact",
    "GPT":       "exact",
    "DeepSeek":  "close",
    "Llama":     "exact",
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
# 6. RUN
# ─────────────────────────────────────────────────────────────────────────────

def run(demo: bool) -> dict:
    prompt = PROMPT_TEMPLATE.format(**ROW)

    responses: list[JudgeResponse] = []
    for name, query_fn in JUDGES:
        raw = DEMO_RESPONSES[name] if demo else query_fn(prompt)
        responses.append(
            JudgeResponse(
                judge_name=name,
                model=MODEL_VERSIONS[name],
                label=normalize(raw),
                raw_response=raw,
            )
        )

    jury = majority_vote(responses)
    gold = ROW["gold_sssom_label"]

    abstained = jury == ABSTAIN
    unknown_count = sum(1 for r in responses if r.label == UNKNOWN)

    return {
        # Identification
        "row": ROW,
        "prompt": prompt,
        "mode": "demo" if demo else "live",

        # Reproducibility
        "model_versions": MODEL_VERSIONS,
        "gen_params": GEN_PARAMS,

        # Per-judge
        "judges": [asdict(r) for r in responses],

        # Aggregation
        "vote_counts": dict(Counter(r.label for r in responses)),
        "jury_label": jury,
        "jury_abstained": abstained,
        "unknown_count": unknown_count,
        "routed_to_human": abstained,   # operational signal: send to curator

        # Evaluation against the (single) gold label
        "gold_label": gold,
        "per_judge_correct": {r.judge_name: (r.label == gold) for r in responses},
        "jury_correct": (False if abstained else jury == gold),

        # Caveats made explicit so they propagate downstream
        "caveats": {
            "single_curator": True,
            "weighted_kappa_required": True,
            "directional_narrow_broad": True,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. PRETTY PRINT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(result: dict) -> None:
    print("\n" + "═" * 72)
    print("  AI-READI SSSOM BENCHMARK — single-row demonstration  (v2)")
    print(f"  Mode: {result['mode'].upper()}    Temperature: {result['gen_params']['temperature']}")
    print("═" * 72)

    print("\n  ROW UNDER TEST")
    print(f"    Source term:   {result['row']['source_term']!r}")
    print(f"    Target OMOP:   {result['row']['target_label']!r}")
    print(f"    Concept id:    {result['row']['target_omop_id']}")
    print(f"    Gold label:    {result['gold_label']}   (hidden from judges; single curator)")

    print("\n  JUDGE RESPONSES")
    print(f"    {'Judge':<10s}  {'Model':<48s}  {'Label':<10s}  vs gold")
    print(f"    {'-' * 10}  {'-' * 48}  {'-' * 10}  {'-' * 6}")
    for j in result["judges"]:
        if j["label"] == UNKNOWN:
            match = "n/a"
        elif j["label"] == result["gold_label"]:
            match = "yes"
        else:
            match = "no"
        print(f"    {j['judge_name']:<10s}  {j['model']:<48s}  {j['label']:<10s}  {match}")

    print("\n  AGGREGATION")
    vc = ", ".join(f"{k}={v}" for k, v in result["vote_counts"].items())
    print(f"    Vote counts:        {vc}")
    print(f"    UNKNOWN responses:  {result['unknown_count']}")
    print(f"    Jury label:         {result['jury_label']}")
    if result["jury_abstained"]:
        print(f"    Jury vs gold:       ABSTAIN — routed to human review")
    else:
        verdict = "MATCH ✓" if result["jury_correct"] else "MISMATCH ✗"
        print(f"    Jury vs gold:       {verdict}")

    print("\n  TAKEAWAY")
    n_correct = sum(result["per_judge_correct"].values())
    n_total = len(result["judges"])
    print(f"    {n_correct} of {n_total} individual judges matched the (single) curator label.")
    if result["jury_abstained"]:
        print("    The jury abstained — this row would be flagged for human re-grade.")
    else:
        outcome = "agrees" if result["jury_correct"] else "disagrees"
        print(f"    The jury (majority vote) {outcome} with the curator.")

    print("\n  CAVEATS")
    print("    - Single curator: HbA1c row has at least one defensible alternative")
    print("      ('close', since the OMOP target adds the /Hemoglobin.total ratio")
    print("      specification not in the source). Second-curator regrade pending.")
    print("    - This script captures per-row data only. Downstream analysis")
    print("      computes: weighted κ (judge↔expert) with directional cost matrix,")
    print("      Fleiss κ and Krippendorff's α (inter-judge), McNemar (jury vs.")
    print("      best single judge), and bootstrap CIs on all headline numbers.")

    print("\n" + "═" * 72 + "\n")


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
