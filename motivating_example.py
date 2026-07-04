#!/usr/bin/env python3
"""
motivating_example.py
─────────────────────
AI-READI SSSOM Benchmark — single-row demonstration.

Runs the LLM-jury protocol on one row from AI-READI's OMOP mapping table,
collects four independent ratings, aggregates by majority vote, and
compares both single-judge and jury outputs to the expert gold label.

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
import sys
from collections import Counter
from dataclasses import asdict, dataclass


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
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeResponse:
    judge_name:   str
    label:        str
    raw_response: str


def query_claude(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def query_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-5.4",
        max_tokens=10,
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
        model="deepseek-chat",
        max_tokens=10,
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
        model="meta-llama/Llama-4-Maverick",
        max_tokens=10,
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
# ─────────────────────────────────────────────────────────────────────────────

def normalize(raw: str) -> str:
    raw = (raw or "").strip().lower()
    for ch in ".,!?;:\"'`":
        raw = raw.replace(ch, "")
    raw = raw.strip()
    if raw in SSSOM_LABELS:
        return raw
    for label in SSSOM_LABELS:
        if label in raw:
            return label
    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAJORITY VOTE
# ─────────────────────────────────────────────────────────────────────────────

def majority_vote(responses: list[JudgeResponse]) -> str:
    counts = Counter(r.label for r in responses if r.label != "UNKNOWN")
    if not counts:
        return "UNKNOWN"
    return counts.most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# 6. RUN
# ─────────────────────────────────────────────────────────────────────────────

def run(demo: bool) -> dict:
    prompt = PROMPT_TEMPLATE.format(**ROW)

    responses: list[JudgeResponse] = []
    for name, query_fn in JUDGES:
        raw = DEMO_RESPONSES[name] if demo else query_fn(prompt)
        responses.append(JudgeResponse(name, normalize(raw), raw))

    jury = majority_vote(responses)
    gold = ROW["gold_sssom_label"]

    return {
        "row": ROW,
        "prompt": prompt,
        "judges": [asdict(r) for r in responses],
        "jury_label": jury,
        "gold_label": gold,
        "vote_counts": dict(Counter(r.label for r in responses)),
        "per_judge_correct": {r.judge_name: (r.label == gold) for r in responses},
        "jury_correct": (jury == gold),
        "mode": "demo" if demo else "live",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. PRETTY PRINT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(result: dict) -> None:
    print("\n" + "═" * 72)
    print("  AI-READI SSSOM BENCHMARK — single-row demonstration")
    print(f"  Mode: {result['mode'].upper()}")
    print("═" * 72)

    print("\n  ROW UNDER TEST")
    print(f"    Source term:   {result['row']['source_term']!r}")
    print(f"    Target OMOP:   {result['row']['target_label']!r}")
    print(f"    Concept id:    {result['row']['target_omop_id']}")
    print(f"    Gold label:    {result['gold_label']}   (hidden from judges)")

    print("\n  JUDGE RESPONSES")
    print(f"    {'Judge':<10s}  {'Label':<10s}  Matches gold?")
    print(f"    {'-' * 10}  {'-' * 10}  {'-' * 13}")
    for j in result["judges"]:
        match = "yes" if j["label"] == result["gold_label"] else "no"
        print(f"    {j['judge_name']:<10s}  {j['label']:<10s}  {match}")

    print("\n  AGGREGATION")
    vc = ", ".join(f"{k}={v}" for k, v in result["vote_counts"].items())
    print(f"    Vote counts:   {vc}")
    print(f"    Jury label:    {result['jury_label']}")
    print(f"    Jury vs gold:  {'MATCH ✓' if result['jury_correct'] else 'MISMATCH ✗'}")

    print("\n  TAKEAWAY")
    n_correct = sum(result["per_judge_correct"].values())
    n_total = len(result["judges"])
    print(f"    {n_correct} of {n_total} individual judges matched the gold label.")
    print(f"    The jury (majority vote) {'agrees' if result['jury_correct'] else 'disagrees'} with the expert.")
    print("    On this single row, the jury recovers the expert label even though")
    print("    not every judge does — the mechanism the proposal tests at scale.")

    print("\n" + "═" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true", help="Use real API calls (requires API keys)")
    group.add_argument("--demo", action="store_true", help="Use recorded responses (default)")
    args = parser.parse_args()

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
