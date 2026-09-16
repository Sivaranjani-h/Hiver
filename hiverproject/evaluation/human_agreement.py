"""
Step A: Generates a human scoring template (same rubric as the LLM judge) for YOU to
fill in by hand on the same sample the LLM judge scored.

Step B: Once you've filled it in, computes agreement between your scores and the
LLM judge's scores - this is the assignment's required "evidence of how well your
judge agrees with a human."

USAGE:
    1. Run: python human_agreement.py --generate
       -> creates human_scoring_template.csv
    2. Open it in Excel/WPS, fill in the "human_*" score columns (1-5) for each row
       by reading the customer message + drafted reply yourself.
    3. Run: python human_agreement.py --compare
       -> prints agreement statistics
"""

import sys
import pandas as pd
import numpy as np

JUDGE_SCORES_FILE = "llm_judge_scores.csv"  # override with --compare llm_judge_scores_v2.csv
TEMPLATE_FILE = "human_scoring_template.csv"


def generate_template():
    judge_df = pd.read_csv(JUDGE_SCORES_FILE)
    template = judge_df[["tweet_id", "customer_message", "drafted_reply"]].copy()
    template["human_relevance"] = ""
    template["human_tone_fit"] = ""
    template["human_groundedness"] = ""
    template["human_actionability"] = ""
    template.to_csv(TEMPLATE_FILE, index=False)
    print(f"Created {TEMPLATE_FILE} with {len(template)} rows.")
    print("\nOpen it and fill in the human_* columns (1-5) for each row using this rubric:")
    print("  RELEVANCE: Does it directly address the customer's actual issue? (1-5)")
    print("  TONE_FIT: Does it match a casual, friendly support-tweet tone? (1-5)")
    print("  GROUNDEDNESS: Does it avoid inventing facts/promises/programs not supported")
    print("                by context? (1-5, LOW if it fabricates specifics)")
    print("  ACTIONABILITY: Does it give a clear next step or DM redirect? (1-5)")
    print(f"\nThen run: python human_agreement.py --compare")


def compare_scores(judge_file=JUDGE_SCORES_FILE):
    judge_df = pd.read_csv(judge_file)
    human_df = pd.read_csv(TEMPLATE_FILE)

    merged = judge_df.merge(human_df, on="tweet_id", suffixes=("_llm", "_human"))

    criteria = [
        ("relevance", "human_relevance"),
        ("tone_fit", "human_tone_fit"),
        ("groundedness", "human_groundedness"),
        ("actionability", "human_actionability"),
    ]

    print(f"Comparing {len(merged)} scored replies (LLM judge vs. human)\n")

    for llm_col, human_col in criteria:
        pair = merged[[llm_col, human_col]].dropna()
        pair[human_col] = pd.to_numeric(pair[human_col], errors="coerce")
        pair = pair.dropna()

        if len(pair) == 0:
            print(f"{llm_col}: no valid pairs to compare (did you fill in the template?)")
            continue

        exact_match = (pair[llm_col] == pair[human_col]).mean()
        within_1 = (abs(pair[llm_col] - pair[human_col]) <= 1).mean()
        mean_abs_diff = abs(pair[llm_col] - pair[human_col]).mean()
        correlation = pair[llm_col].corr(pair[human_col])

        print(f"{llm_col.upper()}:")
        print(f"  Exact match rate: {exact_match:.1%}")
        print(f"  Within 1 point:   {within_1:.1%}")
        print(f"  Mean abs diff:    {mean_abs_diff:.2f}")
        print(f"  Correlation:      {correlation:.2f}" if not pd.isna(correlation) else "  Correlation: N/A (no variance)")
        print()

    merged.to_csv("judge_vs_human_comparison.csv", index=False)
    print("Saved full comparison to judge_vs_human_comparison.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("--generate", "--compare"):
        print("Usage: python human_agreement.py --generate   (first, to create the template)")
        print("       python human_agreement.py --compare [judge_scores_file.csv]")
        print("       (default judge file: llm_judge_scores.csv; pass llm_judge_scores_v2.csv to compare the sharpened judge)")
        sys.exit(1)

    if sys.argv[1] == "--generate":
        generate_template()
    else:
        judge_file = sys.argv[2] if len(sys.argv) > 2 else JUDGE_SCORES_FILE
        compare_scores(judge_file)