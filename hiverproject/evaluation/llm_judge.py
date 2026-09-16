"""
LLM-as-Judge: scores a sample of drafted replies on a 4-criteria rubric using Groq.
Judges on: Relevance, Brand tone fit, Groundedness (no fabricated claims), Actionability.

Run on a SAMPLE (default 20) to conserve Groq quota - judging all 86 isn't necessary
to demonstrate the harness works, and the assignment explicitly allows subsampling.

SETUP:
    pip install groq pandas

Set your API key:
    Windows PowerShell:
        $env:GROQ_API_KEY="your_key_here"

Then run:
    python llm_judge.py
"""

import os
import time
import re
import pandas as pd
from groq import Groq

PIPELINE_OUTPUT_FILE = "final_pipeline_output.csv"
OUTPUT_FILE = "llm_judge_scores_v2.csv"
SAMPLE_SIZE = 20
RANDOM_STATE = 42
GROQ_MODEL = "openai/gpt-oss-20b"
DELAY_BETWEEN_CALLS = 2.0

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

JUDGE_PROMPT_TEMPLATE = """You are evaluating a customer support reply drafted for SpotifyCares
(Spotify's Twitter support account). Rate the DRAFTED REPLY on 4 criteria, each 1-5
(1=poor, 5=excellent):

1. RELEVANCE: Does it directly address the customer's actual issue?
2. TONE_FIT: Does it match a casual, friendly, professional support-tweet tone?
3. GROUNDEDNESS: Does the reply invent SPECIFIC facts, numbers, policies, programs, or
   claims about what the company is currently doing, that are not something a real
   support agent could safely say without checking?
   - LOW score (1-2) examples of fabrication: naming a specific eligibility rule
     ("this promo is only for new subscribers"), describing an internal distribution
     process ("codes are sent to a select group of top fans"), claiming a specific team
     is actively working on something right now ("our tech team is on it"), stating a
     concrete cause with confidence when it wasn't verified.
   - HIGH score (4-5) examples of acceptable, safe generic language: "we don't have a
     release date to share right now", "sometimes tracks are removed for licensing
     reasons", asking the customer for more info, generic reassurance without inventing
     specifics ("we'll keep you posted", "hopefully it'll be back soon").
   - The test: could this exact sentence get someone in trouble for being wrong? If yes,
     it's a low score even if it "sounds" helpful and confident.
4. ACTIONABILITY: Does it give the customer a clear next step (or clearly redirect to DM
   when account info is needed)?

CUSTOMER MESSAGE: "{customer_message}"

DRAFTED REPLY: "{drafted_reply}"

Respond in EXACTLY this format, nothing else:
RELEVANCE: <1-5>
TONE_FIT: <1-5>
GROUNDEDNESS: <1-5>
ACTIONABILITY: <1-5>
NOTES: <one short sentence explaining the lowest score>"""


def parse_scores(raw_text):
    scores = {}
    for key in ["RELEVANCE", "TONE_FIT", "GROUNDEDNESS", "ACTIONABILITY"]:
        match = re.search(rf"{key}:\s*(\d)", raw_text)
        scores[key.lower()] = int(match.group(1)) if match else None
    notes_match = re.search(r"NOTES:\s*(.+)", raw_text)
    scores["notes"] = notes_match.group(1).strip() if notes_match else ""
    return scores


def judge_reply(customer_message, drafted_reply):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        customer_message=customer_message, drafted_reply=drafted_reply
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
        reasoning_effort="low",
    )
    raw = (response.choices[0].message.content or "").strip()
    return parse_scores(raw), raw


def main():
    pipeline_output = pd.read_csv(PIPELINE_OUTPUT_FILE)
    auto = pipeline_output[pipeline_output["decision"] == "AUTO_HANDLE"].copy()

    sample = auto.sample(min(SAMPLE_SIZE, len(auto)), random_state=RANDOM_STATE)
    print(f"Judging {len(sample)} drafted replies...\n")

    results = []
    for i, row in sample.iterrows():
        try:
            scores, raw = judge_reply(row["text"], row["drafted_reply"])
        except Exception as e:
            err_text = str(e)
            print(f"Error: {err_text}")
            if any(kw in err_text.lower() for kw in ["quota", "rate limit", "429"]):
                print("*** Quota/rate limit reached. Stopping safely. ***")
                break
            scores = {"relevance": None, "tone_fit": None, "groundedness": None,
                      "actionability": None, "notes": "ERROR"}

        result = {
            "tweet_id": row["tweet_id"],
            "customer_message": row["text"],
            "drafted_reply": row["drafted_reply"],
            **scores,
        }
        results.append(result)
        print(f"[{row['tweet_id']}] R={scores['relevance']} T={scores['tone_fit']} "
              f"G={scores['groundedness']} A={scores['actionability']} - {scores['notes']}")
        time.sleep(DELAY_BETWEEN_CALLS)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(results_df)} judged replies to {OUTPUT_FILE}\n")
    for col in ["relevance", "tone_fit", "groundedness", "actionability"]:
        valid = results_df[col].dropna()
        if len(valid) > 0:
            print(f"Mean {col}: {valid.mean():.2f} / 5")


if __name__ == "__main__":
    main()