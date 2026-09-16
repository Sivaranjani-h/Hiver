"""
SpotifyCares Intent Classifier using Groq API (few-shot prompting)

SETUP:
    pip install groq pandas

Set your API key:
    Windows PowerShell:
        $env:GROQ_API_KEY="your_key_here"

Then run:
    python classify_with_groq.py
"""

import os
import time
import pandas as pd
from groq import Groq

# ---- CONFIG ----
MODEL = "openai/gpt-oss-20b"
REFERENCE_FILE = "reference_examples.csv"
TEST_FILE = "test_set.csv"
OUTPUT_FILE = "classifier_predictions.csv"
DELAY_BETWEEN_CALLS = 2.0

# Any prediction matching these is treated as a FAILURE, not a real result,
# so those rows will be retried on resume instead of being skipped.
INVALID_PREDICTIONS = {"EMPTY_RESPONSE", "ERROR", "", "nan", "NaN"}

INTENTS = [
    "Account & Login",
    "Premium & Subscription",
    "Billing & Payment",
    "Playback & Audio",
    "App & Technical Issues",
    "Playlist & Library",
    "Content Availability",
    "Ads / Free Tier",
    "Feature Request / Feedback",
    "How-to / General Question",
    "Noise / Unclassifiable",
]

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def build_prompt(reference_df, message, max_examples_per_intent=1):
    trimmed = (
        reference_df.groupby("final_label", group_keys=False)
        .apply(lambda g: g.head(max_examples_per_intent))
    )
    examples_text = ""
    for _, row in trimmed.iterrows():
        examples_text += f'Message: "{row["text"]}"\nIntent: {row["final_label"]}\n\n'

    intents_list = "\n".join(f"- {i}" for i in INTENTS)

    prompt = f"""You are an intent classifier for SpotifyCares customer support tweets.

Classify the customer message into EXACTLY ONE of these intents:

{intents_list}

IMPORTANT:
Use "Noise / Unclassifiable" for messages that are NOT a clear, standalone support request.

This includes:
- thank-you/praise messages
- mid-conversation fragments
- device information without a clear problem
- "same issue as here"
- vague statements
- off-topic chatter
- messages too short or vague to identify a real issue

Only choose a specific intent if the message clearly states an actual problem,
question, or request.

Examples:

{examples_text}

Respond with ONLY the intent label.

Message: "{message}"

Intent:"""
    return prompt


def clean_prediction(raw):
    text = raw.strip()
    for prefix in ["Intent:", "intent:", "Answer:", "Label:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.strip('"').strip("'").strip(".").strip()
    for intent in INTENTS:
        if intent.lower() in text.lower():
            return intent
    return text


def classify_message(reference_df, message, retries=2):
    prompt = build_prompt(reference_df, message)
    for attempt in range(retries + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,       # widened - gpt-oss-20b needs headroom for reasoning tokens
            reasoning_effort="low",  # CRITICAL - prevents silent empty responses
        )
        raw = response.choices[0].message.content or ""
        if raw.strip():
            return clean_prediction(raw)
        time.sleep(1)
    return "EMPTY_RESPONSE"


def main():
    reference_df = pd.read_csv(REFERENCE_FILE)
    test_df = pd.read_csv(TEST_FILE)

    if os.path.exists(OUTPUT_FILE):
        results_df = pd.read_csv(OUTPUT_FILE)
        print(f"Found existing results file with {len(results_df)} rows.")

        # Split into genuinely valid results vs failed ones (EMPTY_RESPONSE/ERROR/blank).
        # Only valid ones count as "already completed" - failed ones get retried.
        results_df["predicted_label"] = results_df["predicted_label"].astype(str)
        is_valid = ~results_df["predicted_label"].isin(INVALID_PREDICTIONS)
        valid_results = results_df[is_valid].copy()
        n_invalid = (~is_valid).sum()

        if n_invalid > 0:
            print(f"  -> {n_invalid} of those were failed runs (EMPTY_RESPONSE/ERROR) - will be retried.")
        print(f"  -> {len(valid_results)} were genuinely valid - will be skipped.")

        results_df = valid_results
    else:
        results_df = pd.DataFrame(
            columns=["tweet_id", "text", "true_label", "predicted_label", "correct"]
        )
        print("No previous results found. Starting from row 1.")

    completed_ids = set(results_df["tweet_id"].astype(str))
    total = len(test_df)

    print(f"Total test messages: {total}")
    print(f"Already validly completed: {len(completed_ids)}")
    print(f"Remaining to process: {total - len(completed_ids)}\n")

    for i, row in test_df.iterrows():
        tweet_id = str(row["tweet_id"])

        if tweet_id in completed_ids:
            print(f"[{i+1}/{total}] SKIP - already completed")
            continue

        message = row["text"]
        true_label = row["final_label"]

        try:
            predicted = classify_message(reference_df, message)
        except Exception as e:
            err_text = str(e)
            print(f"\nError on row {i}: {err_text}")
            if any(kw in err_text.lower() for kw in ["quota", "rate limit", "429", "tokens per day"]):
                print("\n*** Groq quota/rate limit reached. Stopping safely. ***")
                print(f"*** Saved {len(results_df)} completed rows. ***")
                results_df.to_csv(OUTPUT_FILE, index=False)
                return
            predicted = "ERROR"

        correct = (str(predicted).strip() == str(true_label).strip())

        new_result = {
            "tweet_id": row["tweet_id"],
            "text": message,
            "true_label": true_label,
            "predicted_label": predicted,
            "correct": correct,
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_result])], ignore_index=True)
        results_df.to_csv(OUTPUT_FILE, index=False)  # save after every row

        print(f"[{i+1}/{total}] true={true_label!r} pred={predicted!r} {'OK' if correct else 'MISS'}")
        time.sleep(DELAY_BETWEEN_CALLS)

    print("\n===================================")
    print("ALL TEST MESSAGES COMPLETED")
    print("===================================\n")

    accuracy = results_df["correct"].mean()
    print(f"Overall accuracy: {accuracy:.2%} ({results_df['correct'].sum()}/{len(results_df)})")

    print("\nPer-intent accuracy:")
    per_intent = results_df.groupby("true_label")["correct"].agg(["mean", "count"])
    per_intent.columns = ["accuracy", "n_examples"]
    print(per_intent.sort_values("accuracy"))

    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()