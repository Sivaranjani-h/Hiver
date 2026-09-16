"""
Automated Metric: Semantic similarity between drafted replies and the REAL historical
replies SpotifyCares actually sent for those same customer messages.

This is a much better automated proxy than raw string matching (difflib), since it
captures meaning rather than exact wording overlap.

SETUP:
    pip install sentence-transformers pandas numpy scikit-learn

Then run:
    python semantic_reply_eval.py
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CUSTOMER_FILE = "spotify_customer_tweets.csv"
REPLY_FILE = "spotify_support_replies.csv"
PIPELINE_OUTPUT_FILE = "final_pipeline_output.csv"  # from Step 7
OUTPUT_FILE = "semantic_reply_eval.csv"

MODEL_NAME = "all-MiniLM-L6-v2"


def find_actual_reply(tweet_id, customer_lookup, reply_lookup):
    if tweet_id not in customer_lookup.index:
        return None
    resp_id = customer_lookup.loc[tweet_id, "response_tweet_id"]
    if pd.isna(resp_id):
        return None
    first_id = str(resp_id).split(",")[0].strip()
    try:
        first_id = int(float(first_id))
    except ValueError:
        return None
    return reply_lookup.get(first_id)


def main():
    print("Loading data...")
    pipeline_output = pd.read_csv(PIPELINE_OUTPUT_FILE)
    customer = pd.read_csv(CUSTOMER_FILE)
    replies = pd.read_csv(REPLY_FILE)

    customer_lookup = customer.set_index("tweet_id")
    reply_lookup = replies.set_index("tweet_id")["text"].to_dict()

    # Only score AUTO_HANDLE rows - escalated ones have no drafted reply
    auto = pipeline_output[pipeline_output["decision"] == "AUTO_HANDLE"].copy()
    print(f"Scoring {len(auto)} auto-handled drafted replies...")

    auto["actual_historical_reply"] = auto["tweet_id"].apply(
        lambda t: find_actual_reply(t, customer_lookup, reply_lookup)
    )
    matched = auto["actual_historical_reply"].notna().sum()
    print(f"Matched real historical replies for {matched}/{len(auto)} messages")

    valid = auto.dropna(subset=["actual_historical_reply", "drafted_reply"]).copy()

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Computing semantic similarity...")
    drafted_emb = model.encode(valid["drafted_reply"].tolist())
    actual_emb = model.encode(valid["actual_historical_reply"].tolist())

    sims = [
        cosine_similarity([drafted_emb[i]], [actual_emb[i]])[0][0]
        for i in range(len(valid))
    ]
    valid["semantic_similarity"] = sims

    valid.to_csv(OUTPUT_FILE, index=False)

    print(f"\nMean semantic similarity: {np.mean(sims):.3f}")
    print(f"Median: {np.median(sims):.3f}")
    print(f"Min: {np.min(sims):.3f} / Max: {np.max(sims):.3f}")

    print("\nLowest-scoring pairs (worth reviewing for the report):")
    worst = valid.nsmallest(5, "semantic_similarity")[
        ["text", "drafted_reply", "actual_historical_reply", "semantic_similarity"]
    ]
    for _, row in worst.iterrows():
        print(f"\n  Customer: {row['text'][:80]}")
        print(f"  Drafted:  {row['drafted_reply'][:80]}")
        print(f"  Actual:   {row['actual_historical_reply'][:80]}")
        print(f"  Similarity: {row['semantic_similarity']:.3f}")

    print(f"\nSaved full results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
