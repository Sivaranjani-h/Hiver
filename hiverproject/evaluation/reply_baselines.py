"""
Reply-quality baselines for comparison against your LLM-grounded reply drafter.

TRIVIAL: the exact same canned reply sent to every message, regardless of content.
SIMPLE: copy the single most similar retrieved historical reply verbatim (retrieval
        only, no LLM rewriting).

Both are scored the same way as your actual system in Step 8 (semantic_reply_eval.py):
semantic similarity to the REAL historical reply that was actually sent.

SETUP:
    pip install sentence-transformers pandas numpy scikit-learn

Then run:
    python reply_baselines.py
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CUSTOMER_FILE = "spotify_customer_tweets.csv"
REPLY_FILE = "spotify_support_replies.csv"
PIPELINE_OUTPUT_FILE = "final_pipeline_output.csv"
KB_FILE = "knowledge_base_sample.csv"
EMBEDDINGS_FILE = "kb_embeddings.npy"
OUTPUT_FILE = "reply_baselines_eval.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
CANNED_REPLY = "Thanks for reaching out! Please DM us your account email so we can help. /RH"


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
    kb = pd.read_csv(KB_FILE)
    kb_embeddings = np.load(EMBEDDINGS_FILE)

    customer_lookup = customer.set_index("tweet_id")
    reply_lookup = replies.set_index("tweet_id")["text"].to_dict()

    auto = pipeline_output[pipeline_output["decision"] == "AUTO_HANDLE"].copy()
    print(f"Evaluating baselines on {len(auto)} auto-handled messages...")

    auto["actual_historical_reply"] = auto["tweet_id"].apply(
        lambda t: find_actual_reply(t, customer_lookup, reply_lookup)
    )
    valid = auto.dropna(subset=["actual_historical_reply"]).copy()
    print(f"Matched real historical replies for {len(valid)} messages")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # --- TRIVIAL BASELINE: same canned reply for everyone ---
    print("\nComputing TRIVIAL baseline (canned reply)...")
    canned_emb = model.encode([CANNED_REPLY])
    actual_embs = model.encode(valid["actual_historical_reply"].tolist())
    trivial_sims = cosine_similarity(canned_emb, actual_embs)[0]
    valid["trivial_baseline_similarity"] = trivial_sims

    # --- SIMPLE BASELINE: retrieval only, copy top-1 match verbatim ---
    print("Computing SIMPLE baseline (retrieval only, no LLM)...")
    query_embs = model.encode(valid["text"].tolist())
    sims_matrix = cosine_similarity(query_embs, kb_embeddings)
    top1_indices = sims_matrix.argmax(axis=1)
    retrieval_only_replies = kb.iloc[top1_indices]["reply_text"].tolist()

    retrieval_only_embs = model.encode(retrieval_only_replies)
    simple_sims = [
        cosine_similarity([retrieval_only_embs[i]], [actual_embs[i]])[0][0]
        for i in range(len(valid))
    ]
    valid["simple_baseline_similarity"] = simple_sims
    valid["simple_baseline_reply"] = retrieval_only_replies

    # --- YOUR SYSTEM (already computed in Step 8, recomputed here for a clean 3-way comparison) ---
    drafted_embs = model.encode(valid["drafted_reply"].tolist())
    system_sims = [
        cosine_similarity([drafted_embs[i]], [actual_embs[i]])[0][0]
        for i in range(len(valid))
    ]
    valid["your_system_similarity"] = system_sims

    valid.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 50)
    print("REPLY QUALITY: 3-WAY COMPARISON (semantic similarity to real reply)")
    print("=" * 50)
    print(f"TRIVIAL (canned reply):        {np.mean(trivial_sims):.3f}")
    print(f"SIMPLE (retrieval only):       {np.mean(simple_sims):.3f}")
    print(f"YOUR SYSTEM (retrieval+LLM):   {np.mean(system_sims):.3f}")
    print(f"\nSaved full results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
