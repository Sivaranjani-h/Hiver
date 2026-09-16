"""
Drafts replies ONLY for messages the escalation engine marked AUTO_HANDLE.
Escalated messages are left for a human agent - no reply is drafted for them,
just the reason they were flagged.

Run escalation_decisions.py FIRST.

SETUP:
    pip install sentence-transformers groq pandas numpy scikit-learn

Set your API key:
    Windows PowerShell:
        $env:GROQ_API_KEY="your_key_here"

Then run:
    python draft_auto_handled.py
"""

import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

DECISIONS_FILE = "escalation_decisions.csv"
KB_FILE = "knowledge_base_sample.csv"
EMBEDDINGS_FILE = "kb_embeddings.npy"
OUTPUT_FILE = "final_pipeline_output.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 4
DELAY_BETWEEN_CALLS = 2.0

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def retrieve_similar(message, kb, embeddings, model, top_k=TOP_K):
    query_emb = model.encode([message])
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    results = kb.iloc[top_indices].copy()
    results["similarity"] = sims[top_indices]
    return results


def build_draft_prompt(message, retrieved):
    examples_text = ""
    for _, row in retrieved.iterrows():
        examples_text += (
            f'Past customer message: "{row["customer_text"]}"\n'
            f'Actual SpotifyCares reply: "{row["reply_text"]}"\n\n'
        )
    prompt = f"""You are drafting a customer support reply for SpotifyCares. Match the tone,
length, and style SpotifyCares actually uses, based on these real past examples:

{examples_text}
Now draft a reply to this NEW customer message, in the same style. Keep it concise
(1-3 sentences, like a real tweet reply). Do not invent specific facts, promises,
programs, or operational status that are not supported by the examples above.

New customer message: "{message}"

Drafted reply:"""
    return prompt


def main():
    decisions = pd.read_csv(DECISIONS_FILE)
    kb = pd.read_csv(KB_FILE)
    kb_embeddings = np.load(EMBEDDINGS_FILE)
    model = SentenceTransformer(MODEL_NAME)

    auto_handle = decisions[decisions["decision"] == "AUTO_HANDLE"].copy()
    escalated = decisions[decisions["decision"] == "ESCALATE"].copy()

    print(f"AUTO_HANDLE messages to draft replies for: {len(auto_handle)}")
    print(f"ESCALATE messages (no reply drafted, flagged for human): {len(escalated)}\n")

    # Resume support: if a previous run already produced valid drafts, don't redo them
    invalid_drafts = {"", "ERROR", "nan", "NaN"}
    already_drafted = {}
    if os.path.exists(OUTPUT_FILE):
        prev = pd.read_csv(OUTPUT_FILE)
        prev["drafted_reply"] = prev["drafted_reply"].astype(str)
        valid_prev = prev[~prev["drafted_reply"].isin(invalid_drafts)]
        valid_prev = valid_prev[valid_prev["decision"] == "AUTO_HANDLE"]
        already_drafted = dict(zip(valid_prev["tweet_id"], valid_prev["drafted_reply"]))
        print(f"Found {len(already_drafted)} already-drafted messages from a previous run - will skip those.\n")

    auto_handle["drafted_reply"] = auto_handle["tweet_id"].map(already_drafted).fillna("")

    for idx, row in auto_handle.iterrows():
        if row["drafted_reply"].strip():
            print(f"[SKIP - already drafted] {row['text'][:50]}...")
            continue

        message = row["text"]
        try:
            retrieved = retrieve_similar(message, kb, kb_embeddings, model)
            prompt = build_draft_prompt(message, retrieved)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
                reasoning_effort="low",
            )
            draft = (response.choices[0].message.content or "").strip()
        except Exception as e:
            err_text = str(e)
            print(f"Error on row: {err_text}")
            if any(kw in err_text.lower() for kw in ["quota", "rate limit", "429", "tokens per day"]):
                print("\n*** Quota/rate limit reached. Stopping safely. ***")
                # save progress so far before stopping
                escalated["drafted_reply"] = "[NO DRAFT - ESCALATED TO HUMAN AGENT]"
                final = pd.concat([auto_handle, escalated], ignore_index=True)
                final.to_csv(OUTPUT_FILE, index=False)
                print(f"Saved progress ({(auto_handle['drafted_reply'].str.strip()!='').sum()} of {len(auto_handle)} drafted) to {OUTPUT_FILE}")
                return
            draft = "ERROR"

        auto_handle.at[idx, "drafted_reply"] = draft
        print(f"[{message[:50]}...] -> {draft[:80]}...")

        # save after every successful draft too, not just at the end
        escalated["drafted_reply"] = "[NO DRAFT - ESCALATED TO HUMAN AGENT]"
        pd.concat([auto_handle, escalated], ignore_index=True).to_csv(OUTPUT_FILE, index=False)

        time.sleep(DELAY_BETWEEN_CALLS)

    escalated["drafted_reply"] = "[NO DRAFT - ESCALATED TO HUMAN AGENT]"
    final = pd.concat([auto_handle, escalated], ignore_index=True)
    final.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved final pipeline output ({len(final)} rows) to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()