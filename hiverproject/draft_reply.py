"""
SpotifyCares Reply Drafter: retrieval (sentence embeddings) + LLM grounding (Groq).

Given a new customer message, finds the most similar past customer messages
(using semantic embeddings) and their REAL SpotifyCares replies, then asks an
LLM to draft a new reply grounded in those real historical resolutions.

SETUP (run build_embeddings.py FIRST, once):
    pip install sentence-transformers groq pandas numpy

Set your API key:
    Windows PowerShell:
        $env:GROQ_API_KEY="your_key_here"

Then run:
    python draft_reply.py
"""

import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

KB_FILE = "knowledge_base_sample.csv"
EMBEDDINGS_FILE = "kb_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 4  # number of similar past examples to retrieve

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def load_index():
    kb = pd.read_csv(KB_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)
    model = SentenceTransformer(MODEL_NAME)
    return kb, embeddings, model


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

    prompt = f"""You are drafting a customer support reply for SpotifyCares (Spotify's official
Twitter support account). Match the tone, length, and style SpotifyCares actually uses,
based on these real past examples of similar issues:

{examples_text}
Now draft a reply to this NEW customer message, in the same style as the examples above.
Keep it concise (1-3 sentences, like a real tweet reply). If the issue needs account-specific
info, ask the customer to DM their account email, matching the pattern shown above.

New customer message: "{message}"

Drafted reply:"""
    return prompt


def draft_reply(message, kb, embeddings, model):
    retrieved = retrieve_similar(message, kb, embeddings, model)
    prompt = build_draft_prompt(message, retrieved)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
        reasoning_effort="low",
    )
    draft = (response.choices[0].message.content or "").strip()
    return draft, retrieved


def main():
    print("Loading knowledge base and embeddings...")
    kb, embeddings, model = load_index()
    print(f"Loaded {len(kb)} pairs.\n")

    # Load some test messages to try this on
    test_df = pd.read_csv("test_set.csv")
    sample = test_df.sample(10, random_state=1)

    results = []
    for i, row in sample.iterrows():
        message = row["text"]
        print(f"\n{'='*70}")
        print(f"CUSTOMER MESSAGE: {message}")
        draft, retrieved = draft_reply(message, kb, embeddings, model)
        print(f"\nTop retrieved similar past message:")
        print(f"  \"{retrieved.iloc[0]['customer_text'][:100]}\" (sim={retrieved.iloc[0]['similarity']:.2f})")
        print(f"\nDRAFTED REPLY: {draft}")

        results.append({
            "customer_message": message,
            "true_intent": row["final_label"],
            "top_retrieved_example": retrieved.iloc[0]["customer_text"],
            "top_retrieved_reply": retrieved.iloc[0]["reply_text"],
            "drafted_reply": draft,
        })
        time.sleep(2)

    pd.DataFrame(results).to_csv("drafted_replies_sample.csv", index=False)
    print(f"\n\nSaved {len(results)} drafted replies to drafted_replies_sample.csv")


if __name__ == "__main__":
    main()
