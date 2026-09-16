"""
Escalation Decision Engine: combines predicted intent (risk category) + retrieval
similarity (grounding confidence) to decide AUTO_HANDLE vs ESCALATE for each message.

Runs entirely locally (embeddings only, no Groq calls needed for this step).

SETUP:
    pip install sentence-transformers pandas numpy

Then run:
    python escalation_decisions.py
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

KB_FILE = "knowledge_base_sample.csv"
EMBEDDINGS_FILE = "kb_embeddings.npy"
CLASSIFIER_RESULTS_FILE = "classifier_predictions.csv"
OUTPUT_FILE = "escalation_decisions.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.75  # below this, treat retrieval as too weak to trust

# Intents that ALWAYS escalate regardless of confidence - money or account access involved
ALWAYS_ESCALATE_INTENTS = {
    "Billing & Payment",
    "Account & Login",
    "Premium & Subscription",
}


def decide(predicted_intent, top_similarity):
    """Returns (decision, reason)."""
    if predicted_intent in ALWAYS_ESCALATE_INTENTS:
        return "ESCALATE", f"Risk category: '{predicted_intent}' always requires human review (money/account access involved)"

    if top_similarity < SIMILARITY_THRESHOLD:
        return "ESCALATE", f"Low retrieval confidence ({top_similarity:.2f} < {SIMILARITY_THRESHOLD}) - no strongly similar historical case found, risk of ungrounded reply"

    return "AUTO_HANDLE", f"Non-risk intent + strong historical grounding ({top_similarity:.2f} similarity)"


def main():
    print("Loading classifier predictions...")
    classifier_df = pd.read_csv(CLASSIFIER_RESULTS_FILE)
    print(f"Loaded {len(classifier_df)} classified messages.")

    print("Loading knowledge base + embeddings for retrieval...")
    kb = pd.read_csv(KB_FILE)
    kb_embeddings = np.load(EMBEDDINGS_FILE)
    model = SentenceTransformer(MODEL_NAME)

    print("Computing retrieval similarity for each test message...")
    messages = classifier_df["text"].tolist()
    query_embeddings = model.encode(messages, show_progress_bar=True, batch_size=64)

    sims_matrix = cosine_similarity(query_embeddings, kb_embeddings)
    top_similarities = sims_matrix.max(axis=1)  # best match per message

    results = []
    for i, row in classifier_df.iterrows():
        predicted_intent = row["predicted_label"]
        top_sim = float(top_similarities[i])
        decision, reason = decide(predicted_intent, top_sim)

        results.append({
            "tweet_id": row["tweet_id"],
            "text": row["text"],
            "true_label": row.get("true_label", ""),
            "predicted_intent": predicted_intent,
            "top_similarity": round(top_sim, 3),
            "decision": decision,
            "reason": reason,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(results_df)} decisions to {OUTPUT_FILE}\n")
    print("Decision distribution:")
    print(results_df["decision"].value_counts())
    print(f"\nEscalation rate: {(results_df['decision']=='ESCALATE').mean():.1%}")

    print("\nEscalation reasons breakdown:")
    escalated = results_df[results_df["decision"] == "ESCALATE"]
    risk_escalations = escalated[escalated["reason"].str.contains("Risk category")]
    confidence_escalations = escalated[escalated["reason"].str.contains("Low retrieval")]
    print(f"  Due to risk category (Billing/Account/Premium): {len(risk_escalations)}")
    print(f"  Due to low retrieval confidence: {len(confidence_escalations)}")


if __name__ == "__main__":
    main()