"""
Build embeddings for the SpotifyCares knowledge base (customer message -> real reply pairs).
Run this ONCE. It creates kb_embeddings.npy, which the reply drafter script will load.

SETUP:
    pip install sentence-transformers pandas numpy

Then run:
    python build_embeddings.py
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

KB_FILE = "knowledge_base_sample.csv"
EMBEDDINGS_FILE = "kb_embeddings.npy"

print("Loading knowledge base...")
kb = pd.read_csv(KB_FILE)
print(f"{len(kb)} customer<->reply pairs loaded.")

print("Loading embedding model (all-MiniLM-L6-v2, ~80MB, downloads once)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding all customer messages... (this may take a few minutes)")
embeddings = model.encode(
    kb["customer_text"].tolist(),
    show_progress_bar=True,
    batch_size=64,
)

np.save(EMBEDDINGS_FILE, embeddings)
print(f"\nSaved {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]}) to {EMBEDDINGS_FILE}")
print("Done. You can now run draft_reply.py")
