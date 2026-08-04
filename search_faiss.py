import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils import clean_text

print("Loading embeddings...")

embeddings = np.load("embeddings.npy").astype("float32")
texts = np.load("texts.npy", allow_pickle=True)

print(f"Loaded {len(texts)} texts")
print(f"Embeddings shape: {embeddings.shape}")

# ---------------------------------
# 1. Normalize embeddings
# ---------------------------------
# Για cosine similarity στο FAISS χρησιμοποιούμε inner product
# πάνω σε normalized vectors.
faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]
n_vectors = embeddings.shape[0]

# ---------------------------------
# 2. Build IVF index
# ---------------------------------
nlist = 50  # αριθμός clusters
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

print("\nTraining FAISS index...")
start_train = time.time()
index.train(embeddings)
end_train = time.time()

print(f"Training time: {end_train - start_train:.4f} sec")

print("Adding embeddings to index...")
start_add = time.time()
index.add(embeddings)
end_add = time.time()

print(f"Add time: {end_add - start_add:.4f} sec")

# Πόσα clusters θα ψάχνει σε κάθε query
index.nprobe = 10

print(f"Index total vectors: {index.ntotal}")

# ---------------------------------
# 3. Load embedding model
# ---------------------------------
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def search(query, top_k=5):
    query = clean_text(query)

    start = time.time()

    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    end = time.time()

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        results.append((texts[idx], float(score)))

    latency = end - start
    return results, latency


while True:
    query = input("\nEnter query (or 'exit'): ")

    if query.lower() == "exit":
        break

    results, latency = search(query, top_k=5)

    print(f"\nFAISS ANN latency: {latency:.6f} sec")
    print("Top results:")

    for i, (text, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(text[:300])