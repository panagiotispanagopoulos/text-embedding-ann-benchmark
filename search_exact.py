import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text

print("Loading embeddings...")

embeddings = np.load("embeddings.npy")
texts = np.load("texts.npy", allow_pickle=True)

print(f"Loaded {len(texts)} texts")

model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, top_k=5):
    query = clean_text(query)

    start = time.time()

    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    end = time.time()

    results = [(texts[i], similarities[i]) for i in top_indices]

    return results, end - start


while True:
    query = input("\nEnter query (or 'exit'): ")

    if query == "exit":
        break

    results, latency = search(query)

    print(f"\nLatency: {latency:.6f} sec")

    for i, (text, score) in enumerate(results, 1):
        print(f"\n{i}. {score:.4f}")
        print(text[:300])