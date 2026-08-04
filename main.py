import time
import html
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Load dataset
# -----------------------------
print("Loading dataset...")
dataset = load_dataset("ag_news", split="train[:2000]")

texts = [item["text"] for item in dataset]

print(f"Loaded {len(texts)} texts.")
print("Sample before cleaning:")
print(texts[0])

# -----------------------------
# 2. Clean texts
# -----------------------------
def clean_text(text):
    text = html.unescape(text)   # μετατρέπει &lt; &gt; κλπ
    text = text.replace("\\n", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()

texts = [clean_text(t) for t in texts if str(t).strip()]

print("\nSample after cleaning:")
print(texts[0])

# -----------------------------
# 3. Load embedding model
# -----------------------------
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 4. Create embeddings
# -----------------------------
print("Creating embeddings...")
start_embed = time.time()

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

end_embed = time.time()

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embedding time: {end_embed - start_embed:.4f} seconds")

# Αποθήκευση embeddings και texts
np.save("embeddings.npy", embeddings)
np.save("texts.npy", np.array(texts, dtype=object))

print("Saved embeddings to embeddings.npy")
print("Saved texts to texts.npy")

# -----------------------------
# 5. Brute-force search
# -----------------------------
def search(query, top_k=5):
    query = clean_text(query)

    start_query = time.time()

    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    end_query = time.time()

    results = []
    for idx in top_indices:
        results.append({
            "text": texts[idx],
            "score": float(similarities[idx])
        })

    latency = end_query - start_query
    return results, latency

# -----------------------------
# 6. Test queries
# -----------------------------
queries = [
    "stock market crash",
    "football match result",
    "new technology product launch"
]

for query in queries:
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")

    results, latency = search(query, top_k=5)

    print(f"Query latency: {latency:.6f} seconds")
    print("Top results:")

    for i, item in enumerate(results, 1):
        print(f"\n{i}. Score: {item['score']:.4f}")
        print(item["text"][:300])  # δείχνουμε μόνο τα πρώτα 300 chars