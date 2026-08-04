import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
print("Loading embeddings and texts...")

embeddings = np.load("embeddings.npy").astype("float32")
texts = np.load("texts.npy", allow_pickle=True)

print(f"Loaded {len(texts)} texts")
print(f"Embeddings shape: {embeddings.shape}")

# Κρατάμε 2 εκδόσεις:
# - exact_embeddings: για brute-force cosine search
# - faiss_embeddings: normalized για FAISS inner-product search
exact_embeddings = embeddings.copy()
faiss_embeddings = embeddings.copy()
faiss.normalize_L2(faiss_embeddings)

dim = faiss_embeddings.shape[1]
n_vectors = faiss_embeddings.shape[0]

# --------------------------------------------------
# 2. Build FAISS IVF index
# --------------------------------------------------
nlist = 50
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

print("\nTraining FAISS index...")
start_train = time.time()
index.train(faiss_embeddings)
end_train = time.time()

print(f"FAISS training time: {end_train - start_train:.6f} sec")

print("Adding embeddings to FAISS index...")
start_add = time.time()
index.add(faiss_embeddings)
end_add = time.time()

print(f"FAISS add time: {end_add - start_add:.6f} sec")

# Ρύθμιση search quality / speed trade-off
index.nprobe = 10
print(f"FAISS index total vectors: {index.ntotal}")
print(f"FAISS nprobe: {index.nprobe}")

# --------------------------------------------------
# 3. Load model
# --------------------------------------------------
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# 4. Define test queries
# --------------------------------------------------
queries = [
    "stock market crash",
    "football match result",
    "new technology product launch",
    "oil prices rise",
    "company earnings report",
    "government election debate",
    "basketball game victory",
    "new software release",
    "international conflict news",
    "scientific discovery"
]

TOP_K = 5

# --------------------------------------------------
# 5. Search functions
# --------------------------------------------------
def exact_search(query, top_k=5):
    query = clean_text(query)

    start = time.time()

    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, exact_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    end = time.time()

    return top_indices.tolist(), (end - start)


def faiss_search(query, top_k=5):
    query = clean_text(query)

    start = time.time()

    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    end = time.time()

    valid_indices = [int(i) for i in indices[0] if i != -1]
    return valid_indices, (end - start)


def recall_at_k(exact_ids, ann_ids):
    exact_set = set(exact_ids)
    ann_set = set(ann_ids)

    if len(exact_set) == 0:
        return 0.0

    return len(exact_set.intersection(ann_set)) / len(exact_set)


# --------------------------------------------------
# 6. Run benchmark
# --------------------------------------------------
exact_latencies = []
faiss_latencies = []
recalls = []

print("\n" + "=" * 80)
print("RUNNING BENCHMARK")
print("=" * 80)

for q in queries:
    exact_ids, exact_latency = exact_search(q, TOP_K)
    faiss_ids, faiss_latency = faiss_search(q, TOP_K)
    recall = recall_at_k(exact_ids, faiss_ids)

    exact_latencies.append(exact_latency)
    faiss_latencies.append(faiss_latency)
    recalls.append(recall)

    print(f"\nQuery: {q}")
    print(f"Exact latency: {exact_latency:.6f} sec")
    print(f"FAISS latency: {faiss_latency:.6f} sec")
    print(f"Recall@{TOP_K}: {recall:.4f}")

    print("Exact IDs:", exact_ids)
    print("FAISS IDs:", faiss_ids)

# --------------------------------------------------
# 7. Final summary
# --------------------------------------------------
avg_exact_latency = np.mean(exact_latencies)
avg_faiss_latency = np.mean(faiss_latencies)
avg_recall = np.mean(recalls)

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(f"Number of queries: {len(queries)}")
print(f"Average Exact Latency: {avg_exact_latency:.6f} sec")
print(f"Average FAISS Latency: {avg_faiss_latency:.6f} sec")
print(f"Average Recall@{TOP_K}: {avg_recall:.4f}")
print(f"FAISS training time: {end_train - start_train:.6f} sec")
print(f"FAISS add time: {end_add - start_add:.6f} sec")

if avg_faiss_latency < avg_exact_latency:
    speedup = avg_exact_latency / avg_faiss_latency
    print(f"FAISS speedup vs Exact: {speedup:.4f}x")
else:
    slowdown = avg_faiss_latency / avg_exact_latency
    print(f"FAISS is slower than Exact by: {slowdown:.4f}x")