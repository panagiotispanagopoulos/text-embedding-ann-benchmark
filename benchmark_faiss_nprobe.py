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

exact_embeddings = embeddings.copy()
faiss_embeddings = embeddings.copy()
faiss.normalize_L2(faiss_embeddings)

dim = faiss_embeddings.shape[1]

# --------------------------------------------------
# 2. Build FAISS IVF index
# --------------------------------------------------
nlist = 50
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

print("\nTraining FAISS index...")
index.train(faiss_embeddings)
index.add(faiss_embeddings)

print(f"FAISS index total vectors: {index.ntotal}")

# --------------------------------------------------
# 3. Load model
# --------------------------------------------------
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# 4. Queries
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
NPROBE_VALUES = [1, 3, 5, 10, 20]

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


def faiss_search(query, nprobe, top_k=5):
    query = clean_text(query)

    index.nprobe = nprobe

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
# 6. Baseline exact once
# --------------------------------------------------
print("\nRunning Exact baseline...")

exact_results = {}
exact_latencies = []

for q in queries:
    exact_ids, exact_latency = exact_search(q, TOP_K)
    exact_results[q] = exact_ids
    exact_latencies.append(exact_latency)

avg_exact_latency = np.mean(exact_latencies)

print(f"Average Exact Latency: {avg_exact_latency:.6f} sec")

# --------------------------------------------------
# 7. Test different nprobe values
# --------------------------------------------------
print("\n" + "=" * 90)
print("FAISS NPROBE TUNING")
print("=" * 90)

all_results = []

for nprobe in NPROBE_VALUES:
    faiss_latencies = []
    recalls = []

    for q in queries:
        faiss_ids, faiss_latency = faiss_search(q, nprobe, TOP_K)
        recall = recall_at_k(exact_results[q], faiss_ids)

        faiss_latencies.append(faiss_latency)
        recalls.append(recall)

    avg_faiss_latency = np.mean(faiss_latencies)
    avg_recall = np.mean(recalls)

    if avg_faiss_latency < avg_exact_latency:
        speedup = avg_exact_latency / avg_faiss_latency
    else:
        speedup = avg_exact_latency / avg_faiss_latency  # θα είναι < 1

    all_results.append((nprobe, avg_faiss_latency, avg_recall, speedup))

    print(f"\nnprobe = {nprobe}")
    print(f"Average FAISS Latency: {avg_faiss_latency:.6f} sec")
    print(f"Average Recall@{TOP_K}: {avg_recall:.4f}")
    print(f"Speedup vs Exact: {speedup:.4f}x")

# --------------------------------------------------
# 8. Final summary table
# --------------------------------------------------
print("\n" + "=" * 90)
print("FINAL SUMMARY")
print("=" * 90)
print(f"{'nprobe':<10}{'faiss_latency':<20}{'recall@5':<15}{'speedup_vs_exact':<20}")

for nprobe, lat, rec, spd in all_results:
    print(f"{nprobe:<10}{lat:<20.6f}{rec:<15.4f}{spd:<20.4f}")