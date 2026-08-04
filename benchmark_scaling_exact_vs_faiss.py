import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text

# --------------------------------------------------
# 1. Load full stored embeddings/texts
# --------------------------------------------------
print("Loading full embeddings and texts...")

all_embeddings = np.load("embeddings.npy").astype("float32")
all_texts = np.load("texts.npy", allow_pickle=True)

print(f"Total loaded texts: {len(all_texts)}")
print(f"Embeddings shape: {all_embeddings.shape}")

# --------------------------------------------------
# 2. Load model
# --------------------------------------------------
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# 3. Experiment settings
# --------------------------------------------------
DATASET_SIZES = [1000, 2000, 5000]
TOP_K = 5
NPROBE = 10
NLIST = 50

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

# --------------------------------------------------
# 4. Search functions
# --------------------------------------------------
def exact_search(query, embeddings, top_k=5):
    query = clean_text(query)

    start = time.time()

    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    end = time.time()

    return top_indices.tolist(), (end - start)


def faiss_search(query, index, top_k=5):
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
# 5. Scaling experiment
# --------------------------------------------------
results_summary = []

print("\n" + "=" * 90)
print("SCALING EXPERIMENT")
print("=" * 90)

for size in DATASET_SIZES:
    print(f"\nRunning experiment for dataset size = {size}")

    embeddings = all_embeddings[:size].copy()
    texts = all_texts[:size].copy()

    # Exact embeddings
    exact_embeddings = embeddings.copy()

    # FAISS embeddings
    faiss_embeddings = embeddings.copy()
    faiss.normalize_L2(faiss_embeddings)

    dim = faiss_embeddings.shape[1]

    # Adjust nlist if dataset is small
    current_nlist = min(NLIST, max(2, size // 100))

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, current_nlist, faiss.METRIC_INNER_PRODUCT)

    # Build FAISS index
    start_train = time.time()
    index.train(faiss_embeddings)
    end_train = time.time()

    start_add = time.time()
    index.add(faiss_embeddings)
    end_add = time.time()

    index.nprobe = min(NPROBE, current_nlist)

    exact_latencies = []
    faiss_latencies = []
    recalls = []

    for q in queries:
        exact_ids, exact_latency = exact_search(q, exact_embeddings, TOP_K)
        faiss_ids, faiss_latency = faiss_search(q, index, TOP_K)
        recall = recall_at_k(exact_ids, faiss_ids)

        exact_latencies.append(exact_latency)
        faiss_latencies.append(faiss_latency)
        recalls.append(recall)

    avg_exact_latency = np.mean(exact_latencies)
    avg_faiss_latency = np.mean(faiss_latencies)
    avg_recall = np.mean(recalls)
    train_time = end_train - start_train
    add_time = end_add - start_add

    if avg_faiss_latency < avg_exact_latency:
        speedup = avg_exact_latency / avg_faiss_latency
    else:
        speedup = avg_exact_latency / avg_faiss_latency

    results_summary.append((
        size,
        avg_exact_latency,
        avg_faiss_latency,
        avg_recall,
        speedup,
        train_time,
        add_time,
        current_nlist,
        index.nprobe
    ))

    print(f"Average Exact Latency: {avg_exact_latency:.6f} sec")
    print(f"Average FAISS Latency: {avg_faiss_latency:.6f} sec")
    print(f"Average Recall@{TOP_K}: {avg_recall:.4f}")
    print(f"Speedup vs Exact: {speedup:.4f}x")
    print(f"FAISS train time: {train_time:.6f} sec")
    print(f"FAISS add time: {add_time:.6f} sec")
    print(f"nlist: {current_nlist}, nprobe: {index.nprobe}")

# --------------------------------------------------
# 6. Final summary
# --------------------------------------------------
print("\n" + "=" * 110)
print("FINAL SCALING SUMMARY")
print("=" * 110)
print(f"{'size':<10}{'exact_latency':<18}{'faiss_latency':<18}{'recall@5':<12}{'speedup':<12}{'train_time':<15}{'add_time':<15}")

for row in results_summary:
    size, exact_lat, faiss_lat, recall, speedup, train_t, add_t, nlist_val, nprobe_val = row
    print(f"{size:<10}{exact_lat:<18.6f}{faiss_lat:<18.6f}{recall:<12.4f}{speedup:<12.4f}{train_t:<15.6f}{add_t:<15.6f}")