import csv
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils import clean_text


# ---------------------------------------------------------
# Αρχεία
# ---------------------------------------------------------

EMBEDDINGS_PATH = Path("embeddings.npy")
TEXTS_PATH = Path("texts.npy")
HNSW_INDEX_PATH = Path("hnsw.index")

RESULTS_PATH = Path(
    "benchmark_exact_ivf_hnsw_results.csv"
)


# ---------------------------------------------------------
# Ρυθμίσεις πειράματος
# ---------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5
NUM_QUERIES = 100
REPEATS = 5
RANDOM_SEED = 42

# Παράμετροι FAISS IVF
NLIST = 100
NPROBE = 10

# Παράμετρος FAISS HNSW
EF_SEARCH = 50


def load_data():
    """
    Φορτώνει τα embeddings και τα αντίστοιχα κείμενα.
    Τα embeddings μετατρέπονται σε float32 και
    κανονικοποιούνται για cosine similarity.
    """

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Δεν βρέθηκε το embeddings.npy. "
            "Τρέξε πρώτα το build_embeddings.py."
        )

    if not TEXTS_PATH.exists():
        raise FileNotFoundError(
            "Δεν βρέθηκε το texts.npy. "
            "Τρέξε πρώτα το build_embeddings.py."
        )

    print("Loading embeddings and texts...")

    embeddings = np.load(EMBEDDINGS_PATH)
    texts = np.load(
        TEXTS_PATH,
        allow_pickle=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    embeddings = np.ascontiguousarray(
        embeddings,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Τα embeddings πρέπει να είναι "
            "δισδιάστατος πίνακας."
        )

    if len(embeddings) != len(texts):
        raise ValueError(
            "Ο αριθμός των embeddings δεν συμφωνεί "
            "με τον αριθμό των κειμένων."
        )

    if len(embeddings) <= TOP_K:
        raise ValueError(
            "Το dataset πρέπει να περιέχει περισσότερα "
            f"από {TOP_K} κείμενα."
        )

    # Αντίγραφο ώστε να μην αλλάξουμε τα δεδομένα
    # που φορτώθηκαν από το αρχείο.
    normalized_embeddings = embeddings.copy()

    # Κανονικοποίηση σε μοναδιαίο μήκος.
    faiss.normalize_L2(normalized_embeddings)

    return normalized_embeddings, texts


def build_ivf_index(
    embeddings: np.ndarray,
):
    """
    Δημιουργεί και εκπαιδεύει ένα FAISS IndexIVFFlat.
    Η κατασκευή του index δεν περιλαμβάνεται
    στον χρόνο αναζήτησης.
    """

    num_vectors, dimension = embeddings.shape

    # Στόχος είναι να έχουμε περίπου τουλάχιστον
    # 40 vectors ανά cluster, ιδιαίτερα όταν
    # χρησιμοποιείται μικρό dataset.
    actual_nlist = min(
        NLIST,
        max(1, num_vectors // 40),
    )

    actual_nprobe = min(
        NPROBE,
        actual_nlist,
    )

    print("\nBuilding FAISS IVF index...")
    print(f"nlist: {actual_nlist}")
    print(f"nprobe: {actual_nprobe}")

    # Exact quantizer με inner product.
    quantizer = faiss.IndexFlatIP(
        dimension,
    )

    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        actual_nlist,
        faiss.METRIC_INNER_PRODUCT,
    )

    build_start = time.perf_counter()

    index.train(embeddings)
    index.add(embeddings)

    build_time = (
        time.perf_counter() - build_start
    )

    index.nprobe = actual_nprobe

    print(
        f"IVF build time: {build_time:.4f} sec"
    )
    print(
        f"Vectors in IVF index: {index.ntotal}"
    )

    return index, actual_nlist, actual_nprobe


def load_hnsw_index(
    num_vectors: int,
    dimension: int,
):
    """
    Φορτώνει το HNSW index που κατασκευάστηκε
    από το build_hnsw_index.py.
    """

    if not HNSW_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Δεν βρέθηκε το hnsw.index. "
            "Τρέξε πρώτα το build_hnsw_index.py."
        )

    print("\nLoading FAISS HNSW index...")

    index = faiss.read_index(
        str(HNSW_INDEX_PATH)
    )

    if index.ntotal != num_vectors:
        raise ValueError(
            "Το HNSW index δεν περιέχει τον ίδιο "
            "αριθμό vectors με το embeddings.npy. "
            "Ξανατρέξε το build_hnsw_index.py."
        )

    if index.d != dimension:
        raise ValueError(
            "Η διάσταση του HNSW index δεν συμφωνεί "
            "με τη διάσταση των embeddings."
        )

    index.hnsw.efSearch = EF_SEARCH

    print(
        f"Vectors in HNSW index: {index.ntotal}"
    )
    print(f"efSearch: {EF_SEARCH}")

    return index


def create_query_embeddings(
    texts: np.ndarray,
):
    """
    Επιλέγει τυχαία κείμενα από το dataset και
    τα μετατρέπει σε query embeddings.

    Κρατάμε επίσης τα IDs τους ώστε να αφαιρέσουμε
    το ίδιο το κείμενο από τα αποτελέσματα.
    """

    number_of_queries = min(
        NUM_QUERIES,
        len(texts),
    )

    random_generator = np.random.default_rng(
        RANDOM_SEED
    )

    query_ids = random_generator.choice(
        len(texts),
        size=number_of_queries,
        replace=False,
    )

    query_texts = []

    for query_id in query_ids:
        cleaned_text = clean_text(
            str(texts[query_id])
        )

        query_texts.append(cleaned_text)

    print("\nLoading Sentence Transformer model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"Creating {number_of_queries} "
        "query embeddings..."
    )

    query_embeddings = model.encode(
        query_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32,
    )

    query_embeddings = np.ascontiguousarray(
        query_embeddings,
    )

    faiss.normalize_L2(query_embeddings)

    return query_ids, query_embeddings


def exact_search(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    query_id: int,
):
    """
    Εκτελεί exhaustive brute-force cosine search.

    Επειδή τα vectors είναι κανονικοποιημένα,
    το dot product ισούται με cosine similarity.
    """

    scores = embeddings @ query_embedding

    # Το query προέρχεται από το ίδιο dataset.
    # Αφαιρούμε το ίδιο το κείμενο ώστε να μην
    # εμφανίζεται ως τετριμμένο πρώτο αποτέλεσμα.
    scores[query_id] = -np.inf

    candidate_indices = np.argpartition(
        scores,
        -TOP_K,
    )[-TOP_K:]

    sorted_indices = candidate_indices[
        np.argsort(
            scores[candidate_indices]
        )[::-1]
    ]

    return sorted_indices.astype(np.int64)


def faiss_search(
    index,
    query_embedding: np.ndarray,
    query_id: int,
):
    """
    Εκτελεί αναζήτηση σε FAISS IVF ή HNSW index
    και αφαιρεί το ίδιο το query από τα αποτελέσματα.
    """

    # Ζητάμε περισσότερα αποτελέσματα ώστε
    # να υπάρχει χώρος για αφαίρεση του query.
    search_k = min(
        index.ntotal,
        TOP_K + 10,
    )

    query_matrix = query_embedding.reshape(
        1,
        -1,
    )

    _, indices = index.search(
        query_matrix,
        search_k,
    )

    filtered_indices = []

    for result_id in indices[0]:
        result_id = int(result_id)

        if result_id == -1:
            continue

        if result_id == query_id:
            continue

        filtered_indices.append(result_id)

        if len(filtered_indices) == TOP_K:
            break

    return np.asarray(
        filtered_indices,
        dtype=np.int64,
    )


def calculate_recall(
    exact_results,
    approximate_results,
):
    """
    Υπολογίζει το μέσο Recall@K.

    Recall@K =
    κοινά αποτελέσματα / K
    """

    recalls = []

    for exact_ids, approximate_ids in zip(
        exact_results,
        approximate_results,
    ):
        exact_set = set(
            int(value) for value in exact_ids
        )

        approximate_set = set(
            int(value)
            for value in approximate_ids
        )

        common_results = exact_set.intersection(
            approximate_set
        )

        recall = (
            len(common_results) / TOP_K
        )

        recalls.append(recall)

    return float(np.mean(recalls))


def benchmark_search_methods(
    embeddings: np.ndarray,
    query_ids: np.ndarray,
    query_embeddings: np.ndarray,
    ivf_index,
    hnsw_index,
):
    """
    Εκτελεί το benchmark για Exact, IVF και HNSW.

    Η παραγωγή των query embeddings έχει ήδη γίνει
    και δεν περιλαμβάνεται στους χρόνους.
    """

    exact_times = []
    ivf_times = []
    hnsw_times = []

    exact_results = []
    ivf_results = []
    hnsw_results = []

    print("\nWarming up indexes...")

    warmup_queries = min(
        5,
        len(query_embeddings),
    )

    for position in range(warmup_queries):
        query_embedding = query_embeddings[position]
        query_id = int(query_ids[position])

        exact_search(
            query_embedding,
            embeddings,
            query_id,
        )

        faiss_search(
            ivf_index,
            query_embedding,
            query_id,
        )

        faiss_search(
            hnsw_index,
            query_embedding,
            query_id,
        )

    print("Running benchmark...")
    print(
        f"Queries: {len(query_embeddings)}"
    )
    print(f"Repeats per query: {REPEATS}")
    print(f"Top-K: {TOP_K}")

    for query_id, query_embedding in zip(
        query_ids,
        query_embeddings,
    ):
        query_id = int(query_id)

        current_exact_results = None
        current_ivf_results = None
        current_hnsw_results = None

        # -------------------------------------------------
        # Exact Search
        # -------------------------------------------------

        for _ in range(REPEATS):
            start_time = time.perf_counter_ns()

            current_exact_results = exact_search(
                query_embedding,
                embeddings,
                query_id,
            )

            elapsed_time = (
                time.perf_counter_ns() - start_time
            )

            exact_times.append(
                elapsed_time / 1_000_000
            )

        # -------------------------------------------------
        # IVF Search
        # -------------------------------------------------

        for _ in range(REPEATS):
            start_time = time.perf_counter_ns()

            current_ivf_results = faiss_search(
                ivf_index,
                query_embedding,
                query_id,
            )

            elapsed_time = (
                time.perf_counter_ns() - start_time
            )

            ivf_times.append(
                elapsed_time / 1_000_000
            )

        # -------------------------------------------------
        # HNSW Search
        # -------------------------------------------------

        for _ in range(REPEATS):
            start_time = time.perf_counter_ns()

            current_hnsw_results = faiss_search(
                hnsw_index,
                query_embedding,
                query_id,
            )

            elapsed_time = (
                time.perf_counter_ns() - start_time
            )

            hnsw_times.append(
                elapsed_time / 1_000_000
            )

        exact_results.append(
            current_exact_results
        )

        ivf_results.append(
            current_ivf_results
        )

        hnsw_results.append(
            current_hnsw_results
        )

    exact_times = np.asarray(
        exact_times,
        dtype=np.float64,
    )

    ivf_times = np.asarray(
        ivf_times,
        dtype=np.float64,
    )

    hnsw_times = np.asarray(
        hnsw_times,
        dtype=np.float64,
    )

    ivf_recall = calculate_recall(
        exact_results,
        ivf_results,
    )

    hnsw_recall = calculate_recall(
        exact_results,
        hnsw_results,
    )

    exact_average = float(
        np.mean(exact_times)
    )

    ivf_average = float(
        np.mean(ivf_times)
    )

    hnsw_average = float(
        np.mean(hnsw_times)
    )

    results = [
        {
            "method": "Exact",
            "average_latency_ms": exact_average,
            "median_latency_ms": float(
                np.median(exact_times)
            ),
            "std_latency_ms": float(
                np.std(exact_times)
            ),
            "recall_at_k": 1.0,
            "speedup": 1.0,
        },
        {
            "method": "FAISS IVF",
            "average_latency_ms": ivf_average,
            "median_latency_ms": float(
                np.median(ivf_times)
            ),
            "std_latency_ms": float(
                np.std(ivf_times)
            ),
            "recall_at_k": ivf_recall,
            "speedup": (
                exact_average / ivf_average
            ),
        },
        {
            "method": "FAISS HNSW",
            "average_latency_ms": hnsw_average,
            "median_latency_ms": float(
                np.median(hnsw_times)
            ),
            "std_latency_ms": float(
                np.std(hnsw_times)
            ),
            "recall_at_k": hnsw_recall,
            "speedup": (
                exact_average / hnsw_average
            ),
        },
    ]

    return results


def print_results(results):
    """
    Εμφανίζει τα αποτελέσματα σε πίνακα.
    """

    print("\n" + "=" * 82)
    print("BENCHMARK RESULTS")
    print("=" * 82)

    recall_title = f"Recall@{TOP_K}"

    print(
        f"{'Method':<16}"
        f"{'Avg (ms)':>12}"
        f"{'Median (ms)':>14}"
        f"{'Std (ms)':>12}"
        f"{recall_title:>12}"
        f"{'Speedup':>12}"
    )

    print("-" * 82)

    for result in results:
        print(
            f"{result['method']:<16}"
            f"{result['average_latency_ms']:>12.6f}"
            f"{result['median_latency_ms']:>14.6f}"
            f"{result['std_latency_ms']:>12.6f}"
            f"{result['recall_at_k']:>12.4f}"
            f"{result['speedup']:>11.2f}x"
        )

    print("=" * 82)


def save_results(
    results,
    actual_nlist: int,
    actual_nprobe: int,
):
    """
    Αποθηκεύει τα συνοπτικά αποτελέσματα σε CSV.
    """

    fieldnames = [
        "method",
        "average_latency_ms",
        "median_latency_ms",
        "std_latency_ms",
        "recall_at_k",
        "speedup",
        "top_k",
        "num_queries",
        "repeats",
        "nlist",
        "nprobe",
        "ef_search",
    ]

    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            row = result.copy()

            row["top_k"] = TOP_K
            row["num_queries"] = NUM_QUERIES
            row["repeats"] = REPEATS
            row["nlist"] = actual_nlist
            row["nprobe"] = actual_nprobe
            row["ef_search"] = EF_SEARCH

            writer.writerow(row)

    print(
        f"\nResults saved to: {RESULTS_PATH}"
    )


def main():
    embeddings, texts = load_data()

    num_vectors, dimension = embeddings.shape

    print(f"Loaded {num_vectors} texts")
    print(
        f"Embeddings shape: {embeddings.shape}"
    )

    ivf_index, actual_nlist, actual_nprobe = (
        build_ivf_index(embeddings)
    )

    hnsw_index = load_hnsw_index(
        num_vectors=num_vectors,
        dimension=dimension,
    )

    query_ids, query_embeddings = (
        create_query_embeddings(texts)
    )

    results = benchmark_search_methods(
        embeddings=embeddings,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        ivf_index=ivf_index,
        hnsw_index=hnsw_index,
    )

    print_results(results)

    save_results(
        results=results,
        actual_nlist=actual_nlist,
        actual_nprobe=actual_nprobe,
    )


if __name__ == "__main__":
    main()