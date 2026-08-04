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

RESULTS_PATH = Path("benchmark_hnsw_efsearch_results.csv")


# ---------------------------------------------------------
# Ρυθμίσεις πειράματος
# ---------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5
NUM_QUERIES = 100
REPEATS = 10
RANDOM_SEED = 42

# Τιμές efSearch που θα εξεταστούν
EF_SEARCH_VALUES = [10, 20, 50, 100, 200]


def load_data():
    """
    Φορτώνει embeddings και κείμενα.

    Τα embeddings μετατρέπονται σε float32
    και κανονικοποιούνται ώστε το dot product
    να αντιστοιχεί στην cosine similarity.
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
            f"Χρειάζονται περισσότερα από {TOP_K} embeddings."
        )

    normalized_embeddings = embeddings.copy()

    faiss.normalize_L2(
        normalized_embeddings
    )

    return normalized_embeddings, texts


def load_hnsw_index(
    num_vectors: int,
    dimension: int,
):
    """
    Φορτώνει το αποθηκευμένο FAISS HNSW index.
    """

    if not HNSW_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Δεν βρέθηκε το hnsw.index. "
            "Τρέξε πρώτα το build_hnsw_index.py."
        )

    print("\nLoading HNSW index...")

    index = faiss.read_index(
        str(HNSW_INDEX_PATH)
    )

    if index.ntotal != num_vectors:
        raise ValueError(
            "Το HNSW index δεν περιέχει τον ίδιο "
            "αριθμό διανυσμάτων με το embeddings.npy. "
            "Ξανατρέξε το build_hnsw_index.py."
        )

    if index.d != dimension:
        raise ValueError(
            "Η διάσταση του HNSW index δεν συμφωνεί "
            "με τη διάσταση των embeddings."
        )

    print(f"Vectors in index: {index.ntotal}")
    print(f"Embedding dimension: {index.d}")

    return index


def create_query_embeddings(
    texts: np.ndarray,
):
    """
    Επιλέγει τυχαία κείμενα από το dataset
    και δημιουργεί τα query embeddings.

    Η παραγωγή των embeddings γίνεται μόνο μία φορά
    και δεν συμπεριλαμβάνεται στο search latency.
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
        original_text = str(texts[query_id])

        cleaned_text = clean_text(
            original_text
        )

        # Προστασία σε περίπτωση που κάποιο κείμενο
        # γίνει κενό μετά τον καθαρισμό.
        if not cleaned_text:
            cleaned_text = original_text.strip()

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

    faiss.normalize_L2(
        query_embeddings
    )

    return query_ids, query_embeddings


def exact_search(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    query_id: int,
):
    """
    Εκτελεί exhaustive cosine similarity search.

    Επιστρέφει τα πραγματικά top-k αποτελέσματα,
    τα οποία χρησιμοποιούνται ως ground truth.
    """

    scores = embeddings @ query_embedding

    # Το query είναι κείμενο του ίδιου dataset.
    # Αφαιρούμε το ίδιο από τα αποτελέσματα.
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

    return sorted_indices.astype(
        np.int64
    )


def hnsw_search(
    index,
    query_embedding: np.ndarray,
    query_id: int,
):
    """
    Εκτελεί HNSW αναζήτηση και αφαιρεί
    το ίδιο το query από τα αποτελέσματα.
    """

    # Ζητάμε λίγα περισσότερα αποτελέσματα,
    # επειδή μέσα σε αυτά πιθανότατα θα υπάρχει
    # και το ίδιο το query.
    search_k = min(
        index.ntotal,
        TOP_K + 5,
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
    seen_indices = set()

    for result_id in indices[0]:
        result_id = int(result_id)

        if result_id == -1:
            continue

        if result_id == query_id:
            continue

        if result_id in seen_indices:
            continue

        seen_indices.add(result_id)
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
    αριθμός κοινών αποτελεσμάτων / K
    """

    recalls = []

    for exact_ids, approximate_ids in zip(
        exact_results,
        approximate_results,
    ):
        exact_set = set(
            int(value)
            for value in exact_ids
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

    return float(
        np.mean(recalls)
    )


def create_exact_ground_truth(
    embeddings: np.ndarray,
    query_ids: np.ndarray,
    query_embeddings: np.ndarray,
):
    """
    Υπολογίζει μία φορά τα Exact Search αποτελέσματα
    για όλα τα queries.
    """

    print("\nCreating Exact Search ground truth...")

    exact_results = []

    for query_id, query_embedding in zip(
        query_ids,
        query_embeddings,
    ):
        result_ids = exact_search(
            query_embedding=query_embedding,
            embeddings=embeddings,
            query_id=int(query_id),
        )

        exact_results.append(
            result_ids
        )

    return exact_results


def benchmark_ef_search(
    index,
    query_ids: np.ndarray,
    query_embeddings: np.ndarray,
    exact_results,
):
    """
    Δοκιμάζει διαφορετικές τιμές efSearch και
    μετρά latency και Recall@K.
    """

    benchmark_results = []

    print("\nRunning efSearch tuning...")
    print(f"Queries: {len(query_embeddings)}")
    print(f"Repeats per query: {REPEATS}")
    print(f"Top-K: {TOP_K}")

    for ef_search in EF_SEARCH_VALUES:
        print(
            f"\nTesting efSearch = {ef_search}..."
        )

        index.hnsw.efSearch = ef_search

        # ---------------------------------------------
        # Warm-up
        # ---------------------------------------------

        warmup_queries = min(
            5,
            len(query_embeddings),
        )

        for position in range(warmup_queries):
            hnsw_search(
                index=index,
                query_embedding=query_embeddings[position],
                query_id=int(query_ids[position]),
            )

        # ---------------------------------------------
        # Benchmark
        # ---------------------------------------------

        search_times = []
        approximate_results = []

        for query_id, query_embedding in zip(
            query_ids,
            query_embeddings,
        ):
            query_id = int(query_id)

            current_results = None

            for _ in range(REPEATS):
                start_time = time.perf_counter_ns()

                current_results = hnsw_search(
                    index=index,
                    query_embedding=query_embedding,
                    query_id=query_id,
                )

                elapsed_time = (
                    time.perf_counter_ns()
                    - start_time
                )

                # Μετατροπή nanoseconds σε milliseconds
                search_times.append(
                    elapsed_time / 1_000_000
                )

            approximate_results.append(
                current_results
            )

        search_times = np.asarray(
            search_times,
            dtype=np.float64,
        )

        recall = calculate_recall(
            exact_results=exact_results,
            approximate_results=approximate_results,
        )

        average_latency = float(
            np.mean(search_times)
        )

        median_latency = float(
            np.median(search_times)
        )

        standard_deviation = float(
            np.std(search_times)
        )

        percentile_95 = float(
            np.percentile(
                search_times,
                95,
            )
        )

        queries_per_second = (
            1000.0 / average_latency
            if average_latency > 0
            else float("inf")
        )

        result = {
            "ef_search": ef_search,
            "average_latency_ms": average_latency,
            "median_latency_ms": median_latency,
            "std_latency_ms": standard_deviation,
            "p95_latency_ms": percentile_95,
            "recall_at_k": recall,
            "queries_per_second": queries_per_second,
        }

        benchmark_results.append(
            result
        )

        print(
            f"Average latency: "
            f"{average_latency:.6f} ms"
        )

        print(
            f"Recall@{TOP_K}: "
            f"{recall:.4f}"
        )

    return benchmark_results


def print_results(results):
    """
    Εμφανίζει τα αποτελέσματα σε μορφή πίνακα.
    """

    print("\n" + "=" * 98)
    print("HNSW efSearch TUNING RESULTS")
    print("=" * 98)

    recall_title = f"Recall@{TOP_K}"

    print(
        f"{'efSearch':>10}"
        f"{'Avg (ms)':>14}"
        f"{'Median (ms)':>16}"
        f"{'Std (ms)':>14}"
        f"{'P95 (ms)':>14}"
        f"{recall_title:>14}"
        f"{'Queries/sec':>16}"
    )

    print("-" * 98)

    for result in results:
        print(
            f"{result['ef_search']:>10}"
            f"{result['average_latency_ms']:>14.6f}"
            f"{result['median_latency_ms']:>16.6f}"
            f"{result['std_latency_ms']:>14.6f}"
            f"{result['p95_latency_ms']:>14.6f}"
            f"{result['recall_at_k']:>14.4f}"
            f"{result['queries_per_second']:>16.2f}"
        )

    print("=" * 98)


def save_results(results):
    """
    Αποθηκεύει τα αποτελέσματα σε αρχείο CSV.
    """

    fieldnames = [
        "ef_search",
        "average_latency_ms",
        "median_latency_ms",
        "std_latency_ms",
        "p95_latency_ms",
        "recall_at_k",
        "queries_per_second",
        "top_k",
        "num_queries",
        "repeats",
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
            row["num_queries"] = min(
                NUM_QUERIES,
                len(result) if False else NUM_QUERIES,
            )
            row["repeats"] = REPEATS

            writer.writerow(row)

    print(
        f"\nResults saved to: {RESULTS_PATH}"
    )


def main():
    embeddings, texts = load_data()

    num_vectors, dimension = embeddings.shape

    print(f"Loaded {len(texts)} texts")
    print(f"Embeddings shape: {embeddings.shape}")

    hnsw_index = load_hnsw_index(
        num_vectors=num_vectors,
        dimension=dimension,
    )

    query_ids, query_embeddings = (
        create_query_embeddings(texts)
    )

    exact_results = create_exact_ground_truth(
        embeddings=embeddings,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
    )

    results = benchmark_ef_search(
        index=hnsw_index,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        exact_results=exact_results,
    )

    print_results(results)

    save_results(results)


if __name__ == "__main__":
    main()