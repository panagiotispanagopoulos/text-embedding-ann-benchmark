import csv
import time
from pathlib import Path

import faiss
import numpy as np


# =========================================================
# ΑΡΧΕΙΑ
# =========================================================

EMBEDDINGS_PATH = Path("embeddings.npy")

RESULTS_PATH = Path(
    "benchmark_scaling_exact_ivf_hnsw_results.csv"
)


# =========================================================
# ΡΥΘΜΙΣΕΙΣ ΠΕΙΡΑΜΑΤΟΣ
# =========================================================

# Μεγέθη dataset που θέλουμε να εξετάσουμε.
REQUESTED_DATASET_SIZES = [
    1000,
    2000,
    5000,
    10000,
]

# Προσθέτει αυτόματα και όλα τα διαθέσιμα embeddings.
INCLUDE_ALL_AVAILABLE = True

TOP_K = 5
NUM_QUERIES = 100
REPEATS = 10
WARMUP_QUERIES = 10

RANDOM_SEED = 42

# Για πιο σταθερά και συγκρίσιμα CPU timings.
NUM_THREADS = 1


# =========================================================
# IVF ΠΑΡΑΜΕΤΡΟΙ
# =========================================================

IVF_MAX_NLIST = 100
IVF_NPROBE = 10

# Περίπου τουλάχιστον 40 vectors ανά cluster.
MIN_VECTORS_PER_LIST = 40


# =========================================================
# HNSW ΠΑΡΑΜΕΤΡΟΙ
# =========================================================

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100

# Βάλε εδώ την καλύτερη τιμή που βρήκες
# από το benchmark_hnsw_efsearch.py.
HNSW_EF_SEARCH = 50


def load_embeddings() -> np.ndarray:
    """
    Φορτώνει και κανονικοποιεί τα embeddings.
    """

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Δεν βρέθηκε το embeddings.npy. "
            "Τρέξε πρώτα το build_embeddings.py."
        )

    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_PATH)

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    embeddings = np.ascontiguousarray(
        embeddings
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Τα embeddings πρέπει να είναι "
            "δισδιάστατος πίνακας."
        )

    if len(embeddings) <= TOP_K:
        raise ValueError(
            f"Χρειάζονται περισσότερα από "
            f"{TOP_K} embeddings."
        )

    normalized_embeddings = embeddings.copy()

    # Κανονικοποίηση σε μοναδιαίο μήκος.
    faiss.normalize_L2(
        normalized_embeddings
    )

    return normalized_embeddings


def select_dataset_sizes(
    total_vectors: int,
) -> list[int]:
    """
    Κρατά μόνο τα μεγέθη που μπορούν πραγματικά
    να χρησιμοποιηθούν με τα διαθέσιμα embeddings.
    """

    sizes = [
        size
        for size in REQUESTED_DATASET_SIZES
        if TOP_K < size <= total_vectors
    ]

    if INCLUDE_ALL_AVAILABLE:
        sizes.append(total_vectors)

    sizes = sorted(set(sizes))

    if not sizes:
        sizes = [total_vectors]

    return sizes


def prepare_nested_dataset(
    embeddings: np.ndarray,
    dataset_sizes: list[int],
):
    """
    Ανακατεύει μία φορά τα embeddings και δημιουργεί
    nested subsets.

    Τα ίδια queries χρησιμοποιούνται σε όλα τα
    μεγέθη dataset για δίκαιη σύγκριση.
    """

    random_generator = np.random.default_rng(
        RANDOM_SEED
    )

    permutation = random_generator.permutation(
        len(embeddings)
    )

    shuffled_embeddings = np.ascontiguousarray(
        embeddings[permutation],
        dtype=np.float32,
    )

    smallest_dataset_size = min(
        dataset_sizes
    )

    number_of_queries = min(
        NUM_QUERIES,
        smallest_dataset_size,
    )

    # Επιλέγουμε queries μόνο από το μικρότερο subset,
    # ώστε να υπάρχουν σε όλα τα επόμενα subsets.
    query_ids = random_generator.choice(
        smallest_dataset_size,
        size=number_of_queries,
        replace=False,
    )

    query_ids = np.asarray(
        query_ids,
        dtype=np.int64,
    )

    query_embeddings = np.ascontiguousarray(
        shuffled_embeddings[query_ids].copy(),
        dtype=np.float32,
    )

    return (
        shuffled_embeddings,
        query_ids,
        query_embeddings,
    )


def serialized_index_size_mb(index) -> float:
    """
    Υπολογίζει το μέγεθος του index σε serialized μορφή.
    """

    serialized_index = faiss.serialize_index(
        index
    )

    number_of_bytes = np.asarray(
        serialized_index
    ).nbytes

    return float(
        number_of_bytes / (1024 ** 2)
    )


def build_exact_index(
    embeddings: np.ndarray,
):
    """
    Δημιουργεί exhaustive exact index.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    start_time = time.perf_counter()

    index.add(embeddings)

    build_time = (
        time.perf_counter() - start_time
    )

    return index, build_time


def build_ivf_index(
    embeddings: np.ndarray,
):
    """
    Δημιουργεί και εκπαιδεύει IndexIVFFlat.
    """

    num_vectors, dimension = embeddings.shape

    actual_nlist = min(
        IVF_MAX_NLIST,
        max(
            1,
            num_vectors // MIN_VECTORS_PER_LIST,
        ),
    )

    actual_nprobe = min(
        IVF_NPROBE,
        actual_nlist,
    )

    quantizer = faiss.IndexFlatL2(
        dimension
    )

    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        actual_nlist,
        faiss.METRIC_L2,
    )

    start_time = time.perf_counter()

    index.train(embeddings)
    index.add(embeddings)

    build_time = (
        time.perf_counter() - start_time
    )

    index.nprobe = actual_nprobe

    return (
        index,
        build_time,
        actual_nlist,
        actual_nprobe,
    )


def build_hnsw_index(
    embeddings: np.ndarray,
):
    """
    Δημιουργεί FAISS IndexHNSWFlat.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(
        dimension,
        HNSW_M,
    )

    index.hnsw.efConstruction = (
        HNSW_EF_CONSTRUCTION
    )

    index.hnsw.efSearch = (
        HNSW_EF_SEARCH
    )

    start_time = time.perf_counter()

    index.add(embeddings)

    build_time = (
        time.perf_counter() - start_time
    )

    return index, build_time


def search_top_k(
    index,
    query_embedding: np.ndarray,
    query_id: int,
) -> np.ndarray:
    """
    Εκτελεί αναζήτηση και αφαιρεί το ίδιο το query
    από τα αποτελέσματα.
    """

    search_k = min(
        index.ntotal,
        TOP_K + 5,
    )

    query_matrix = np.ascontiguousarray(
        query_embedding.reshape(1, -1),
        dtype=np.float32,
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

        # Το query υπάρχει μέσα στο dataset,
        # επομένως αφαιρούμε το self-match.
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


def benchmark_index(
    index,
    query_ids: np.ndarray,
    query_embeddings: np.ndarray,
):
    """
    Μετρά latency και κρατά τα top-k αποτελέσματα.
    """

    warmup_count = min(
        WARMUP_QUERIES,
        len(query_embeddings),
    )

    # Warm-up
    for position in range(warmup_count):
        search_top_k(
            index=index,
            query_embedding=query_embeddings[position],
            query_id=int(query_ids[position]),
        )

    search_times = []
    all_results = []

    for query_id, query_embedding in zip(
        query_ids,
        query_embeddings,
    ):
        query_id = int(query_id)

        current_results = None

        for _ in range(REPEATS):
            start_time = time.perf_counter_ns()

            current_results = search_top_k(
                index=index,
                query_embedding=query_embedding,
                query_id=query_id,
            )

            elapsed_ns = (
                time.perf_counter_ns()
                - start_time
            )

            # Nanoseconds σε milliseconds.
            search_times.append(
                elapsed_ns / 1_000_000
            )

        all_results.append(
            current_results
        )

    search_times = np.asarray(
        search_times,
        dtype=np.float64,
    )

    statistics = {
        "average_latency_ms": float(
            np.mean(search_times)
        ),
        "median_latency_ms": float(
            np.median(search_times)
        ),
        "std_latency_ms": float(
            np.std(search_times)
        ),
        "p95_latency_ms": float(
            np.percentile(search_times, 95)
        ),
    }

    average_latency = statistics[
        "average_latency_ms"
    ]

    statistics["queries_per_second"] = (
        1000.0 / average_latency
        if average_latency > 0
        else float("inf")
    )

    return all_results, statistics


def calculate_recall(
    exact_results,
    approximate_results,
) -> float:
    """
    Υπολογίζει το μέσο Recall@K.
    """

    recalls = []

    for exact_ids, approximate_ids in zip(
        exact_results,
        approximate_results,
    ):
        exact_set = {
            int(value)
            for value in exact_ids
        }

        approximate_set = {
            int(value)
            for value in approximate_ids
        }

        common_results = exact_set.intersection(
            approximate_set
        )

        recalls.append(
            len(common_results) / TOP_K
        )

    return float(
        np.mean(recalls)
    )


def create_result_row(
    dataset_size: int,
    method: str,
    build_time: float,
    index_size_mb: float,
    statistics: dict,
    recall: float,
    speedup: float,
    nlist=None,
    nprobe=None,
):
    """
    Δημιουργεί μία γραμμή αποτελεσμάτων.
    """

    return {
        "dataset_size": dataset_size,
        "method": method,
        "build_time_sec": build_time,
        "index_size_mb": index_size_mb,
        "average_latency_ms": statistics[
            "average_latency_ms"
        ],
        "median_latency_ms": statistics[
            "median_latency_ms"
        ],
        "std_latency_ms": statistics[
            "std_latency_ms"
        ],
        "p95_latency_ms": statistics[
            "p95_latency_ms"
        ],
        "queries_per_second": statistics[
            "queries_per_second"
        ],
        "recall_at_k": recall,
        "speedup_vs_exact": speedup,
        "top_k": TOP_K,
        "num_queries": len_global_queries,
        "repeats": REPEATS,
        "nlist": nlist,
        "nprobe": nprobe,
        "hnsw_m": (
            HNSW_M
            if method == "FAISS HNSW"
            else None
        ),
        "ef_construction": (
            HNSW_EF_CONSTRUCTION
            if method == "FAISS HNSW"
            else None
        ),
        "ef_search": (
            HNSW_EF_SEARCH
            if method == "FAISS HNSW"
            else None
        ),
    }


def print_size_results(
    dataset_size: int,
    rows: list[dict],
):
    """
    Εμφανίζει τα αποτελέσματα ενός dataset size.
    """

    print("\n" + "=" * 112)
    print(
        f"RESULTS FOR DATASET SIZE: {dataset_size}"
    )
    print("=" * 112)

    print(
        f"{'Method':<16}"
        f"{'Build (s)':>12}"
        f"{'Size (MB)':>12}"
        f"{'Avg (ms)':>14}"
        f"{'P95 (ms)':>14}"
        f"{f'Recall@{TOP_K}':>12}"
        f"{'Speedup':>12}"
        f"{'Queries/sec':>16}"
    )

    print("-" * 112)

    for row in rows:
        print(
            f"{row['method']:<16}"
            f"{row['build_time_sec']:>12.4f}"
            f"{row['index_size_mb']:>12.3f}"
            f"{row['average_latency_ms']:>14.6f}"
            f"{row['p95_latency_ms']:>14.6f}"
            f"{row['recall_at_k']:>12.4f}"
            f"{row['speedup_vs_exact']:>11.2f}x"
            f"{row['queries_per_second']:>16.2f}"
        )

    print("=" * 112)


def save_results(
    results: list[dict],
):
    """
    Αποθηκεύει όλα τα αποτελέσματα σε CSV.
    """

    fieldnames = [
        "dataset_size",
        "method",
        "build_time_sec",
        "index_size_mb",
        "average_latency_ms",
        "median_latency_ms",
        "std_latency_ms",
        "p95_latency_ms",
        "queries_per_second",
        "recall_at_k",
        "speedup_vs_exact",
        "top_k",
        "num_queries",
        "repeats",
        "nlist",
        "nprobe",
        "hnsw_m",
        "ef_construction",
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
        writer.writerows(results)

    print(
        f"\nResults saved to: {RESULTS_PATH}"
    )


# Ορίζεται στο main και χρησιμοποιείται
# κατά τη δημιουργία των CSV rows.
len_global_queries = 0


def main():
    global len_global_queries

    # Μειώνει τις αποκλίσεις από multithreading.
    faiss.omp_set_num_threads(
        NUM_THREADS
    )

    embeddings = load_embeddings()

    total_vectors, dimension = embeddings.shape

    print(f"Available embeddings: {total_vectors}")
    print(f"Embedding dimension: {dimension}")
    print(f"FAISS threads: {NUM_THREADS}")

    dataset_sizes = select_dataset_sizes(
        total_vectors
    )

    print(
        f"Dataset sizes: {dataset_sizes}"
    )

    (
        shuffled_embeddings,
        query_ids,
        query_embeddings,
    ) = prepare_nested_dataset(
        embeddings=embeddings,
        dataset_sizes=dataset_sizes,
    )

    len_global_queries = len(
        query_embeddings
    )

    print(
        f"Fixed queries for all sizes: "
        f"{len_global_queries}"
    )

    all_results = []

    for dataset_size in dataset_sizes:
        print("\n" + "#" * 70)
        print(
            f"TESTING DATASET SIZE: {dataset_size}"
        )
        print("#" * 70)

        corpus = np.ascontiguousarray(
            shuffled_embeddings[:dataset_size],
            dtype=np.float32,
        )

        # -------------------------------------------------
        # EXACT INDEX
        # -------------------------------------------------

        print("\nBuilding Exact index...")

        exact_index, exact_build_time = (
            build_exact_index(corpus)
        )

        exact_size_mb = serialized_index_size_mb(
            exact_index
        )

        print("Benchmarking Exact Search...")

        (
            exact_results,
            exact_statistics,
        ) = benchmark_index(
            index=exact_index,
            query_ids=query_ids,
            query_embeddings=query_embeddings,
        )

        exact_average = exact_statistics[
            "average_latency_ms"
        ]

        exact_row = create_result_row(
            dataset_size=dataset_size,
            method="Exact",
            build_time=exact_build_time,
            index_size_mb=exact_size_mb,
            statistics=exact_statistics,
            recall=1.0,
            speedup=1.0,
        )

        # -------------------------------------------------
        # IVF INDEX
        # -------------------------------------------------

        print("\nBuilding IVF index...")

        (
            ivf_index,
            ivf_build_time,
            actual_nlist,
            actual_nprobe,
        ) = build_ivf_index(corpus)

        print(f"nlist: {actual_nlist}")
        print(f"nprobe: {actual_nprobe}")

        ivf_size_mb = serialized_index_size_mb(
            ivf_index
        )

        print("Benchmarking IVF Search...")

        (
            ivf_results,
            ivf_statistics,
        ) = benchmark_index(
            index=ivf_index,
            query_ids=query_ids,
            query_embeddings=query_embeddings,
        )

        ivf_recall = calculate_recall(
            exact_results=exact_results,
            approximate_results=ivf_results,
        )

        ivf_average = ivf_statistics[
            "average_latency_ms"
        ]

        ivf_speedup = (
            exact_average / ivf_average
            if ivf_average > 0
            else float("inf")
        )

        ivf_row = create_result_row(
            dataset_size=dataset_size,
            method="FAISS IVF",
            build_time=ivf_build_time,
            index_size_mb=ivf_size_mb,
            statistics=ivf_statistics,
            recall=ivf_recall,
            speedup=ivf_speedup,
            nlist=actual_nlist,
            nprobe=actual_nprobe,
        )

        # -------------------------------------------------
        # HNSW INDEX
        # -------------------------------------------------

        print("\nBuilding HNSW index...")
        print(f"M: {HNSW_M}")
        print(
            f"efConstruction: "
            f"{HNSW_EF_CONSTRUCTION}"
        )
        print(f"efSearch: {HNSW_EF_SEARCH}")

        hnsw_index, hnsw_build_time = (
            build_hnsw_index(corpus)
        )

        hnsw_size_mb = serialized_index_size_mb(
            hnsw_index
        )

        print("Benchmarking HNSW Search...")

        (
            hnsw_results,
            hnsw_statistics,
        ) = benchmark_index(
            index=hnsw_index,
            query_ids=query_ids,
            query_embeddings=query_embeddings,
        )

        hnsw_recall = calculate_recall(
            exact_results=exact_results,
            approximate_results=hnsw_results,
        )

        hnsw_average = hnsw_statistics[
            "average_latency_ms"
        ]

        hnsw_speedup = (
            exact_average / hnsw_average
            if hnsw_average > 0
            else float("inf")
        )

        hnsw_row = create_result_row(
            dataset_size=dataset_size,
            method="FAISS HNSW",
            build_time=hnsw_build_time,
            index_size_mb=hnsw_size_mb,
            statistics=hnsw_statistics,
            recall=hnsw_recall,
            speedup=hnsw_speedup,
        )

        current_rows = [
            exact_row,
            ivf_row,
            hnsw_row,
        ]

        all_results.extend(
            current_rows
        )

        print_size_results(
            dataset_size=dataset_size,
            rows=current_rows,
        )

        # Απελευθέρωση indexes πριν το επόμενο size.
        del exact_index
        del ivf_index
        del hnsw_index

    save_results(
        all_results
    )

    print("\nScaling benchmark completed successfully.")


if __name__ == "__main__":
    main()