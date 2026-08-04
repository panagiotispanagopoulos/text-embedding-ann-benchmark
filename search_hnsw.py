import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils import clean_text


HNSW_INDEX_PATH = Path("hnsw.index")
TEXTS_PATH = Path("texts.npy")

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5
EF_SEARCH = 50


def load_texts() -> np.ndarray:
    """
    Φορτώνει τα κείμενα που αντιστοιχούν
    στα embeddings του HNSW index.
    """
    if not TEXTS_PATH.exists():
        raise FileNotFoundError(
            "Δεν βρέθηκε το texts.npy. "
            "Τρέξε πρώτα το build_embeddings.py."
        )

    return np.load(
        TEXTS_PATH,
        allow_pickle=True,
    )


def load_hnsw_index():
    """
    Φορτώνει το αποθηκευμένο FAISS HNSW index.
    """
    if not HNSW_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Δεν βρέθηκε το hnsw.index. "
            "Τρέξε πρώτα το build_hnsw_index.py."
        )

    index = faiss.read_index(
        str(HNSW_INDEX_PATH)
    )

    # Πόσους υποψήφιους κόμβους εξετάζει
    # το HNSW κατά την αναζήτηση.
    index.hnsw.efSearch = EF_SEARCH

    return index


def create_query_embedding(
    query: str,
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Μετατρέπει το query σε embedding float32
    και το κανονικοποιεί.
    """
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    query_embedding = np.ascontiguousarray(
        query_embedding
    )

    faiss.normalize_L2(query_embedding)

    return query_embedding


def search_hnsw(
    query_embedding: np.ndarray,
    index,
    texts: np.ndarray,
    top_k: int = TOP_K,
):
    """
    Εκτελεί HNSW αναζήτηση και επιστρέφει
    τα top-k κοντινότερα αποτελέσματα.
    """
    top_k = min(
        top_k,
        index.ntotal,
    )

    start_time = time.perf_counter()

    distances, indices = index.search(
        query_embedding,
        top_k,
    )

    search_latency = (
        time.perf_counter() - start_time
    )

    results = []

    for idx, squared_l2_distance in zip(
        indices[0],
        distances[0],
    ):
        if idx == -1:
            continue

        # Για κανονικοποιημένα διανύσματα ισχύει:
        #
        # squared_L2 = 2 - 2 * cosine_similarity
        #
        # άρα:
        #
        # cosine_similarity = 1 - squared_L2 / 2
        cosine_similarity = (
            1.0 - float(squared_l2_distance) / 2.0
        )

        results.append(
            (
                int(idx),
                texts[idx],
                cosine_similarity,
                float(squared_l2_distance),
            )
        )

    return results, search_latency


def main() -> None:
    print("Loading texts...")

    texts = load_texts()

    print(f"Loaded {len(texts)} texts")

    print("\nLoading HNSW index...")

    index = load_hnsw_index()

    if index.ntotal != len(texts):
        raise ValueError(
            "Ο αριθμός των embeddings του index "
            "δεν συμφωνεί με τον αριθμό των κειμένων."
        )

    print("HNSW index loaded successfully.")
    print(f"Vectors in index: {index.ntotal}")
    print(f"efSearch: {EF_SEARCH}")

    print("\nLoading Sentence Transformer model...")

    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input(
            "\nEnter query (or 'exit'): "
        ).strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        if not query:
            print("Query cannot be empty.")
            continue

        cleaned_query = clean_text(query)

        if not cleaned_query:
            print(
                "The query is empty after text cleaning."
            )
            continue

        # Χρόνος δημιουργίας query embedding.
        embedding_start = time.perf_counter()

        query_embedding = create_query_embedding(
            cleaned_query,
            model,
        )

        embedding_latency = (
            time.perf_counter() - embedding_start
        )

        # Χρόνος αποκλειστικά της HNSW αναζήτησης.
        results, search_latency = search_hnsw(
            query_embedding=query_embedding,
            index=index,
            texts=texts,
            top_k=TOP_K,
        )

        total_latency = (
            embedding_latency + search_latency
        )

        print(f"\nQuery: {query}")
        print(
            f"Embedding latency: "
            f"{embedding_latency:.6f} sec"
        )
        print(
            f"HNSW search latency: "
            f"{search_latency:.6f} sec"
        )
        print(
            f"Total latency: "
            f"{total_latency:.6f} sec"
        )

        print("\nTop results:")

        for rank, (
            idx,
            text,
            score,
            distance,
        ) in enumerate(results, start=1):

            print(f"\n{rank}. Index: {idx}")
            print(f"Cosine similarity: {score:.4f}")
            print(
                f"Squared L2 distance: "
                f"{distance:.4f}"
            )
            print(text[:300])


if __name__ == "__main__":
    main()