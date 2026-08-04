import time
from pathlib import Path

import faiss
import numpy as np


EMBEDDINGS_PATH = Path("embeddings.npy")
HNSW_INDEX_PATH = Path("hnsw.index")

# Παράμετροι κατασκευής HNSW
M = 16
EF_CONSTRUCTION = 100


def load_embeddings() -> np.ndarray:
    """
    Φορτώνει τα αποθηκευμένα embeddings και τα μετατρέπει
    σε float32, όπως απαιτεί η FAISS.
    """
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Δεν βρέθηκε το embeddings.npy. "
            "Τρέξε πρώτα το build_embeddings.py."
        )

    embeddings = np.load(EMBEDDINGS_PATH)

    if embeddings.ndim != 2:
        raise ValueError(
            "Τα embeddings πρέπει να έχουν μορφή "
            "(αριθμός_κειμένων, διαστάσεις)."
        )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    embeddings = np.ascontiguousarray(embeddings)

    return embeddings


def main() -> None:
    print("Loading embeddings...")

    embeddings = load_embeddings()

    num_elements, dimension = embeddings.shape

    print(f"Number of embeddings: {num_elements}")
    print(f"Embedding dimension: {dimension}")

    # Δημιουργούμε αντίγραφο, ώστε να μην αλλάξουμε
    # τα embeddings που φορτώθηκαν από το αρχείο.
    normalized_embeddings = embeddings.copy()

    # Κανονικοποίηση των embeddings σε μοναδιαίο μήκος.
    faiss.normalize_L2(normalized_embeddings)

    print("\nBuilding FAISS HNSW index...")
    print(f"M: {M}")
    print(f"efConstruction: {EF_CONSTRUCTION}")

    # Δημιουργία HNSW index.
    # Από προεπιλογή χρησιμοποιεί squared L2 distance.
    index = faiss.IndexHNSWFlat(
        dimension,
        M,
    )

    # Πόσο αναλυτικά εξερευνάται ο γράφος
    # κατά την κατασκευή του index.
    index.hnsw.efConstruction = EF_CONSTRUCTION

    start_time = time.perf_counter()

    index.add(normalized_embeddings)

    build_time = time.perf_counter() - start_time

    # Αποθήκευση του HNSW index στον δίσκο.
    faiss.write_index(
        index,
        str(HNSW_INDEX_PATH),
    )

    print("\nHNSW index created successfully.")
    print(f"Vectors stored: {index.ntotal}")
    print(f"Build time: {build_time:.4f} sec")
    print(f"Saved index: {HNSW_INDEX_PATH}")


if __name__ == "__main__":
    main()