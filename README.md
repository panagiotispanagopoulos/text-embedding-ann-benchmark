# Efficient Similarity Search in Text Embeddings

Experimental framework developed for a diploma thesis on exact and Approximate Nearest Neighbor (ANN) search over text embeddings.

## Methods

- Exact Search / brute-force baseline
- FAISS IVF (`IndexIVFFlat`)
- FAISS HNSW (`IndexHNSWFlat`)

## Dataset and embeddings

- Dataset: AG News
- Final collection: 100,000 texts
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384
- Embeddings are L2-normalized before indexing

## Evaluation

The benchmark measures:

- Recall@5
- Mean, median, standard deviation and P95 search latency
- Queries per second
- Speedup over Exact Search
- Index build time
- Serialized index size
- Scaling from 1,000 to 100,000 embeddings

The final experiments also study the IVF `nprobe` and HNSW `efSearch` parameters and compare the two ANN methods at a common Recall@5 operating point.

## Final thesis benchmark settings

- 100 query vectors
- 10 timed repetitions per query
- 10 warm-up queries
- Fixed random seed: 42
- One FAISS CPU thread for controlled timing
- Nested dataset subsets for scaling
- Exact Search used as the ground truth

## Results files

The repository includes the CSV outputs used for the final thesis analysis:

- `benchmark_scaling_100k_exact_ivf_hnsw_results.csv`
- `benchmark_ivf_nprobe_100k_results.csv`
- `benchmark_hnsw_efsearch_100k_results.csv`

## Installation

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Data files

Generated embeddings, text arrays and serialized indexes are intentionally excluded from Git because they are reproducible and can be large. They are covered by `.gitignore`.

## Reproducibility

The project separates embedding generation, index construction, benchmarking and plotting so that experiments can be rerun independently. Parameter values used in the final evaluation are documented in the thesis and in the benchmark scripts/results included here.
