\# Efficient Similarity Search in Text Embeddings



Experimental framework for evaluating exact and approximate nearest-neighbor search methods on text embeddings.



\## Implemented Methods



\- Exact Search

\- FAISS IVF

\- FAISS HNSW



\## Dataset and Embeddings



\- AG News dataset

\- Sentence Transformers

\- Model: all-MiniLM-L6-v2

\- Embedding dimension: 384



\## Evaluation Metrics



\- Average, median and P95 search latency

\- Recall@5

\- Queries per second

\- Speedup over Exact Search

\- Index build time

\- Index size

\- Scaling with dataset size



\## Main Experiments



\- Exact vs IVF vs HNSW comparison

\- IVF nprobe tuning

\- HNSW efSearch tuning

\- Scaling experiments

\- Automatic CSV and plot generation



\## Installation



1\. Create a virtual environment:



&#x20;  python -m venv venv



2\. Activate it on Windows:



&#x20;  venv\\Scripts\\activate



3\. Install the required packages:



&#x20;  python -m pip install -r requirements.txt



\## Current Status



The core implementation and the initial experimental evaluation have been completed. Larger-scale experiments and additional reproducibility tests are planned.

