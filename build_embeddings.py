import time
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from utils import clean_text

print("Loading dataset...")
dataset = load_dataset("ag_news", split="train[:5000]")

texts = [clean_text(item["text"]) for item in dataset if item["text"]]

print(f"Loaded {len(texts)} texts")

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")
start = time.time()

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

end = time.time()

print(f"Embedding time: {end - start:.4f} sec")

np.save("embeddings.npy", embeddings)
np.save("texts.npy", np.array(texts, dtype=object))

print("Saved embeddings and texts.")