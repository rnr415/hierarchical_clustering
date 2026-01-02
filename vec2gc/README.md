# Vec2GC — Vector-to-Graph Clustering (vec2gc) 🔧

A small library for memory-efficient hierarchical clustering of vector embeddings using a graph-based approach and NetworKit community detection.

## What it does ✅

- Builds a sparse graph from pairwise cosine similarities of embeddings (edges created when similarity > threshold).
- Uses weighted edges (weight = 1 / (1 - similarity)) and NetworKit's Louvain (PLM) algorithm to detect communities.
- Applies recursive, hierarchical clustering while processing each subgraph sequentially to drastically reduce peak memory usage.

## Key modules 🔍

- `vec2gc.py` — Core algorithm:
  - `HierarchicalSequentialClustering`: main class that builds the graph (batched streaming mode or chunked-parallel streaming for very large datasets), detects communities, and recursively splits them.
  - `generate_sample_embeddings`: helper to create synthetic clusterable embeddings for quick tests.

- `embeddings.py` — Helpers for turning text into sentence embeddings using `sentence-transformers` and HuggingFace `datasets`.
  - `create_sentence_embeddings(dataset, model_name, text_column=None, ...)` accepts a HuggingFace `Dataset` (or similar mapping with a text column) and returns a `torch.Tensor` of embeddings.

- `main.py` — Example usage demonstrating embedding creation and clustering.

## Installation ⚠️

Recommended (conda helps for `networkit` on macOS):

```bash
# Core Python deps
pip install numpy joblib sentence-transformers datasets torch

# Networkit (binary from conda-forge recommended on macOS)
conda install -c conda-forge networkit
# or try: pip install networkit
```

## Quick start — synthetic data (recommended for testing) 💡

```python
from vec2gc.vec2gc import HierarchicalSequentialClustering, generate_sample_embeddings

# Create sample embeddings
embeddings = generate_sample_embeddings(n_items=200, embedding_dim=128, n_clusters=4)

# Initialize clusterer
clusterer = HierarchicalSequentialClustering(
    similarity_threshold=0.6,  # cosine similarity threshold for creating edges
    min_cluster_size=10,       # stop recursion for clusters smaller than this
    min_modularity=0.5         # require modularity to continue splitting
)

# Run clustering
clusters = clusterer.fit_predict(embeddings, batch_size=1000)

# Print or inspect results
clusterer.print_clusters()
```

## Quick start — using text and Sentence-Transformers 📝

```python
from datasets import load_dataset
from vec2gc.embeddings import create_sentence_embeddings
from vec2gc.vec2gc import HierarchicalSequentialClustering

# Load a small subset of a HuggingFace dataset
dataset = load_dataset('ag_news', split='train[:500]')

# Create embeddings (auto-detects text column if not provided)
emb = create_sentence_embeddings(dataset, model_name='all-MiniLM-L6-v2', text_column='text', batch_size=64)

# `create_sentence_embeddings` returns a torch.Tensor — convert to numpy for the clusterer
emb_np = emb.numpy()

# Cluster
clusterer = HierarchicalSequentialClustering(similarity_threshold=0.6, min_cluster_size=20)
clusters = clusterer.fit_predict(emb_np, batch_size=512)
clusterer.print_clusters()
```

Notes:
- `create_sentence_embeddings` expects a dataset-like object with a text column (e.g., HuggingFace `Dataset`).
- If your usage needs different inputs (list of strings), convert to a `datasets.Dataset` or adapt the function.

## Parameters & tuning 🔧

- `similarity_threshold` (float): higher values → fewer edges, tighter clusters.
- `batch_size` / `chunk_size`: control memory vs speed trade-off for graph construction.
- `min_cluster_size` and `min_modularity`: control recursion stopping criteria.

## Notebooks & examples 📚

- `vec2gc_clustering.ipynb` contains an exploratory notebook showing example runs.
- `examples/run_clustering.py` — a small runnable demo that shows how to run the algorithm on synthetic data (and optionally a small HF dataset).

## Troubleshooting & tips ⚠️

- Building `networkit` from pip can be difficult on macOS; prefer `conda` from `conda-forge` if you encounter build errors.
- For very large datasets, adjust `chunk_size`, `batch_size`, and `n_jobs` to balance memory and speed.
- The implementation is explicitly designed to keep only one subgraph in memory during recursion to reduce peak memory usage.

## License

This repository is provided under the project license (see top-level `LICENSE`).

---

If you'd like, I can also add a short runnable example script under `examples/` or a top-level README summarizing the whole repo. Would you like that? ✨
