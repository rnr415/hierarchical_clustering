"""Package-level example runner (kept for convenience).

This script assumes it's executed as a module (e.g., `python -m vec2gc.main`) or
that the package is installed in editable mode.
"""
from __future__ import annotations

from vec2gc.embeddings import create_sentence_embeddings
from vec2gc.vec2gc import HierarchicalSequentialClustering
import numpy as np


if __name__ == "__main__":
    print("Creating embeddings...")
    # Note: create_sentence_embeddings expects a HuggingFace dataset object if used
    # embeddings = create_sentence_embeddings(dataset_name='sst2', model_name='all-MiniLM-L6-v2')

    # Simple demo using synthetic embeddings
    from vec2gc.vec2gc import generate_sample_embeddings

    embeddings = generate_sample_embeddings(n_items=100, embedding_dim=64, n_clusters=4)

    # Use with clustering
    clusterer = HierarchicalSequentialClustering(
        similarity_threshold=0.6,
        min_cluster_size=20,
        min_modularity=0.5
    )
    clusters = clusterer.fit_predict(embeddings)
    clusterer.print_clusters()
