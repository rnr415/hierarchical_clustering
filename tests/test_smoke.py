import pytest

# networkit is an optional dependency that can be hard to install on some platforms (e.g., macOS); skip test if not available
pytest.importorskip("networkit")

import numpy as np

from vec2gc.vec2gc import HierarchicalSequentialClustering, generate_sample_embeddings


def test_smoke_clustering_runs():
    # Small synthetic run to ensure clustering code can be imported and executed quickly
    emb = generate_sample_embeddings(n_items=40, embedding_dim=32, n_clusters=2)

    clusterer = HierarchicalSequentialClustering(similarity_threshold=0.6, min_cluster_size=5, min_modularity=0.1)
    clusters = clusterer.fit_predict(emb, batch_size=128)

    assert isinstance(clusters, dict)
    # Ensure that at least one cluster was produced
    assert len(clusters) >= 1
    # final_clusters attribute should reflect the result
    assert clusterer.final_clusters == clusters
